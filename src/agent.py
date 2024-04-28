import sys
from pathlib import Path

from numpy.lib.format import open_memmap

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange

from src.qt_new import QRoboticTransformer

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Iterator, Tuple, Union

from tqdm import tqdm
import numpy as np
import time
import wandb

# just force training on 64 bit systems

assert sys.maxsize > (2 ** 32), 'you need to be on 64 bit system to store > 2GB experience for your q-transformer agent'

# constants


# helpers

def exists(v):
    return v is not None

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# replay memory dataset


class ReplayMemoryDataset(Dataset):
    def __init__(self, file_path: str, num_timesteps: int = 1, binary_rewards: bool = False):
        assert num_timesteps >= 1, "Number of timesteps must be at least 1"
        self.num_timesteps = num_timesteps
        self.is_single_timestep = num_timesteps == 1

        # Validate the file path
        file_path = Path(file_path)
        assert file_path.exists() and file_path.is_file(), "The file path does not exist or is not a file"

        # Load data from npz file
        data = np.load(file_path, allow_pickle=True)
        self.states = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        if binary_rewards:
            self.rewards = (self.rewards == 10).astype(int)
        self.dones = data['dones']

        done_indices = np.where(self.dones == 1)[0]
        # Calculate differences between consecutive indices
        differences = np.diff(done_indices)
        if not np.all(differences == differences[0]):
            raise ValueError("Inconsistent episode lenghts")
        self.episode_length = differences

        assert len(self.dones) > 0, 'No trainable episodes'

        # Calculate the maximum episode length and number of episodes
        self.num_episodes = len(differences)
        self.max_episode_len = max(self.episode_length)

        # Create indices for data sampling
        timestep_arange = torch.arange(self.max_episode_len)
        timestep_indices = torch.stack(torch.meshgrid(
            torch.arange(self.num_episodes),
            timestep_arange
        ), dim=-1)

        trainable_mask = timestep_arange < (rearrange(torch.from_numpy(self.episode_length) - num_timesteps, 'e -> e 1'))
        self.indices = timestep_indices[trainable_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_index, timestep_index = self.indices[idx]
        start_index = episode_index*self.max_episode_len + timestep_index
        end_index = start_index + self.num_timesteps

        states = self.states[start_index: end_index].copy()
        actions = self.actions[start_index: end_index].copy()
        rewards = self.rewards[start_index: end_index].copy()
        dones = self.dones[start_index: end_index].copy()

        next_state = self.states[end_index : end_index + 1].copy()

        return states, actions, next_state, rewards, dones

# base environment class to extend

class BaseEnvironment(Module):
    @beartype
    def __init__(
        self,
        *,
        state_shape: Tuple[int, ...],
    ):
        super().__init__()
        self.state_shape = state_shape
        self.register_buffer('dummy', torch.zeros(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def init(self) -> Tensor: # (initial state)
        raise NotImplementedError

    def forward(
        self,
        actions: Tensor
    ) -> Tuple[
        TensorType[(), float],     # reward
        Tensor,                    # next state
        TensorType[(), bool]       # done
    ]:
        raise NotImplementedError

# agent class

class Agent(Module):
    @beartype
    def __init__(
        self,
        q_transformer: QRoboticTransformer,
        *,
        environment: BaseEnvironment,
        num_episodes: int = 1000,
        max_num_steps_per_episode: int = 10000,
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.001,
        num_steps_to_target_epsilon: int = 1000
    ):
        super().__init__()
        self.q_transformer = q_transformer

        self.environment = environment

        assert hasattr(environment, 'state_shape')

        assert 0. <= epsilon_start <= 1.
        assert 0. <= epsilon_end <= 1.
        assert epsilon_start >= epsilon_end

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_steps_to_target_epsilon = num_steps_to_target_epsilon
        self.epsilon_slope = (epsilon_end - epsilon_start) / num_steps_to_target_epsilon

        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode


        # states_path = mem_path / STATES_FILENAME
        #actions_path = "actions.npz"
        # rewards_path = mem_path / REWARDS_FILENAME
        # dones_path = mem_path / DONES_FILENAME

        prec_shape = (num_episodes, max_num_steps_per_episode)
        num_actions = q_transformer.num_actions
        state_shape = environment.state_shape

        # if condition_on_text:
        #     text_embeds_path = mem_path / TEXT_EMBEDS_FILENAME
        #     text_embed_shape = environment.text_embed_shape
        #     self.text_embed_shape = text_embed_shape
        #     self.text_embeds = open_memmap(str(text_embeds_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *text_embed_shape))

        # self.states      = open_memmap(str(states_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *state_shape))
        #self.actions     = open_memmap(str(actions_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, num_actions))
        # self.rewards     = open_memmap(str(rewards_path), dtype = 'float32', mode = 'w+', shape = prec_shape)
        # self.dones       = open_memmap(str(dones_path), dtype = 'bool', mode = 'w+', shape = prec_shape)

    def get_epsilon(self, step):
        return max(self.epsilon_end, self.epsilon_slope * float(step) + self.epsilon_start)

    @beartype
    @torch.no_grad()
    def forward(self):
        self.q_transformer.eval()

        for episode in range(self.num_episodes):
            print(f'episode {episode}')

            curr_state = self.environment.init()
            reward_one_cnt = 0
            for step_agent in tqdm(range(self.max_num_steps_per_episode)):
                last_step = step_agent == (self.max_num_steps_per_episode - 1)

                # epsilon = self.get_epsilon(step)


                actions = self.q_transformer.get_actions(
                    rearrange(curr_state, '... -> 1 ...'),
                    prob_random_action = 0.0 #epsilon
                )
                actions = (actions / 128) -1
                reward, next_state, done = self.environment(actions)
                if reward > 9:
                    reward_one_cnt +=1

               

                done = done | last_step

                # store memories using memmap, for later reflection and learning

                # if self.condition_on_text:
                #     assert text_embed.shape[1:] == self.text_embed_shape
                #     self.text_embeds[episode, step] = text_embed.cpu().numpy()

                # self.states[episode, step]      = curr_state.cpu().numpy()
                #self.actions[episode, step]     = actions.cpu().numpy()
                # self.rewards[episode, step]     = reward.cpu().numpy()
                # self.dones[episode, step]       = done.cpu().numpy()

                # if done, move onto next episode

                if done:
                    break

                # set next state

                curr_state = next_state

            # if self.condition_on_text:
            #     self.text_embeds.flush()

            # self.states.flush()
            #self.actions.flush()
            # self.rewards.flush()
            # self.dones.flush()

            wandb.log({"episode": episode, "target_achieved_agent": reward_one_cnt,"done_agent": int(done), "last_step_agent": int(last_step)})

        # close memmap

        # if self.condition_on_text:
        #     del self.text_embeds

        # del self.states
        #del self.actions
        # del self.rewards
        # del self.dones