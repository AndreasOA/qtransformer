import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import os

class MetaworldDataset(Dataset):
    def __init__(self, files, discount_factor, context_length, binary_reward, rescale_reward):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.discount_factor = discount_factor
        self.context_length = context_length
        self.states, self.next_states, self.actions, self.rewards, self.dones = [], [], [], [], []

        # Load data from multiple files
        for file_path in files:
            data = np.load(file_path)
            self.states.append(data['observations'])
            self.next_states.append(data['next_observations'])
            self.actions.append(data['actions'])
            self.rewards.append(data['rewards'])
            self.dones.append(data['dones'])

        # Concatenate data from all files
        self.states = np.concatenate(self.states, axis=0)
        self.next_states = np.concatenate(self.next_states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.dones = np.concatenate(self.dones, axis=0)

        if binary_reward:
            #self.rewards = np.where(self.rewards == 0, -0.5, self.rewards)
            self.rewards = np.where(self.rewards > 5, 1, 0)
            #self.rewards = np.where((self.rewards != 1) & (self.rewards != -0.5), 0, self.rewards)
            
        if rescale_reward:
            self.rewards /= 10

        done_indices = np.where(self.dones == 1)[0]
        differences = np.diff(done_indices)
        if not np.all(differences == differences[0]):
            raise ValueError("Inconsistent episode lenghts")
        self.episode_length = differences
        self.num_episodes = len(differences)
        self.max_episode_len = max(self.episode_length)


    def __len__(self):
        return len(self.states)


    def __getitem__(self, idx):
        # Calculate the start index of the current episode
        start_id = idx
        episode_context_spacer = (idx % self.max_episode_len) - (self.max_episode_len - self.context_length)
        if episode_context_spacer > 0:
            start_id -= episode_context_spacer
        end_id = start_id + self.context_length
        #episode_start_idx = start_id - (start_id % 200)
        
        # Calculate the time steps within the episode
        #episode_length = idx - episode_start_idx + 1
        discount_factors = np.power(self.discount_factor, np.arange(self.context_length))
         
        # Calculate the cumulative return from the episode start to the current idx with discounting
        cum_reward = torch.tensor([(self.rewards[start_id: end_id].T * discount_factors).sum()], dtype=torch.float32, device=self.device)   
        state = torch.tensor(self.states[start_id: end_id], dtype=torch.float32, requires_grad=True).squeeze()
        next_state = torch.tensor(self.next_states[start_id: end_id], dtype=torch.float32, requires_grad=True).squeeze()
        action = torch.tensor(self.actions[end_id - 1], dtype=torch.float32).squeeze()
        reward = torch.tensor(self.rewards[end_id - 1], dtype=torch.float32)
        #reward = reward.mean(dim=0)
        done = torch.tensor(self.dones[end_id - 1], dtype=torch.int)
        return state, action, reward, next_state, done

def get_dataloader(file_path, batch_size, discount_factor=0.98, context_length=10, shuffle=True, binary_reward=False, rescale_reward=False):
    dataset = MetaworldDataset(file_path, discount_factor=discount_factor, context_length=context_length, binary_reward=binary_reward, rescale_reward=rescale_reward)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if "__main__" == __name__:
    dataloader = get_dataloader("metaworld/", 4, binary_reward=True)
    for states, actions, rewards, next_states, dones in dataloader:
        print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        print("="*50)
#        print("="*50)
        print(rewards)
        print("="*50)
#        print(actions)