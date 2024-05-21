from pathlib import Path
from functools import partial
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

from torchtyping import TensorType

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Optional, Union, List, Tuple

from src.qt_new import QRoboticTransformer

from src.optimizer import get_adam_optimizer

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

import wandb

# constants

QIntermediates = namedtuple('QIntermediates', [
    'q_pred_all_actions',
    'q_pred',
    'q_next',
    'q_target'
])

Losses = namedtuple('Losses', [
    'td_loss',
    'conservative_reg_loss'
])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_divisible(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# tensor helpers

def batch_select_indices(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    indices = indices.type(torch.int64)
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')

# Q learning on robotic transformer

class QLearner(Module):

    @beartype
    def __init__(
        self,
        model: Union[QRoboticTransformer, Module],
        *,
        dataset: Dataset,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float,
        min_reward: float = 0.,
        grad_accum_every: int = 1,
        monte_carlo_return: Optional[float] = None,
        weight_decay: float = 0.,
        accelerator: Optional[Accelerator] = None,
        accelerator_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(
            shuffle = False
        ),
        q_target_ema_kwargs: dict = dict(
            beta = 0.5,
            update_after_step = 1,
            update_every = 1
        ),
        max_grad_norm = 0.5,
        n_step_q_learning = False,
        discount_factor_gamma = 0.98,
        conservative_reg_loss_weight: float = 1., # they claim 1. is best in paper
        optimizer_kwargs: dict = dict(),
        test: bool = False,
        test_every: int = 1000,
        agent = None
    ):
        super().__init__()

        self.is_multiple_actions = model.num_actions > 1

        # q-learning related hyperparameters

        self.discount_factor_gamma = discount_factor_gamma
        self.n_step_q_learning = n_step_q_learning

        self.has_conservative_reg_loss = conservative_reg_loss_weight > 0.
        self.conservative_reg_loss_weight = conservative_reg_loss_weight

        self.register_buffer('discount_matrix', None, persistent = False)

        # online (evaluated) Q model

        self.model = model
        
        self.test = test
        self.test_every = test_every
        self.agent = agent


        # ema (target) Q model

        self.ema_model = EMA(
            model,
            include_online_model = False,
            **q_target_ema_kwargs
        )

        self.max_grad_norm = max_grad_norm

        self.optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            **optimizer_kwargs
        )

        if not exists(accelerator):
            accelerator = Accelerator(
                kwargs_handlers = [
                    DistributedDataParallelKwargs(find_unused_parameters = True)
                ],
                **accelerator_kwargs
            )

        self.accelerator = accelerator

        self.min_reward = min_reward
        self.monte_carlo_return = monte_carlo_return

        self.dataloader = DataLoader(dataset, batch_size = batch_size,**dataloader_kwargs)
        (
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        )

        # checkpointing related

        # dummy loss

        self.register_buffer('zero', torch.tensor(0.))

        # training step related

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        self.register_buffer('step', torch.tensor(0))


    def wait(self):
        return self.accelerator.wait_for_everyone()

    def autoregressive_q_learn_handle_single_timestep(
        self,
        states,
        actions,
        next_states,
        rewards,
        dones,
        *,
        monte_carlo_return = None
    ):
        """
        simply detect and handle single timestep
        and use `autoregressive_q_learn` as more general function
        """
        print("Actions autoregress handler", actions)
        if states.ndim == 4:
            states = states[0]

        if next_states.ndim == 4:
            next_states = next_states[0]

        if actions.ndim == 4:
            actions =actions[0]

        return self.autoregressive_q_learn(states, actions, next_states, rewards, dones, monte_carlo_return = monte_carlo_return)

    def autoregressive_q_learn(
        self,
        states:         TensorType['b', 't', 'c', 'f', 'h', 'w', float],
        actions:        TensorType['b', 't', 'n', int],
        next_states:    TensorType['b', 'c', 'f', 'h', 'w', float],
        rewards:        TensorType['b', 't', float],
        dones:          TensorType['b', 't', bool],
        *,
        monte_carlo_return = None

    ) -> Tuple[TensorType[()], QIntermediates]:
        """
        einops

        b - batch
        c - channels
        f - frames
        h - height
        w - width
        t - timesteps
        n - number of actions
        a - action bins
        q - q values
        d - text cond dimension
        """
        monte_carlo_return = default(monte_carlo_return, -1e4)
        num_timesteps, device = states.shape[1], states.device

        # fold time steps into batch
        print("Actions autoregress q learner", actions) 
        states, time_ps = pack_one(states, '* f')
        actions, _ = pack_one(actions.detach(), '* n')
        print("Actions autoregress q learner", actions)

        # repeat text embeds per timestep
        # anything after the first done flag will be considered terminal

        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # rewards should not be given on and after terminal step

        rewards = rewards * not_terminal

        # because greek unicode is nice to look at

        γ = self.discount_factor_gamma

        # get predicted Q for each action

        q_pred_all_actions = self.model(states, actions = actions)
        print("PRED:", q_pred_all_actions.argmax(dim = -1))
        q_pred = batch_select_indices(q_pred_all_actions, actions)
        q_pred = unpack_one(q_pred, time_ps, '* n')
        
        # get q_next

        q_next = self.ema_model(next_states)
        q_next = q_next.max(dim = -1).values
        q_next.clamp_(min = monte_carlo_return)

        # get target Q

        q_target_all_actions = self.ema_model(states, actions = actions).detach()
        #one_hot_actions = torch.zeros_like(q_target_all_actions)
        #actions = actions.type(torch.int64)
        #q_target_all_actions = one_hot_actions.scatter_(2, actions.unsqueeze(-1), 1)
        print("TARGET:", q_target_all_actions.argmax(dim = -1))
        q_target = q_target_all_actions.max(dim = -1).values
        q_target.clamp_(min = monte_carlo_return)

        # main contribution of the paper is the following logic
        # section 4.1 - eq. 1

        # first take care of the loss for all actions except for the very last one

        q_pred_rest_actions, q_pred_last_action      = q_pred[..., :-1], q_pred[..., -1]
        q_target_first_action, q_target_rest_actions = q_target[..., 0], q_target[..., 1:]

        losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions, reduction = 'none')

        # next take care of the very last action, which incorporates the rewards

        #q_target_last_action, _ = pack([q_target_first_action, q_next], 'b *')
        q_target_last_action = rewards + γ * q_next[0]
        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action, reduction = 'none')

        # flatten and average

        losses, _ = pack([losses_all_actions_but_last, losses_last_action], '*')

        return losses.mean(), QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def learn(
        self,
        *args,
        min_reward: Optional[float] = None,
        monte_carlo_return: Optional[float] = None
    ):
        states, actions, next_state, rewards, dones = args
        print("True Action: ", actions)
        scaled_tensor = (actions + 1) / 2 * 1024
        actions = scaled_tensor.floor().to(torch.int)
        print("True Actions converted: ", actions)

        # q-learn kwargs

        q_learn_kwargs = dict(
            monte_carlo_return = monte_carlo_return
        )

        # main q-learning loss, respectively
        # proposed autoregressive q-learning for multiple actions - (handles single or n-step automatically)

        td_loss, q_intermediates = self.autoregressive_q_learn_handle_single_timestep(states, actions, next_state, rewards, dones, **q_learn_kwargs)
        num_timesteps = actions.shape[1]

        if not self.has_conservative_reg_loss:
            return loss, Losses(td_loss, self.zero)

        # calculate conservative regularization
        # section 4.2 in paper, eq 2

        batch = actions.shape[0]

        q_preds = q_intermediates.q_pred_all_actions
        q_preds = rearrange(q_preds, '... a -> (...) a')

        num_action_bins = q_preds.shape[-1]
        num_non_dataset_actions = num_action_bins - 1

        actions = rearrange(actions, '... -> (...) 1')
        actions = actions.type(torch.int64)
        dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))

        q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
        q_actions_not_taken = rearrange(q_actions_not_taken, '(b t a) -> b t a', b = batch, a = num_non_dataset_actions)

        conservative_reg_loss = ((q_actions_not_taken - (min_reward * num_timesteps)) ** 2).sum() / num_non_dataset_actions

        # total loss

        loss =  0.5 * td_loss + \
                0.5 * conservative_reg_loss * self.conservative_reg_loss_weight

        loss_breakdown = Losses(td_loss, conservative_reg_loss)

        return loss, loss_breakdown

    def forward(
        self,
        *,
        monte_carlo_return: Optional[float] = None,
        min_reward: Optional[float] = None
    ):
        monte_carlo_return = default(monte_carlo_return, self.monte_carlo_return)
        min_reward = default(min_reward, self.min_reward)

        step = self.step.item()

        replay_buffer_iter = cycle(self.dataloader)

        self.model.train()
        self.model.embedding_layer.train()
        self.ema_model.train()

        wandb.watch(self.model, self.ema_model, log="all", log_freq=10)

        while step < self.num_train_steps:

            # zero grads

            self.optimizer.zero_grad()

            # main q-learning algorithm

            for grad_accum_step in range(self.grad_accum_every):
                is_last = grad_accum_step == (self.grad_accum_every - 1)
                context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                with self.accelerator.autocast(), context():

                    loss, (td_loss, conservative_reg_loss) = self.learn(
                        *next(replay_buffer_iter),
                        min_reward = min_reward,
                        monte_carlo_return = monte_carlo_return
                    )

                    self.accelerator.backward(loss / self.grad_accum_every)

            if (step + 1) % 50 == 0:
                print(f'Step: {step}, loss: {loss:.3f}, td loss: {td_loss.item():.3f}, conservative reg loss: {conservative_reg_loss.item():.3f}')

            # clip gradients (transformer best practices)

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # take optimizer step

            self.optimizer.step()
            
            # update target ema

            self.wait()

            self.ema_model.update()

            # increment step
            wandb.log({"epoch": step, "loss": loss,"td loss": td_loss.item(), "conservative reg loss": conservative_reg_loss.item()})
            step += 1
            self.step.add_(1)

            # whether to checkpoint or not

            self.wait()

            if self.test and self.agent is not None and (step + 1) % self.test_every == 0:
                self.agent()

            

        print('training complete')
