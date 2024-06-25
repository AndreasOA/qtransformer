import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from functools import partial
from contextlib import nullcontext
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.model import SimpleNet, QRoboticTransformer
from src.data_loader import get_dataloader
import wandb
from ema_pytorch import EMA
from einops import rearrange, pack
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from src.test import test_model


def select_q_values(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')


def default(val, d):
    return val if val != "None" else d

def exists(val):
    return val is not None

class QLearner:
    def __init__(self, model, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        if cfg.train.train_single_task:
            self.files = [cfg.train.folder_path + cfg.train.file_name]
        else:
            self.files = [os.path.join(cfg.train.folder_path, f) for f in os.listdir(cfg.train.folder_path) if f.endswith('.npz')][:cfg.train.num_files]
        self.dataloader = get_dataloader(self.files,
                                         cfg.train.batch_size, 
                                         cfg.train.discount_factor_gamma,
                                         cfg.train.context_length,
                                         cfg.train.shuffle, 
                                         cfg.train.binary_reward,
                                         cfg.train.rescale_reward
                                        )
        self.epochs = cfg.train.epochs
        self.save_path = cfg.train.model_path
        self.model = model
        self.ema_model = self.init_ema_model()
        self.discount_factor_gamma = cfg.train.discount_factor_gamma
        self.conservative_reg_loss_weight = cfg.train.conservative_reg_loss_weight
        self.min_reward = cfg.train.min_reward
        self.optimizer = self.init_optimizer()
        self.max_grad_norm = cfg.train.max_grad_norm
        self.monte_carlo_return = cfg.train.monte_carlo_return
        self.grad_accum_every = cfg.train.grad_accum_every
        self.accelerator = Accelerator(kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters = True)])
        self.initial_lr = cfg.model.learning_rate
        self.num_steps = len(self.dataloader) * self.epochs
        self.scheduler = self.init_scheduler()

    def init_scheduler(self):
        # Warm-up for the first 1000 steps, then cosine annealing
        warmup_steps = 1000
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_steps)
        
        def combined_scheduler(step):
            if step < warmup_steps:
                scheduler.step()
            else:
                cosine_scheduler.step()
        
        return combined_scheduler


    def init_ema_model(self):
        return EMA(
            self.model,
            include_online_model = False,
            beta = self.cfg.train.ema_beta,
            update_after_step = self.cfg.train.ema_update_after_step,
            update_every = self.cfg.train.ema_update_every
        )


    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay)


    def save_model(self, name=None):
        if name is None:
            name = self.save_path
        torch.save(self.model.state_dict(), name)
        print(f'Model saved to {self.save_path}')

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def q_learn(self, states, actions, rewards, next_states, dones):
        not_terminal = (1 - dones).float()
        # rewards should not be given on and after terminal step
        rewards = rewards * not_terminal

        # first make a prediction with online q robotic transformer
        # select out the q-values for the action that was taken
        q_pred_all_actions = self.model(states, actions=actions)
        q_pred = select_q_values(q_pred_all_actions, actions)

        # get q_next

        q_next = self.ema_model(next_states)
        q_next = q_next.max(dim = -1).values
        q_next = torch.clamp(q_next, min = default(self.monte_carlo_return, -1e4))

        # get target Q
        # unpack back to - (b, t, n)

        q_target_all_actions = self.ema_model(states, actions = actions)
        q_target = q_target_all_actions.max(dim = -1).values
        q_target = torch.clamp(q_target, min = default(self.monte_carlo_return, -1e4))

        # main contribution of the paper is the following logic
        # section 4.1 - eq. 1

        # first take care of the loss for all actions except for the very last one

        q_pred_rest_actions, q_pred_last_action      = q_pred[:, :-1], q_pred[:, -1]
        q_target_first_action, q_target_rest_actions = q_target[:, 0], q_target[:, 1:]
        #q_target_rest_actions = torch.max(q_target_rest_actions, rewards)

        losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions, reduction = 'none')

        # next take care of the very last action, which incorporates the rewards
        #q_target_last_action, _ = pack([q_target_first_action, q_next], 'b *')

        q_target_last_action = rewards.squeeze() + self.discount_factor_gamma * q_next[:, 0]
        #q_target_last_action = torch.max(q_target_last_action, rewards)

        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action, reduction = 'none')

        # flatten and average
        losses, _ = pack([losses_all_actions_but_last, losses_last_action], '*')

        return losses.mean(), q_pred_all_actions, q_pred, q_next, q_target


    def learn(self, states, actions, rewards, next_states, dones):
        td_loss, q_pred_all_actions, q_pred, q_next, q_target = self.q_learn(states, actions, rewards, next_states, dones)

        # calculate conservative regularization
        # section 4.2 in paper, eq 2

        batch = actions.shape[0]

        q_preds = q_pred_all_actions
        num_non_dataset_actions = q_preds.shape[-1] - 1
        actions = rearrange(actions, '... -> ... 1')
        dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))

        q_actions_not_taken = q_preds[(1 - dataset_action_mask).bool()]
        #q_actions_not_taken = rearrange(q_actions_not_taken, '(b t a) -> b t a', b = batch, a = num_non_dataset_actions)

        # Min Reward in the paper is 0, formula (2s)
        conservative_reg_loss = ((q_actions_not_taken - self.min_reward) ** 2).sum() / num_non_dataset_actions

        # total loss

        loss =  0.5 * td_loss + 0.5 * conservative_reg_loss * self.conservative_reg_loss_weight

        return loss, (td_loss, conservative_reg_loss)


    # def train_model_qlearn(self):
    #     self.model.train()
    #     self.ema_model.train()
    #     if self.cfg.wandb.use_wandb:
    #         wandb.watch(self.model, self.ema_model, log="all", log_freq=10)

    #     for epoch in range(self.cfg.train.epochs):
    #         for i, (states, actions, rewards, next_states, dones) in enumerate(self.dataloader):    
    #             states = states.to(self.device)
    #             actions = actions.to(self.device)
    #             rewards = rewards.to(self.device)
    #             next_states = next_states.to(self.device)
    #             dones = dones.to(self.device)
    #             # zero grads
    #             self.optimizer.zero_grad()
    #             # main q-learning algorithm
    #             with torch.no_grad():
    #                 actions = self.model.discretize_actions(actions)
    #                 #actions = actions[:, -1]
    #             loss, (td_loss, conservative_reg_loss) = self.learn(states, actions, rewards, next_states, dones)
    #             loss.backward()

    #             # clip gradients (transformer best practices)
    #             nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    #             # take optimizer step
    #             self.optimizer.step()

    #             # update target ema
    #             self.ema_model.update()


    #             if (i+1) % 100 == 0:  # Print every 100 batches
    #                 print(f'Epoch [{epoch+1}/{self.cfg.train.epochs}], Step [{i+1}/{len(self.dataloader)}], TDLoss: {td_loss.item():.4f} Conservative Loss: {conservative_reg_loss.item():.4f}')
    #             if self.cfg.wandb.use_wandb:
    #                 wandb.log({"epoch": epoch, "loss": loss,"td loss": td_loss.item(), "conservative reg loss": conservative_reg_loss.item()})

    #             if self.cfg.train.save_model and (i + 1) % 200 == 0:
    #                 self.save_model(f"trafo_cp_{epoch}_{i}.pth")

    #     if self.cfg.train.save_model:
    #         self.save_model()

    #     print('training complete')


    def train_model_qlearn_repo(self):
        self.model.train()
        self.ema_model.train()
        if self.cfg.wandb.use_wandb:
            wandb.watch(self.model, self.ema_model, log="all", log_freq=10)

        for epoch in range(self.cfg.train.epochs):
            for i, (states, actions, rewards, next_states, dones) in enumerate(self.dataloader):    
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                with torch.no_grad():
                    actions = self.model.discretize_actions(actions)

                # zero grads
                self.optimizer.zero_grad()

                # main q-learning algorithm
                for grad_accum_step in range(self.grad_accum_every):
                    is_last = grad_accum_step == (self.grad_accum_every - 1)
                    context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                    with self.accelerator.autocast(), context():

                        loss, (td_loss, conservative_reg_loss) = self.learn(states, actions, rewards, next_states, dones)
                        self.accelerator.backward(loss / self.grad_accum_every)


                if (i+1) % 100 == 0:  # Print every 100 batches
                    print(f'Epoch [{epoch+1}/{self.cfg.train.epochs}], Step [{i+1}/{len(self.dataloader)}], TDLoss: {td_loss.item():.4f} Conservative Loss: {conservative_reg_loss.item():.4f}')
                if self.cfg.wandb.use_wandb:
                    wandb.log({"epoch": epoch, "loss": loss,"td loss": td_loss.item(), "conservative reg loss": conservative_reg_loss.item()})


                # clip gradients (transformer best practices)

                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # take optimizer step

                self.optimizer.step()

                # update target ema

                self.wait()

                self.ema_model.update()

                self.wait()
                self.scheduler(i + len(self.dataloader)*epoch)

                if self.cfg.train.save_model and (i + 1) % 200 == 0:
                    file_reward = []
                    for file in self.files:
                        file = file.split("/")[1].split(".")[0]
                        file_reward.append(test_model(self.cfg, self.model, mode="train", env_name=file))
                    self.model.train()
                    self.save_model(f"trafo_cp_{epoch}_{i}_{torch.tensor(file_reward).mean():.3f}.pth")

        if self.cfg.train.save_model:
            self.save_model()

        print('training complete')

    ##############################################################################################################
    # Simplified Approach

    def conservative_q_learning_loss(self, q_values, next_q_values, rewards, actions, dones):
        q_next = next_q_values.amax(dim=-1)
        q_pred = select_q_values(q_values, actions)
        bellman_errors = rewards + self.discount_factor_gamma * q_next * (1 - dones) - q_pred
        td_loss = bellman_errors.pow(2).mean()

        # Conservative loss
        conservative_loss = (q_values.pow(2).mean() * self.conservative_reg_loss_weight)

        return td_loss, conservative_loss


    def train_q_transformer(self):
        self.model.train()

        if self.cfg.wandb.use_wandb:
            wandb.watch(self.model, log="all", log_freq=10)

        for epoch in range(self.epochs):
            for i, (states, actions, rewards, cum_rewards, next_states, dones) in enumerate(self.dataloader):
                
                self.optimizer.zero_grad()
                q_values = self.model(states)

                with torch.no_grad():
                    actions = self.model.discretize_actions(actions)
                    next_q_values = self.model(next_states)

                td_loss, conservative_loss = self.conservative_q_learning_loss(q_values, next_q_values, cum_rewards, actions, dones)
                loss = td_loss + conservative_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                if (i+1) % 100 == 0:  # Print every 100 batches
                    print(f'Epoch [{epoch+1}/{self.cfg.train.epochs}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f} Conservative Loss: {conservative_loss.item():.4f}')
                if self.cfg.wandb.use_wandb:
                    wandb.log({"epoch": epoch, "loss": loss,"td loss": td_loss.item(), "conservative reg loss": conservative_loss.item(), "reward": rewards.mean()})

        if self.cfg.train.save_model:
            self.save_model()

        print('training complete')        


##############################################################################################################################      


def train_model_simple(cfg, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    model.train()
    # Training loop
    for epoch in range(cfg.train.epochs):
        for i, (states, actions, rewards) in enumerate(dataloader):          
            # Forward pass
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print('Training complete')
    save_model(model, cfg.train.model_path)
    return model


