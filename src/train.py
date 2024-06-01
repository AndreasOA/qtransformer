import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from src.model import SimpleNet, QRoboticTransformer
from src.data_loader import get_dataloader
import wandb
from ema_pytorch import EMA
from einops import rearrange


def select_q_values(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')


def default(val, d):
    return val if val != "None" else d

class QLearner:
    def __init__(self, model, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.dataloader = get_dataloader(cfg.train.file, cfg.train.batch_size, cfg.train.shuffle)
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


    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f'Model saved to {self.save_path}')


    def q_learn(self, states, actions, rewards, next_states, dones):
        # 'next' stands for the very next time step (whether state, q, actions etc)
        not_terminal = (1 - dones).float()     # ~dones does not make sense imo replaced by 1 - dones

        # first make a prediction with online q robotic transformer
        # select out the q-values for the action that was taken
        q_pred_all_actions = self.model(states)
        q_pred = select_q_values(q_pred_all_actions, actions)

        # use an exponentially smoothed copy of model for the future q target. more stable than setting q_target to q_eval after each batch
        # the max Q value is taken as the optimal action is implicitly the one with the highest Q score

        q_next = self.ema_model(next_states).amax(dim = -1)
        q_next = torch.clamp(q_next, min = default(self.monte_carlo_return, -1e4))

        # Bellman's equation. most important line of code, hopefully done correctly

        q_target = rewards + not_terminal * (self.discount_factor_gamma * q_next)

        # now just force the online model to be able to predict this target

        loss = F.mse_loss(q_pred, q_target)

        # that's it. ~5 loc for the heart of q-learning
        # return loss and some of the intermediates for logging

        return loss, q_pred_all_actions, q_pred, q_next, q_target


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


    def train_model_qlearn(self):
        self.model.train()
        self.ema_model.train()
        if self.cfg.wandb.use_wandb:
            wandb.watch(self.model, self.ema_model, log="all", log_freq=10)

        for epoch in range(self.cfg.train.epochs):
            for i, (states, actions, rewards, next_states, dones) in enumerate(self.dataloader):    

                # zero grads
                self.optimizer.zero_grad()
                # main q-learning algorithm
                actions = self.model.discretize_actions(actions)
                loss, (td_loss, conservative_reg_loss) = self.learn(states, actions, rewards, next_states, dones)
                loss.backward()

                # clip gradients (transformer best practices)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # take optimizer step
                self.optimizer.step()

                # update target ema
                self.ema_model.update()


                if (i+1) % 100 == 0:  # Print every 100 batches
                    print(f'Epoch [{epoch+1}/{self.cfg.train.epochs}], Step [{i+1}/{len(self.dataloader)}], TDLoss: {loss.item():.4f} Conservative Loss: {conservative_reg_loss.item():.4f}')
                if self.cfg.wandb.use_wandb:
                    wandb.log({"epoch": epoch, "loss": loss,"td loss": td_loss.item(), "conservative reg loss": conservative_reg_loss.item()})

        if self.cfg.train.save_model:
            self.save_model()

        print('training complete')        


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


