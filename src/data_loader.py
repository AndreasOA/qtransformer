import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MetaworldDataset(Dataset):
    def __init__(self, file_path, discount_factor, binary_reward=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.discount_factor = discount_factor
        data = np.load(file_path)
        self.states = data['observations']
        self.next_states = data['next_observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        if binary_reward:
            self.rewards = np.where(self.rewards > 8.5, 1, 0)
        self.dones = data['dones']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Calculate the start index of the current episode
        episode_start_idx = idx - (idx % 200)
        
        # Calculate the time steps within the episode
        episode_length = idx - episode_start_idx + 1
        discount_factors = np.power(self.discount_factor, np.arange(episode_length))
         
        # Calculate the cumulative return from the episode start to the current idx with discounting
        cum_reward = torch.tensor([(self.rewards[episode_start_idx:idx + 1].T * discount_factors).sum()], dtype=torch.float32, device=self.device)   
        state = torch.tensor(self.states[idx][0], dtype=torch.float32, device=self.device, requires_grad=True)
        next_state = torch.tensor(self.next_states[idx][0], dtype=torch.float32, device=self.device, requires_grad=True)
        action = torch.tensor(self.actions[idx][0], dtype=torch.float32, device=self.device)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device)
        done = torch.tensor(self.dones[idx], dtype=torch.int, device=self.device)
        return state, action, reward, cum_reward, next_state, done

def get_dataloader(file_path, batch_size, discount_factor=0.98, shuffle=False, binary_reward=False):
    dataset = MetaworldDataset(file_path, discount_factor=discount_factor, binary_reward=binary_reward)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if "__main__" == __name__:
    dataloader = get_dataloader("metaworld/button-press-topdown-wall-v2.npz", 4, binary_reward=True)
    for states, actions, rewards, cum_reward, next_states, dones in dataloader:
        print(states.shape, actions.shape, rewards.shape, cum_reward.shape, next_states.shape, dones.shape)
        print("="*50)
        print(cum_reward)
#        print("="*50)
        print(rewards)
        print("="*50)
#        print(actions)