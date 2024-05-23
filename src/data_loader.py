import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MetaworldDataset(Dataset):
    def __init__(self, file_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = np.load(file_path)
        self.states = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx][0], dtype=torch.float32, device=self.device)
        action = torch.tensor(self.actions[idx][0], dtype=torch.float32, device=self.device)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device)
        return state, action, reward

def get_dataloader(file_path, batch_size, shuffle=False):
    dataset = MetaworldDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if "__main__" == __name__:
    dataloader = get_dataloader("metaworld/button-press-topdown-wall-v2.npz", 4)
    for states, actions, rewards in dataloader:
        print(states.shape, actions.shape, rewards.shape)
#        print("="*50)
#        print(states)
#        print("="*50)
#        print(rewards)
#        print("="*50)
#        print(actions)