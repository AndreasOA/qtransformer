import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MetaworldDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.states = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return state, action, reward

def get_dataloader(file_path, batch_size):
    dataset = MetaworldDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader