from random import randrange

import torch
from torch.utils.data import Dataset
import cv2
from beartype.typing import Tuple, Optional
import metaworld
from torchtyping import TensorType
from src.agent import BaseEnvironment


class MetaworldEnvironment(BaseEnvironment):
    def __init__(self, env_name='door-open-v2', render=False, action_bins=256, state_shape=(1, 39)):
        super().__init__(state_shape=state_shape)
        # Load MetaWorld Env
        self.ml1 = metaworld.ML1(env_name)
        self.env = self.ml1.train_classes[env_name]()
        if render:
            self.env.render_mode = "human"
        task = self.ml1.train_tasks[0]
        self.env.set_task(task)
        self.action_bins = action_bins
        # Reset environment to its initial state
        self.reset()

    def init(self):
        return self.reset()

    def reset(self):
        return torch.tensor(self.env.reset()[0], device=self.device, dtype=torch.float32)

    def norm_action_bins(self, actions):
        return (actions/(self.action_bins/2)) - 1

    def forward(self, action):
        action = action.cpu().numpy()  # Assuming action needs to be a numpy array for MetaWorld
        next_state, reward, done, truncated, _ = self.env.step(self.norm_action_bins(action[0]))
        self.env.render()
        if done:
            print("hallöö")
        return torch.tensor(reward, device=self.device, dtype=torch.float32), torch.tensor(next_state, device=self.device, dtype=torch.float32), torch.tensor(done, device=self.device, dtype=torch.bool)

    @property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MockEnvironment(BaseEnvironment):
    def init(self) -> Tuple[
        Optional[str],
        TensorType[float]
    ]:
        return 'please clean the kitchen', torch.randn(self.state_shape, device = self.device)

    def forward(self, actions) -> Tuple[
        TensorType[(), float],
        TensorType[float],
        TensorType[(), bool]
    ]:
        rewards = torch.randn((), device = self.device)
        next_states = torch.randn(self.state_shape, device = self.device)
        done = torch.zeros((), device = self.device, dtype = torch.bool)

        return rewards, next_states, done