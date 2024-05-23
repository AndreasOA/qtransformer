import torch
import torch.optim as optim
import torch.nn as nn
import random
from src.model import SimpleNet
from src.env import MetaworldEnvironment
import wandb


def test_model(cfg, model):
    # Instantiate the MetaWorldEnv class
    meta_env = MetaworldEnvironment(cfg.test.env_name, render=cfg.test.render)
    model.eval()

    # Reset the environment
    obs = meta_env.reset()
    print("Initial Observation:", obs)

    # Perform actions using the model
    for _ in range(cfg.test.steps):
        with torch.no_grad():
            action = model(obs.unsqueeze(0)).squeeze(0)

        reward, next_state, done, truncated = meta_env.forward(action)
        print(f"Reward: {reward.item()}, Done: {done.item()}, Truncated: {truncated.item()}")

        if done.item() or truncated.item():
            obs = meta_env.reset()
        else:
            obs = next_state