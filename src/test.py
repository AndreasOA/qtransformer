import torch
import torch.optim as optim
import torch.nn as nn
import random
from src.model import SimpleNet
from src.env import MetaworldEnvironment
import wandb


def test_model(cfg, model, mode="test", env_name=""):
    if env_name == "":
        env_name = cfg.test.env_name
    # Instantiate the MetaWorldEnv class
    if mode == "train":
        meta_env = MetaworldEnvironment(env_name)
    else:
        meta_env = MetaworldEnvironment(env_name, render=cfg.test.render)
    model.eval()

    # Reset the environment
    obs = meta_env.reset()
    if mode != "train":
        print("Initial Observation:", obs)

    # Perform actions using the model
    rewards = []
    for episode in range(cfg.test.episodes):
        reward_episode = []
        obs = meta_env.reset()
        if mode != "train":
            print("Initial Observation:", obs)
        for _ in range(cfg.test.steps):
            with torch.no_grad():
                action = model.get_optimal_actions(obs.unsqueeze(0).unsqueeze(1)).squeeze(0)
                action = model.undiscretize_actions(action)
            reward, next_state, done, truncated = meta_env.forward(action)
            rewards.append(reward)
            reward_episode.append(reward)
            if mode != "train":
                print(f"Reward: {reward.item()}, Done: {done.item()}, Truncated: {truncated.item()}, Action: {action}")

            if done.item() or truncated.item():
                obs = meta_env.reset()
            else:
                obs = next_state
        print("Episode Mean reward for ", env_name, ": ", torch.tensor(reward_episode, dtype=torch.float32).mean())

    rewards_mean = torch.tensor(rewards, dtype=torch.float32).mean()
    print("Mean Reward:", rewards_mean)
    return rewards_mean