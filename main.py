import torch
import numpy as np
import metaworld
import torch
import gym
import matplotlib.pyplot as plt
import time
import wandb
import os
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.qt_new import QRoboticTransformer
from src.q_learner import QLearner
from src.agent import Agent
from src.agent import ReplayMemoryDataset
from src.mocks import MetaworldEnvironment

PIPELINE = None

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_wandb(self):
        wandb.login()
        wandb.init(project=self.config["model_name"], config=dict(self.config["model_config"]))

    def setup_qlearner(self):
        config = self.config["training_config"]
        if config["test"]:
            agent = self.setup_agent()
        else:
            agent = None

        self.q_learner = QLearner(
            self.model,
            dataset = ReplayMemoryDataset(config["dataset"], binary_rewards=config["binary_rewards"]),
            num_train_steps = config["num_train_steps"],
            learning_rate = config["learning_rate"],
            batch_size = config["batch_size"],
            grad_accum_every = config["grad_accum_every"],
            test = config["test"],
            agent = agent,
            conservative_reg_loss_weight = config["conservative_reg_loss_weight"],
            test_every= config["test_every"]
        )
        self.q_learner = self.q_learner.to(self.device)

    def setup_model(self):
        config = self.config["model_config"]
        self.model = QRoboticTransformer(
            state_dim= config["state_dim"],
            num_actions = config["num_actions"],
            action_bins = config["action_bins"],
            depth = config["depth"],
            heads = config["heads"],
            dim_head = config["dim_head"],
            dueling = config["dueling"]
        )

    def train_model(self):
        self.setup_qlearner()
        self.q_learner()

        if self.config["training_config"]["save_model"]:
            torch.save(self.model.state_dict(), self.config["model_name"] + ".pth")


    def setup_agent(self):
        config = self.config["test_config"]
        if config["load_model"]:
            self.model.load_state_dict(torch.load(self.config["model_name"] + ".pth"))
            self.model.to(self.device)

        self.model.eval()
        self.model.embedding_layer.eval()
        
        env = MetaworldEnvironment(
            env_name = config["env_name"],
            render= config["render"]
        )
        env.to(self.device)
        agent = Agent(
            self.model,
            environment = env,
            num_episodes = config["num_episodes"],
            max_num_steps_per_episode = config["max_num_steps_per_episode"],
        )
        agent.to(self.device)

        return agent

    def test_model(self):
        agent = self.setup_agent()
        agent()

    def run(self):
        if self.config["use_wandb"]:
            self.setup_wandb()
        self.setup_model()

        if self.config["mode"] == "train":
            self.train_model()
        elif self.config["mode"] == "test":
            self.test_model()
        else:
            self.train_model()
            self.test_model()
        
        wandb.finish()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name="qtransformer"):
        cfg = compose(config_name="config")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    pipeline = Pipeline(cfg)
    pipeline.run()
