import torch
import numpy as np
import metaworld
import torch
import time
import wandb
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.train import train_model
from src.test import test_model
from src.model import SimpleNet, QRoboticTransformer

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_wandb(self):
        wandb.login()
        wandb.init(project=self.config["wandb"]["model_name"])

    def run(self):
        if self.config["wandb"]["use_wandb"]:
            self.setup_wandb()

        self.model = self.setup_model()

        if self.config["wandb"]["mode"] == "train":
            train_model(self.config, self.model)
        if self.config["wandb"]["mode"] == "test":
            if self.config.test.load_model:
                self.model.load_state_dict(torch.load(self.config.test.model_path))
            test_model(self.config, self.model)
        
        wandb.finish()

    def setup_model(self):
        return QRoboticTransformer()



if __name__ == "__main__":
    with initialize(version_base=None, config_path="configs", job_name="qtransformer"):
        cfg = compose(config_name="config")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    pipeline = Pipeline(cfg)
    pipeline.run()