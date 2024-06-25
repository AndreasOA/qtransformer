import torch
import numpy as np
import metaworld
import torch
import time
import wandb
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
from src.train import train_model_simple, QLearner
from src.test import test_model
#from src.model import SimpleNet, QRoboticTransformer
#from src.model_qt import QRoboticTransformer as QTAdvanced
from src.model_repo import QRoboticTransformer

torch.manual_seed(0)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_wandb(self):
        wandb.login()
        wandb.init(project=self.config["wandb"]["model_name"], config=dict(self.config))

    def run(self):
        if self.config["wandb"]["use_wandb"]:
            self.setup_wandb()

        self.model = self.setup_model()
        self.q_learner = QLearner(self.model, self.config)

        if self.config["wandb"]["mode"] == "train":
            #self.q_learner.train_q_transformer()
            self.q_learner.train_model_qlearn_repo()
        if self.config["wandb"]["mode"] == "test":
            if self.config.test.load_model:
                self.model.load_state_dict(torch.load(self.config.test.model_path))
            test_model(self.config, self.model)
        
        wandb.finish()

    def setup_model(self):
        return QRoboticTransformer(state_dim=self.config.model.state_dim,
                                   num_actions=self.config.model.action_dim,  
                                   action_bins=self.config.model.action_bins, 
                                   depth=self.config.model.transformer_depth, 
                                   heads=self.config.model.heads,
                                   dim_head=self.config.model.dim_head, 
                                    ).to(self.device)



if __name__ == "__main__":
    with initialize(version_base=None, config_path="configs", job_name="qtransformer"):
        cfg = compose(config_name="config")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    pipeline = Pipeline(cfg)
    pipeline.run()