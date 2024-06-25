# MetaWorld Project

This project demonstrates how to train and test a neural network model using the MetaWorld environment. The project includes scripts for defining, training, testing the model, and interacting with the MetaWorld environment.

## Table of Contents
1. [Data Download](#data-download)
2. [Environment Setup](#environment-setup)
3. [WANDB Configuration](#wandb-configuration)
4. [Training the Model](#training-the-model)
5. [Testing the Model](#testing-the-model)

## Data Download
To download the MetaWorld dataset, use the following `wget` command:

```bash
wget --recursive --no-parent --no-host-directories --cut-dirs=2 -R "index.html*" https://ml.jku.at/research/l2m/metaworld
```

This command will download the dataset recursively while excluding `index.html` files.

## Environment Setup
Ensure you have the necessary dependencies installed. You can use `pip` to install the required packages:
```bash
pip install torch metaworld gym wandb
```

## Training the Model

To begin training the model, ensure all configurations are set in the config.yaml file. This file allows you to adjust the training parameters such as learning rate, batch size, and number of epochs. Here's an example configuration:

```yaml
defaults:
  - _self_

wandb:
  model_name: qtransformer
  mode: train
  use_wandb: True

model:
  state_dim: 39
  action_dim: 4
  action_bins: 1024
  transformer_depth: 6
  heads: 8
  dim_head: 64
  dropout: 0.15
  learning_rate: 0.00075
  weight_decay: 0.1

train:
  save_model: True
  batch_size: 48
  context_length: 20
  epochs: 5
  shuffle: True
  folder_path: "metaworld/"
  train_single_task: False
  file_name: "button-press-v2.npz"
  num_files: 10
  model_path: "trafo.pth"
  ema_beta: 0.99
  ema_update_after_step: 10
  ema_update_every: 5
  discount_factor_gamma: 0.98
  conservative_reg_loss_weight: 1.0
  min_reward: 0
  max_grad_norm: 1.0
  monte_carlo_return: None
  binary_reward: True
  rescale_reward: False
  grad_accum_every: 8

test:
  load_model: True
  env_name: "button-press-v2"
  model_path: "test.pth"
  render: True
  steps: 200
  episodes: 3
```


Use the run.py script to start training. The script reads settings from config.yaml and applies them to the chosen process:

```bash
python run.py
```