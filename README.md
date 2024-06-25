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

Environment Setup
Ensure you have the necessary dependencies installed. You can use `pip` to install the required packages:
```bash
pip install torch metaworld gym wandb
```

