import torch
import torch.optim as optim
import torch.nn as nn
import random
from src.model import SimpleNet
from src.data_loader import get_dataloader
import wandb


def save_model(model, path='simple_net.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def train_model(cfg):

    model = SimpleNet(39, 256, 4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    dataloader = get_dataloader(cfg.train.file, cfg.train.batch_size, cfg.train.shuffle)

    model.train()
    # Training loop
    for epoch in range(cfg.train.epochs):
        for i, (states, actions, rewards) in enumerate(dataloader):          
            # Forward pass
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print('Training complete')
    save_model(model, cfg.train.model_path)
    return model


