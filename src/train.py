import torch
import torch.optim as optim
import torch.nn as nn
import random
from src.model import QTransformer
from src.data_loader import get_dataloader
import wandb

def epsilon_greedy_policy(q_values, epsilon):
    if random.random() < epsilon:
        return random.choice(range(q_values.size(1)))
    else:
        return q_values.argmax(dim=1).item()

def discretize_actions(actions, num_bins):
    """
    Discretize continuous actions into bins.
    Args:
        actions (Tensor): Continuous actions in the range (-1, 1).
        num_bins (int): Number of discrete bins.
    Returns:
        Tensor: Discretized actions.
    """
    # Rescale actions from (-1, 1) to (0, num_bins-1)
    actions = ((actions + 1) * 0.5 * (num_bins - 1)).long()
    return actions

def undiscretize_actions(discrete_actions, num_bins):
    """
    Convert discrete actions back to continuous values in the range (-1, 1).
    Args:
        discrete_actions (Tensor): Discretized actions.
        num_bins (int): Number of discrete bins.
    Returns:
        Tensor: Continuous actions in the range (-1, 1).
    """
    # Rescale discrete actions from (0, num_bins-1) back to (-1, 1)
    continuous_actions = ((discrete_actions.float() / (num_bins - 1)) * 2) - 1
    return continuous_actions

def train_model(cfg):

    model = QTransformer(cfg.model.state_dim, cfg.model.action_dim, cfg.model.transformer_layers, cfg.model.transformer_heads, cfg.model.num_bins)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    dataloader = get_dataloader(cfg.train.file, cfg.train.batch_size)
    num_bins = cfg.model.num_bins

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0

        for states, actions, rewards in dataloader:
            q_values = model(states)
            #actions = discretize_actions(actions, num_bins)
            actions_pred = torch.argmax(q_values, dim=2, keepdim=True)
            actions_shape = actions_pred.shape
            actions_pred = actions_pred.reshape(actions_shape[0], actions_shape[2], actions_shape[1])
            #action_q_values = q_values.gather(2, actions).squeeze(2)
            actions_pred_cont = undiscretize_actions(actions_pred, num_bins)
            loss = criterion(actions_pred_cont, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}, Loss: {avg_loss}")

        # Exploration
        epsilon = cfg.train.epsilon
        predicted_actions = []
        for states, _, _ in dataloader:
            q_values = model(states)
            for q in q_values:
                action = epsilon_greedy_policy(q.unsqueeze(0), epsilon)
                predicted_actions.append(action)
        
        wandb.log({"predicted_actions": predicted_actions})

