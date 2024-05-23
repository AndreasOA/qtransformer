import torch
import torch.nn as nn
import torch.nn.functional as F

class QTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, n_heads, num_bins):
        super(QTransformer, self).__init__()
        self.num_bins = num_bins
        encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(state_dim, action_dim * num_bins)  # Output: action_dim * num_bins


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


    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, state_dim] -> [seq_len, batch_size, state_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Pooling over the sequence length dimension
        q_values = self.fc(x)
        q_values = q_values.view(q_values.size(0), -1, self.num_bins)  # Reshape to [batch_size, action_dim, num_bins]
        return q_values


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer to prevent overfitting
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
