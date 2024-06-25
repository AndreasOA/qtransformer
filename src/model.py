import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)

# Simplified RMSNorm
class RMSNorm(Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

# Transformer Attention
class TransformerAttention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.MultiheadAttention(inner_dim, heads, dropout=dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        out, _ = self.attend(q, k, v)
        return self.to_out(out) + x

# FeedForward network
class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x

# Transformer
class Transformer(Module):
    def __init__(self, dim, depth=6, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.layers = ModuleList([
            ModuleList([
                TransformerAttention(dim, heads, dim_head, dropout),
                FeedForward(dim, dropout=dropout)
            ]) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return self.norm(x)

# QHead for multiple actions
class QHeadMultipleActions(Module):
    def __init__(self, dim, num_actions=4, action_bins=256):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins

        self.to_q_values = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_actions * action_bins),
            nn.Sigmoid()
        )

    def forward(self, x):
        q_values = self.to_q_values(x)
        return q_values.view(x.shape[0], self.num_actions, self.action_bins)

    def get_optimal_actions(self, state):
        q_values = self.forward(state)
        return q_values.argmax(dim=-1)

# QRoboticTransformer simplified for Metaworld
class QRoboticTransformer(Module):
    def __init__(self, state_dim=39, action_bins=256, num_actions=4, transformer_dim=128, transformer_depth=6, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.action_bins = action_bins
        self.state_encoder = nn.Linear(state_dim, transformer_dim)
        self.transformer = Transformer(transformer_dim, depth=transformer_depth, heads=heads, dim_head=dim_head, dropout=dropout)
        self.q_head = QHeadMultipleActions(transformer_dim, num_actions=num_actions, action_bins=action_bins)

    def forward(self, state):
        x = self.state_encoder(state)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.q_head(x)

    def get_random_actions(self, batch_size=1):
        return torch.randint(0, self.q_head.action_bins, (batch_size, self.q_head.num_actions), device=self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def get_optimal_actions(self, state):
        x = self.forward(state)
        x = x.argmax(dim=-1)
        return self.undiscretize_actions(x)

    def discretize_actions(self, actions):
        """
        Discretize continuous actions into bins.
        Args:
            actions (Tensor): Continuous actions in the range (-1, 1).
            num_bins (int): Number of discrete bins.
        Returns:
            Tensor: Discretized actions.
        """
        # Rescale actions from (-1, 1) to (0, num_bins-1)
        actions = ((actions + 1) * 0.5 * (self.action_bins - 1)).long()
        return actions

    def undiscretize_actions(self, discrete_actions):
        """
        Convert discrete actions back to continuous values in the range (-1, 1).
        Args:
            discrete_actions (Tensor): Discretized actions.
            num_bins (int): Number of discrete bins.
        Returns:
            Tensor: Continuous actions in the range (-1, 1).
        """
        # Rescale discrete actions from (0, num_bins-1) back to (-1, 1)
        continuous_actions = ((discrete_actions / (self.action_bins - 1)) * 2) - 1
        return continuous_actions


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
