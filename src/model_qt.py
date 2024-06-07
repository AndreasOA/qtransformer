import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

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

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# QRoboticTransformer with autoregressive action prediction
class QRoboticTransformer(Module):
    def __init__(self, state_dim=39, action_bins=256, num_actions=4, transformer_depth=6, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.action_bins = action_bins
        self.num_actions = num_actions
        self.state_encoder = nn.Linear(state_dim, action_bins)
        self.transformer = Transformer(action_bins, depth=transformer_depth, heads=heads, dim_head=dim_head, dropout=dropout)
        self.positional_encoding = PositionalEncoding(action_bins)
        self.q_head = QHeadMultipleActions(action_bins, num_actions=1, action_bins=action_bins)

    def forward(self, state):
        action_seq = self.state_encoder(state).unsqueeze(1)  # (batch_size, 1, transformer_dim)

        for i in range(self.num_actions):
            action_seq = self.positional_encoding(action_seq)
            transformer_out = self.transformer(action_seq)
            q_values = self.q_head(transformer_out[:, -1, :])  # Get Q-values for the last token
            action_indices = q_values.argmax(dim=-1)  # (batch_size, 1, action_bins)
            one_hot_actions = F.one_hot(action_indices, num_classes=self.action_bins).float()
            action_seq = torch.cat((action_seq, one_hot_actions), dim=1)  # Append the new action

        return action_seq[:, 1:]     # first entry is state encoded followed by num_actions actions

    def get_random_actions(self, batch_size=1):
        return torch.randint(0, self.action_bins, (batch_size, self.num_actions), device=self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def get_optimal_actions(self, state):
        x = self.forward(state)
        x = x.argmax(dim=-1)
        return self.undiscretize_actions(x)

    def discretize_actions(self, actions):
        actions = ((actions + 1) * 0.5 * (self.action_bins - 1)).long()
        return actions

    def undiscretize_actions(self, discrete_actions):
        continuous_actions = ((discrete_actions / (self.action_bins - 1)) * 2) - 1
        return continuous_actions
