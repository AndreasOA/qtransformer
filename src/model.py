import torch
import torch.nn as nn
from transformers import BertModel

class QTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, n_heads, num_bins):
        super(QTransformer, self).__init__()
        self.num_bins = num_bins
        encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(state_dim, action_dim * num_bins)  # Output: action_dim * num_bins

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, state_dim] -> [seq_len, batch_size, state_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Pooling over the sequence length dimension
        q_values = self.fc(x)
        q_values = q_values.view(q_values.size(0), -1, self.num_bins)  # Reshape to [batch_size, action_dim, num_bins]
        return q_values