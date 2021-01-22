import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def block(self, x, mask, padding_mask):
        x = self.norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask = mask, key_padding_mask = padding_mask)
        x = self.dropout(x)
        return x

    def forward(self, x, mask, padding_mask):
        return x + self.block(x, mask, padding_mask)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def block(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        return x + self.block(x)

