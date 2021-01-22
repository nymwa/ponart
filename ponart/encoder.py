import torch
import torch.nn as nn
from ponart.block import SelfAttentionBlock, FeedForwardBlock

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, attention_dropout, dropout):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, nhead, attention_dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_feedforward, dropout)

    def forward(self, x, padding_mask = None):
        x = self.self_attn(x, None, padding_mask)
        x = self.feed_forward(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, num_layers, attention_dropout, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_feedforward, attention_dropout, dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask = None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        return x

