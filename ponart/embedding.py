import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 256, denom=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model).requires_grad_(False)
        pos = torch.arange(0.0, max_len).unsqueeze(-1)
        div = torch.exp(-torch.arange(0, d_model, 2.0) / d_model * torch.log(torch.tensor(denom)))
        self.embedding.weight[:, 0::2] = torch.sin(div * pos)
        self.embedding.weight[:, 1::2] = torch.cos(div * pos)

    def forward(self, x):
        return self.embedding(x)

class TransformerEmbedding(nn.Module):
    def __init__(self, d_vocab, d_model, dropout, padding_idx = 0, max_seq_len = 256):
        super().__init__()
        self.token_embedding = nn.Embedding(d_vocab, d_model, padding_idx = 0)
        self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len, 10000.0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, position_ids = None):
        x = self.token_embedding(x)
        if position_ids is None:
            position_ids = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        x = x + self.position_embedding(position_ids)
        x = self.norm(x)
        x = self.dropout(x)
        return x

