import torch

def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), 1)
    mask.masked_fill_(mask == 1, float('-inf'))
    return mask

