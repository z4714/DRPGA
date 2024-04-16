import os
import torch

def load_params(params, path):
    for filename in os.listdir(path):
        if filename.endswith('.pth'):
            params.append(torch.load(os.path.join(path, filename)))

    return params

def square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size,size)*float('-inf'),diagonal=1)
    return mask