import os
import torch

def load_params(params, path):
    for filename in os.listdir(path):
        if filename.endswith('.pth'):
            params.append(torch.load(os.path.join(path, filename)))

    return params