import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time



class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_layers, dropout=0.1) -> None:
        super(TransformerEncoder, self).__init__()
        self.linear_proj = nn.Linear(input_dim, out_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=out_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)


    
    def forward(self, x):
        x = self.linear_proj(x)
        x = self.transformer_encoder(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_layers, dropout=0.1) -> None:
        super(TransformerDecoder, self).__init__()
        self.linear_proj = nn.Linear(input_dim, out_dim)
        encoder_layers = nn.TransformerDecoderLayer(d_model=out_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(encoder_layers, num_layers=num_layers)


    
    def forward(self, x):
        x = self.linear_proj(x)
        x = self.transformer_decoder(x)
        return x