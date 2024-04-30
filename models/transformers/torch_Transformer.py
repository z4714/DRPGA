import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math


class nlu_tf(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class TRTF_encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, 
    d_ff, max_seq_length=4096, dropout=0.1):
        super(TRTF_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1,max_seq_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src += self.pos_encoder[:, :src.size(1)]
        output = self.transformer_encoder(src, src_mask)
        return output


class Fixpooling_TRTF(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, 
    d_ff, max_seq_length=4096, dropout=0.1):
        super(Fixpooling_TRTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1,max_seq_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.out_layer = nn.Linear(d_model, max_seq_length)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = output.permute(1,2,0)
        output = self.pooling(output).squeeze(-1)
        output = self.out_layer(output)
        return output
    

class paddingT_FTRTF_encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, 
    d_ff, out_dim, max_seq_length=512, dropout=0.1):
        super(paddingT_FTRTF_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1,max_seq_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

        #self.pooling = nn.AdaptiveAvgPool1d(out_dim)
        self.out_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src += self.pos_encoder[:, :src.size(1)]

        output = self.transformer_encoder(src, src_mask)
        #output = output.permute(0,2,1)
        #output = self.pooling(output).squeeze(-1)
        output = self.out_layer(output)
        return output    


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