import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):
    """
    features: 特徴量の次元
    eps: layer normalization の分母の補正項(1e-6)
    """
    def __init__(self, features, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
