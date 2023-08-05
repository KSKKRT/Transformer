import torch.nn as nn

from SubLayers import SublayerConnection, LayerNorm
from Process import clones


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x:  self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.ffn)