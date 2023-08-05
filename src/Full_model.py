import torch.nn as nn
import numpy as np
import copy

from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
from EncoderDecoder import EncoderDecoder, Generator
from SubLayers import MultiHeadedAttention, PositionwiseFeedForward
from Process import Embeddings, PositionalEncoding

def full_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, n_head=8, dropout=0.1
):
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_head, d_model)
    ffn = PositionwiseFeedForward(d_model, dropout)
    pos = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ffn), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ffn), dropout)),
        nn.Sequential(Embeddings(d_model, src_vocab), c(pos)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos)),
        Generator(d_model, tgt_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
    