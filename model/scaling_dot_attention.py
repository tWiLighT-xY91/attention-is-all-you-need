# This is following the exact calcualtioon of notes I made with the tutorial. Just implementing it in code.
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, heads, seq_len, d_k)
        mask:    (batch, 1, 1, seq_len_k) or broadcastable
        """

        d_k = Q.size(-1) # initializing the d_k parameter

        # (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # calculating the values

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # masking before softmax

        attention_weights = torch.softmax(scores, dim=-1) # softmax applied to scores

        # (batch, heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, V) # final output calculation

        return output, attention_weights
