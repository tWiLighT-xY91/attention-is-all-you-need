import torch
import torch.nn as nn
import math
from model.scaling_dot_attention import ScaledDotProductAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__() # Is a must in nn.Module subclasses from torch, or else your model will train, but will give no output

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" # An alternative for if...: raiseValueError()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # double slash for integer division

        self.W_q = nn.Linear(d_model, d_model) # the W matrices for Q, K and V which actually gives the meaning for Multi-head Attention
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        """
        x: (batch, seq_len, d_model)
        return: (batch, heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        x: (batch, heads, seq_len, d_k)
        return: (batch, seq_len, d_model)
        """
        batch_size, heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous() # contiguos to ensure memory is contiguos and the model doesn't learn garbage
        return x.view(batch_size, seq_len, heads * d_k)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = self.attention(Q, K, V, mask)

        concat = self.combine_heads(attention_output)

        output = self.W_o(concat)

        return output, attention_weights
