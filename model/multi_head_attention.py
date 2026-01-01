import torch
import torch.nn as nn
from model.scaling_dot_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
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
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, heads * d_k)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, q_len, d_model)
        key:   (batch, k_len, d_model)
        value: (batch, v_len, d_model)
        mask:  (batch, 1, q_len, k_len) or None
        """

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Combine heads
        attn_output = self.combine_heads(attn_output)

        # Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights
