import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward_network import PositionwiseFeedForward
import torch

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads) # Initializing the MultiHeadAttention class
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff) # Initializing the PositionwiseFeedForward class

        self.norm1 = nn.LayerNorm(d_model) # Linnear normalization layer 1
        self.norm2 = nn.LayerNorm(d_model) # Linnear normalization layer 2

    def forward(self, x, mask=None):
        # Self-attention sublayer
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x
