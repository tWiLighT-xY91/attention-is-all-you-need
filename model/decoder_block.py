from model.multi_head_attention import MultiHeadAttention
from model.feed_forward_network import PositionwiseFeedForward
import torch.nn as nn
import torch
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention
        attn1, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn1)

        # 2. Encoder-decoder attention
        attn2, _ = self.cross_attention(
            x,              # queries
            enc_output,     # keys
            enc_output,     # values
            src_mask
        )
        x = self.norm2(x + attn2)

        # 3. Feed-forward network
        ff = self.feed_forward(x)
        x = self.norm3(x + ff)

        return x
