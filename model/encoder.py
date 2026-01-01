# The below code explains the structure of an Encoder, consisting of the Token embedding, 
# the positional embedding, and then the encoder block, which contains all the multi-head attention part 
# and the complicated part of the encoder, and this part is just integrating all of that together.
import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding
from model.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__( self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) — token indices
        mask: (batch, 1, 1, seq_len) — padding mask
        """

        # 1. Token → vector
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # 2. Scale embeddings (paper detail people forget)
        x = x * (self.d_model ** 0.5)

        # 3. Add positional encoding
        x = self.positional_encoding(x)

        x = self.dropout(x)

        # 4. Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        return x
