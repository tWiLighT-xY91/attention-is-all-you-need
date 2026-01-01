import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding
from model.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x: (batch, tgt_seq_len)           — target token indices
        enc_output: (batch, src_seq_len, d_model)
        src_mask: (batch, 1, 1, src_seq_len)
        tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len)
        """

        # 1. Token → vector
        x = self.embedding(x)  # (batch, tgt_seq_len, d_model)

        # 2. Scale embeddings
        x = x * (self.d_model ** 0.5)

        # 3. Add positional encoding
        x = self.positional_encoding(x)

        x = self.dropout(x)

        # 4. Pass through decoder blocks
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x
