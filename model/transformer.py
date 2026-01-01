import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.output import OutputProjection


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.output_projection = OutputProjection(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_seq_len)
        tgt: (batch, tgt_seq_len)

        src_mask: (batch, 1, 1, src_seq_len)
        tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len)
        """

        # Encoder
        enc_output = self.encoder(src, src_mask)

        # Decoder
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # Output projection
        logits = self.output_projection(dec_output)

        return logits
