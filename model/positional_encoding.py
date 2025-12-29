import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1) # makes it into a row vector, to match the dimensions for calculation
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model) # making terms ready for calculation according to the paper formula
        )

        pe[:, 0::2] = torch.sin(position * div_term) # slicing off only the even indexes, the pe[:, 0::2] means start at 0, go to the end, step by 2, only for columns, the row part is empty
        pe[:, 1::2] = torch.cos(position * div_term) # slicing off only the odd indexes, the pe[:, 1::2] means start at 1, go to the end, step by 2, only for columns, the row part is empty

        pe = pe.unsqueeze(0)  # (1, max_len, d_model), this converts pe to a column vector for batch process

        self.register_buffer("pe", pe) # register_buffer is used to save the tensor as part of the model, but not as a parameter to be trained

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]
