class OutputProjection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, vocab_size)
        """
        return self.linear(x)
