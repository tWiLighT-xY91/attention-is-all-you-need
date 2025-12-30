
def create_padding_mask(seq, pad_idx = 0):
    """
    seq: (batch, seq_len)
    returns: (batch, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask