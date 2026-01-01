import torch

def create_padding_mask(seq, pad_idx = 0):
    """
    seq: (batch, seq_len)
    returns: (batch, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask




def create_look_ahead_mask(seq_len):
    """
    returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
    return mask
