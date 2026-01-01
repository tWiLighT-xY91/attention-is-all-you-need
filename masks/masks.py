import torch

def create_padding_mask(seq, pad_idx=0):
    """
    seq: (batch, seq_len)
    returns: (batch, 1, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len, device):
    """
    returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(1)


def create_decoder_mask(tgt_input, pad_idx=0):
    """
    tgt_input: (batch, tgt_seq_len)
    returns: (batch, 1, tgt_seq_len, tgt_seq_len)
    """
    device = tgt_input.device

    padding_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)
    look_ahead_mask = create_look_ahead_mask(tgt_input.size(1), device)

    return padding_mask & look_ahead_mask
