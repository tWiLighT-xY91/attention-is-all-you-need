import torch.nn as nn

def get_loss_fn(pad_idx=0):
    """
    CrossEntropyLoss that ignores padding tokens
    """
    return nn.CrossEntropyLoss(ignore_index=pad_idx)
