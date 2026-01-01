import torch
from masks.masks import create_padding_mask, create_decoder_mask


def train_step(model, batch, loss_fn, optimizer, device, pad_idx=0):
    """
    batch:
        src: (batch, src_len)
        tgt_input: (batch, tgt_len)
        tgt_output: (batch, tgt_len)
    """

    model.train()

    src, tgt_input, tgt_output = batch
    src = src.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)

    # Masks
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_decoder_mask(tgt_input, pad_idx)

    # Forward pass
    logits = model(
        src=src,
        tgt=tgt_input,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )
    # logits: (batch, tgt_len, vocab_size)

    # Loss expects (batch*tgt_len, vocab_size)
    logits = logits.view(-1, logits.size(-1))
    tgt_output = tgt_output.view(-1)

    loss = loss_fn(logits, tgt_output)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
