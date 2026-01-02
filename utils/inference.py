import torch
from utils.tokenizer import BOS_IDX, EOS_IDX, PAD_IDX

def greedy_decode(model, src, src_mask, max_len, device):
    model.eval()

    enc_output = model.encoder(src, src_mask)

    tgt = torch.tensor([[BOS_IDX]], device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)

        dec_output = model.decoder(
            tgt,
            enc_output,
            src_mask,
            tgt_mask
        )

        logits = model.output_projection(dec_output)
        next_token = logits[:, -1].argmax(dim=-1).unsqueeze(1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == EOS_IDX:
            break

    return tgt.squeeze(0)
