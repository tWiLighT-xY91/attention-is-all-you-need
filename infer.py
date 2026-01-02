import torch
from model.transformer import Transformer
from utils.tokenizer import build_vocab
from utils.inference import greedy_decode
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vocab (same as training!)
src_vocab = build_vocab("data/raw/train.de")
tgt_vocab = build_vocab("data/raw/train.en")

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    dropout=config.dropout
).to(device)

model.load_state_dict(torch.load("checkpoints/transformer_epoch10.pt", map_location=device))

def translate(sentence):
    tokens = sentence.strip().split()
    src_ids = src_vocab.encode(tokens)

    src = torch.tensor([src_ids], device=device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    out = greedy_decode(model, src, src_mask, max_len=50, device=device)
    return " ".join(tgt_vocab.decode(out.tolist()))

print(translate("ich liebe maschinelles lernen ."))
print(translate("das ist ein kleines beispiel ."))
