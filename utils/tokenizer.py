# utils/tokenizer.py

from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


class Vocab:
    def __init__(self, token_to_idx, idx_to_token):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens):
        return [
            self.token_to_idx.get(token, UNK_IDX)
            for token in tokens
        ]

    def decode(self, ids):
        return [
            self.idx_to_token[idx]
            for idx in ids
        ]


def build_vocab(file_path, min_freq=2):
    counter = Counter()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    token_to_idx = {}
    idx_to_token = {}

    # Add special tokens
    for idx, token in enumerate(SPECIAL_TOKENS):
        token_to_idx[token] = idx
        idx_to_token[idx] = token

    idx = len(SPECIAL_TOKENS)
    for token, freq in counter.items():
        if freq >= min_freq:
            token_to_idx[token] = idx
            idx_to_token[idx] = token
            idx += 1

    return Vocab(token_to_idx, idx_to_token)
