# utils/dataset.py

import torch
from torch.utils.data import Dataset

from utils.tokenizer import BOS_IDX, EOS_IDX


class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab):
        self.src_sentences = open(src_path, encoding="utf-8").read().strip().split("\n")
        self.tgt_sentences = open(tgt_path, encoding="utf-8").read().strip().split("\n")

        assert len(self.src_sentences) == len(self.tgt_sentences)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = self.src_sentences[idx].split()
        tgt_tokens = self.tgt_sentences[idx].split()

        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = [BOS_IDX] + self.tgt_vocab.encode(tgt_tokens) + [EOS_IDX]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = []
    padded_tgt_input = []
    padded_tgt_output = []

    for src, tgt in zip(src_batch, tgt_batch):
        src_pad = torch.cat([
            src,
            torch.full((max_src - len(src),), pad_idx)
        ])
        padded_src.append(src_pad)

        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        tgt_input_pad = torch.cat([
            tgt_input,
            torch.full((max_tgt - len(tgt_input),), pad_idx)
        ])
        tgt_output_pad = torch.cat([
            tgt_output,
            torch.full((max_tgt - len(tgt_output),), pad_idx)
        ])

        padded_tgt_input.append(tgt_input_pad)
        padded_tgt_output.append(tgt_output_pad)

    return (
        torch.stack(padded_src),
        torch.stack(padded_tgt_input),
        torch.stack(padded_tgt_output),
    )
