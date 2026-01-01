import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader
import torch.optim as optim
import os
from model.transformer import Transformer
from training.train import train_model
from training.loss import get_loss_fn
from training.scheduler import TransformerLRScheduler
from utils.dataset import TranslationDataset, collate_fn
from utils.tokenizer import build_vocab, PAD_IDX
import config
import argparse



def main(train: bool):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabularies (from training data only)
    src_vocab = build_vocab("data/raw/train.de")
    tgt_vocab = build_vocab("data/raw/train.en")

    # Dataset + DataLoader
    dataset = TranslationDataset(
        src_path="data/raw/train.de",
        tgt_path="data/raw/train.en",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )

    # Model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout
    )

    # Optimizer (Adam with paper settings)
    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )

    loss_fn = get_loss_fn(PAD_IDX)

    # Train
    if train:
           train_model(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epochs=config.epochs,
            pad_idx=PAD_IDX
        )
          
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
         model.state_dict(),
         "checkpoints/transformer_epoch10.pt"
         )
        print("✅ Model checkpoint saved. You may now sleep.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    main(train=args.train)

