"""Training von Mini-CLIP.

Beispiele:
  # Sanity-Check auf synthetischen Daten (Sekunden bis Minuten auf CPU):
  python train.py --dataset synthetic --epochs 15 --n-train 2000

  # Echte Daten (Flickr8k zuvor herunterladen, siehe scripts/download_flickr8k.py):
  python train.py --dataset flickr8k --data-root data/flickr8k --epochs 20
"""
import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data import SyntheticShapes, Flickr8kDataset, Collate, all_captions
from src.model import MiniCLIP, clip_loss
from src.tokenizer import SimpleTokenizer
from src.evaluate import encode_dataset, retrieval_recall, zero_shot_classify, pretty


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_datasets(args, cfg):
    if args.dataset == "synthetic":
        train = SyntheticShapes(args.n_train, cfg.image_size, seed=0)
        val = SyntheticShapes(max(200, args.n_train // 5), cfg.image_size, seed=1)
        classnames = SyntheticShapes.classnames
    elif args.dataset == "flickr8k":
        train = Flickr8kDataset(args.data_root, cfg.image_size, split="train")
        val = Flickr8kDataset(args.data_root, cfg.image_size, split="val")
        classnames = None
    else:
        raise ValueError(args.dataset)
    return train, val, classnames


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "flickr8k"], default="synthetic")
    ap.add_argument("--data-root", default="data/flickr8k")
    ap.add_argument("--n-train", type=int, default=2000, help="nur fuer synthetic")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:     cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr = args.lr
    if args.ckpt:       cfg.ckpt_path = args.ckpt
    set_seed(cfg.seed)

    train_ds, val_ds, classnames = build_datasets(args, cfg)
    print(f"Train: {len(train_ds)} Paare | Val: {len(val_ds)} Paare")

    # Vokabular aus den TRAININGS-Captions lernen
    tok = SimpleTokenizer(cfg.max_text_len).build_vocab(
        all_captions(train_ds), min_freq=cfg.min_word_freq)
    print(f"Vokabulargroesse: {tok.vocab_size}")
    collate = Collate(tok)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=cfg.num_workers, drop_last=True)

    model = MiniCLIP(cfg, tok.vocab_size, tok.pad_id).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modellparameter: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, cfg.warmup_steps, total_steps))

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            images = batch["image"].to(cfg.device)
            tokens = batch["text"].to(cfg.device)
            logits_i, logits_t = model(images, tokens)
            loss = clip_loss(logits_i, logits_t)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{sched.get_last_lr()[0]:.1e}")

        # --- Validierung ---
        img_emb, txt_emb, _ = encode_dataset(model, val_ds, collate,
                                             cfg.batch_size, cfg.device)
        rec = retrieval_recall(img_emb, txt_emb)
        msg = f"  [val] avg_loss={running/len(loader):.3f}  {pretty(rec)}"
        if classnames:
            acc = zero_shot_classify(model, val_ds, collate, classnames,
                                     batch_size=cfg.batch_size, device=cfg.device)
            msg += f"  zero_shot_acc={acc}%"
        print(msg)

    os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "tokenizer": tok.state_dict(),
                "config": cfg.__dict__}, cfg.ckpt_path)
    print(f"Checkpoint gespeichert: {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
