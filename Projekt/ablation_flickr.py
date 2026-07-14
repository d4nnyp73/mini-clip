"""Ablation: Einfluss der Batchgroesse (= Anzahl In-Batch-Negative) auf die
Retrieval-Qualitaet von Mini-CLIP. CLIP profitiert laut Paper von sehr grossen
Batches, weil jedes Paar gegen alle anderen Texte des Batches kontrastiert wird.

Trainiert je Konfiguration ein frisches Modell (gleiche Daten, gleiche Epochen)
auf einem Subset und misst Image->Text Recall auf dem 1000er-Test-Split.
"""
import argparse, math, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import Config
from src.data import CachedFlickr, Collate, all_captions
from src.model import MiniCLIP, clip_loss
from src.tokenizer import SimpleTokenizer
from src.evaluate import encode_dataset, retrieval_recall


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def train_one(bs, epochs, train_ds, test_ds, tok, cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    collate = Collate(tok)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        collate_fn=collate, drop_last=True)
    total = epochs * len(loader)
    model = MiniCLIP(cfg, tok.vocab_size, tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    for _ in range(epochs):
        model.train()
        for b in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg.lr, cfg.warmup_steps, total)
            li, lt = model(b["image"], b["text"]); loss = clip_loss(li, lt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
    ie, te, _ = encode_dataset(model, test_ds, collate, 256, "cpu")
    return retrieval_recall(ie, te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256])
    args = ap.parse_args()

    cfg = Config()
    full = CachedFlickr(split="train", eval=False)
    train_ds = Subset(full, list(range(min(args.n_train, len(full)))))
    test_ds = CachedFlickr(split="test", eval=True)
    tok = SimpleTokenizer(cfg.max_text_len).build_vocab(
        all_captions(full), min_freq=2)

    print(f"Ablation: n_train={len(train_ds)}, epochs={args.epochs}, "
          f"test=1000, vocab={tok.vocab_size}\n", flush=True)
    print(f"{'batch':>6} | {'i2t_R@1':>7} {'i2t_R@5':>7} {'i2t_R@10':>8} | {'sek':>5}")
    print("-" * 45)
    for bs in args.batches:
        t = time.time()
        rec = train_one(bs, args.epochs, train_ds, test_ds, tok, cfg)
        print(f"{bs:>6} | {rec['i2t_R@1']:>7} {rec['i2t_R@5']:>7} "
              f"{rec['i2t_R@10']:>8} | {time.time()-t:>5.1f}", flush=True)


if __name__ == "__main__":
    main()
