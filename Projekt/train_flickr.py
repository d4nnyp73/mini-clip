"""Training von Mini-CLIP auf Flickr8k (aus dem npy-Cache).

Resumebar: jede Epoche wird ein Checkpoint gespeichert; mit --resume wird der
Trainingszustand (Modell, Optimizer, Step, Tokenizer) fortgesetzt. Zusaetzlich
bricht das Skript bei Erreichen von --time-budget Sekunden kontrolliert ab und
speichert -- so laesst sich ein langes Training in mehreren Aufrufen fahren.

Beispiel:
    python train_flickr.py --epochs 30 --batch-size 256
    python train_flickr.py --epochs 30 --batch-size 256 --resume   # fortsetzen
"""
import argparse, math, os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from src.data import CachedFlickr, Collate, all_captions
from src.model import MiniCLIP, clip_loss
from src.tokenizer import SimpleTokenizer
from src.evaluate import encode_dataset, retrieval_recall, pretty

CKPT = "checkpoints/mini_clip_flickr.pt"


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--time-budget", type=float, default=40.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = Config(); cfg.batch_size = args.batch_size; cfg.lr = args.lr
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    train_ds = CachedFlickr(split="train", eval=False)
    test_ds = CachedFlickr(split="test", eval=True)
    print(f"Train {len(train_ds)} | Test {len(test_ds)}", flush=True)

    # Tokenizer: bei Resume aus Checkpoint, sonst neu aus Trainings-Captions
    if args.resume and os.path.exists(CKPT):
        state = torch.load(CKPT, map_location="cpu")
        tok = SimpleTokenizer().load_state_dict(state["tokenizer"])
    else:
        state = None
        tok = SimpleTokenizer(cfg.max_text_len).build_vocab(
            all_captions(train_ds), min_freq=2)
    print(f"Vokabular: {tok.vocab_size}", flush=True)
    collate = Collate(tok)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch

    model = MiniCLIP(cfg, tok.vocab_size, tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, gstep, best = 1, 0, 0.0
    if state is not None:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1; gstep = state["gstep"]; best = state.get("best", 0.0)
        print(f"Resume ab Epoche {start_epoch} (gstep={gstep}, best i2t_R@1={best})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_params/1e6:.2f}M | steps/epoch={steps_per_epoch}", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep, cfg.lr, cfg.warmup_steps, total_steps)
            li, lt = model(batch["image"], batch["text"])
            loss = clip_loss(li, lt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); gstep += 1; running += loss.item()

        img_emb, txt_emb, _ = encode_dataset(model, test_ds, collate, 256, "cpu")
        rec = retrieval_recall(img_emb, txt_emb)
        best = max(best, rec["i2t_R@1"])
        print(f"E{epoch:02d} loss={running/steps_per_epoch:.3f} {pretty(rec)}", flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "tokenizer": tok.state_dict(), "epoch": epoch,
                    "gstep": gstep, "best": best, "config": cfg.__dict__}, CKPT)

        if time.time() - t0 > args.time_budget and epoch < args.epochs:
            print(f"ZEITBUDGET erreicht nach Epoche {epoch} -> Checkpoint gespeichert, "
                  f"erneut mit --resume aufrufen.", flush=True)
            return
    print("DONE: Zielepochen erreicht. best i2t_R@1 =", best, flush=True)


if __name__ == "__main__":
    main()
