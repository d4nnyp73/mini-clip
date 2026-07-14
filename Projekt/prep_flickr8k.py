"""Laedt jxie/flickr8k von HuggingFace und legt Cache an:

    data/flickr8k_cache/
        images_<split>.npy   -- [N, 64, 64, 3] uint8
        caps_<split>.json    -- Liste[ Liste[str] ]  (5 Captions je Bild)

Damit entfaellt lanngsames JPEG-Decoding/Resizing waehrend des CPU-Trainings.
Fortschritt wird nach stdout geflusht (fuer Hintergrund-Logging).
"""
import json
import os
import sys

import numpy as np
from datasets import load_dataset

SIZE = 64
OUT = "data/flickr8k_cache"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def process_split(split_ds, name):
    n = len(split_ds)
    imgs = np.zeros((n, SIZE, SIZE, 3), dtype=np.uint8)
    caps = []
    cap_keys = [k for k in split_ds.column_names if k.startswith("caption")]
    log(f"[{name}] {n} Bilder, Caption-Spalten: {cap_keys}")
    for i in range(n):
        ex = split_ds[i]
        img = ex["image"].convert("RGB").resize((SIZE, SIZE))
        imgs[i] = np.asarray(img, dtype=np.uint8)
        caps.append([ex[k] for k in cap_keys if ex[k]])
        if (i + 1) % 500 == 0:
            log(f"[{name}] {i+1}/{n}")
    np.save(os.path.join(OUT, f"images_{name}.npy"), imgs)
    with open(os.path.join(OUT, f"caps_{name}.json"), "w") as f:
        json.dump(caps, f)
    log(f"[{name}] FERTIG -> {imgs.shape}")


def main():
    log("Lade jxie/flickr8k ...")
    d = load_dataset("jxie/flickr8k")
    log("Splits:", {k: len(v) for k, v in d.items()})
    for name, split in d.items():
        process_split(split, name)
    log("ALLES FERTIG")


if __name__ == "__main__":
    main()
