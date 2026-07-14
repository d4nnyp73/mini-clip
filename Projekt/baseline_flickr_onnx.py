"""Baseline: vortrainiertes CLIP (ViT-B/32, OpenAI) via ONNX-Runtime.

Nutzt die quantisierten ONNX-Encoder von 'Xenova/clip-vit-base-patch32'
(vision_model_quantized + text_model_quantized) -- klein genug fuer die
Sandbox und schnell auf CPU. Bewertet Bild<->Text-Retrieval auf demselben
Flickr8k-Test-Split (caption_0) wie unser Mini-CLIP.

Aufruf: python baseline_flickr_onnx.py --n 1000
"""
import argparse, glob, io, os, time
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image
from transformers import CLIPTokenizerFast

from src.evaluate import retrieval_recall, pretty

ONNX = "clip_onnx/onnx"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def find_test_parquet():
    snap = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--jxie--flickr8k/snapshots")
    hits = glob.glob(f"{snap}/*/data/test-*.parquet")
    if not hits:
        raise FileNotFoundError("Test-Parquet nicht gefunden (erst prep ausfuehren).")
    return hits[0]


def preprocess(pil):
    """CLIP-Preprocessing: shortest-edge 224 (bicubic) -> center-crop 224 ->
    /255 -> normalisieren. Ausgabe [3,224,224] float32."""
    img = pil.convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    arr = np.asarray(img, np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    table = pq.read_table(find_test_parquet())
    imgs_raw = table.column("image").to_pylist()
    caps = table.column("caption_0").to_pylist()
    n = min(args.n, len(caps))

    vis = ort.InferenceSession(f"{ONNX}/vision_model_quantized.onnx",
                               providers=["CPUExecutionProvider"])
    txt = ort.InferenceSession(f"{ONNX}/text_model_quantized.onnx",
                               providers=["CPUExecutionProvider"])
    tok = CLIPTokenizerFast.from_pretrained("clip_onnx")

    # Bild-Embeddings
    img_emb = []
    for i in range(0, n, args.batch):
        batch = [imgs_raw[j]["bytes"] for j in range(i, min(i + args.batch, n))]
        px = np.stack([preprocess(Image.open(io.BytesIO(b))) for b in batch])
        img_emb.append(vis.run(None, {"pixel_values": px})[0])
    img_emb = l2(np.concatenate(img_emb))

    # Text-Embeddings
    ids = tok(caps[:n], padding="max_length", max_length=77, truncation=True,
              return_tensors="np")["input_ids"].astype(np.int64)
    txt_emb = []
    for i in range(0, n, args.batch):
        txt_emb.append(txt.run(None, {"input_ids": ids[i:i + args.batch]})[0])
    txt_emb = l2(np.concatenate(txt_emb))

    import torch
    rec = retrieval_recall(torch.from_numpy(img_emb), torch.from_numpy(txt_emb))
    print(f"Pretrained CLIP (ViT-B/32, ONNX int8) auf {n} Test-Bildern, {time.time()-t0:.1f}s")
    print("  " + pretty(rec))


if __name__ == "__main__":
    main()
