"""Dekodiert EINE Flickr8k-Parquet-Datei in einen npy/json-Cache.
Aufruf: python prep_parquet.py <parquet> <out_prefix>
Erzeugt: data/flickr8k_cache/images_<prefix>.npy  [N,64,64,3] uint8
         data/flickr8k_cache/caps_<prefix>.json   Liste[Liste[str]]
"""
import io, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SIZE = 64
OUT = "data/flickr8k_cache"
os.makedirs(OUT, exist_ok=True)


def main(path, prefix):
    t0 = time.time()
    table = pq.read_table(path)
    cap_cols = [c for c in table.column_names if c.startswith("caption")]
    images_col = table.column("image").to_pylist()
    caps_cols = {c: table.column(c).to_pylist() for c in cap_cols}
    n = len(images_col)
    imgs = np.zeros((n, SIZE, SIZE, 3), dtype=np.uint8)
    caps = []
    for i in range(n):
        rec = images_col[i]
        b = rec["bytes"] if isinstance(rec, dict) else rec
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE))
        imgs[i] = np.asarray(img, dtype=np.uint8)
        caps.append([caps_cols[c][i] for c in cap_cols if caps_cols[c][i]])
    np.save(os.path.join(OUT, f"images_{prefix}.npy"), imgs)
    with open(os.path.join(OUT, f"caps_{prefix}.json"), "w") as f:
        json.dump(caps, f)
    print(f"{prefix}: {imgs.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
