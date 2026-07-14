"""Hilfe zum Beschaffen von Flickr8k.

Flickr8k hat keine voellig stabile offizielle URL. Empfohlener Weg ueber Kaggle:

  1) Kaggle-Account anlegen, dann das Kaggle-CLI installieren:
         pip install kaggle
     und den API-Token (kaggle.json) unter ~/.kaggle/ ablegen.

  2) Datensatz laden:
         kaggle datasets download -d adityajn105/flickr8k
         unzip flickr8k.zip -d data/flickr8k

  Danach sollte die Struktur so aussehen:
         data/flickr8k/Images/*.jpg
         data/flickr8k/captions.txt      (Spalten: image,caption)

Der Loader in src/data.py erwartet genau dieses Layout. Alternativ funktioniert
JEDER Bild-Text-Datensatz, sofern eine captions.txt im Format 'image,caption'
und ein Ordner Images/ vorliegen.

Dieses Skript prueft nur, ob alles am richtigen Platz liegt.
"""
import os
import sys


def check(root: str) -> bool:
    img_dir = os.path.join(root, "Images")
    caps = os.path.join(root, "captions.txt")
    ok = os.path.isdir(img_dir) and os.path.isfile(caps)
    if ok:
        n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        with open(caps, encoding="utf-8") as f:
            n_caps = sum(1 for _ in f)
        print(f"OK: {n_imgs} Bilder, {n_caps} Caption-Zeilen unter '{root}'.")
    else:
        print(f"FEHLT: erwarte '{img_dir}/' und '{caps}'.")
        print(__doc__)
    return ok


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/flickr8k"
    sys.exit(0 if check(root) else 1)
