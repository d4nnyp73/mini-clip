# Mini-CLIP – Abgabe (KDDL, SoSe 2026)

Die beiliegende PDF ist die Projektdokumentation. Im Ordner `Projekt/` kann das Jupyter-Notebook
`Mini_CLIP_Exploration.ipynb` direkt gestartet und komplett ausgeführt werden
(einmalig `pip install -r requirements.txt`; läuft in ca. 3 Minuten vollständig auf der CPU,
alle Daten und Checkpoints liegen bei).

Aus Platzgründen (Upload-Limit) sind zwei Dinge ausgelassen, die das Notebook **nicht** benötigt –
sie sind nur für den optionalen vollen Trainings-/Baseline-Lauf (`RUN_HEAVY = True`) relevant:

1. `Projekt/clip_onnx/` – vortrainierte ONNX-Encoder (150 MB) für `baseline_flickr_onnx.py`.
   Wiederherstellbar von HuggingFace, Repo `Xenova/clip-vit-base-patch32`
   (Tokenizer-Dateien nach `clip_onnx/`, `onnx/*_model_quantized.onnx` nach `clip_onnx/onnx/`).
2. Die Trainings- und Validierungs-Caches in `Projekt/data/flickr8k_cache/`
   (`images_train0/1.npy`, `images_val.npy` samt Captions, 82 MB). Neu erzeugbar mit
   `python prep_parquet.py`. Der Test-Split, den das Notebook lädt, liegt vollständig bei.
