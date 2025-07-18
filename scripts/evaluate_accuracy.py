import os
import torch
import pandas as pd
import numpy as np
import librosa

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

# ─── 1) Locate everything ────────────────────────────────────────────
project_root = os.getcwd()  # assume you run it from C:\Users\aadit\thia-audio-sentiment
ckpt_dir     = os.path.join(project_root, "models", "wav2vec2-ser", "checkpoint-10241")
csv_file     = os.path.join(project_root, "data", "metadata.csv")

assert os.path.isdir(ckpt_dir), f"Checkpoint dir not found: {ckpt_dir}"
assert os.path.isfile(csv_file),  f"CSV not found: {csv_file}"

# ─── 2) Prepare model & extractor ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use the base extractor (no need for preprocessor_config.json)
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model     = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt_dir, local_files_only=True)
model.to(device).eval()

# build label2id from the checkpoint’s config
id2label = model.config.id2label              # e.g. {'0':'angry','1':'disgust',…}
label2id = {v:int(k) for k,v in id2label.items()}

print(f"→ Using checkpoint: {ckpt_dir}")
print(f"→ Device: {device}")
print(f"→ CSV: {csv_file}")

# ─── 3) Read metadata ─────────────────────────────────────────────────
df = pd.read_csv(csv_file)
df["path"] = df["path"].str.replace(r"\\", "/", regex=True)

# ─── 4) Inference loop ────────────────────────────────────────────────
correct = 0
for rel, true_lbl in zip(df["path"], df["label"]):
    # resolve absolute WAV path
    wav_path = rel if os.path.isabs(rel) else os.path.join(project_root, rel)
    # load & resample
    wav, sr = librosa.load(wav_path, sr=extractor.sampling_rate)
    # extract features
    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    input_vals    = inputs["input_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    # forward
    with torch.no_grad():
        logits = model(input_vals, attention_mask=attention_mask).logits
    pred_id = int(logits.argmax(-1).cpu())
    if pred_id == label2id[true_lbl]:
        correct += 1

# ─── 5) Report ────────────────────────────────────────────────────────
total = len(df)
acc   = correct / total
print(f"\n✅ Total:   {total}")
print(f"✅ Correct: {correct}")
print(f"✅ Accuracy: {acc:.4f} ({acc*100:.1f}%)")
