#!/usr/bin/env python3
import os
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    AutoConfig
)

# ─── 0) Resolve paths ─────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(__file__)                             # .../thia-audio-sentiment/scripts
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))  # .../thia-audio-sentiment

TOP_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "wav2vec2-ser")
CKPT_DIR      = os.path.join(TOP_MODEL_DIR, "checkpoint-10241")
CSV_PATH      = os.path.join(PROJECT_ROOT, "data", "metadata.csv")

assert os.path.isdir(TOP_MODEL_DIR), f"Model dir not found: {TOP_MODEL_DIR}"
assert os.path.isdir(CKPT_DIR),      f"Checkpoint not found: {CKPT_DIR}"
assert os.path.isfile(CSV_PATH),     f"CSV not found: {CSV_PATH}"

# ─── 1) Load processor, config & model ───────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processor knows how to produce both input_values & attention_mask
processor = Wav2Vec2Processor.from_pretrained(
    TOP_MODEL_DIR,
    local_files_only=True
)

# Load correct config (num_labels=6) from top-level save
config  = AutoConfig.from_pretrained(TOP_MODEL_DIR, local_files_only=True)

# Load checkpoint weights into that config
model   = Wav2Vec2ForSequenceClassification.from_pretrained(
    CKPT_DIR,
    config=config,
    local_files_only=True
)

model.to(device).eval()

print(f"✅ Loaded model from {CKPT_DIR} on {device}")
print(f"✅ Reading metadata from {CSV_PATH}")

# ─── 2) Read CSV & normalize paths ────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["path"] = df["path"].str.replace(r"\\", "/", regex=True)

# Build label2id map (config.id2label is {'0':'angry',…})
label2id = {v:int(k) for k,v in model.config.id2label.items()}

# ─── 3) Inference & accuracy loop ───────────────────────────────────────
correct = 0
for rel_path, true_lbl in tqdm(zip(df["path"], df["label"]),
                               total=len(df),
                               desc="Evaluating"):
    # resolve absolute path
    wav_path = rel_path if os.path.isabs(rel_path) else os.path.join(PROJECT_ROOT, rel_path)

    # load + resample
    wav, sr = librosa.load(wav_path, sr=processor.feature_extractor.sampling_rate)

    # tokenize (this yields input_values + attention_mask)
    inputs = processor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k:v.to(device) for k,v in inputs.items()}

    # forward
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = int(logits.argmax(-1).cpu())
    if pred_id == label2id[true_lbl]:
        correct += 1

# ─── 4) Report ───────────────────────────────────────────────────────────
total = len(df)
acc   = correct / total
print(f"\n🎯 Total samples: {total}")
print(f"✅ Correct     : {correct}")
print(f"⚡️ Accuracy   : {acc:.4f} ({acc*100:.1f}%)")
