#!/usr/bin/env python3
import os
import torch
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Settings - adjust if needed
MODEL_DIR = "models/wav2vec2-ser"    # Top-level folder with config.json, model.safetensors, etc.
CSV_PATH  = "data/metadata_ravdess.csv"      # Path to your metadata CSV

def main():
    # 1) Setup device and load model + processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    model.to(device).eval()

    # Build label2id map from the saved model’s config
    id2label = model.config.id2label              # e.g. {'0':'angry', '1':'disgust', ...}
    label2id = {v: int(k) for k, v in id2label.items()}

    print(f"→ Loaded model from {MODEL_DIR} on {device}")
    print(f"→ Reading metadata from {CSV_PATH}")

    # 2) Read CSV & normalize paths
    df = pd.read_csv(CSV_PATH)
    df["path"] = df["path"].str.replace(r"\\", "/", regex=True)

    # 3) Inference loop
    correct = 0
    for wav_rel, true_label in tqdm(zip(df["path"], df["label"]),
                                    total=len(df),
                                    desc="Evaluating"):
        # Resolve to absolute path if relative
        wav_path = wav_rel if os.path.isabs(wav_rel) else os.path.join(os.getcwd(), wav_rel)
        # Load and resample
        wav, sr = librosa.load(wav_path, sr=processor.feature_extractor.sampling_rate)
        # Tokenize and prepare inputs
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = model(**inputs).logits

        pred = int(torch.argmax(logits, dim=-1)[0].cpu())
        if pred == label2id[true_label]:
            correct += 1

    # 4) Report
    total = len(df)
    acc = correct / total
    print(f"\n✅ Total samples: {total}")
    print(f"✅ Correct      : {correct}")
    print(f"✅ Accuracy     : {acc:.4f} ({acc*100:.1f}%)")

if __name__ == "__main__":
    main()
