import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)

import torchaudio

# ─── Determine project root based on script location ───
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

MODEL_NAME = "facebook/wav2vec2-base"
DATA_CSV = os.path.join(ROOT_DIR, "data", "metadata.csv")

# --- Check CSV ---
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Metadata file not found at {DATA_CSV}. Please run preprocessing first.")

df = pd.read_csv(DATA_CSV)
if df.empty:
    raise ValueError("Metadata CSV is empty. Check your preprocessing step.")
print(f"✅ Loaded metadata: {len(df)} samples.")

# 🔷 Fix Windows backslashes in CSV paths
# (ensures forward slashes inside 'path' column)
df["path"] = df["path"].str.replace("\\", "/", regex=False)

label_list = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# --- PyTorch Dataset ---
class SERDataset(Dataset):
    def __init__(self, dataframe, processor, label2id, sampling_rate=16000):
        self.dataframe = dataframe
        self.processor = processor
        self.label2id = label2id
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = os.path.join(ROOT_DIR, row["path"])
        label = self.label2id[row["label"]]

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)
        waveform = waveform.squeeze().numpy()

        inputs = self.processor(
            waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )

        return {"input_values": inputs.input_values[0], "labels": torch.tensor(label)}

# --- Prepare Dataset ---
print("✅ Preparing dataset…")
dataset = SERDataset(df, processor, label2id)

# --- Data Collator ---
def custom_collate(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded = processor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )
    padded["labels"] = torch.tensor(labels)
    return padded

# --- Model ---
print("✅ Loading model…")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label
)

# --- Training Args ---
training_args = TrainingArguments(
    output_dir=os.path.join(ROOT_DIR, "models", "wav2vec2-ser"),
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=12,
    weight_decay=0.01,
    logging_dir=os.path.join(ROOT_DIR, "logs"),
    logging_steps=10,
    save_strategy="epoch",
    logging_strategy="steps",
    report_to="none",
    gradient_checkpointing=True
)

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(preds == labels)}

# --- Trainer ---
print("✅ Starting training…")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # optionally split your df
    tokenizer=processor,
    data_collator=custom_collate,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(os.path.join(ROOT_DIR, "models", "wav2vec2-ser"))
processor.save_pretrained(os.path.join(ROOT_DIR, "models", "wav2vec2-ser"))

print("🎉 Training complete. Model saved to models/wav2vec2-ser/")
