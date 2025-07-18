# scripts/evaluate_accuracy.py
import torch
import pandas as pd
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

MODEL_DIR = "models/wav2vec2-ser/checkpoint-931"
CSV_PATH  = "data/metadata.csv"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load the original base processor (it contains the feature-extractor config)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# 2) Load your fine-tuned model (with label mappings in config.json)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device).eval()

# 3) Prepare label→ID mapping
df = pd.read_csv(CSV_PATH)
labels = sorted(df["label"].unique())
label2id = {l:i for i,l in enumerate(labels)}

# 4) Split out 10% test
raw = load_dataset("csv", data_files=CSV_PATH)["train"]
raw = raw.cast_column("path", Audio(sampling_rate=16_000))
test_ds = raw.train_test_split(test_size=0.1, seed=42)["test"]

# 5) Inference loop
correct = 0
for ex in tqdm(test_ds, desc="Evaluating"):
    inp = processor(
        ex["path"]["array"],
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True
    )
    inp = {k:v.to(device) for k,v in inp.items()}
    with torch.no_grad():
        logits = model(**inp).logits

    pred_id = logits.argmax(dim=-1).item()
    true_id = label2id[ex["label"]]
    correct += (pred_id == true_id)

acc = correct / len(test_ds)
print(f"\n▶︎ Test set accuracy: {acc*100:.2f}%")
