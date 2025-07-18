# scripts/create_metadata_ravdess.py
import os
import pandas as pd

# Map RAVDESS code → emotion in your 6-class head
emotion_map = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "neutral"
}

rows = []
for root, _, files in os.walk("data/RAVDESS"):
    for fn in files:
        if fn.lower().endswith(".wav"):
            parts = fn.split("-")
            emo   = emotion_map.get(parts[2], None)
            if emo is None:
                continue                # skip anything unmapped
            rel_p = os.path.join(root, fn).replace("\\", "/")
            rows.append({"path": rel_p, "label": emo})

df = pd.DataFrame(rows)
df.to_csv("data/metadata_ravdess.csv", index=False)
print(f"Wrote data/metadata_ravdess.csv with {len(df)} rows")
