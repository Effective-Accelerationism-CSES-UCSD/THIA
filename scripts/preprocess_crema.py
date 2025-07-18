import torchaudio
import os
import pandas as pd

# ✅ Make sure this matches the folder where your .wav files are
DATA_DIR = "data/CREMA-D"
OUTPUT_CSV = "data/metadata.csv"

label_map = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "DIS": "disgust",
    "FEA": "fear"
}

output = []

print(f"✅ Scanning directory: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory {DATA_DIR} does not exist!")

files = os.listdir(DATA_DIR)
if not files:
    raise FileNotFoundError(f"No files found in {DATA_DIR}!")

for fname in files:
    print(f"Found: {fname}")
    if fname.endswith(".wav"):
        parts = fname.split("_")
        if len(parts) < 3:
            print(f"⚠️ Skipping invalid file name: {fname}")
            continue
        label_code = parts[2]
        label = label_map.get(label_code, "unknown")
        path = os.path.join(DATA_DIR, fname)

        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            torchaudio.save(path, waveform, 16000)

        output.append({"path": path, "label": label})

if not output:
    raise RuntimeError(f"✅ Found files but none processed. Check file names and extensions!")

df = pd.DataFrame(output)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Processed {len(output)} files.")
print(f"✅ Metadata saved to {OUTPUT_CSV}")
