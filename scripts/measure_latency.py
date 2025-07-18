#!/usr/bin/env python
"""
Measure Wav2Vec2 inference latency (single-shot & averaged) on a 16 kHz WAV file.
By default, if --audio is omitted, the script picks the first .wav file under data/CREMA-D.
"""
import argparse
import time
import torch
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ─── Project paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "CREMA-D"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "wav2vec2-ser" / "checkpoint-10241"

# ─── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Wav2Vec2 inference latency (single-shot & averaged)."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help=(
            "Path to a 16 kHz WAV file under data/CREMA-D. "
            "If omitted, the first .wav in data/CREMA-D will be used."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to your fine-tuned checkpoint folder"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of inference repetitions for averaging (default: 50)"
    )
    return parser.parse_args()

# ─── Model & processor loading ─────────────────────────────────────────────────
def load_model_and_processor(model_dir: Path):
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# ─── Prepare a single example input ───────────────────────────────────────────
def prepare_input(audio_path: Path, processor, device):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"Sampling rate must be 16 kHz, but got {sr}")
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    return {k: v.to(device) for k, v in inputs.items()}

# ─── Latency measurement ───────────────────────────────────────────────────────
def measure_latency(model, inputs, device, num_runs: int):
    # Warm-up (especially important on GPU)
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)

    # Single-shot timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) * 1000

    # Averaged timing
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t_start)
    avg_ms = (sum(times) / len(times)) * 1000

    return single_ms, avg_ms

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve audio path
    if args.audio:
        audio_path = Path(args.audio)
    else:
        wavs = sorted(AUDIO_DIR.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {AUDIO_DIR}")
        audio_path = wavs[0]

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load model & processor
    processor, model, device = load_model_and_processor(Path(args.checkpoint))

    # Prepare input tensor
    inputs = prepare_input(audio_path, processor, device)

    # Measure latency
    single_ms, avg_ms = measure_latency(model, inputs, device, args.runs)

    print(f"▶︎ Single-shot latency: {single_ms:.1f} ms")
    print(f"▶︎ Avg latency over {args.runs} runs: {avg_ms:.1f} ms")

if __name__ == "__main__":
    main()
