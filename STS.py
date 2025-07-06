from faster_whisper import WhisperModel

# Initialize the model
model = WhisperModel("large-v2", device="cpu", compute_type="int8")

# Transcribe the audio file
segments, info = model.transcribe("sample-3.mp3", beam_size=5)

# Print the detected language and transcription
print(f"Detected language: {info.language} ({info.language_probability:.2f})")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
