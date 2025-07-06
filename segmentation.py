from faster_whisper import WhisperModel
from pydub import AudioSegment
import os

def transcribe_in_1s_chunks(audio_path: str,
                            model_name: str = "large-v2",
                            device: str = "cpu",
                            compute_type: str = "int8",
                            beam_size: int = 5) -> None:

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_ms = 1000  # 1 s

    all_text = []
    detected_language = None
    language_prob = None

    # First pass: detect language on the very first chunk only
    # -------------------------------------------------------
    first_chunk = audio[:chunk_ms]
    tmp0 = "__tmp0.wav"
    first_chunk.export(tmp0, format="wav")
    # This first call will detect language
    segments, info = model.transcribe(tmp0, beam_size=beam_size)
    detected_language = info.language
    language_prob    = info.language_probability
    os.remove(tmp0)

    # Collect text from the first chunk
    text0 = " ".join(seg.text for seg in segments).strip()
    all_text.append(text0)

    # Now transcribe the rest, telling the model the language
    # so it won't run detect_language again
    for start_ms in range(chunk_ms, duration_ms, chunk_ms):
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk = audio[start_ms:end_ms]
        tmp = f"__tmp_{start_ms}.wav"
        chunk.export(tmp, format="wav")

        # Pass language=detected_language to skip detection
        segments, _ = model.transcribe(
            tmp,
            beam_size=beam_size,
            language=detected_language
        )

        all_text.append(" ".join(seg.text for seg in segments).strip())
        os.remove(tmp)

    # Final transcript
    transcript = " ".join(all_text).replace("  ", " ").strip()

    print(f"Detected language: {detected_language} ({language_prob:.2f})\n")
    print(transcript)


if __name__ == "__main__":
    transcribe_in_1s_chunks("sample-3.mp3")
