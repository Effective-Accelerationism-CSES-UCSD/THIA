import subprocess
import sys

def synthesize(text):
    # Adjust these paths and parameters as needed
    restore_step = "900000"
    mode = "single"
    preprocess_config = "config/LJSpeech/preprocess.yaml"
    model_config = "config/LJSpeech/model.yaml"
    train_config = "config/LJSpeech/train.yaml"

    cmd = [
        "python3", "synthesize.py",
        "--text", text,
        "--restore_step", restore_step,
        "--mode", mode,
        "-p", preprocess_config,
        "-m", model_config,
        "-t", train_config
    ]

    # Run the synthesis command and wait for it to finish
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Synthesis successful! Output saved to output/result/")
    else:
        print("Error during synthesis:")
        print(result.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 TTS.py 'Your text here'")
        sys.exit(1)

    input_text = sys.argv[1]
    synthesize(input_text)
