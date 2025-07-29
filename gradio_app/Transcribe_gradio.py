import os
import sys
import tempfile
from faster_whisper import WhisperModel

model_size = "large-v3"

def transcribe_file(audio_input, model_instance):
    # audio_input can be a file path or a NamedString-like object with .name and .value (bytes)
    if hasattr(audio_input, 'value') and hasattr(audio_input, 'name'):
        # Gradio-style NamedString: write bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_input.name)[-1]) as tmp:
            tmp.write(audio_input.value)
            tmp_path = tmp.name
        audio_path = tmp_path
    else:
        # Assume it's a file path
        audio_path = audio_input
    try:
        segments, _ = model_instance.transcribe(audio_path, beam_size=5, language="en", condition_on_previous_text=False)
        transcript = "".join(segment.text for segment in segments).strip()
        print(transcript)
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Transcribe_gradio.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    print(f"Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Model loaded successfully.")
    transcribe_file(audio_file, model) 