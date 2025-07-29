import gradio as gr
import tempfile
import os
import torch
import traceback
import argparse
import threading
import sys
from faster_whisper import WhisperModel

# Utility functions
def allowed_file(filename: str) -> bool:
    allowed_extensions = {"wav", "mp3", "flac", "m4a"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def secure_filename(filename: str) -> str:
    # Simple filename sanitizer
    return os.path.basename(filename).replace(" ", "_").replace("..", "_")

# Global model variable
whisper_model = None
current_filename = None

def load_model(model_size="base"):
    """Load the Whisper model with appropriate device settings"""
    global whisper_model
    
    # Determine the best available device and compute type
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"Using CUDA with {compute_type}")
    else:
        device = "cpu"
        compute_type = "int8"
        print(f"Using CPU with {compute_type}")
    
    try:
        print(f"Loading Whisper model '{model_size}' on {device}...")
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def transcribe_audio(file, model_size, language, beam_size, word_timestamps, vad_filter):
    print("transcribe_audio called")
    
    global whisper_model, current_filename
    
    if file is None:
        return "", "No file uploaded."
    
    # Check if we need to reload the model
    if whisper_model is None:
        success = load_model(model_size)
        if not success:
            return "", "Error: Failed to load Whisper model."
    
    try:
        # Handle Gradio file upload
        if hasattr(file, 'name'):
            filename = secure_filename(file.name)
            if not allowed_file(filename):
                return "", "Unsupported file type. Please upload WAV, MP3, FLAC, or M4A files."
            
            # Store the filename for download (without extension)
            current_filename = os.path.splitext(filename)[0]
            
            # Get the file path from Gradio
            audio_path = file.name if hasattr(file, 'name') else file
            
            print(f"Transcribing file: {audio_path}")
            
            # Set up transcription parameters
            transcribe_params = {
                "beam_size": int(beam_size),
                "language": None if language == "auto" else language,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
                "condition_on_previous_text": False
            }
            
            print(f"Transcription parameters: {transcribe_params}")
            
            # Perform transcription
            segments, info = whisper_model.transcribe(audio_path, **transcribe_params)
            
            # Extract transcript
            transcript_parts = []
            for segment in segments:
                if word_timestamps:
                    # Include timestamps if requested
                    transcript_parts.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                else:
                    transcript_parts.append(segment.text)
            
            transcript = "\n".join(transcript_parts).strip()
            
            if not transcript:
                return "", "No speech detected in the audio file."
            
            # Return transcript and success message (order matters for UI)
            success_msg = f"Transcription completed successfully!\nDetected language: {info.language}\nDuration: {info.duration:.2f}s"
            return transcript, success_msg
            
        else:
            return "", "Error: Invalid file upload format."
            
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return "", error_msg

def download_txt(transcript):
    global current_filename
    
    if not transcript or transcript.strip() == "":
        return None
    
    # Use the stored filename or default
    base_name = current_filename if current_filename else "transcript"
    filename = f"{base_name}.txt"
    
    # Create a temporary file with the proper name
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    return temp_path

def get_model_sizes():
    return ["tiny", "base", "small", "medium", "large-v3"]

def get_languages():
    return ["auto", "en", "fr", "de", "es", "it", "zh"]

with gr.Blocks() as demo:
    gr.Markdown("# Faster Whisper Transcription UI")
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(label="Upload WAV/MP3/FLAC/M4A file")
            model_size = gr.Dropdown(get_model_sizes(), value="base", label="Model Size")
            language = gr.Dropdown(get_languages(), value="auto", label="Language")
            beam_size = gr.Slider(1, 10, value=5, step=1, label="Beam Size")
            word_timestamps = gr.Checkbox(label="Word Timestamps", value=False)
            vad_filter = gr.Checkbox(label="VAD Filter", value=True)
            transcribe_btn = gr.Button("Transcribe")
        with gr.Column():
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            save_btn = gr.DownloadButton("Save as .txt")
            status_output = gr.Textbox(label="Status", lines=3)

    transcribe_btn.click(
        transcribe_audio,
        inputs=[audio_input, model_size, language, beam_size, word_timestamps, vad_filter],
        outputs=[transcript_output, status_output],
    )
    save_btn.click(
        download_txt,
        inputs=transcript_output,
        outputs=save_btn,
    )

def start_api_server():
    """Start the Flask API server in a separate thread"""
    from flask import Flask, request, jsonify
    import base64
    from werkzeug.utils import secure_filename as werkzeug_secure_filename
    
    app = Flask(__name__)
    
    def api_allowed_file(filename: str) -> bool:
        allowed_extensions = {"wav", "mp3", "flac", "m4a", "ogg", "webm"}
        return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

    def api_load_model(model_size="base"):
        """Load the Whisper model with appropriate device settings"""
        global whisper_model
        
        # Don't reload if same model is already loaded
        if whisper_model is not None:
            return True
        
        return load_model(model_size)

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "model_loaded": whisper_model is not None,
            "current_model": "base" if whisper_model is not None else None
        })

    @app.route('/transcribe', methods=['POST'])
    def transcribe_endpoint():
        """Main transcription endpoint"""
        try:
            # Get parameters
            model_size = request.form.get('model_size', request.json.get('model_size', 'base') if request.json else 'base')
            language = request.form.get('language', request.json.get('language', 'auto') if request.json else 'auto')
            beam_size = int(request.form.get('beam_size', request.json.get('beam_size', 5) if request.json else 5))
            word_timestamps = request.form.get('word_timestamps', request.json.get('word_timestamps', False) if request.json else False)
            vad_filter = request.form.get('vad_filter', request.json.get('vad_filter', True) if request.json else True)
            
            # Convert string booleans to actual booleans
            if isinstance(word_timestamps, str):
                word_timestamps = word_timestamps.lower() in ('true', '1', 'yes')
            if isinstance(vad_filter, str):
                vad_filter = vad_filter.lower() in ('true', '1', 'yes')
            
            # Load model if needed
            if not api_load_model(model_size):
                return jsonify({"error": "Failed to load Whisper model"}), 500
            
            # Handle file input
            audio_path = None
            temp_file = None
            
            if 'audio' in request.files:
                # File upload via form-data
                file = request.files['audio']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                if not api_allowed_file(file.filename):
                    return jsonify({"error": "Unsupported file type. Supported: WAV, MP3, FLAC, M4A, OGG, WEBM"}), 400
                
                # Save uploaded file temporarily
                filename = werkzeug_secure_filename(file.filename)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                file.save(temp_file.name)
                audio_path = temp_file.name
                
            elif request.json and 'audio_data' in request.json:
                # Base64 encoded audio data
                try:
                    audio_data = base64.b64decode(request.json['audio_data'])
                    filename = request.json.get('filename', 'audio.wav')
                    
                    if not api_allowed_file(filename):
                        return jsonify({"error": "Unsupported file type. Supported: WAV, MP3, FLAC, M4A, OGG, WEBM"}), 400
                    
                    # Save decoded data to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                    temp_file.write(audio_data)
                    temp_file.close()
                    audio_path = temp_file.name
                    
                except Exception as e:
                    return jsonify({"error": f"Failed to decode base64 audio data: {str(e)}"}), 400
            
            else:
                return jsonify({"error": "No audio file provided. Use 'audio' form field or 'audio_data' JSON field"}), 400
            
            # Set up transcription parameters
            transcribe_params = {
                "beam_size": beam_size,
                "language": None if language == "auto" else language,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
                "condition_on_previous_text": False
            }
            
            print(f"API Transcribing file: {audio_path}")
            print(f"API Transcription parameters: {transcribe_params}")
            
            # Perform transcription
            segments, info = whisper_model.transcribe(audio_path, **transcribe_params)
            
            # Extract transcript and segments
            transcript_parts = []
            segments_data = []
            
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }
                
                if word_timestamps:
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        } for word in segment.words
                    ] if hasattr(segment, 'words') and segment.words else []
                
                segments_data.append(segment_data)
                
                if word_timestamps:
                    transcript_parts.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")
                else:
                    transcript_parts.append(segment.text.strip())
            
            transcript = "\n".join(transcript_parts).strip()
            
            # Clean up temporary file
            if temp_file and os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if not transcript:
                return jsonify({
                    "error": "No speech detected in the audio file",
                    "info": {
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                        "duration_after_vad": info.duration_after_vad
                    }
                }), 400
            
            # Return successful response
            response = {
                "transcript": transcript,
                "segments": segments_data,
                "info": {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "duration_after_vad": info.duration_after_vad
                },
                "parameters": {
                    "model_size": model_size,
                    "language": language,
                    "beam_size": beam_size,
                    "word_timestamps": word_timestamps,
                    "vad_filter": vad_filter
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up temporary file on error
            if 'temp_file' in locals() and temp_file and 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)
            
            error_msg = f"Error during transcription: {str(e)}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            return jsonify({"error": error_msg}), 500

    @app.route('/models', methods=['GET'])
    def get_models():
        """Get available model sizes"""
        return jsonify({
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "current_model": "base" if whisper_model is not None else None,
            "model_loaded": whisper_model is not None
        })

    @app.route('/languages', methods=['GET'])
    def get_languages():
        """Get supported languages"""
        return jsonify({
            "languages": {
                "auto": "Auto-detect",
                "en": "English",
                "es": "Spanish", 
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "ar": "Arabic",
                "hi": "Hindi",
                "tr": "Turkish",
                "pl": "Polish",
                "nl": "Dutch",
                "sv": "Swedish",
                "da": "Danish",
                "no": "Norwegian",
                "fi": "Finnish"
            }
        })
    
    print("Starting API server on http://localhost:8002")
    app.run(host='127.0.0.1', port=8002, debug=False, use_reloader=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Faster Whisper Gradio UI")
    parser.add_argument("--api", action="store_true", help="Also start the API server")
    args = parser.parse_args()
    
    if args.api:
        # Start API server in a separate thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        print("API server started in background on http://localhost:8002")
        print("Starting Gradio UI on http://localhost:8001")
    
    demo.launch(server_name="127.0.0.1", server_port=8001)
