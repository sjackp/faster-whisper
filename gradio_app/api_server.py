from flask import Flask, request, jsonify
import tempfile
import os
import torch
import traceback
from faster_whisper import WhisperModel
import base64
import mimetypes
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global model variable
whisper_model = None
current_model_size = None

def allowed_file(filename: str) -> bool:
    allowed_extensions = {"wav", "mp3", "flac", "m4a", "ogg", "webm"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def load_model(model_size="base"):
    """Load the Whisper model with appropriate device settings"""
    global whisper_model, current_model_size
    
    # Don't reload if same model is already loaded
    if whisper_model is not None and current_model_size == model_size:
        return True
    
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
        current_model_size = model_size
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": whisper_model is not None,
        "current_model": current_model_size
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    """
    Main transcription endpoint
    
    Accepts:
    - File upload via form-data with key 'audio'
    - Base64 encoded audio in JSON with key 'audio_data' and 'filename'
    - Audio file URL (future enhancement)
    
    Parameters:
    - model_size: str (default: "base") - "tiny", "base", "small", "medium", "large-v3"
    - language: str (default: "auto") - language code or "auto"
    - beam_size: int (default: 5) - beam search size (1-10)
    - word_timestamps: bool (default: false) - include word-level timestamps
    - vad_filter: bool (default: true) - use voice activity detection
    """
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
        if not load_model(model_size):
            return jsonify({"error": "Failed to load Whisper model"}), 500
        
        # Handle file input
        audio_path = None
        temp_file = None
        
        if 'audio' in request.files:
            # File upload via form-data
            file = request.files['audio']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not allowed_file(file.filename):
                return jsonify({"error": "Unsupported file type. Supported: WAV, MP3, FLAC, M4A, OGG, WEBM"}), 400
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            file.save(temp_file.name)
            audio_path = temp_file.name
            
        elif request.json and 'audio_data' in request.json:
            # Base64 encoded audio data
            try:
                audio_data = base64.b64decode(request.json['audio_data'])
                filename = request.json.get('filename', 'audio.wav')
                
                if not allowed_file(filename):
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
        
        print(f"Transcribing file: {audio_path}")
        print(f"Transcription parameters: {transcribe_params}")
        
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
        "current_model": current_model_size,
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

if __name__ == '__main__':
    # Load default model on startup
    print("Starting Whisper API Server...")
    load_model("base")
    
    app.run(host='127.0.0.1', port=8002, debug=False)
