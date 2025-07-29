# Faster Whisper API Documentation

This API provides audio transcription capabilities using OpenAI's Whisper model via the faster-whisper implementation.

## Quick Start

### Start the API Server
```bash
cd gradio_app
python api_server.py
```

The API will be available at `http://localhost:8000`

### Start the Gradio UI
```bash
cd gradio_app  
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

### Start Both Services Together
```bash
cd gradio_app
python app.py --api
```

This will start:
- Gradio UI on `http://localhost:8001`
- API server on `http://localhost:8002`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API server is running and if a model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "base"
}
```

### 2. Transcribe Audio
**POST** `/transcribe`

Main endpoint for audio transcription.

#### Input Methods

**Method 1: File Upload (Form Data)**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@your_audio_file.mp3" \
  -F "model_size=base" \
  -F "language=auto" \
  -F "beam_size=5" \
  -F "word_timestamps=false" \
  -F "vad_filter=true"
```

**Method 2: Base64 JSON**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_data_here",
    "filename": "audio.mp3",
    "model_size": "base",
    "language": "auto",
    "beam_size": 5,
    "word_timestamps": false,
    "vad_filter": true
  }'
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | string | "base" | Model size: "tiny", "base", "small", "medium", "large-v3" |
| `language` | string | "auto" | Language code or "auto" for auto-detection |
| `beam_size` | integer | 5 | Beam search size (1-10) |
| `word_timestamps` | boolean | false | Include word-level timestamps |
| `vad_filter` | boolean | true | Use voice activity detection |

#### Supported Audio Formats
- WAV
- MP3  
- FLAC
- M4A
- OGG
- WEBM

#### Response

**Success (200):**
```json
{
  "transcript": "Hello world, this is a test transcription.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello world, this is a test transcription.",
      "tokens": [50364, 2425, 1002, 11, 341, 307, 257, 1500, 35288, 13, 50542],
      "temperature": 0.0,
      "avg_logprob": -0.3,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01,
      "words": [] // Only included if word_timestamps=true
    }
  ],
  "info": {
    "language": "en",
    "language_probability": 0.99,
    "duration": 3.5,
    "duration_after_vad": 3.2
  },
  "parameters": {
    "model_size": "base",
    "language": "auto",
    "beam_size": 5,
    "word_timestamps": false,
    "vad_filter": true
  }
}
```

**Error (400/500):**
```json
{
  "error": "Error message describing what went wrong"
}
```

### 3. Get Available Models
**GET** `/models`

Get list of available Whisper model sizes.

**Response:**
```json
{
  "models": ["tiny", "base", "small", "medium", "large-v3"],
  "current_model": "base",
  "model_loaded": true
}
```

### 4. Get Supported Languages
**GET** `/languages`

Get list of supported languages.

**Response:**
```json
{
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
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Filipino",
    "sw": "Swahili",
    "af": "Afrikaans",
    "zu": "Zulu",
    "xh": "Xhosa",
    "st": "Southern Sotho",
    "tn": "Tswana",
    "ts": "Tsonga",
    "ve": "Venda",
    "nr": "Southern Ndebele",
    "ss": "Swati",
    "sn": "Shona",
    "rw": "Kinyarwanda",
    "lg": "Ganda",
    "ak": "Akan",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "so": "Somali",
    "am": "Amharic",
    "or": "Odia",
    "bn": "Bengali",
    "ur": "Urdu",
    "si": "Sinhala",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "mn": "Mongolian",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "uz": "Uzbek",
    "tg": "Tajik",
    "tk": "Turkmen",
    "fa": "Persian",
    "ps": "Pashto",
    "sd": "Sindhi",
    "ne": "Nepali",
    "dz": "Dzongkha",
    "bo": "Tibetan",
    "ug": "Uyghur",
    "jv": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
    "war": "Waray",
    "ilo": "Ilocano",
    "pam": "Kapampangan",
    "bik": "Bikol",
    "hil": "Hiligaynon",
    "ceb": "Cebuano",
    "war": "Waray",
    "ilo": "Ilocano",
    "pam": "Kapampangan",
    "bik": "Bikol",
    "hil": "Hiligaynon"
  }
}
```

## Python Client Examples

### Basic Usage
```python
import requests

# File upload
with open('audio.mp3', 'rb') as f:
    files = {'audio': f}
    data = {
        'model_size': 'base',
        'language': 'auto',
        'word_timestamps': False
    }
    response = requests.post('http://localhost:8000/transcribe', files=files, data=data)
    result = response.json()
    print(result['transcript'])
```

### Base64 Encoding
```python
import requests
import base64

# Read and encode audio file
with open('audio.mp3', 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Send request
data = {
    'audio_data': audio_data,
    'filename': 'audio.mp3',
    'model_size': 'base',
    'language': 'auto'
}
response = requests.post('http://localhost:8000/transcribe', json=data)
result = response.json()
print(result['transcript'])
```

### Advanced Usage with Error Handling
```python
import requests
import json

def transcribe_audio(file_path, model_size='base', language='auto'):
    try:
        with open(file_path, 'rb') as f:
            files = {'audio': f}
            data = {
                'model_size': model_size,
                'language': language,
                'beam_size': 5,
                'word_timestamps': True,
                'vad_filter': True
            }
            
            response = requests.post('http://localhost:8000/transcribe', 
                                   files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'transcript': result['transcript'],
                    'segments': result['segments'],
                    'language': result['info']['language'],
                    'duration': result['info']['duration']
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Could not connect to server'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Usage
result = transcribe_audio('audio.mp3', model_size='large-v3', language='en')
if result['success']:
    print(f"Transcript: {result['transcript']}")
    print(f"Language: {result['language']}")
    print(f"Duration: {result['duration']}s")
else:
    print(f"Error: {result['error']}")
```

## N8N Integration Examples

### Example 1: Basic Audio File Transcription

```json
{
  "nodes": [
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/transcribe",
        "sendBody": true,
        "specifyBody": "multipart-form-data",
        "bodyParameters": {
          "parameters": [
            {
              "name": "audio",
              "value": "={{ $binary.audio.data }}"
            },
            {
              "name": "model_size",
              "value": "base"
            },
            {
              "name": "language",
              "value": "auto"
            },
            {
              "name": "word_timestamps",
              "value": "false"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [500, 300],
      "id": "transcribe-audio",
      "name": "Transcribe Audio"
    }
  ]
}
```

### Example 2: Advanced Transcription with Custom Parameters

```json
{
  "nodes": [
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "model_config",
              "name": "transcription_params",
              "value": "={\n  \"model_size\": \"large-v3\",\n  \"language\": \"en\",\n  \"beam_size\": 8,\n  \"word_timestamps\": true,\n  \"vad_filter\": true\n}",
              "type": "object"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [300, 300],
      "id": "set-params",
      "name": "Set Parameters"
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/transcribe",
        "sendBody": true,
        "specifyBody": "multipart-form-data",
        "bodyParameters": {
          "parameters": [
            {
              "name": "audio",
              "value": "={{ $binary.audio.data }}"
            },
            {
              "name": "model_size",
              "value": "={{ $json.transcription_params.model_size }}"
            },
            {
              "name": "language",
              "value": "={{ $json.transcription_params.language }}"
            },
            {
              "name": "beam_size",
              "value": "={{ $json.transcription_params.beam_size }}"
            },
            {
              "name": "word_timestamps",
              "value": "={{ $json.transcription_params.word_timestamps }}"
            },
            {
              "name": "vad_filter",
              "value": "={{ $json.transcription_params.vad_filter }}"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [500, 300],
      "id": "transcribe-advanced",
      "name": "Advanced Transcribe"
    }
  ],
  "connections": {
    "Set Parameters": {
      "main": [
        [
          {
            "node": "Advanced Transcribe",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Example 3: Base64 JSON Method (for URL-downloaded audio)

```json
{
  "nodes": [
    {
      "parameters": {
        "url": "https://example.com/audio.mp3",
        "options": {
          "response": {
            "response": {
              "responseFormat": "file"
            }
          }
        }
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [300, 300],
      "id": "download-audio",
      "name": "Download Audio"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "base64_data",
              "name": "audio_base64",
              "value": "={{ $binary.data.data.toString('base64') }}",
              "type": "string"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [500, 300],
      "id": "convert-base64",
      "name": "Convert to Base64"
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/transcribe",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={\n  \"audio_data\": \"{{ $json.audio_base64 }}\",\n  \"filename\": \"audio.mp3\",\n  \"model_size\": \"base\",\n  \"language\": \"auto\",\n  \"word_timestamps\": false\n}",
        "options": {}
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [700, 300],
      "id": "transcribe-base64",
      "name": "Transcribe Base64"
    }
  ],
  "connections": {
    "Download Audio": {
      "main": [
        [
          {
            "node": "Convert to Base64",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Convert to Base64": {
      "main": [
        [
          {
            "node": "Transcribe Base64",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Example 4: Complete Workflow with Error Handling

```json
{
  "nodes": [
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/transcribe",
        "sendBody": true,
        "specifyBody": "multipart-form-data",
        "bodyParameters": {
          "parameters": [
            {
              "name": "audio",
              "value": "={{ $binary.audio.data }}"
            },
            {
              "name": "model_size",
              "value": "base"
            },
            {
              "name": "language",
              "value": "auto"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [500, 300],
      "id": "transcribe-request",
      "name": "Transcribe Request"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "success_result",
              "name": "transcript",
              "value": "={{ $json.transcript }}",
              "type": "string"
            },
            {
              "id": "language_detected",
              "name": "detected_language",
              "value": "={{ $json.info.language }}",
              "type": "string"
            },
            {
              "id": "duration",
              "name": "audio_duration",
              "value": "={{ $json.info.duration }}",
              "type": "number"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [700, 200],
      "id": "process-success",
      "name": "Process Success"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "error_msg",
              "name": "error_message",
              "value": "=Transcription failed: {{ $json.error || 'Unknown error' }}",
              "type": "string"
            }
          ]
        }
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [700, 400],
      "id": "process-error",
      "name": "Process Error"
    }
  ],
  "connections": {
    "Transcribe Request": {
      "main": [
        [
          {
            "node": "Process Success",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Process Error",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

## Docker Usage

### Build and Run API Server
```bash
# Build Docker image
docker build -t whisper-api .

# Run API server
docker run -p 8000:8000 whisper-api

# Run with GPU support (if available)
docker run --gpus all -p 8000:8000 whisper-api
```

### Docker Compose Example
```yaml
version: '3.8'
services:
  whisper-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_SIZE=base
    volumes:
      - ./temp:/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance Tips

1. **Model Selection**: 
   - `tiny`: Fastest, least accurate (39M parameters)
   - `base`: Good balance of speed and accuracy (74M parameters)
   - `small`: Better accuracy, moderate speed (244M parameters)
   - `medium`: High accuracy, slower (769M parameters)
   - `large-v3`: Most accurate, slowest (1550M parameters)

2. **Hardware**: 
   - GPU (CUDA) significantly faster than CPU
   - More RAM allows larger models
   - SSD storage for faster model loading

3. **Parameters**:
   - Lower `beam_size` for faster processing (1-5 recommended)
   - Disable `word_timestamps` if not needed
   - Use `vad_filter=true` to skip silent parts
   - Set specific `language` instead of "auto" for better performance

4. **Batch Processing**: 
   - Keep the API server running to avoid model reload overhead
   - Process multiple files through the same server instance
   - Use connection pooling for multiple requests

5. **Memory Management**:
   - Use `int8` quantization for CPU inference
   - Use `float16` for GPU inference
   - Monitor memory usage with larger models

## Error Handling

Common error responses:

- `400 Bad Request`: Invalid input (unsupported file format, missing audio, invalid parameters)
- `500 Internal Server Error`: Server error (model loading failed, transcription error, out of memory)

Always check the `error` field in JSON responses for detailed error messages.

### Common Issues and Solutions

1. **Model Loading Fails**:
   - Check available disk space for model download
   - Verify internet connection for model download
   - Ensure sufficient RAM for model loading

2. **CUDA Out of Memory**:
   - Use smaller model size
   - Enable `int8` quantization
   - Reduce `beam_size`
   - Process shorter audio files

3. **Slow Transcription**:
   - Use GPU if available
   - Choose smaller model size
   - Reduce `beam_size`
   - Enable `vad_filter`

4. **File Format Issues**:
   - Ensure audio file is not corrupted
   - Convert to supported format (WAV, MP3, FLAC, M4A, OGG, WEBM)
   - Check file size (very large files may timeout)

## Requirements

- Python 3.8+
- torch
- faster-whisper
- flask
- gradio (for UI)

Install dependencies:
```bash
pip install torch faster-whisper flask gradio
```

For GPU support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Security Considerations

1. **File Upload**: The API accepts file uploads - implement proper validation in production
2. **Rate Limiting**: Consider implementing rate limiting for production use
3. **Authentication**: Add authentication for production deployments
4. **CORS**: Configure CORS settings for web applications
5. **Input Validation**: Validate all input parameters and file types
6. **Resource Limits**: Set appropriate timeouts and file size limits

## Production Deployment

For production deployment, consider:

1. **Reverse Proxy**: Use nginx or Apache as reverse proxy
2. **Process Manager**: Use gunicorn, uwsgi, or systemd for process management
3. **Load Balancing**: Use multiple API instances behind a load balancer
4. **Monitoring**: Implement health checks and monitoring
5. **Logging**: Configure proper logging and error tracking
6. **SSL/TLS**: Use HTTPS for secure communication
7. **Container Orchestration**: Use Kubernetes or Docker Swarm for scaling
