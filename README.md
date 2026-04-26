# Emotion AI - AI-Powered Emotion Detection System

A production-ready, modular backend system for emotion detection using AI. Supports image-based facial emotion recognition, multilingual voice-based emotion analysis, and contextual motivational quote generation.

---

## Features

### 1. Image-Based Emotion Detection
- Face detection using MTCNN (via FER library)
- CNN-based emotion classification: Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust
- Returns confidence scores for all emotions
- Multi-face detection support

### 2. Voice-Based Emotion Detection (Multilingual)
- Offline speech recognition using Vosk
- Supported languages: **English, Hindi, Marathi, Tamil, Telugu**
- Sentiment analysis using `nlptown/bert-base-multilingual-uncased-sentiment`
- Maps sentiment polarity to emotion labels

### 3. Quote Generation
- Generates motivational quotes using HuggingFace `distilgpt2`
- Emotion-specific prompt engineering
- Fallback to curated quotes for quality assurance
- Clean output with incomplete sentence filtering

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Image Emotion | FER (OpenCV + MTCNN + CNN) |
| Voice Recognition | Vosk (offline) |
| NLP Sentiment | `nlptown/bert-base-multilingual-uncased-sentiment` |
| Text Generation | `distilgpt2` (HuggingFace Transformers) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── routes/
│   │   ├── image.py               # /predict-image endpoint
│   │   ├── voice.py               # /predict-voice endpoint
│   │   └── quote.py               # /generate-quote endpoint
│   ├── services/
│   │   ├── image_emotion.py       # Image emotion detection logic
│   │   ├── voice_emotion.py       # Voice emotion detection logic
│   │   └── quote_generator.py     # Quote generation logic
│   ├── utils/
│   │   └── preprocessing.py       # Image/audio preprocessing utilities
│   └── models/
│       └── vosk_model/            # Vosk language models (download separately)
├── frontend/
│   └── index.html                 # Testing UI
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Minimum 8GB RAM
- GPU optional but recommended

### 1. Clone the Repository
```bash
git clone <repo-url>
cd emotion-ai
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Vosk Models

Download the required Vosk models from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models):

**Recommended models:**
- `vosk-model-small-en-us-0.15` (English)
- `vosk-model-small-hi-0.22` (Hindi)

```bash
# Create model directory
mkdir -p backend/models/vosk_model

# Download and extract English model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d backend/models/vosk_model/
mv backend/models/vosk_model/vosk-model-small-en-us-0.15 backend/models/vosk_model/en

# Download and extract Hindi model
wget https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
unzip vosk-model-small-hi-0.22.zip -d backend/models/vosk_model/
mv backend/models/vosk_model/vosk-model-small-hi-0.22 backend/models/vosk_model/hi
```

### 5. Run the Server
```bash
python -m backend.main
```

The server will start at `http://localhost:8000`.

---

## API Endpoints

### POST `/predict-image`
Detect emotion from a face image.

**Input:** `multipart/form-data` with `file` field (JPG, PNG, BMP, WebP)

**Response:**
```json
{
  "status": "success",
  "data": {
    "emotion": "Happy",
    "confidence": 0.9823,
    "all_emotions": {
      "Happy": 0.9823,
      "Sad": 0.0012,
      "Angry": 0.0034,
      "Surprise": 0.0056,
      "Neutral": 0.0045,
      "Fear": 0.0018,
      "Disgust": 0.0012
    },
    "bounding_box": {"x": 120, "y": 80, "width": 200, "height": 200},
    "faces_detected": 1
  }
}
```

### POST `/predict-voice`
Detect emotion from voice audio.

**Input:** `multipart/form-data` with `file` (WAV, mono, 16-bit PCM) and `language` field

**Supported languages:** `en`, `hi`, `mr`, `ta`, `te`

**Response:**
```json
{
  "status": "success",
  "data": {
    "text": "I am feeling very happy today",
    "emotion": "Happy",
    "sentiment_label": "positive",
    "sentiment_score": 0.9234,
    "language": "English"
  }
}
```

### POST `/generate-quote`
Generate a motivational quote based on emotion.

**Input:** `application/json`
```json
{
  "emotion": "Happy"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "emotion": "Happy",
    "quote": "Happiness is not something ready-made. It comes from your own actions.",
    "source": "fallback"
  }
}
```

### GET `/supported-languages`
List supported languages for voice analysis.

### GET `/health`
Health check endpoint.

---

## Testing

### Using the UI
Open `http://localhost:8000` in your browser to access the testing UI.

### Using curl

**Test image emotion:**
```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@test_image.jpg"
```

**Test voice emotion:**
```bash
curl -X POST http://localhost:8000/predict-voice \
  -F "file=@test_audio.wav" \
  -F "language=en"
```

**Test quote generation:**
```bash
curl -X POST http://localhost:8000/generate-quote \
  -H "Content-Type: application/json" \
  -d '{"emotion": "Happy"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## Environment Requirements

| Requirement | Value |
|-------------|-------|
| Python | 3.10+ |
| OS | Windows / Linux |
| RAM | 8GB minimum |
| GPU | Optional (recommended for faster inference) |

---

## Constraints

- No paid APIs used
- Works fully offline (after model download)
- Supports multilingual voice input
- Minimal dependencies for core functionality

---

## License

MIT
