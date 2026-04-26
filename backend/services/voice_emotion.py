"""
Voice-based emotion detection service.
Uses Vosk for offline speech recognition and HuggingFace multilingual
sentiment model for emotion classification.
Supports: English, Hindi, Marathi, Tamil, Telugu.
"""

import json
import logging
import os
import wave
from typing import Dict, Optional

from vosk import KaldiRecognizer, Model
from transformers import pipeline

from backend.utils.preprocessing import save_temp_file, cleanup_temp_file

logger = logging.getLogger(__name__)

VOSK_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "vosk_model")

SENTIMENT_TO_EMOTION = {
    1: "Sad",
    2: "Angry",
    3: "Neutral",
    4: "Happy",
    5: "Happy",
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}


class VoiceEmotionDetector:
    _instance: Optional["VoiceEmotionDetector"] = None

    def __init__(self):
        self._vosk_models: Dict[str, Model] = {}
        self._sentiment_pipeline = None

    @classmethod
    def get_instance(cls) -> "VoiceEmotionDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_vosk_model(self, lang: str = "en") -> Optional[Model]:
        if lang in self._vosk_models:
            return self._vosk_models[lang]

        model_path = os.path.join(VOSK_MODELS_DIR, lang)
        if not os.path.exists(model_path):
            alt_names = [
                f"vosk-model-small-{lang}",
                f"vosk-model-{lang}",
            ]
            for alt in alt_names:
                alt_path = os.path.join(VOSK_MODELS_DIR, alt)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                logger.warning(f"Vosk model not found for language: {lang}")
                return None

        logger.info(f"Loading Vosk model for language: {lang} from {model_path}")
        model = Model(model_path)
        self._vosk_models[lang] = model
        return model

    @property
    def sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            logger.info("Loading multilingual sentiment model...")
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1,
            )
            logger.info("Multilingual sentiment model loaded.")
        return self._sentiment_pipeline

    def transcribe_audio(self, audio_bytes: bytes, lang: str = "en") -> str:
        temp_path = save_temp_file(audio_bytes, suffix=".wav")
        try:
            model = self._get_vosk_model(lang)
            if model is None:
                available = self._list_available_models()
                raise FileNotFoundError(
                    f"Vosk model for language '{lang}' not found. "
                    f"Available models: {available}. "
                    f"Download models from https://alphacephei.com/vosk/models "
                    f"and place them in {VOSK_MODELS_DIR}/{lang}/"
                )

            wf = wave.open(temp_path, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise ValueError(
                    "Audio must be mono WAV, 16-bit PCM. "
                    "Please convert your audio file to the correct format."
                )

            sample_rate = wf.getframerate()
            rec = KaldiRecognizer(model, sample_rate)
            rec.SetWords(True)

            full_text = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        full_text.append(result["text"])

            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                full_text.append(final_result["text"])

            wf.close()
            return " ".join(full_text).strip()

        finally:
            cleanup_temp_file(temp_path)

    def _list_available_models(self) -> list:
        if not os.path.exists(VOSK_MODELS_DIR):
            return []
        return [
            d for d in os.listdir(VOSK_MODELS_DIR)
            if os.path.isdir(os.path.join(VOSK_MODELS_DIR, d))
        ]

    def analyze_sentiment(self, text: str) -> Dict:
        if not text or text.strip() == "":
            return {
                "emotion": "Neutral",
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
            }

        result = self.sentiment_pipeline(text[:512])[0]
        label = result["label"]
        score = result["score"]

        star_rating = int(label.split()[0])
        emotion = SENTIMENT_TO_EMOTION.get(star_rating, "Neutral")

        if star_rating <= 2:
            sentiment_label = "negative"
        elif star_rating == 3:
            sentiment_label = "neutral"
        else:
            sentiment_label = "positive"

        return {
            "emotion": emotion,
            "sentiment_label": sentiment_label,
            "sentiment_score": round(float(score), 4),
            "raw_label": label,
        }

    def predict(self, audio_bytes: bytes, lang: str = "en") -> Dict:
        lang = lang.lower().strip()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: '{lang}'. "
                f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        transcribed_text = self.transcribe_audio(audio_bytes, lang)

        if not transcribed_text:
            return {
                "text": "",
                "emotion": "Neutral",
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "language": SUPPORTED_LANGUAGES[lang],
                "message": "No speech detected in the audio.",
            }

        sentiment_result = self.analyze_sentiment(transcribed_text)

        return {
            "text": transcribed_text,
            "emotion": sentiment_result["emotion"],
            "sentiment_label": sentiment_result["sentiment_label"],
            "sentiment_score": sentiment_result["sentiment_score"],
            "language": SUPPORTED_LANGUAGES[lang],
        }
