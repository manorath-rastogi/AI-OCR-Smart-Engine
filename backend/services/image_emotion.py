"""
Image-based emotion detection service.
Uses OpenCV for face detection and FER library for emotion classification.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from fer import FER

from backend.utils.preprocessing import load_image_from_bytes, preprocess_face

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class ImageEmotionDetector:
    _instance: Optional["ImageEmotionDetector"] = None

    def __init__(self):
        self._detector: Optional[FER] = None

    @classmethod
    def get_instance(cls) -> "ImageEmotionDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def detector(self) -> FER:
        if self._detector is None:
            logger.info("Loading FER emotion detector (MTCNN backend)...")
            self._detector = FER(mtcnn=True)
            logger.info("FER emotion detector loaded successfully.")
        return self._detector

    def detect_emotions(self, image_bytes: bytes) -> List[Dict]:
        img = load_image_from_bytes(image_bytes)
        results = self.detector.detect_emotions(img)

        if not results:
            return []

        detections = []
        for face_result in results:
            box = face_result["box"]
            emotions = face_result["emotions"]

            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            detections.append({
                "bounding_box": {
                    "x": int(box[0]),
                    "y": int(box[1]),
                    "width": int(box[2]),
                    "height": int(box[3]),
                },
                "dominant_emotion": dominant_emotion.capitalize(),
                "confidence": round(float(confidence), 4),
                "all_emotions": {
                    k.capitalize(): round(float(v), 4) for k, v in emotions.items()
                },
            })

        return detections

    def predict_single(self, image_bytes: bytes) -> Dict:
        detections = self.detect_emotions(image_bytes)
        if not detections:
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "message": "No face detected in the image.",
                "faces_detected": 0,
            }

        primary = detections[0]
        return {
            "emotion": primary["dominant_emotion"],
            "confidence": primary["confidence"],
            "all_emotions": primary["all_emotions"],
            "bounding_box": primary["bounding_box"],
            "faces_detected": len(detections),
        }
