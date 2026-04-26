"""
Preprocessing utilities for image and audio data.
"""

import os
import tempfile
from typing import Tuple

import cv2
import numpy as np


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB = 10


def validate_file_extension(filename: str, allowed: set) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed


def validate_file_size(content: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
    return len(content) <= max_mb * 1024 * 1024


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return img


def preprocess_face(face_roi: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    resized = cv2.resize(gray, target_size)
    normalized = resized.astype("float32") / 255.0
    return normalized


def save_temp_file(content: bytes, suffix: str = ".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path


def cleanup_temp_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
