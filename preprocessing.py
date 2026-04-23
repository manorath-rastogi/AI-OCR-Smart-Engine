"""
Image preprocessing module for OCR.
Uses OpenCV for grayscale conversion, denoising, thresholding, and resizing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR or grayscale) to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply Non-Local Means Denoising."""
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for better text extraction."""
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def resize_image(image: np.ndarray, max_dimension: int = 4000) -> np.ndarray:
    """Resize image if it exceeds max dimension while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image
    scale = max_dimension / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew in the image using minimum area rectangle."""
    gray = to_grayscale(image)
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 50:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_text_regions(image: np.ndarray) -> list[dict]:
    """
    Detect text regions and classify them as handwritten or printed.
    Uses contour analysis and variance-based heuristics.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    h_img, w_img = image.shape[:2]
    min_area = (h_img * w_img) * 0.001  # minimum 0.1% of image area

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue

        # Extract region for analysis
        roi = gray[y : y + h, x : x + w]
        text_type = classify_text_region(roi)

        regions.append(
            {"bbox": (x, y, w, h), "type": text_type, "confidence": 0.0}
        )

    return regions


def classify_text_region(roi: np.ndarray) -> str:
    """
    Classify a text region as 'handwritten' or 'printed' using heuristics:
    - Stroke width variance (handwriting has more variation)
    - Line regularity (printed text is more uniform)
    - Edge density patterns
    """
    if roi.size == 0:
        return "printed"

    # Binarize
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stroke width variance via distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_pixels = dist[dist > 0]
    if len(stroke_pixels) < 10:
        return "printed"

    stroke_variance = np.var(stroke_pixels)
    stroke_mean = np.mean(stroke_pixels)
    cv_ratio = stroke_variance / (stroke_mean + 1e-6)

    # Edge density
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1] + 1e-6)

    # Horizontal projection profile regularity
    h_proj = np.sum(binary, axis=1)
    h_proj_nonzero = h_proj[h_proj > 0]
    if len(h_proj_nonzero) > 2:
        line_regularity = np.std(h_proj_nonzero) / (np.mean(h_proj_nonzero) + 1e-6)
    else:
        line_regularity = 0

    # Handwriting indicators:
    # - Higher stroke width variance
    # - Higher edge density
    # - Less regular line spacing
    handwriting_score = 0
    if cv_ratio > 1.5:
        handwriting_score += 1
    if edge_density > 0.15:
        handwriting_score += 1
    if line_regularity > 0.8:
        handwriting_score += 1

    return "handwritten" if handwriting_score >= 2 else "printed"


def preprocess_for_ocr(
    image: Image.Image,
    apply_denoise: bool = True,
    apply_deskew: bool = True,
    apply_threshold: bool = False,
    max_dimension: int = 4000,
) -> Image.Image:
    """
    Full preprocessing pipeline for OCR.
    Returns a preprocessed PIL Image.
    """
    cv_img = pil_to_cv2(image)
    cv_img = resize_image(cv_img, max_dimension)

    if apply_deskew:
        cv_img = deskew(cv_img)

    if apply_denoise:
        cv_img = denoise(cv_img)

    if apply_threshold:
        cv_img = adaptive_threshold(cv_img)

    return cv2_to_pil(cv_img)


def preprocess_for_handwriting(image: Image.Image) -> Image.Image:
    """
    Specialized preprocessing for handwritten text.
    Enhances contrast and applies lighter denoising to preserve strokes.
    """
    cv_img = pil_to_cv2(image)
    cv_img = resize_image(cv_img, 4000)

    # Lighter denoising to preserve handwriting strokes
    cv_img = denoise(cv_img, strength=5)

    # Enhance contrast
    gray = enhance_contrast(cv_img)

    return cv2_to_pil(gray)
