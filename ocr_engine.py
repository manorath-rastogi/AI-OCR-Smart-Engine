"""
OCR Engine module with smart routing.
- PaddleOCR: main engine for detection + multilingual text
- TrOCR: handwritten text recognition (line-by-line)
- Tesseract: fallback for simple printed text
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from preprocessing import (
    classify_text_region,
    pil_to_cv2,
    preprocess_for_handwriting,
    preprocess_for_ocr,
    to_grayscale,
)

logger = logging.getLogger(__name__)

# ─── Lazy-loaded singletons ──────────────────────────────────────────────────
_paddle_ocr = None
_trocr_model = None
_trocr_processor = None


def _get_paddle_ocr():
    """Lazy-load PaddleOCR instance."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR

        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
        )
        logger.info("PaddleOCR loaded successfully")
    return _paddle_ocr


def _get_trocr():
    """Lazy-load TrOCR model and processor."""
    global _trocr_model, _trocr_processor
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        model_name = "microsoft/trocr-base-handwritten"
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        logger.info("TrOCR loaded successfully")
    return _trocr_processor, _trocr_model


# ─── Individual OCR engines ────────────────────────────────────────────────


def ocr_with_paddle(image: Image.Image) -> dict:
    """
    Run PaddleOCR on the image.
    Returns detection boxes + recognized text with confidence.
    """
    ocr = _get_paddle_ocr()
    img_array = np.array(image.convert("RGB"))

    results = ocr.ocr(img_array, cls=True)

    extracted = []
    full_text_parts = []

    if results and results[0]:
        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = float(line[1][1])
            extracted.append(
                {"text": text, "confidence": confidence, "bbox": bbox, "engine": "PaddleOCR"}
            )
            full_text_parts.append(text)

    return {
        "engine": "PaddleOCR",
        "text": "\n".join(full_text_parts),
        "details": extracted,
    }


def _trocr_recognize_line(line_image: Image.Image) -> str:
    """Run TrOCR on a single cropped text line image."""
    processor, model = _get_trocr()

    if line_image.mode != "RGB":
        line_image = line_image.convert("RGB")

    # Resize to a reasonable height while keeping aspect ratio
    w, h = line_image.size
    if h < 32:
        scale = 32 / h
        line_image = line_image.resize((int(w * scale), 32), Image.LANCZOS)
    elif h > 128:
        scale = 128 / h
        line_image = line_image.resize((int(w * scale), 128), Image.LANCZOS)

    pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def ocr_with_trocr(image: Image.Image) -> dict:
    """
    Run TrOCR on the image for handwritten text recognition.
    Uses PaddleOCR for text detection (bounding boxes), then runs TrOCR
    on each detected text line for better accuracy.
    """
    # Use PaddleOCR just for detection (finding text line bounding boxes)
    ocr = _get_paddle_ocr()
    img_array = np.array(image.convert("RGB"))
    det_results = ocr.ocr(img_array, cls=True)

    extracted = []
    full_text_parts = []

    if det_results and det_results[0]:
        for line in det_results[0]:
            bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # Crop the text line region
            cropped = _crop_from_polygon(image, bbox)
            if cropped is None:
                continue

            # Run TrOCR on each cropped line
            text = _trocr_recognize_line(cropped)
            if text:
                extracted.append(
                    {"text": text, "confidence": 0.0, "bbox": bbox, "engine": "TrOCR"}
                )
                full_text_parts.append(text)
    else:
        # No lines detected by PaddleOCR, try full image
        processed = preprocess_for_handwriting(image)
        text = _trocr_recognize_line(processed)
        if text:
            extracted.append({"text": text, "confidence": 0.0, "engine": "TrOCR"})
            full_text_parts.append(text)

    return {
        "engine": "TrOCR",
        "text": "\n".join(full_text_parts),
        "details": extracted,
    }


def ocr_with_tesseract(image: Image.Image) -> dict:
    """
    Run Tesseract OCR on the image (fallback for printed text).
    """
    import pytesseract

    processed = preprocess_for_ocr(image, apply_threshold=True)

    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    extracted = []
    full_text_parts = []

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 0:
            continue
        extracted.append(
            {
                "text": text,
                "confidence": conf / 100.0,
                "bbox": [
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                ],
                "engine": "Tesseract",
            }
        )
        full_text_parts.append(text)

    return {
        "engine": "Tesseract",
        "text": " ".join(full_text_parts),
        "details": extracted,
    }


# ─── Helpers ────────────────────────────────────────────────────────────────


def _crop_from_polygon(image: Image.Image, polygon: list) -> Optional[Image.Image]:
    """Crop a text line from a 4-point polygon bounding box."""
    try:
        pts = np.array(polygon, dtype=np.float32)
        x_min = max(0, int(pts[:, 0].min()))
        y_min = max(0, int(pts[:, 1].min()))
        x_max = min(image.width, int(pts[:, 0].max()))
        y_max = min(image.height, int(pts[:, 1].max()))

        if x_max <= x_min or y_max <= y_min:
            return None

        # Add small padding
        pad = 2
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(image.width, x_max + pad)
        y_max = min(image.height, y_max + pad)

        return image.crop((x_min, y_min, x_max, y_max))
    except Exception:
        return None


# ─── Smart routing ──────────────────────────────────────────────────────────


def smart_ocr(image: Image.Image, use_smart_routing: bool = True) -> dict:
    """
    Smart OCR that routes text regions to appropriate engines.

    Strategy:
    1. Classify the overall image as handwritten or printed
    2. Use PaddleOCR for detection + initial recognition
    3. If handwritten detected, re-process each line with TrOCR
    4. Use Tesseract as fallback for low-confidence printed text
    5. Merge all results
    """
    # PaddleOCR handles its own preprocessing internally,
    # so pass the original image for best detection results.
    # Only preprocess for TrOCR/Tesseract when needed.

    # Step 1: PaddleOCR for detection + recognition (on original image)
    paddle_result = ocr_with_paddle(image)

    if not use_smart_routing:
        return paddle_result

    # Step 2: Classify overall image
    overall_type = _classify_overall_image(image)
    has_handwritten = overall_type == "handwritten"

    # Step 3: Also check individual PaddleOCR-detected lines
    if not has_handwritten and paddle_result["details"]:
        handwritten_line_count = 0
        for detail in paddle_result["details"]:
            bbox = detail.get("bbox")
            if bbox:
                cropped = _crop_from_polygon(image, bbox)
                if cropped:
                    cv_crop = pil_to_cv2(cropped)
                    gray_crop = to_grayscale(cv_crop)
                    line_type = classify_text_region(gray_crop)
                    if line_type == "handwritten":
                        handwritten_line_count += 1
        if handwritten_line_count > len(paddle_result["details"]) * 0.3:
            has_handwritten = True

    all_results = []
    engines_used = ["PaddleOCR"]

    # Add PaddleOCR results
    all_results.extend(paddle_result["details"])

    # Step 4: If handwritten, run TrOCR on each detected line
    if has_handwritten:
        try:
            trocr_result = ocr_with_trocr(image)
            if trocr_result["text"].strip():
                all_results.extend(trocr_result["details"])
                engines_used.append("TrOCR")
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")

    # Step 5: Tesseract fallback for low-confidence printed text
    paddle_confidences = [
        d["confidence"] for d in paddle_result["details"] if d["confidence"] > 0
    ]
    avg_confidence = (
        sum(paddle_confidences) / len(paddle_confidences)
        if paddle_confidences
        else 0
    )

    if avg_confidence < 0.7 and not has_handwritten:
        try:
            tesseract_result = ocr_with_tesseract(image)
            if tesseract_result["text"].strip():
                all_results.extend(tesseract_result["details"])
                engines_used.append("Tesseract")
        except Exception as e:
            logger.warning(f"Tesseract fallback failed: {e}")

    # Merge results
    final_text = _merge_results(paddle_result["text"], all_results, engines_used)

    return {
        "engines_used": engines_used,
        "text": final_text,
        "details": all_results,
        "handwritten_detected": has_handwritten,
    }


def _classify_overall_image(image: Image.Image) -> str:
    """Classify the overall image as handwritten or printed."""
    cv_img = pil_to_cv2(image)
    gray = to_grayscale(cv_img)
    return classify_text_region(gray)


def _merge_results(
    paddle_text: str, all_results: list[dict], engines_used: list[str]
) -> str:
    """
    Merge results from multiple engines.
    Prioritize TrOCR for handwritten text, PaddleOCR for structured text.
    """
    if "TrOCR" in engines_used:
        trocr_texts = [
            r["text"] for r in all_results if r.get("engine") == "TrOCR" and r["text"]
        ]
        paddle_texts = [
            r["text"] for r in all_results if r.get("engine") == "PaddleOCR" and r["text"]
        ]

        combined_parts = []
        if paddle_texts:
            combined_parts.append("\n".join(paddle_texts))
        if trocr_texts:
            combined_parts.append("\n--- Handwritten Text (TrOCR) ---")
            combined_parts.append("\n".join(trocr_texts))

        return "\n".join(combined_parts)

    if "Tesseract" in engines_used and not paddle_text.strip():
        tesseract_texts = [
            r["text"] for r in all_results if r.get("engine") == "Tesseract" and r["text"]
        ]
        return " ".join(tesseract_texts)

    return paddle_text
