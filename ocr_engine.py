"""
OCR Engine module with smart routing.
- PaddleOCR: main engine for detection + multilingual text
- TrOCR: handwritten text recognition (line-by-line)
- Tesseract: fallback for simple printed text
"""

import logging
import os
import re
from typing import Optional

_MAX_TROCR_LINES = 35  # cap to keep latency acceptable

# Pre-import torch before PaddleOCR to ensure correct DLL loading order on Windows.
# If torch is not installed, TrOCR will be unavailable but other engines still work.
try:
    import torch  # noqa: F401
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image

from preprocessing import (
    classify_text_region,
    pil_to_cv2,
    preprocess_for_handwriting,
    preprocess_for_ocr,
    preprocess_for_paddle,
    preprocess_trocr_line,
    to_grayscale,
)

logger = logging.getLogger(__name__)

# ─── Lazy-loaded singletons ──────────────────────────────────────────────────
_paddle_ocr = None
_trocr_model = None
_trocr_processor = None
_trocr_device = "cpu"  # updated to "cuda" at load time if GPU is available


def _get_paddle_ocr():
    """Lazy-load PaddleOCR instance (GPU-enabled when CUDA is available)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("FLAGS_enable_pir_inference", "0")
        os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
        from paddleocr import PaddleOCR

        try:
            import torch as _t
            _use_gpu = _t.cuda.is_available()
        except Exception:
            _use_gpu = False

        try:
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                det_db_thresh=0.2,       # lower = detect more lines (helps faint handwriting)
                det_db_box_thresh=0.4,  # lower = accept more bounding boxes
                use_gpu=_use_gpu,
            )
        except TypeError:
            try:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
            except TypeError:
                _paddle_ocr = PaddleOCR()
        logger.info(f"PaddleOCR loaded (GPU={_use_gpu})")
    return _paddle_ocr


def _get_trocr():
    """Lazy-load TrOCR model and processor (GPU-accelerated when CUDA is available)."""
    global _trocr_model, _trocr_processor, _trocr_device
    if _trocr_model is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
        logger.info(f"Loading TrOCR model: {model_name} on {_trocr_device}")
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _trocr_model = _trocr_model.to(_trocr_device)  # ← GPU inference
        logger.info(f"TrOCR loaded on {_trocr_device}")
    return _trocr_processor, _trocr_model


# ─── Individual OCR engines ────────────────────────────────────────────────


def ocr_with_paddle(image: Image.Image, skip_preprocess: bool = False) -> dict:
    """
    Run PaddleOCR on the image.
    Returns detection boxes + recognized text with confidence.
    Set skip_preprocess=True if the image has already been through preprocess_for_paddle.
    """
    ocr = _get_paddle_ocr()
    if not skip_preprocess:
        try:
            image = preprocess_for_paddle(image)
        except Exception as e:
            logger.debug(f"preprocess_for_paddle skipped: {e}")
    img_array = np.array(image.convert("RGB"))

    results = ocr.ocr(img_array)

    extracted = []

    if results and results[0]:
        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = float(line[1][1])
            extracted.append(
                {"text": text, "confidence": confidence, "bbox": bbox, "engine": "PaddleOCR"}
            )

    # Re-order by reading sequence (top-to-bottom, left-to-right)
    extracted = _sort_by_reading_order(extracted)

    return {
        "engine": "PaddleOCR",
        "text": "\n".join(d["text"] for d in extracted),
        "details": extracted,
    }


def _trocr_recognize_line(line_image: Image.Image) -> str:
    """
    Run TrOCR on a single cropped text line image.

    Decoding: beam-8 search with no-repeat-ngram filtering.
    Post-decoding: word-level confidence filtering — words whose average
    token probability is below 0.15 are dropped to reduce garbled output.
    """
    processor, model = _get_trocr()

    if line_image.mode != "RGB":
        line_image = line_image.convert("RGB")

    import torch as _torch

    pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(_trocr_device)

    with _torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=8,
            early_stopping=True,
            no_repeat_ngram_size=2,
            max_length=128,
            length_penalty=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )

    full_text = processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )[0].strip()

    if not full_text:
        return ""

    # ── Word-level confidence annotation (no words dropped) ──────────────────
    # Goal: sentence completeness > aggressive filtering.
    # Words below WORD_CONF_THRESHOLD are kept in output as-is.
    # If >50% of words would have fallen below threshold → return raw text
    # immediately (heavy noise; filtering would break sentence structure).
    try:
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            outputs.beam_indices,
            normalize_logits=True,
        )
        token_probs = _torch.exp(transition_scores[0]).cpu().tolist()

        tokenizer = processor.tokenizer
        gen_ids = outputs.sequences[0].tolist()[1:]  # skip decoder BOS token
        eos = tokenizer.eos_token_id
        if eos in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos)]
        token_probs = token_probs[: len(gen_ids)]

        WORD_CONF_THRESHOLD = 0.05   # very low — only catches near-zero confidence
        words_with_conf: list[tuple[str, float]] = []
        cur_ids: list[int] = []
        cur_probs: list[float] = []

        for tid, prob in zip(gen_ids, token_probs):
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            if tok.startswith("Ġ") and cur_ids:
                word = tokenizer.decode(cur_ids).strip()
                if word:
                    avg = sum(cur_probs) / len(cur_probs)
                    words_with_conf.append((word, avg))
                cur_ids, cur_probs = [], []
            cur_ids.append(tid)
            cur_probs.append(prob)

        if cur_ids:
            word = tokenizer.decode(cur_ids).strip()
            if word:
                avg = sum(cur_probs) / len(cur_probs)
                words_with_conf.append((word, avg))

        if not words_with_conf:
            return full_text

        low_conf = sum(1 for _, c in words_with_conf if c < WORD_CONF_THRESHOLD)
        # Safety: if >50% words are near-zero confidence, output is likely garbage
        # — return raw decoded text so spellcheck/LLM can still work on it.
        if low_conf / len(words_with_conf) > 0.50:
            return full_text

        # Always keep all words — preserve sentence structure
        return " ".join(w for w, _ in words_with_conf)

    except Exception:
        return full_text


def ocr_with_trocr(image: Image.Image) -> dict:
    """
    Run TrOCR on the image for handwritten text recognition.
    Uses PaddleOCR for text detection (bounding boxes), then runs TrOCR
    on each detected text line for better accuracy.
    """
    # Use PaddleOCR just for detection (finding text line bounding boxes)
    ocr = _get_paddle_ocr()
    img_array = np.array(image.convert("RGB"))
    det_results = ocr.ocr(img_array)

    extracted = []

    if det_results and det_results[0]:
        lines_to_process = det_results[0][:_MAX_TROCR_LINES]
        for line in lines_to_process:
            bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            cropped = _crop_from_polygon(image, bbox)
            if cropped is None:
                continue

            # Per-line preprocessing: enhance contrast + sharpen for TrOCR
            try:
                cropped = preprocess_trocr_line(cropped)
            except Exception:
                pass

            text = _trocr_recognize_line(cropped)
            if text:
                extracted.append(
                    {"text": text, "confidence": 0.0, "bbox": bbox, "engine": "TrOCR"}
                )
    else:
        # No lines detected by PaddleOCR — try full image
        processed = preprocess_for_handwriting(image)
        text = _trocr_recognize_line(processed)
        if text:
            extracted.append({"text": text, "confidence": 0.0, "engine": "TrOCR"})

    # Re-order by reading sequence
    extracted = _sort_by_reading_order(extracted)

    return {
        "engine": "TrOCR",
        "text": "\n".join(d["text"] for d in extracted),
        "details": extracted,
    }


def trocr_ocr(image: Image.Image) -> dict:
    """Public wrapper for TrOCR handwritten recognition."""
    return ocr_with_trocr(image)


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


def _sort_by_reading_order(details: list[dict]) -> list[dict]:
    """
    Sort OCR results into natural reading order: top-to-bottom, then
    left-to-right within each line.  Uses greedy y-axis clustering:
    boxes whose y-centres are within 60 % of the median box height are
    merged into the same line cluster.
    """
    if len(details) < 2:
        return details

    def _metrics(d: dict) -> tuple[float, float, float]:
        bbox = d.get("bbox", [])
        if not bbox:
            return 0.0, 0.0, 20.0
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return float(min(xs)), float((min(ys) + max(ys)) / 2), float(max(ys) - min(ys))

    annotated = [(d, *_metrics(d)) for d in details]  # (dict, x_left, y_mid, height)
    annotated.sort(key=lambda t: t[2])  # sort by y_mid

    heights = sorted(t[3] for t in annotated if t[3] > 0)
    median_h = heights[len(heights) // 2] if heights else 20.0
    threshold = median_h * 0.6

    lines: list[list] = []
    cur: list = [annotated[0]]
    cur_y: float = annotated[0][2]

    for item in annotated[1:]:
        if abs(item[2] - cur_y) <= threshold:
            cur.append(item)
            cur_y = sum(t[2] for t in cur) / len(cur)
        else:
            lines.append(cur)
            cur = [item]
            cur_y = item[2]
    lines.append(cur)

    result: list[dict] = []
    for line in lines:
        for item in sorted(line, key=lambda t: t[1]):  # left-to-right
            result.append(item[0])
    return result


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

        # Add generous padding so TrOCR sees full ascenders/descenders
        pad = 10
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(image.width, x_max + pad)
        y_max = min(image.height, y_max + pad)

        return image.crop((x_min, y_min, x_max, y_max))
    except Exception:
        return None


def _bbox_ratio_stats(details: list[dict], image_width: int, image_height: int) -> tuple[float, float]:
    """Return median width-ratio and height-ratio of OCR boxes."""
    if image_width <= 0 or image_height <= 0:
        return 0.0, 0.0

    width_ratios: list[float] = []
    height_ratios: list[float] = []
    for d in details:
        bbox = d.get("bbox")
        if not bbox:
            continue
        pts = np.array(bbox, dtype=np.float32)
        box_w = float(pts[:, 0].max() - pts[:, 0].min())
        box_h = float(pts[:, 1].max() - pts[:, 1].min())
        if box_w > 0 and box_h > 0:
            width_ratios.append(box_w / float(image_width))
            height_ratios.append(box_h / float(image_height))

    if not width_ratios or not height_ratios:
        return 0.0, 0.0

    width_ratios.sort()
    height_ratios.sort()
    med_w = width_ratios[len(width_ratios) // 2]
    med_h = height_ratios[len(height_ratios) // 2]
    return float(med_w), float(med_h)


def _looks_quarter_rotated(details: list[dict], image_width: int, image_height: int) -> bool:
    """Heuristic signal for 90-degree rotated handwriting/photos."""
    if len(details) < 4:
        return False
    med_w, med_h = _bbox_ratio_stats(details, image_width, image_height)
    return med_w < 0.12 and med_h > (med_w * 1.5)


def _paddle_layout_score(details: list[dict], image_width: int, image_height: int) -> float:
    """Score Paddle detection layout quality for orientation selection."""
    confs = [float(d.get("confidence", 0.0) or 0.0) for d in details]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    med_w, _ = _bbox_ratio_stats(details, image_width, image_height)

    # Wider boxes usually indicate proper horizontal line detection.
    return (avg_conf * 1.0) + min(med_w * 1.8, 0.8) + min(len(details) / 40.0, 0.3)


def _auto_orient_for_paddle(image_proc: Image.Image, paddle_result: dict) -> tuple[Image.Image, dict]:
    """Try 90° rotations only when detections look sideways and keep the best layout."""
    details = paddle_result.get("details", [])
    if not _looks_quarter_rotated(details, image_proc.width, image_proc.height):
        return image_proc, paddle_result

    base_score = _paddle_layout_score(details, image_proc.width, image_proc.height)
    best_image = image_proc
    best_result = paddle_result
    best_score = base_score
    best_angle = 0

    for angle in (90, 270):
        try:
            rotated = image_proc.rotate(angle, expand=True)
            rotated_result = ocr_with_paddle(rotated, skip_preprocess=True)
            score = _paddle_layout_score(rotated_result.get("details", []), rotated.width, rotated.height)
            if score > best_score:
                best_score = score
                best_image = rotated
                best_result = rotated_result
                best_angle = angle
        except Exception as exc:
            logger.debug(f"Orientation check {angle}° failed: {exc}")

    if best_angle and best_score >= (base_score + 0.10):
        logger.info(f"Auto-orientation: applied {best_angle}° rotation for OCR")
        return best_image, best_result

    return image_proc, paddle_result


# ─── Smart routing ──────────────────────────────────────────────────────────


def smart_ocr(image: Image.Image, use_smart_routing: bool = True) -> dict:
    """
    Smart OCR that routes text regions to appropriate engines.

    Strategy:
    1. Run PaddleOCR for detection + recognition
    2. Classify image (handwritten vs printed)
    3. Handwritten: merge Paddle boxes to line-level, then refine per-line with TrOCR
       only when TrOCR looks better; otherwise keep Paddle text.
    4. Tesseract fallback for low-confidence printed text
    """
    # Step 1: Preprocess image once so bbox coords stay consistent across all steps.
    # ocr_with_paddle does its own internal preprocessing (resize+deskew), so we
    # replicate that here to get the same coordinate space for TrOCR crops.
    try:
        image_proc = preprocess_for_paddle(image)
    except Exception:
        image_proc = image

    # Step 1: PaddleOCR (skip_preprocess=True since image_proc is already preprocessed)
    paddle_result = ocr_with_paddle(image_proc, skip_preprocess=True)

    # Step 1b: recover 90° orientation when detections look sideways
    image_proc, paddle_result = _auto_orient_for_paddle(image_proc, paddle_result)

    if not use_smart_routing:
        return paddle_result

    # Step 2: Classify overall image
    has_handwritten = _classify_overall_image(image) == "handwritten"

    paddle_confs = [d["confidence"] for d in paddle_result["details"] if d["confidence"] > 0]
    avg_conf = sum(paddle_confs) / len(paddle_confs) if paddle_confs else 0

    # Step 3: Secondary signal — low PaddleOCR confidence indicates handwriting
    # (Per-line loop removed: O(n) classify calls were ~300 ms overhead and
    # redundant given the global classifier and confidence fallback below.)
    if not has_handwritten and avg_conf < 0.65 and len(paddle_confs) > 3:
        has_handwritten = True
        logger.info(f"Handwriting inferred from low avg confidence ({avg_conf:.2f})")

    # Step 4: TrOCR routing
    #   Handwritten → line-level TrOCR refinement with Paddle fallback.
    #   Printed low-conf → run TrOCR only on lines below 0.60 threshold
    if has_handwritten:
        lines = _ocr_handwriting_lines(image_proc, paddle_result["details"])
        if lines:
            lines = _sort_by_reading_order(lines)
            refined_text = "\n".join(d["text"] for d in lines)
            refined_score = _text_plausibility_score(refined_text)
            paddle_score = _text_plausibility_score(paddle_result["text"])

            if refined_score + 0.01 < paddle_score:
                logger.info(
                    f"Handwriting safety fallback: refined score {refined_score:.3f} < "
                    f"paddle score {paddle_score:.3f}"
                )
                return {
                    "engines_used": ["PaddleOCR"],
                    "text": paddle_result["text"],
                    "details": paddle_result["details"],
                    "handwritten_detected": True,
                }

            engines_used = ["PaddleOCR"]
            if any(d.get("engine") == "TrOCR" for d in lines):
                engines_used.append("TrOCR")
            return {
                "engines_used": engines_used,
                "text": refined_text,
                "details": lines,
                "handwritten_detected": True,
            }
    elif avg_conf < 0.60:
        # Printed but low confidence: selectively improve weak lines with TrOCR
        try:
            improved = _trocr_improve_lines(image_proc, paddle_result["details"], threshold=0.60)
            trocr_touched = [d for d in improved if d.get("engine") == "TrOCR"]
            if trocr_touched:
                improved = _sort_by_reading_order(improved)
                return {
                    "engines_used": ["PaddleOCR", "TrOCR"],
                    "text": "\n".join(d["text"] for d in improved),
                    "details": improved,
                    "handwritten_detected": False,
                }
        except Exception as e:
            logger.warning(f"TrOCR improvement failed: {e}")

    # Step 5: Tesseract fallback for low-confidence printed text
    if avg_conf < 0.6 and not has_handwritten:
        try:
            tess_result = ocr_with_tesseract(image)
            if tess_result["text"].strip():
                return {
                    "engines_used": ["PaddleOCR", "Tesseract"],
                    "text": tess_result["text"],
                    "details": paddle_result["details"] + tess_result["details"],
                    "handwritten_detected": has_handwritten,
                }
        except Exception as e:
            logger.warning(f"Tesseract fallback failed: {e}")

    # Default: PaddleOCR result
    return {
        "engines_used": ["PaddleOCR"],
        "text": paddle_result["text"],
        "details": paddle_result["details"],
        "handwritten_detected": has_handwritten,
    }


def _is_trocr_garbage(text: str) -> bool:
    """
    Return True only for clear noise / blank-region TrOCR output.
    Conservative: preserves short but valid words ("It", "AI", "1.").
    Only drops:
      - Empty / single non-letter characters
      - All-bare-single-digit tokens: "0", "0 0", "1 2 3"
        (NOT "1.", "1st", "A1" which are valid)
    """
    t = text.strip()
    if not t:
        return True
    if len(t) == 1 and not t.isalpha():
        return True  # single non-letter (punctuation/digit alone)
    tokens = t.split()
    # All tokens are bare single digits — e.g. "0", "0 0", "1 2 3"
    if all(re.match(r'^\d$', tok) for tok in tokens):
        return True
    return False


def _text_plausibility_score(text: str) -> float:
    """Cheap OCR-text quality proxy used for Paddle vs TrOCR line choice."""
    t = text.strip()
    if not t:
        return 0.0

    chars = [c for c in t if not c.isspace()]
    if not chars:
        return 0.0

    alpha_ratio = sum(ch.isalpha() for ch in chars) / len(chars)
    punctuation = sum(ch in ".,;:?!'\"()[]{}-" for ch in chars) / len(chars)
    weird = len(re.findall(r"[^\w\s\.,;:?!'\"()\[\]{}\-/#&%+]", t)) / len(chars)

    token_count = max(len(t.split()), 1)
    length_bonus = min(token_count / 12.0, 1.0)

    score = (alpha_ratio * 0.55) + (length_bonus * 0.35) + (punctuation * 0.10) - (weird * 0.80)
    return float(score)


def _should_use_trocr_text(trocr_text: str, paddle_text: str, paddle_confidence: float = 0.0) -> bool:
    """Use TrOCR only when it is clearly at least as plausible as Paddle output."""
    t = trocr_text.strip()
    p = paddle_text.strip()

    if not t or _is_trocr_garbage(t):
        return False
    if not p:
        return True

    # If Paddle line starts cleanly but TrOCR starts with odd punctuation,
    # prefer Paddle to avoid noisy substitutions like "# what ...".
    t_head = t.lstrip("'\"“”` ")
    p_head = p.lstrip("'\"“”` ")
    if t_head and p_head and (not t_head[0].isalnum()) and p_head[0].isalnum():
        return False

    # Guard against long hallucinations replacing a short stable Paddle line.
    if len(t) > max(120, int(len(p) * 2.2)):
        return False

    t_score = _text_plausibility_score(t)
    p_score = _text_plausibility_score(p)

    margin = 0.01
    if paddle_confidence >= 0.90:
        margin = 0.06
    elif paddle_confidence >= 0.82:
        margin = 0.03

    return t_score >= (p_score + margin)


def _should_merge_line_boxes(det_details: list[dict], image_width: int, image_height: int) -> bool:
    """Heuristic: merge only when Paddle detections look like word fragments."""
    if len(det_details) < 4 or image_width <= 0 or image_height <= 0:
        return False

    width_ratios: list[float] = []
    height_ratios: list[float] = []
    for d in det_details:
        bbox = d.get("bbox")
        if not bbox:
            continue
        pts = np.array(bbox, dtype=np.float32)
        box_w = float(pts[:, 0].max() - pts[:, 0].min())
        box_h = float(pts[:, 1].max() - pts[:, 1].min())
        if box_w > 0 and box_h > 0:
            width_ratios.append(box_w / float(image_width))
            height_ratios.append(box_h / float(image_height))

    if len(width_ratios) < 4 or len(height_ratios) < 4:
        return False

    width_ratios.sort()
    height_ratios.sort()
    med = width_ratios[len(width_ratios) // 2]
    p75 = width_ratios[(3 * len(width_ratios)) // 4]
    med_h = height_ratios[len(height_ratios) // 2]

    # Sideways pages often produce very narrow-but-tall boxes; merging them
    # collapses many lines into one and hurts output.
    if med < 0.12 and med_h > (med * 0.9):
        return False

    # Fragment-heavy detections are narrow; line-level detections are wider.
    return med < 0.30 or p75 < 0.45


def _merge_line_boxes(det_details: list[dict], y_tolerance_ratio: float = 0.6) -> list[dict]:
    """
    Merge PaddleOCR word/fragment boxes into full line-width boxes.

    PaddleOCR on handwriting often returns word-level fragments instead of
    full text lines.  TrOCR needs a FULL line crop to decode correctly —
    a tiny word crop fed to TrOCR produces garbage (e.g. "sp", "pigs", "fi").

    Algorithm:
      1. Sort boxes top-to-bottom by Y-centre.
      2. Group boxes whose Y-centres are within (median_height * tolerance).
      3. Merge each group into one wide bbox spanning the full line width.
    """
    if not det_details:
        return det_details

    items = []
    for d in det_details:
        bbox = d.get("bbox")
        if not bbox:
            continue
        pts = np.array(bbox, dtype=np.float32)
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        y_mid = (y_min + y_max) / 2.0
        h = max(y_max - y_min, 1.0)
        items.append((d, x_min, x_max, y_min, y_max, y_mid, h))

    if not items:
        return det_details

    items.sort(key=lambda t: t[5])  # sort by y_mid

    heights = sorted(t[6] for t in items)
    median_h = heights[len(heights) // 2]
    tolerance = median_h * y_tolerance_ratio

    # Group by Y proximity
    groups: list[list] = [[items[0]]]
    for item in items[1:]:
        if abs(item[5] - groups[-1][-1][5]) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])

    # Build one merged bbox per group
    merged: list[dict] = []
    for group in groups:
        group_sorted = sorted(group, key=lambda t: t[1])
        gx_min = min(t[1] for t in group)
        gx_max = max(t[2] for t in group)
        gy_min = min(t[3] for t in group)
        gy_max = max(t[4] for t in group)

        parts = [
            (t[0].get("text") or "").strip()
            for t in group_sorted
            if (t[0].get("text") or "").strip()
        ]
        merged_text = " ".join(parts).strip()

        confs = [
            float(t[0].get("confidence", 0.0) or 0.0)
            for t in group_sorted
            if isinstance(t[0].get("confidence", None), (int, float))
        ]
        merged_conf = (sum(confs) / len(confs)) if confs else 0.0

        merged_bbox = [
            [gx_min, gy_min], [gx_max, gy_min],
            [gx_max, gy_max], [gx_min, gy_max],
        ]
        merged.append({
            "text": merged_text,
            "confidence": merged_conf,
            "bbox": merged_bbox,
            "engine": "PaddleOCR",
        })

    logger.debug(f"_merge_line_boxes: {len(det_details)} fragments → {len(merged)} lines")
    return merged


def _ocr_handwriting_lines(
    image: Image.Image,
    det_details: list[dict],
) -> list[dict]:
    """
    Handwritten OCR pipeline with safe fallback.

    - Use PaddleOCR geometry as the line source (merged to line-level when needed).
    - Run TrOCR per line.
    - Keep Paddle line text unless TrOCR output is clearly better.
    """
    if _should_merge_line_boxes(det_details, image.width, image.height):
        line_dicts = _merge_line_boxes(det_details)
        if line_dicts:
            logger.info(
                f"Handwriting: merged Paddle boxes ({len(det_details)} -> {len(line_dicts)})"
            )
        else:
            line_dicts = det_details
    else:
        line_dicts = det_details

    if not line_dicts:
        return []

    try:
        _get_trocr()
    except Exception:
        return line_dicts

    logger.info(f"Handwriting: refining {len(line_dicts)} line(s) with TrOCR")

    results: list[dict] = []
    for d in line_dicts[:_MAX_TROCR_LINES]:
        bbox = d.get("bbox")
        if not bbox:
            continue

        paddle_text = (d.get("text") or "").strip()
        final_text = paddle_text
        final_engine = d.get("engine") or "PaddleOCR"
        final_confidence = float(d.get("confidence", 0.0) or 0.0)

        cropped = _crop_from_polygon(image, bbox)
        if cropped is not None and np.array(cropped).std() >= 8:
            try:
                trocr_crop = preprocess_trocr_line(cropped)
                trocr_text = _trocr_recognize_line(trocr_crop).strip()
                if _should_use_trocr_text(trocr_text, paddle_text, final_confidence):
                    final_text = trocr_text
                    final_engine = "TrOCR"
                    final_confidence = 0.0
            except Exception as exc:
                logger.debug(f"TrOCR line error: {exc}")

        if not final_text:
            continue

        results.append({
            "text": final_text,
            "engine": final_engine,
            "bbox": bbox,
            "confidence": final_confidence,
        })

    return results


def _classify_overall_image(image: Image.Image) -> str:
    """Classify the overall image as handwritten or printed."""
    cv_img = pil_to_cv2(image)
    gray = to_grayscale(cv_img)
    return classify_text_region(gray)


def _trocr_improve_lines(
    image: Image.Image,
    details: list[dict],
    threshold: float = 0.70,
) -> list[dict]:
    """
    Selectively re-process only low-confidence PaddleOCR lines with TrOCR.
    High-confidence lines keep their original text — no wasted inference.

    Returns a new details list with TrOCR text substituted where confidence
    was below `threshold`.
    """
    # Pre-load TrOCR; if unavailable return unchanged
    try:
        _get_trocr()
    except Exception:
        return details

    improved: list[dict] = []
    for d in details:
        conf = d.get("confidence", 1.0)
        bbox = d.get("bbox")
        # Attempt TrOCR on all PaddleOCR lines below threshold (incl. conf=0)
        if d.get("engine") == "PaddleOCR" and conf < threshold and bbox:
            try:
                cropped = _crop_from_polygon(image, bbox)
                if cropped:
                    # Skip crops that are mostly blank (table lines, borders, noise)
                    crop_arr = np.array(cropped)
                    if crop_arr.std() < 12:
                        improved.append(d)  # blank crop → keep PaddleOCR
                        continue

                    cropped = preprocess_trocr_line(cropped)
                    ttext = _trocr_recognize_line(cropped)
                    ttext_clean = ttext.strip()

                    # Quality guard: detect '0 0'-style garbage from TrOCR
                    # (all tokens are single-char digits → blank/noise region)
                    if ttext_clean:
                        tokens = ttext_clean.split()
                        is_garbage = (
                            len(tokens) <= 4
                            and all(len(t) <= 2 and t.isdigit() for t in tokens)
                        )
                        if not is_garbage:
                            improved.append({
                                **d,
                                "text": ttext_clean,
                                "engine": "TrOCR",
                                "paddle_confidence": conf,
                            })
                            continue
            except Exception as exc:
                logger.debug(f"Line TrOCR improvement skipped: {exc}")
        improved.append(d)
    return improved


