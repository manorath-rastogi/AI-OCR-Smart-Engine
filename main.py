"""
FastAPI server for the Advanced OCR System.
Supports image and PDF uploads with smart OCR routing.
"""

import io
import logging
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ocr_engine import ocr_with_paddle, ocr_with_tesseract, ocr_with_trocr, smart_ocr
from postprocessing import compute_ocr_quality, normalize_whitespace, spellcheck
from llm import llm_correct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _text_similarity(a: str, b: str) -> float:
    """SequenceMatcher character-level similarity ratio (0.0–1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:3000], b[:3000]).ratio()


def _extract_pdf_text_direct(content: bytes) -> tuple[str, int]:
    """
    Try to extract text directly from a PDF using PyMuPDF.
    Returns (extracted_text, page_count).
    extracted_text is empty string when PDF is scanned (no text layer).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — cannot extract PDF text directly.")
        return "", 0

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        pages_text: list[str] = []
        total_chars = 0

        for page_num in range(page_count):
            text = doc[page_num].get_text("text").strip()
            total_chars += len(text)
            if text:
                pages_text.append(f"--- Page {page_num + 1} ---\n{text}")

        doc.close()

        if total_chars > 50:
            return "\n\n".join(pages_text), page_count
        return "", page_count

    except Exception as exc:
        logger.warning(f"Direct PDF text extraction failed: {exc}")
        return "", 0


def _light_clean_pdf_text(text: str) -> str:
    """
    Minimal normalisation for digitally-extracted PDF text.
    Only fixes whitespace — no spellcheck, fuzzy correction, or rewriting.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n")]
    result: list[str] = []
    blanks = 0
    for line in lines:
        if line == "":
            blanks += 1
            if blanks <= 2:
                result.append(line)
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result).strip()


app = FastAPI(
    title="AI-OCR-Smart-Engine",
    description="Production-level OCR with PaddleOCR, TrOCR, and Tesseract",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff", "image/webp"}
ALLOWED_PDF_TYPE = "application/pdf"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images."""
    import os
    from pdf2image import convert_from_bytes

    poppler_path = r"G:\poppler\poppler-24.02.0\Library\bin"
    kwargs = {"dpi": 300}
    if os.path.isdir(poppler_path):
        kwargs["poppler_path"] = poppler_path

    images = convert_from_bytes(pdf_bytes, **kwargs)
    return images


def validate_file(file: UploadFile) -> str:
    """Validate uploaded file and return its type category."""
    content_type = file.content_type or ""

    if content_type in ALLOWED_IMAGE_TYPES:
        return "image"
    if content_type == ALLOWED_PDF_TYPE:
        return "pdf"

    # Fallback: check extension
    filename = (file.filename or "").lower()
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        return "image"
    if filename.endswith(".pdf"):
        return "pdf"

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {content_type}. Supported: images (PNG, JPEG, BMP, TIFF, WebP) and PDF.",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Advanced OCR System</h1><p>Frontend not found. Use /docs for API.</p>"
    )


@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    engine: Optional[str] = None,
    smart_routing: Optional[bool] = True,
    use_llm: bool = False,
    use_spellcheck: bool = False,
):
    """
    Main OCR endpoint.

    - **file**: Image or PDF file to process
    - **engine**: Optional engine override ('paddle', 'trocr', 'tesseract', or 'smart')
    - **smart_routing**: Enable smart routing (default: True)
    - **use_llm**: Run Ollama LLM correction (requires Ollama running)
    - **use_spellcheck**: Run word-level + TextBlob spell correction

    Returns extracted text and metadata.
    """
    file_type = validate_file(file)

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size: 50MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        if file_type == "pdf":
            return await _process_pdf(content, engine, smart_routing, use_llm, use_spellcheck)
        else:
            return await _process_image(content, engine, smart_routing, use_llm, use_spellcheck)
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


async def _process_image(
    content: bytes,
    engine: Optional[str],
    smart_routing: bool,
    use_llm: bool = False,
    use_spellcheck: bool = False,
) -> dict:
    """Process a single image through OCR."""
    image = Image.open(io.BytesIO(content))
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    # ── OCR pass ─────────────────────────────────────────────────────────
    result = _run_engine(image, engine, smart_routing)
    raw_text = result.get("text", "")

    # ── Minimal cleanup: whitespace only (no correction) ────────────────────
    cleaned_text = normalize_whitespace(raw_text) if raw_text else ""

    quality = compute_ocr_quality(raw_text)  # score the raw OCR output

    # ── Spellcheck: opt-in only ───────────────────────────────────────
    sc_text = ""
    if cleaned_text and use_spellcheck:
        sc_text = spellcheck(cleaned_text)

    # ── LLM: opt-in only, and only when quality is low ──────────────────
    llm_text, llm_method, llm_used = "", "none", False
    if use_llm and quality < 70:
        llm_input = sc_text or cleaned_text
        try:
            raw_llm, llm_method = await llm_correct(llm_input)
            if raw_llm and _text_similarity(raw_llm, cleaned_text) >= 0.4:
                llm_text = raw_llm
                llm_used = True
            else:
                llm_method = "rejected_low_similarity"
                logger.info("LLM output rejected (similarity < 0.4)")
        except Exception as exc:
            logger.warning(f"LLM correction failed: {exc}")
    elif use_llm:
        llm_method = "skipped_high_quality"

    result["raw_text"] = raw_text
    result["cleaned_text"] = cleaned_text
    result["formatted_text"] = cleaned_text
    result["spellchecked_text"] = sc_text
    result["llm_text"] = llm_text
    result["llm_method"] = llm_method
    result["quality_score"] = quality
    result["llm_used"] = llm_used

    return {
        "success": True,
        "file_type": "image",
        "pages": 1,
        "result": result,
    }


async def _process_pdf(
    content: bytes,
    engine: Optional[str],
    smart_routing: bool,
    use_llm: bool = False,
    use_spellcheck: bool = False,
) -> dict:
    """
    Process a PDF file.
    ─ Digital (searchable) PDF → extract text directly with PyMuPDF. No OCR.
    ─ Scanned PDF             → convert pages to images and run OCR pipeline.
    """
    # ── Step 1: Try direct extraction ─────────────────────────────────────────
    direct_text, page_count = _extract_pdf_text_direct(content)

    if direct_text:
        logger.info(f"[PDF MODE]: digital — {page_count} pages, {len(direct_text)} chars")

        # HARD RULE: digital PDF text must never go through OCR or heavy processing.
        # Only normalize whitespace — nothing else.
        cleaned = normalize_whitespace(direct_text)

        # Safety check: if cleaning somehow shortened text by >30%, revert to raw
        if len(cleaned) < len(direct_text) * 0.7:
            logger.warning("[PDF SAFETY] cleaned_text < 70% of raw — reverting to raw_text")
            cleaned = direct_text

        quality = compute_ocr_quality(cleaned)
        return {
            "success": True,
            "file_type": "pdf",
            "pdf_type": "digital",
            "pdf_mode": "direct_text",
            "pages": page_count,
            "result": {
                "text": direct_text,
                "raw_text": direct_text,
                "cleaned_text": cleaned,
                "formatted_text": cleaned,
                "spellchecked_text": "",
                "llm_text": "",
                "llm_method": "skipped_digital_pdf",
                "quality_score": quality,
                "llm_used": False,
            },
        }

    # ── Step 2: Scanned PDF ─ convert to images, run OCR ─────────────────────
    logger.info("[PDF MODE]: ocr (scanned PDF — no text layer detected)")
    try:
        images = pdf_to_images(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    if not images:
        raise HTTPException(status_code=400, detail="PDF contains no pages.")

    all_text_parts: list[str] = []
    page_results: list[dict] = []

    for i, image in enumerate(images):
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        result = _run_engine(image, engine, smart_routing)
        page_text = result.get("text", "")
        all_text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
        page_results.append({"page": i + 1, "result": result})

    raw_text = "\n\n".join(all_text_parts)
    cleaned_text = normalize_whitespace(raw_text) if raw_text else ""
    quality = compute_ocr_quality(raw_text)

    sc_text = ""
    if cleaned_text and (use_spellcheck or use_llm):
        sc_text = spellcheck(cleaned_text)

    llm_text, llm_method, llm_used = "", "none", False
    if use_llm and quality < 70:
        llm_input = sc_text or cleaned_text
        try:
            raw_llm, llm_method = await llm_correct(llm_input)
            if raw_llm and _text_similarity(raw_llm, cleaned_text) >= 0.4:
                llm_text = raw_llm
                llm_used = True
            else:
                llm_method = "rejected_low_similarity"
        except Exception as exc:
            logger.warning(f"LLM correction failed: {exc}")
    elif use_llm:
        llm_method = "skipped_high_quality"

    return {
        "success": True,
        "file_type": "pdf",
        "pdf_type": "scanned",
        "pdf_mode": "ocr",
        "pages": len(images),
        "result": {
            "text": raw_text,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "formatted_text": cleaned_text,
            "spellchecked_text": sc_text,
            "llm_text": llm_text,
            "llm_method": llm_method,
            "quality_score": quality,
            "llm_used": llm_used,
            "page_results": page_results,
        },
    }


def _run_engine(
    image: Image.Image,
    engine: Optional[str],
    smart_routing: bool,
) -> dict:
    """Route to the appropriate OCR engine."""
    if engine == "paddle":
        return ocr_with_paddle(image)
    elif engine == "trocr":
        return ocr_with_trocr(image)
    elif engine == "tesseract":
        return ocr_with_tesseract(image)
    elif engine == "smart" or smart_routing:
        return smart_ocr(image, use_smart_routing=True)
    else:
        return ocr_with_paddle(image)


@app.post("/ocr/download")
async def download_text(file: UploadFile = File(...)):
    """
    Process file and return extracted text as a downloadable .txt file.
    """
    file_type = validate_file(file)
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        if file_type == "pdf":
            result = await _process_pdf(content, None, True)
        else:
            result = await _process_image(content, None, True)

        text = result["result"].get("text", "")

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()

        return FileResponse(
            tmp.name,
            media_type="text/plain",
            filename="ocr_output.txt",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/ocr/refine")
async def refine_text(payload: dict):
    """
    LLM-based text correction (Ollama → rule-based fallback).
    Body: {"text": "...", "model": "llama3"}
    """
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")

    model = payload.get("model") or None
    refined, method = await llm_correct(text, model=model)

    if method == "unavailable":
        refined = postprocess(text)
        method = "rule-based"

    return {"refined_text": refined, "method": method}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
