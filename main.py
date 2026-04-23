"""
FastAPI server for the Advanced OCR System.
Supports image and PDF uploads with smart OCR routing.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ocr_engine import ocr_with_paddle, ocr_with_tesseract, ocr_with_trocr, smart_ocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced OCR System",
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
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, dpi=300)
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
):
    """
    Main OCR endpoint.

    - **file**: Image or PDF file to process
    - **engine**: Optional engine override ('paddle', 'trocr', 'tesseract', or 'smart')
    - **smart_routing**: Enable smart routing (default: True)

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
            return await _process_pdf(content, engine, smart_routing)
        else:
            return await _process_image(content, engine, smart_routing)
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


async def _process_image(
    content: bytes,
    engine: Optional[str],
    smart_routing: bool,
) -> dict:
    """Process a single image through OCR."""
    image = Image.open(io.BytesIO(content))

    # Convert RGBA/palette to RGB
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    result = _run_engine(image, engine, smart_routing)

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
) -> dict:
    """Process a PDF by converting to images and running OCR on each page."""
    try:
        images = pdf_to_images(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    if not images:
        raise HTTPException(status_code=400, detail="PDF contains no pages.")

    all_text_parts = []
    page_results = []

    for i, image in enumerate(images):
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        result = _run_engine(image, engine, smart_routing)
        page_text = result.get("text", "")
        all_text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
        page_results.append({"page": i + 1, "result": result})

    return {
        "success": True,
        "file_type": "pdf",
        "pages": len(images),
        "result": {
            "text": "\n\n".join(all_text_parts),
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
