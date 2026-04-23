# Advanced OCR System

Production-level OCR system with smart routing between multiple engines.

## Features

- **PaddleOCR** - Main engine for multilingual text detection and recognition
- **TrOCR** - Microsoft's transformer-based model for handwritten text
- **Tesseract** - Fallback engine for simple printed text
- **Smart Routing** - Automatically detects handwritten vs printed text and routes to the best engine
- **PDF Support** - Converts PDF pages to images and processes each page
- **Image Preprocessing** - OpenCV-based pipeline (grayscale, denoise, deskew, threshold, contrast enhancement)
- **Clean Web UI** - Upload files, view results, download as .txt

## Quick Start

### Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1
```

### Install Python Dependencies

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.8.1
pip install -r requirements.txt
```

### Run the Server

```bash
python main.py
```

Open http://localhost:8000 in your browser.

## API

### POST /ocr

Upload an image or PDF for OCR processing.

**Parameters:**
- `file` (required): Image or PDF file
- `engine` (optional): `smart`, `paddle`, `trocr`, `tesseract`
- `smart_routing` (optional): `true`/`false` (default: `true`)

**Example:**
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@document.png" \
  -F "engine=smart"
```

### GET /health

Health check endpoint.

## Project Structure

```
ocr-document-comparator/
├── main.py              # FastAPI server
├── ocr_engine.py        # OCR engines (PaddleOCR, TrOCR, Tesseract)
├── preprocessing.py     # Image preprocessing (OpenCV)
├── frontend/
│   └── index.html       # Web UI
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker support
└── README.md
```

## Architecture

```
Input (Image/PDF)
    │
    v
Preprocessing (OpenCV)
    │
    v
Smart Router
    ├── Handwritten? -> TrOCR
    ├── Printed? -> Tesseract
    └── Default -> PaddleOCR
    │
    v
Merge Results
    │
    v
Output (Text + Metadata)
```
