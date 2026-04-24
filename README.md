# 🧠 AI-OCR-Smart-Engine

> Production-level OCR system that extracts text from PDFs, images, and handwritten documents — with smart routing, spellcheck, and optional LLM correction.

Built with **FastAPI · PaddleOCR · TrOCR · Tesseract · Ollama**

---

## 🔥 Features

| Feature | Details |
|---|---|
| 🖼️ **Image → Text (Printed)** | PaddleOCR with CLAHE + deskew preprocessing |
| ✍️ **Handwritten Text** | Microsoft TrOCR (large model, beam search) |
| 📄 **PDF → Text** | Poppler page rendering + per-page OCR |
| 🔀 **Smart Routing Engine** | Auto-detects handwriting vs print; routes to best engine |
| 📊 **Line-level Confidence** | Per-line confidence scores; only low-conf lines sent to TrOCR |
| 🔤 **Spellcheck (Offline)** | `pyspellchecker` + TextBlob — no internet needed |
| 🤖 **AI Correction (Optional)** | Local Ollama LLM (llama3 → mistral fallback) with hallucination guard |
| 📐 **Structure Reconstruction** | Numbered lists, reading order, merged word fixes |
| 🎨 **Clean Web UI** | Upload · Preview · Extract · Copy · Download · Multi-view |
| 📤 **Export** | `.txt` (view-aware) · Structured JSON (with confidence + engine info) |
| 🐳 **Docker Support** | One-command deploy |

---

## 🧠 Smart Routing Logic

```
Input Image
    │
    ▼
PaddleOCR (detection + recognition)
    │
    ├─ Handwriting detected OR avg_confidence < 0.5
    │       └──► Full TrOCR (all lines)
    │
    ├─ 0.5 ≤ avg_confidence < 0.7
    │       └──► Line-level TrOCR (only weak lines re-processed)
    │
    └─ avg_confidence ≥ 0.7
            └──► PaddleOCR result (fast path)
    │
    ▼
Post-processing Pipeline
    (list fix → word corrections → acronym norm → spellcheck → structure)
    │
    ▼
Optional: LLM Correction (Ollama)  ← similarity guard (≥ 40% or fallback)
    │
    ▼
Output (Raw · Formatted · Spellchecked · AI · Confidence View)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **OCR — Printed** | PaddleOCR ≥ 2.8 |
| **OCR — Handwritten** | Microsoft TrOCR (`trocr-large-handwritten`) |
| **OCR — Fallback** | Tesseract ≥ 5 |
| **ML Runtime** | PyTorch + HuggingFace Transformers |
| **Image Processing** | OpenCV, Pillow, NumPy |
| **PDF Processing** | pdf2image + Poppler |
| **Spellcheck** | pyspellchecker + TextBlob |
| **AI Correction** | Ollama (local LLM — llama3 / mistral) |
| **Frontend** | HTML + CSS + Vanilla JS |

---

## 📦 Local Setup (Windows)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ocr-document-comparator.git
cd ocr-document-comparator
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PaddleOCR first-run** will download model weights (~300 MB).  
> ⚠️ **TrOCR first-run** will download `trocr-large-handwritten` (~1.3 GB).

### 4. Install Poppler (Required for PDF support)

1. Download the latest release from:  
   👉 https://github.com/oschwartz10612/poppler-windows/releases
2. Extract and add the `bin/` folder to your **system PATH**
3. Verify: `pdftoppm -v`

### 5. Install Tesseract (Required for fallback engine)

1. Download installer from:  
   👉 https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to **system PATH**
3. Verify: `tesseract --version`

### 6. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open 👉 **http://127.0.0.1:8000**

---

## 📦 Local Setup (Linux / macOS)

```bash
# System dependencies
sudo apt-get update && sudo apt-get install -y \
    tesseract-ocr poppler-utils libmagic1 libgl1-mesa-glx

# Python setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker

```bash
docker build -t ai-ocr-smart-engine .
docker run -p 8000:8000 ai-ocr-smart-engine
```

---

## 🤖 Optional: AI Correction via Ollama

```bash
# 1. Install Ollama
# https://ollama.ai

# 2. Pull a model (choose one)
ollama pull llama3       # recommended (~4 GB)
ollama pull mistral      # lighter alternative (~4 GB)

# 3. Ollama runs at http://localhost:11434 — no extra config needed
```

In the UI, check **"AI Correct"** before clicking Extract Text.

**Safety guard built-in:** If the LLM output deviates more than 60% from the input (hallucination detected), the system automatically falls back to the original OCR text.

---

## ⚙️ API Reference

### `POST /ocr`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image (PNG/JPG/BMP/TIFF/WebP) or PDF |
| `engine` | string | `smart` | `smart` · `paddle` · `trocr` · `tesseract` |
| `smart_routing` | bool | `true` | Enable smart engine routing |
| `use_llm` | bool | `false` | Run Ollama LLM correction |
| `use_spellcheck` | bool | `false` | Run offline spell correction |

**Example (Python):**

```python
import requests

with open("document.jpg", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/ocr",
        files={"file": f},
        params={"engine": "smart", "use_spellcheck": True}
    )

print(r.json())
```

**Example (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/ocr?engine=smart&use_spellcheck=true" \
     -F "file=@document.png"
```

### `POST /ocr/refine`

Re-run AI correction on already-extracted text.

```bash
curl -X POST "http://127.0.0.1:8000/ocr/refine" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ths is a tset sentance"}'
```

### `GET /health`

Returns server status and loaded engine info.

---

## 📊 Response Format

```json
{
  "success": true,
  "file_type": "image",
  "pages": 1,
  "result": {
    "text": "raw OCR output...",
    "formatted_text": "cleaned + structured output...",
    "spellchecked_text": "spell-corrected output...",
    "llm_text": "AI-corrected output (if enabled)...",
    "llm_method": "ollama/llama3",
    "engines_used": ["PaddleOCR", "TrOCR"],
    "handwritten_detected": false,
    "details": [
      {
        "text": "line of text",
        "confidence": 0.93,
        "engine": "PaddleOCR",
        "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
      }
    ]
  }
}
```

---

## 🗂️ Project Structure

```
ocr-document-comparator/
├── main.py              # FastAPI server + API endpoints
├── ocr_engine.py        # OCR engines, smart routing, reading-order sort
├── preprocessing.py     # Image preprocessing (deskew, CLAHE, upscale, sharpen)
├── postprocessing.py    # Text cleanup, list fix, acronym norm, spellcheck
├── llm.py               # Ollama LLM correction with similarity guard
├── frontend/
│   └── index.html       # Full web UI (multi-view, export, confidence highlight)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker support
└── README.md
```

---

## 🖥️ UI Views

| Button | Shows |
|---|---|
| **Formatted** | Cleaned, structured text (default) |
| **Raw** | Direct OCR output, no corrections |
| **SC** | Spellcheck-corrected text (when enabled) |
| **AI** | LLM-corrected text (when enabled) |
| **Conf** | Per-line confidence heat-map (🟢 high · 🟡 medium · 🔴 low) |

---

## ⚠️ Known Limitations

- Handwritten accuracy depends heavily on image quality / scan resolution
- TrOCR model downloads ~1.3 GB on first use
- Poppler is required for PDF support (Windows: manual PATH setup)
- LLM correction requires Ollama running locally
- TextBlob spellcheck is capped at 150 words per call for performance

---

## 🚀 Planned Improvements

- [ ] Table extraction (OpenCV grid detection)
- [ ] `.docx` structured export
- [ ] Multi-language OCR (Hindi, Arabic, etc.)
- [ ] Batch file processing
- [ ] Cloud deployment (Render / Railway / AWS)
- [ ] Word-level bounding box + per-word confidence

---

## 👨‍💻 Author

**Manorath Rastogi**


---

⭐ **If this project helped you, give it a star!**
