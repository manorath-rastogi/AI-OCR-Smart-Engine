"""
Emotion AI - Main FastAPI Application

An AI-powered emotion detection system supporting:
- Image-based emotion detection (OpenCV + FER/CNN)
- Voice-based emotion detection (Vosk + multilingual NLP)
- Contextual quote generation (distilgpt2)
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.image import router as image_router
from backend.routes.voice import router as voice_router
from backend.routes.quote import router as quote_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion AI",
    description=(
        "AI-powered Emotion Detection System with image analysis, "
        "voice recognition, and motivational quote generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router, tags=["Image Emotion"])
app.include_router(voice_router, tags=["Voice Emotion"])
app.include_router(quote_router, tags=["Quote Generation"])

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": "Emotion AI",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict-image",
            "POST /predict-voice",
            "POST /generate-quote",
            "GET /supported-languages",
            "GET /health",
        ],
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Emotion AI",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
