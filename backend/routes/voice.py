"""
Voice emotion detection API route.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.voice_emotion import VoiceEmotionDetector, SUPPORTED_LANGUAGES
from backend.utils.preprocessing import (
    ALLOWED_AUDIO_EXTENSIONS,
    validate_file_extension,
    validate_file_size,
)

router = APIRouter()


@router.post("/predict-voice")
async def predict_voice_emotion(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
):
    if not validate_file_extension(file.filename or "", ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    content = await file.read()

    if not validate_file_size(content):
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB.",
        )

    try:
        detector = VoiceEmotionDetector.get_instance()
        result = detector.predict(content, lang=language)
        return {"status": "success", "data": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")


@router.get("/supported-languages")
async def get_supported_languages():
    return {
        "status": "success",
        "data": {
            "languages": SUPPORTED_LANGUAGES,
        },
    }
