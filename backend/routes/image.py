"""
Image emotion detection API route.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.image_emotion import ImageEmotionDetector
from backend.utils.preprocessing import (
    ALLOWED_IMAGE_EXTENSIONS,
    validate_file_extension,
    validate_file_size,
)

router = APIRouter()


@router.post("/predict-image")
async def predict_image_emotion(file: UploadFile = File(...)):
    if not validate_file_extension(file.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    content = await file.read()

    if not validate_file_size(content):
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB.",
        )

    try:
        detector = ImageEmotionDetector.get_instance()
        result = detector.predict_single(content)
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
