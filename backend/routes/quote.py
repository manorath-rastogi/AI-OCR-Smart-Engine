"""
Quote generation API route.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.quote_generator import QuoteGenerator

router = APIRouter()


class QuoteRequest(BaseModel):
    emotion: str = Field(
        ...,
        description="Emotion label (Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust)",
        examples=["Happy", "Sad", "Angry"],
    )


@router.post("/generate-quote")
async def generate_quote(request: QuoteRequest):
    valid_emotions = {"Happy", "Sad", "Angry", "Surprise", "Neutral", "Fear", "Disgust"}
    emotion = request.emotion.strip().capitalize()

    if emotion not in valid_emotions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid emotion '{request.emotion}'. "
            f"Valid emotions: {', '.join(sorted(valid_emotions))}",
        )

    try:
        generator = QuoteGenerator.get_instance()
        result = generator.generate(emotion)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote generation failed: {str(e)}")
