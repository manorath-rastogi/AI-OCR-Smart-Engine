"""
Quote generation service using HuggingFace distilgpt2.
Generates motivational or contextual quotes based on detected emotion.
"""

import logging
import random
import re
from typing import Dict, Optional

from transformers import pipeline, set_seed

logger = logging.getLogger(__name__)

FALLBACK_QUOTES = {
    "Happy": [
        "Happiness is not something ready-made. It comes from your own actions.",
        "The purpose of our lives is to be happy.",
        "Keep smiling, because life is a beautiful thing and there's so much to smile about.",
    ],
    "Sad": [
        "Every storm runs out of rain. Hold on, brighter days are ahead.",
        "It's okay to feel sad. Let yourself heal, and know that you're never alone.",
        "The sun will rise again, and so will you.",
    ],
    "Angry": [
        "For every minute you remain angry, you give up sixty seconds of peace of mind.",
        "Take a deep breath. Let it go. You deserve peace.",
        "Anger is an acid that does more harm to the vessel in which it is stored.",
    ],
    "Surprise": [
        "Life is full of surprises. Embrace the unexpected with open arms.",
        "The best things in life are often the ones we never expected.",
        "Surprise is the greatest gift which life can grant us.",
    ],
    "Neutral": [
        "Peace comes from within. Do not seek it without.",
        "In the stillness of the mind, you find clarity.",
        "Balance is not something you find, it's something you create.",
    ],
    "Fear": [
        "Courage is not the absence of fear, but the triumph over it.",
        "Everything you've ever wanted is on the other side of fear.",
        "Fear is only as deep as the mind allows.",
    ],
    "Disgust": [
        "Let go of what doesn't serve you. Focus on what brings you light.",
        "You have the power to choose your response to anything.",
        "Rise above the negativity. Your peace is worth more.",
    ],
    "Unknown": [
        "Every moment is a fresh beginning.",
        "You are stronger than you think.",
        "Believe in yourself and all that you are.",
    ],
}


class QuoteGenerator:
    _instance: Optional["QuoteGenerator"] = None

    def __init__(self):
        self._generator = None

    @classmethod
    def get_instance(cls) -> "QuoteGenerator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def generator(self):
        if self._generator is None:
            logger.info("Loading distilgpt2 text generation pipeline...")
            self._generator = pipeline(
                "text-generation",
                model="distilgpt2",
                device=-1,
            )
            logger.info("distilgpt2 loaded successfully.")
        return self._generator

    @staticmethod
    def _clean_generated_text(text: str, prompt: str) -> str:
        cleaned = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

        cleaned = re.sub(r"[#@\[\]{}|\\<>]", "", cleaned)
        cleaned = cleaned.replace("  ", " ").strip()

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[-1] in ".!?":
                complete_sentences.append(sentence)

        if complete_sentences:
            result = " ".join(complete_sentences[:2])
        elif cleaned:
            last_period = cleaned.rfind(".")
            last_exclaim = cleaned.rfind("!")
            last_question = cleaned.rfind("?")
            last_end = max(last_period, last_exclaim, last_question)
            if last_end > 10:
                result = cleaned[: last_end + 1]
            else:
                result = cleaned.rstrip(".,!?;: ") + "."
        else:
            result = ""

        return result

    def generate(self, emotion: str) -> Dict:
        emotion = emotion.strip().capitalize()
        if emotion not in FALLBACK_QUOTES:
            emotion = "Neutral"

        prompt = f"Generate a short motivational quote for a person feeling {emotion.lower()}:"

        try:
            set_seed(random.randint(1, 10000))
            outputs = self.generator(
                prompt,
                max_new_tokens=30,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
            )

            raw_text = outputs[0]["generated_text"]
            quote = self._clean_generated_text(raw_text, prompt)

            if len(quote) < 10 or quote.lower() == prompt.lower():
                quote = random.choice(FALLBACK_QUOTES[emotion])
                source = "fallback"
            else:
                source = "generated"

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            quote = random.choice(FALLBACK_QUOTES[emotion])
            source = "fallback"

        return {
            "emotion": emotion,
            "quote": quote,
            "source": source,
        }
