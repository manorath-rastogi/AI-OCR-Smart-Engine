"""
LLM-based OCR correction via local Ollama API (free, no external calls).

Model priority: llama3 → mistral → llama2
Gracefully falls back if Ollama is not running.
"""

import logging
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL_PRIORITY = ["llama3", "mistral", "llama2"]

# Skip LLM for very short texts (no meaningful correction possible)
_MIN_TEXT_LEN = 50

# If LLM output is less than this similar to input, assume hallucination and fall back
_MIN_SIMILARITY = 0.40


def _similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings (0–1)."""
    # Compare only the first 3000 chars for speed
    return SequenceMatcher(None, a[:3000], b[:3000]).ratio()

_PROMPT_TEMPLATE = (
    "You are an OCR text correction assistant.\n"
    "Fix only genuine OCR errors (spelling, broken words, wrong characters).\n"
    "Rules:\n"
    "  - Do NOT add new content or change the meaning.\n"
    "  - Preserve all line breaks, numbered lists, and bullet points.\n"
    "  - Keep acronyms like CIBIL, KYC, PAN, LAP, EMI in uppercase.\n"
    "  - Return ONLY the corrected text — no explanations, no preamble.\n\n"
    "OCR text:\n{text}"
)


async def llm_correct(
    text: str,
    model: Optional[str] = None,
) -> tuple[str, str]:
    """
    Correct OCR text with a local Ollama LLM.

    Returns:
        (corrected_text, method_string)
        method_string is e.g. "ollama/llama3" or "unavailable".
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — LLM correction unavailable")
        return text, "unavailable"

    if not text.strip():
        return text, "none"

    # Skip LLM for very short text — nothing meaningful to correct
    if len(text.strip()) < _MIN_TEXT_LEN:
        logger.debug("Text too short for LLM correction — skipped")
        return text, "skipped"

    prompt = _PROMPT_TEMPLATE.format(text=text)
    models_to_try: list[str] = list(
        dict.fromkeys([model] + _MODEL_PRIORITY if model else _MODEL_PRIORITY)
    )

    for m in models_to_try:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    _OLLAMA_URL,
                    json={"model": m, "prompt": prompt, "stream": False},
                )
            if resp.status_code == 200:
                corrected = resp.json().get("response", "").strip()
                if corrected:
                    sim = _similarity(text, corrected)
                    if sim < _MIN_SIMILARITY:
                        logger.warning(
                            f"LLM output too different (similarity={sim:.2f}) — "
                            "falling back to original text"
                        )
                        return text, f"ollama/{m}+hallucination_fallback"
                    logger.info(f"LLM correction done via {m} (similarity={sim:.2f})")
                    return corrected, f"ollama/{m}"
        except Exception as exc:
            logger.debug(f"Ollama model '{m}' failed: {exc}")
            continue

    logger.info("Ollama unavailable — LLM correction skipped")
    return text, "unavailable"
