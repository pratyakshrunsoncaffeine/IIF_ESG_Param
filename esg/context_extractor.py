"""Sentence and neighboring-sentence evidence extraction."""
import re
from .text_cleaner import clean_text


def sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    # Keep decimal points and abbreviations inside sentences. PDF lines are joined first.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z₹\d])|(?<=;)\s+(?=[A-Z])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def context_at(parts: list[str], index: int) -> tuple[str, str, str]:
    return (parts[index - 1] if index else "", parts[index],
            parts[index + 1] if index + 1 < len(parts) else "")
