"""Adjustable extraction settings. No external service or API is required."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    minimum_embedded_characters: int = 20
    progress_every_pages: int = 50
    ocr_dpi: int = 180
    ocr_language: str = "eng"
    minimum_confidence: float = 0.0


DEFAULTS = Settings()
