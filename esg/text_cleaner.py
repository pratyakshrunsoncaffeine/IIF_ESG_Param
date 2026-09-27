"""Normalize PDF artifacts for matching while retaining unmodified source text."""
import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)  # line-end word wrapping
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def normalize_for_match(text: str) -> str:
    text = clean_text(text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
