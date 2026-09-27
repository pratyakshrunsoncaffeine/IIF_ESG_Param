"""Page-by-page PyMuPDF extraction; optional OCR only for empty text pages."""
from __future__ import annotations

import logging
from pathlib import Path
import fitz

from .text_cleaner import clean_text

LOGGER = logging.getLogger(__name__)


def infer_company(pdf_path: Path, metadata: dict | None = None) -> str:
    title = (metadata or {}).get("title", "") or ""
    if title and title.casefold() not in {"untitled", "document", "annual report"}:
        return title.strip()
    return pdf_path.stem.replace("_", " ").replace("-", " ").strip()


def extract_pages(pdf_path: str | Path, *, ocr: bool = False, minimum_characters: int = 20,
                  ocr_dpi: int = 180, ocr_language: str = "eng",
                  progress_every: int = 50, errors: list[dict] | None = None):
    pdf_path = Path(pdf_path)
    errors = errors if errors is not None else []
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        errors.append(_error(pdf_path, None, "PDF extraction failure", exc))
        return
    try:
        if document.is_encrypted:
            errors.append(_error(pdf_path, None, "encrypted PDF", "Password required"))
            return
        if len(document) == 0:
            errors.append(_error(pdf_path, None, "empty PDF", "No pages"))
            return
        company = infer_company(pdf_path, document.metadata)
        total = len(document)
        for index in range(total):
            page_number = index + 1
            if page_number == 1 or page_number % progress_every == 0 or page_number == total:
                LOGGER.info("Page %s/%s", page_number, total)
            try:
                page = document.load_page(index)
                raw = page.get_text("text", sort=True)
            except Exception as exc:
                errors.append(_error(pdf_path, page_number, "PDF extraction failure", exc))
                continue
            used_ocr = False
            if len("".join(c for c in raw if c.isalnum())) < minimum_characters:
                if ocr:
                    try:
                        import pytesseract
                        from PIL import Image
                        from io import BytesIO
                        pix = page.get_pixmap(dpi=ocr_dpi)
                        raw = pytesseract.image_to_string(Image.open(BytesIO(pix.tobytes("png"))), lang=ocr_language)
                        used_ocr = True
                    except Exception as exc:
                        errors.append(_error(pdf_path, page_number, "OCR failure", exc))
                if not raw.strip():
                    errors.append(_error(pdf_path, page_number, "empty page", "No usable embedded text"))
            yield {"source_file": pdf_path.name, "company": company,
                   "page_number": page_number, "raw_text": raw,
                   "cleaned_text": clean_text(raw), "ocr_used": used_ocr}
    finally:
        document.close()


def _error(path: Path, page: int | None, kind: str, message) -> dict:
    return {"source_file": path.name, "page_number": page, "error_type": kind,
            "message": str(message)}
