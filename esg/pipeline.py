"""Coordinate page text and table evidence; keep failures local to a PDF/page."""
from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import Counter
from pathlib import Path

import pdfplumber

from .classifier import classify
from .context_extractor import context_at, sentences
from .deduplicator import deduplicate
from .keyword_matcher import KeywordMatcher
from .metric_extractor import extract_metrics, periods
from .pdf_extractor import extract_pages

LOGGER = logging.getLogger(__name__)


def primary_keyword(words: list[str]) -> str:
    return max(words, key=lambda word: (len(word.split()), len(word)))


def confidence(words: list[str], sentence: str, metrics: dict, *, ocr: bool = False,
               table: bool = False) -> float:
    score = 0.52
    score += 0.10 if len(primary_keyword(words).split()) > 1 else 0
    score += 0.10 if len(words) > 1 else 0
    score += 0.10 if metrics.get("value") is not None or metrics.get("target_value") else 0
    score += 0.07 if metrics.get("unit") else 0
    score += 0.06 if metrics.get("reporting_period") else 0
    score -= 0.16 if ocr else 0
    score -= 0.17 if table else 0
    score -= 0.12 if primary_keyword(words).casefold() in {"energy", "water", "board", "risk", "waste", "salary", "training"} else 0
    return round(max(0.0, min(1.0, score)), 2)


def _base(page: dict, pillar: str, topic: str, words: list[str], statement: str,
          before: str = "", after: str = "") -> dict:
    primary = primary_keyword(words)
    metrics = extract_metrics(statement, primary)
    return {"company": page["company"], "source_file": page["source_file"],
            "page_number": page["page_number"], "pillar": pillar, "topic": topic,
            "keyword": primary, "matched_keywords": words,
            "claim_type": classify(statement, metrics), "matched_sentence": statement,
            "context_before": before, "context_after": after,
            # Keep evidence snippets rather than copying the full page into
            # every keyword match. Large annual reports otherwise multiply
            # page text across thousands of rows and can exhaust Cloud memory.
            **metrics, "ocr_used": page["ocr_used"], "table_index": None,
            "table_row": None, "column_headers": None, "raw_table_data": None,
            "confidence": confidence(words, statement, metrics, ocr=page["ocr_used"])}


def text_records(page: dict, matcher: KeywordMatcher) -> list[dict]:
    parts = sentences(page["raw_text"])
    records = []
    for i, statement in enumerate(parts):
        hits = matcher.find(statement)
        if not hits:
            continue
        before, _, after = context_at(parts, i)
        for (pillar, topic), words in hits.items():
            records.append(_base(page, pillar, topic, words, statement, before, after))
    return records


def table_records(page: dict, pdf_page, matcher: KeywordMatcher, errors: list[dict]) -> list[dict]:
    records = []
    try:
        tables = pdf_page.extract_tables() or []
    except Exception as exc:
        errors.append({"source_file": page["source_file"], "page_number": page["page_number"],
                       "error_type": "table extraction failure", "message": str(exc)})
        return records
    for table_index, table in enumerate(tables, 1):
        if not table or len(table) < 2:
            continue
        headers = [str(c or "").strip() for c in table[0]]
        if len(headers) < 2:
            errors.append({"source_file": page["source_file"], "page_number": page["page_number"],
                           "error_type": "malformed table", "message": f"Table {table_index}: fewer than two columns"})
            continue
        for row_number, row in enumerate(table[1:], 2):
            if len(row) != len(headers):
                errors.append({"source_file": page["source_file"], "page_number": page["page_number"],
                               "error_type": "malformed table", "message": f"Table {table_index} row {row_number}: column count mismatch"})
                continue
            cells = [str(c or "").strip() for c in row]
            label = cells[0]
            hits = matcher.find(label)
            if not hits:
                continue
            for column, cell in enumerate(cells[1:], 1):
                if not cell:
                    continue
                header = headers[column]
                # Only a clearly numeric value under a labeled column is structured.
                numeric = re.fullmatch(r"\s*(?:₹|INR\s*)?\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*(%|[A-Za-z][A-Za-z0-9³]*)?\s*", cell)
                if not numeric:
                    continue
                value = float(numeric.group(1).replace(",", ""))
                period = (periods(header) or [None])[0]
                statement = f"{label} | {header}: {cell}"
                for (pillar, topic), words in hits.items():
                    record = _base(page, pillar, topic, words, statement)
                    record.update(value=value, unit=numeric.group(2), reporting_period=period,
                                  current_value=value, current_period=period,
                                  table_index=table_index, table_row=" | ".join(cells),
                                  column_headers=json.dumps(headers, ensure_ascii=False),
                                  raw_table_data=json.dumps(table, ensure_ascii=False),
                                  claim_type="MEASURED_RESULT")
                    record["confidence"] = confidence(words, statement, record, table=True,
                                                       ocr=page["ocr_used"])
                    records.append(record)
    return records


def process_pdf(path: Path, matcher: KeywordMatcher, settings, *, ocr: bool,
                errors: list[dict], source_name: str | None = None,
                company_name: str | None = None,
                page_status: list[dict] | None = None) -> list[dict]:
    records = []
    display_name = source_name or path.name
    try:
        table_document = pdfplumber.open(path)
    except Exception as exc:
        table_document = None
        errors.append({"source_file": display_name, "page_number": None,
                       "error_type": "table extraction failure", "message": str(exc)})
    try:
        pages = extract_pages(path, ocr=ocr, minimum_characters=settings.minimum_embedded_characters,
                              ocr_dpi=settings.ocr_dpi, ocr_language=settings.ocr_language,
                              progress_every=settings.progress_every_pages, errors=errors)
        for page in pages:
            if source_name:
                page["source_file"] = source_name
            if company_name:
                page["company"] = company_name
            if page_status is not None:
                page_status.append({
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "characters": len(page["raw_text"]),
                    "ocr_used": page["ocr_used"],
                })
            records.extend(text_records(page, matcher))
            if table_document is not None and page["page_number"] <= len(table_document.pages):
                records.extend(table_records(page, table_document.pages[page["page_number"] - 1],
                                             matcher, errors))
    except Exception as exc:
        errors.append({"source_file": display_name, "page_number": None,
                       "error_type": "PDF extraction failure", "message": str(exc)})
    finally:
        if table_document is not None:
            table_document.close()
    records = deduplicate(records)
    counts = Counter(r["pillar"] for r in records)
    LOGGER.info("Environmental matches: %s | Social matches: %s | Governance matches: %s",
                counts["Environmental"], counts["Social"], counts["Governance"])
    return [r for r in records if r["confidence"] >= settings.minimum_confidence]


def process_pdf_bytes(pdf_bytes: bytes, source_name: str, company_name: str,
                      matcher: KeywordMatcher, settings, *, ocr: bool,
                      errors: list[dict], page_status: list[dict] | None = None) -> list[dict]:
    """Process an uploaded PDF from memory while keeping temporary files isolated."""
    safe_name = Path(source_name).name or "uploaded_report.pdf"
    with tempfile.TemporaryDirectory(prefix="iif_esg_") as temp_dir:
        path = Path(temp_dir) / "uploaded_report.pdf"
        path.write_bytes(pdf_bytes)
        first_new_error = len(errors)
        records = process_pdf(path, matcher, settings, ocr=ocr, errors=errors,
                              source_name=safe_name, company_name=company_name,
                              page_status=page_status)
        for error in errors[first_new_error:]:
            if error.get("source_file") == path.name:
                error["source_file"] = safe_name
        return records
