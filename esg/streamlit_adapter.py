"""Adapters from the detailed extraction dataset to the app's score inputs."""
from __future__ import annotations

import re

PILLAR_CODES = {"Environmental": "E", "Social": "S", "Governance": "G"}
TRACEABILITY_CUES = re.compile(
    r"\b(?:scope|baseline|intensity|per employee|per unit|per tonne|assured|"
    r"assurance|verified|audited|boundary|coverage)\b",
    re.I,
)

SCORE_COLUMNS = [
    "ticker", "company_name", "report_name", "document_type", "fiscal_year", "source_file", "pillar", "pillar_name", "topic",
    "page", "keyword_hits", "numeric_signal", "period_signal", "progress_signal",
    "traceability_signal", "evidence_excerpt",
]


def records_to_score_evidence(records: list[dict], ticker: str,
                              company_name: str, report_name: str,
                              document_type: str = "annual_report",
                              fiscal_year: str = "") -> list[dict]:
    """Keep score inputs tied to the original page and extracted statement."""
    rows = []
    for record in records:
        pillar_name = record.get("pillar", "")
        statement = str(record.get("matched_sentence", ""))
        claim_type = record.get("claim_type", "")
        numeric = any(record.get(key) is not None for key in
                      ("value", "current_value", "target_value"))
        period = any(record.get(key) is not None for key in
                     ("reporting_period", "previous_period", "current_period",
                      "target_year", "baseline_year"))
        rows.append({
            "ticker": ticker,
            "company_name": company_name or ticker,
            "report_name": report_name,
            "document_type": document_type,
            "fiscal_year": fiscal_year or record.get("fiscal_year", ""),
            "source_file": record.get("source_file", report_name),
            "pillar": PILLAR_CODES.get(pillar_name, ""),
            "pillar_name": pillar_name,
            "topic": record.get("topic", ""),
            "page": record.get("page_number"),
            "keyword_hits": len(record.get("matched_keywords") or []),
            "numeric_signal": numeric,
            "period_signal": period,
            "progress_signal": claim_type in {"MEASURED_RESULT", "TARGET", "INITIATIVE"}
                               or bool(record.get("direction")),
            "traceability_signal": bool(TRACEABILITY_CUES.search(statement)),
            "evidence_excerpt": statement[:420],
        })
    return rows
