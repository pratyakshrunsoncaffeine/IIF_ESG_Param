"""Validation and grouping for annual reports and optional standalone BRSR PDFs."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict


class DocumentMappingError(ValueError):
    """Raised when uploaded documents cannot be assigned unambiguously."""


def _clean(value) -> str:
    return str(value or "").strip()


def _identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _clean(value).casefold())


def _aliases(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [x.strip() for x in value if _clean(x)]
    return [x.strip() for x in _clean(value).split(";") if x.strip()]


def group_documents(annual_rows: list[dict], brsr_rows: list[dict] | None = None) -> list[dict]:
    """Validate mappings and return one company row with its PDFs grouped by ticker.

    Expected document rows contain ``file_name``, ``ticker``, ``content`` and
    optional identity/year fields. Supplied BRSR names must match an annual
    company name or alias. An omitted BRSR name/year uses the explicit ticker
    assignment and inherits the annual mapping year; the file's issuer is not
    independently verified from its PDF contents.
    """
    brsr_rows = brsr_rows or []
    if not annual_rows:
        raise DocumentMappingError("Upload at least one annual report.")
    names = [_clean(row.get("file_name")) for row in [*annual_rows, *brsr_rows]]
    if any(not name for name in names):
        raise DocumentMappingError("Every uploaded PDF needs a filename.")
    if len(names) != len(set(names)):
        raise DocumentMappingError("Duplicate filenames are ambiguous. Rename the PDFs and upload again.")

    groups: dict[str, dict] = {}
    bytes_seen: dict[str, tuple[str, str]] = {}
    for row in annual_rows:
        ticker = _clean(row.get("ticker")).upper()
        if not ticker:
            raise DocumentMappingError(f"Assign a ticker to annual report {row['file_name']}.")
        if ticker in groups:
            raise DocumentMappingError(f"Ticker {ticker} has more than one annual report.")
        company = _clean(row.get("company_name"))
        aliases = _aliases(row.get("aliases"))
        identity_names = {x for x in (company, *aliases) if _identity(x)}
        if not company:
            company = ticker
            identity_names.add(ticker)
        year = _clean(row.get("fiscal_year"))
        doc = {"document_type": "annual_report", "source_file": row["file_name"],
               "content": row.get("content", b""), "fiscal_year": year}
        groups[ticker] = {"ticker": ticker, "company_name": company,
                          "aliases": sorted(set(aliases)), "fiscal_year": year,
                          "report_name": row["file_name"], "documents": [doc],
                          "_identity_names": identity_names}
        digest = hashlib.sha256(doc["content"]).hexdigest()
        bytes_seen[digest] = (ticker, row["file_name"])

    for row in brsr_rows:
        file_name = _clean(row.get("file_name"))
        ticker = _clean(row.get("ticker")).upper()
        if not ticker:
            raise DocumentMappingError(f"Assign a ticker to BRSR document {file_name}.")
        if ticker not in groups:
            raise DocumentMappingError(f"BRSR document {file_name} is assigned to {ticker}, which has no uploaded annual report.")
        group = groups[ticker]
        name = _clean(row.get("company_name"))
        if name and _identity(name) not in {_identity(x) for x in group["_identity_names"]}:
            raise DocumentMappingError(f"Company name for BRSR document {file_name} does not match {ticker}'s annual-report company or aliases.")
        year = _clean(row.get("fiscal_year")) or group["fiscal_year"]
        if group["fiscal_year"] and year and year != group["fiscal_year"]:
            raise DocumentMappingError(f"Fiscal year for BRSR document {file_name} conflicts with {ticker}'s annual report.")
        digest = hashlib.sha256(row.get("content", b"")).hexdigest()
        if digest in bytes_seen:
            prior_ticker, prior_name = bytes_seen[digest]
            raise DocumentMappingError(f"{file_name} has identical PDF content to {prior_name}; remove the duplicate upload.")
        bytes_seen[digest] = (ticker, file_name)
        group["documents"].append({"document_type": "brsr", "source_file": file_name,
                                   "content": row.get("content", b""), "fiscal_year": year})
        if not group["fiscal_year"] and year:
            group["fiscal_year"] = year
            group["documents"][0]["fiscal_year"] = year

    # A declared identity cannot alias two tickers in the same comparison.
    owner: dict[str, str] = {}
    for ticker, group in groups.items():
        for alias in group["_identity_names"]:
            norm = _identity(alias)
            if norm in owner and owner[norm] != ticker:
                raise DocumentMappingError(f"Company name or alias {alias!r} is shared by {owner[norm]} and {ticker}.")
            owner[norm] = ticker
        group.pop("_identity_names")
    return list(groups.values())
