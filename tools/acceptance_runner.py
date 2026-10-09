"""Run the app acceptance path and optional local model/report smoke checks.

Run from the repository root with ``.venv/Scripts/python tools/acceptance_runner.py``.
The default path creates small local PDFs and uses Streamlit AppTest with a
deterministic empty news search. ``--real-finbert`` downloads/loads FinBERT;
``--real-reports`` processes the two named local PDFs without OCR or tables.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import pandas as pd
import pymupdf
import requests
import streamlit
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import esg_engine
from config import DEFAULTS
from esg.keyword_matcher import KeywordMatcher
from esg.pipeline import process_pdf_bytes
from esg.news import retrieve_full_article, score_whole_article

def make_pdf(text: str) -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    data = pdf.tobytes()
    pdf.close()
    return data


def _app_run(annual_docs: list[tuple[str, bytes]], brsr_docs: list[tuple[str, bytes]],
             mappings: dict[str, dict], calls: list[str]) -> dict:
    """Drive the real app script through upload, mapping, run, and result state."""
    def data_editor(data, *args, key=None, **kwargs):
        table = data.copy()
        if key and (key.startswith("annual_company_mapping") or key.startswith("brsr_company_mapping")):
            for index, row in table.iterrows():
                supplied = mappings.get(str(row.get("PDF file", "")), {})
                for column, value in supplied.items():
                    if column in table.columns:
                        table.at[index, column] = value
        return table

    def empty_search(company, *_args, **_kwargs):
        calls.append(company["ticker"])
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    # AppTest can upload actual bytes; only data-editor typing needs a small
    # shim because its public testing API currently omits the editor widget.
    with patch.object(streamlit, "data_editor", side_effect=data_editor), \
            patch.object(esg_engine, "scan_news_for_company", side_effect=empty_search):
        app.run()
        for name, content in annual_docs:
            app.file_uploader(key="annual_report_uploads").upload(name, content, "application/pdf")
        for name, content in brsr_docs:
            app.file_uploader(key="optional_brsr_uploads").upload(name, content, "application/pdf")
        app.run()
        assert not app.exception, f"App raised before analysis: {app.exception}"
        assert app.checkbox(key="use_ocr").value is False
        assert app.checkbox(key="use_tables").value is False
        app.button[0].click().run()
    assert not app.exception, f"App raised during analysis: {app.exception}"
    assert not app.error, [item.value for item in app.error]
    result = app.session_state["iif_esg_analysis"]
    return result


def run_app_acceptance() -> dict:
    annual_hdfc = make_pdf("Scope 1 emissions fell 12% in FY2025 after an audited baseline review.")
    annual_icici = make_pdf("Board oversight included 7 independent directors in FY2025.")
    brsr_hdfc = make_pdf("Scope 1 emissions fell 12% in FY2025 after an audited baseline review. BRSR report.")
    # Annual-only input confirms the optional BRSR uploader can stay empty.
    annual_only_calls: list[str] = []
    annual_only = _app_run(
        [("hdfc_annual.pdf", annual_hdfc)], [],
        {"hdfc_annual.pdf": {"Ticker": "HDFCBANK.NS", "Company name": "HDFC Bank",
                              "Aliases (separate with ;)": "HDFC", "Fiscal year": "FY2025"}},
        annual_only_calls,
    )
    assert len(annual_only["companies"]) == 1
    assert set(annual_only["dataset"]["document_type"]) == {"annual_report"}
    assert set(annual_only["dataset"]["fiscal_year"]) == {"FY2025"}
    assert annual_only_calls == ["HDFCBANK.NS"]

    grouped_calls: list[str] = []
    grouped = _app_run(
        [("hdfc_annual.pdf", annual_hdfc), ("icici_annual.pdf", annual_icici)],
        [("hdfc_brsr.pdf", brsr_hdfc)],
        {
            "hdfc_annual.pdf": {"Ticker": "HDFCBANK.NS", "Company name": "HDFC Bank",
                                "Aliases (separate with ;)": "HDFC", "Fiscal year": "FY2025"},
            "icici_annual.pdf": {"Ticker": "ICICIBANK.NS", "Company name": "ICICI Bank",
                                 "Aliases (separate with ;)": "ICICI", "Fiscal year": "FY2025"},
            "hdfc_brsr.pdf": {"Ticker": "HDFCBANK.NS", "Company name": "HDFC Bank", "Fiscal year": "FY2025"},
        }, grouped_calls,
    )
    assert len(grouped["companies"]) == 2
    assert set(grouped["dataset"]["document_type"]) == {"annual_report", "brsr"}
    assert set(grouped["dataset"].loc[grouped["dataset"]["ticker"].eq("HDFCBANK.NS"), "source_file"]) == {
        "hdfc_annual.pdf", "hdfc_brsr.pdf",
    }
    assert sorted(grouped_calls) == ["HDFCBANK.NS", "ICICIBANK.NS"]
    assert len(grouped_calls) == len(set(grouped_calls)) == 2
    return {
        "annual_only_tickers": [company["ticker"] for company in annual_only["companies"]],
        "annual_only_dataset_rows": len(annual_only["dataset"]),
        "grouped_tickers": [company["ticker"] for company in grouped["companies"]],
        "grouped_document_types": sorted(grouped["dataset"]["document_type"].dropna().unique().tolist()),
        "news_calls_per_company": grouped_calls,
    }


def run_real_finbert_smoke() -> dict:
    import torch
    from transformers import pipeline

    torch.set_num_threads(min(2, torch.get_num_threads()))
    classifier = pipeline("text-classification", model="ProsusAI/finbert",
                          tokenizer="ProsusAI/finbert", device=-1)
    # The decisive disclosure appears well beyond both the old 5,000-character
    # clip and the first 512-token model window.
    text = ("HDFC Bank published operating results and described ordinary lending activity. " * 230
            + "The Reserve Bank of India issued a regulatory fine after finding that HDFC Bank's "
              "consumer fraud harmed customers through mis-selling. This final finding followed an investigation.")
    result = score_whole_article(text, classifier)
    assert result["full_text_scored"]
    assert result["chunk_count"] > 1
    assert result["chunks_scored"] == result["chunk_count"]
    assert result["token_count"] > 512
    assert math.isfinite(result["finbert_negative_probability"])
    return {key: result[key] for key in ("chunk_count", "chunks_scored", "token_count",
                                         "finbert_label", "finbert_negative_probability")}


def run_real_report_smoke(paths: list[Path]) -> list[dict]:
    matcher = KeywordMatcher(esg_engine.ESG_DICTIONARY)
    output = []
    for path in paths:
        data = path.read_bytes()
        errors, pages = [], []
        records = process_pdf_bytes(data, path.name, path.stem, matcher, DEFAULTS,
                                    ocr=False, extract_tables=False, errors=errors,
                                    page_status=pages)
        assert pages and any(row["characters"] > 0 for row in pages), path
        output.append({"file": path.name, "bytes": len(data), "pages": len(pages),
                       "dataset_rows": len(records), "extraction_errors": len(errors)})
    return output


def run_live_news_smoke() -> dict:
    """Resolve and retrieve one live Google News RSS item from an approved outlet."""
    query = '"HDFC Bank" RBI regulatory action site:business-standard.com'
    rss_url = "https://news.google.com/rss/search?" + urlencode(
        {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
    )
    response = requests.get(rss_url, timeout=25, headers={"User-Agent": "IIF-ESG-Research/0.1"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    selected = None
    for item in root.findall(".//item"):
        source = item.find("source")
        publisher = esg_engine.publisher_group(source.text.strip() if source is not None and source.text else "")
        link = item.findtext("link", default="").strip()
        if publisher == "Business Standard" and link:
            selected = (publisher, link, item.findtext("title", default="").strip())
            break
    if selected is None:
        return {"rss_items": len(root.findall(".//item")), "publisher_item_found": False,
                "note": "No Business Standard item appeared in this one-query RSS sample."}
    publisher, link, title = selected
    domains = esg_engine.APPROVED_NEWS_PUBLISHERS[publisher]
    retrieved = retrieve_full_article(link, domains)
    state = retrieved["retrieval_state"]
    assert state in {"retrieved", "blocked", "failed", "too_short", "truncated",
                     "paywalled", "rejected_redirect", "unresolved_google_news"}
    relevance = {"decision": "not evaluated", "reason": "No usable full article body was retrieved", "pillar": ""}
    if state == "retrieved":
        assert esg_engine.publisher_group(retrieved["url"]) == publisher
        assert len(retrieved.get("article_text", "").strip()) >= 500
        _accepted, decision, reason, pillar, attribution_evidence = esg_engine.relevance_and_pillar(
            retrieved["article_text"],
            {"company_name": "HDFC Bank", "aliases": ["HDFC"]},
            "HDFCBANK.NS", "G", esg_engine.NEWS_EVENT_TERMS,
        )
        relevance = {"decision": decision, "reason": reason, "pillar": pillar,
                     "attribution_evidence_present": bool(attribution_evidence)}
    return {"rss_items": len(root.findall(".//item")), "publisher": publisher,
            "title": title, "rss_link_host": link.split("/", 3)[2],
            "resolved_url": retrieved.get("url", ""),
            "retrieval_state": state, "target_relevance": relevance,
            "body_characters": len(retrieved.get("article_text", "")),
            "retrieval_reason": retrieved.get("retrieval_reason", "")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-finbert", action="store_true", help="Load ProsusAI/finbert and score a long full-body fixture")
    parser.add_argument("--live-news", action="store_true", help="Fetch one Google News RSS item from one approved-publisher query")
    parser.add_argument("--real-reports", nargs="*", type=Path, default=[],
                        help="Process local PDFs with searchable text, OCR and table extraction disabled")
    args = parser.parse_args()
    result = {"app_acceptance": run_app_acceptance()}
    if args.real_finbert:
        result["real_finbert"] = run_real_finbert_smoke()
    if args.live_news:
        result["live_news"] = run_live_news_smoke()
    if args.real_reports:
        result["real_reports"] = run_real_report_smoke(args.real_reports)
    print(pd.Series(result).to_json(indent=2, force_ascii=False))


if __name__ == "__main__":
    main()
