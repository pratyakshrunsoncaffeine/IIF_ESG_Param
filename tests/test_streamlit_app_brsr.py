"""Mocked Streamlit AppTest checks for document mapping and rerun state."""
from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

import streamlit
from streamlit.testing.v1 import AppTest

import esg.pipeline as pipeline

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _mapping_editor(mapping):
    def edit(data, *, key=None, **_kwargs):
        if not key or not (key.startswith("annual_company_mapping_") or key.startswith("brsr_company_mapping_")):
            return data
        rows = mapping["annual"] if key.startswith("annual_company_mapping_") else mapping["brsr"]
        edited = data.copy()
        for index, values in enumerate(rows):
            for column, value in values.items():
                if column in edited.columns:
                    edited.loc[index, column] = value
        return edited
    return edit


def _processor(content, source_name, company_name, _matcher, _settings, **kwargs):
    kwargs["page_status"].append({"source_file": source_name, "page_number": 1,
                                  "characters": 84, "ocr_used": False})
    return [{
        "company": company_name, "source_file": source_name, "page_number": 1,
        "pillar": "Environmental", "topic": "Climate Change & GHG Emissions",
        "matched_sentence": "Scope 1 emissions fell 10% in FY2025 after an audited baseline review.",
        "matched_keywords": ["Scope 1"], "claim_type": "MEASURED_RESULT",
        "value": 10.0, "unit": "%", "reporting_period": "FY2025",
        "direction": "DECREASE", "confidence": 0.91, "ocr_used": False,
    }]


def _set_news_off(at):
    next(widget for widget in at.get("checkbox") if widget.key == "run_news").set_value(False)


def _submit(at, mapping):
    with patch.object(streamlit, "data_editor", side_effect=_mapping_editor(mapping)), \
            patch.object(pipeline, "process_pdf_bytes", side_effect=_processor):
        at.run(timeout=30)
        at.get("button")[0].click().run(timeout=60)


def test_apptest_annual_only_no_news_ranks_and_changed_mapping_hides_stale_results():
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.get("file_uploader")[0].upload("annual.pdf", b"mock annual PDF", "application/pdf").run(timeout=30)
    _set_news_off(at)
    mapping = {"annual": [{"Ticker": "ABC.NS", "Company name": "Example Ltd",
                            "Aliases (separate with ;)": "Example", "Fiscal year": "FY2025"}],
               "brsr": []}
    _submit(at, mapping)

    assert not at.exception
    assert "iif_esg_analysis" in at.session_state
    analysis = at.session_state["iif_esg_analysis"]
    assert [company["ticker"] for company in analysis["companies"]] == ["ABC.NS"]
    assert analysis["document_scores"]["ticker"].nunique() == 1
    assert set(analysis["dataset"]["document_type"]) == {"annual_report"}
    assert analysis["news"].empty
    assert any(item.value == "4. Company ranking" for item in at.subheader)

    mapping["annual"][0]["Ticker"] = "XYZ.NS"
    with patch.object(streamlit, "data_editor", side_effect=_mapping_editor(mapping)):
        at.run(timeout=30)
    assert not at.exception
    assert all(item.value != "4. Company ranking" for item in at.subheader)


def test_apptest_groups_optional_brsr_with_one_ticker_across_multiple_companies():
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.get("file_uploader")[0].set_value([
        ("alpha-annual.pdf", b"alpha annual PDF", "application/pdf"),
        ("beta-annual.pdf", b"beta annual PDF", "application/pdf"),
    ])
    at.get("file_uploader")[1].upload("alpha-brsr.pdf", b"alpha BRSR PDF", "application/pdf").run(timeout=30)
    _set_news_off(at)
    mapping = {
        "annual": [
            {"Ticker": "AAA.NS", "Company name": "Alpha Ltd", "Aliases (separate with ;)": "Alpha", "Fiscal year": "FY2025"},
            {"Ticker": "BBB.NS", "Company name": "Beta Ltd", "Aliases (separate with ;)": "Beta", "Fiscal year": "FY2025"},
        ],
        "brsr": [{"Ticker": "AAA.NS", "Company name": "Alpha", "Fiscal year": "FY2025"}],
    }
    _submit(at, mapping)

    assert not at.exception
    analysis = at.session_state["iif_esg_analysis"]
    assert len(analysis["companies"]) == 2
    assert len(analysis["document_scores"]) == 6
    assert set(analysis["dataset"].query("ticker == 'AAA.NS'")["document_type"]) == {"annual_report", "brsr"}
    assert set(analysis["dataset"].query("ticker == 'AAA.NS'")["source_file"]) == {"alpha-annual.pdf", "alpha-brsr.pdf"}
    assert set(analysis["dataset"].query("ticker == 'BBB.NS'")["document_type"]) == {"annual_report"}


def test_apptest_news_fallback_reaches_ranking_and_downloads(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pandas as pd
    import esg_engine

    tokenizer = SimpleNamespace(model_max_length=512, encode=lambda text, **_: text.split(),
                                decode=lambda tokens, **_: " ".join(tokens))

    class Classifier:
        def __call__(self, texts, **kwargs):
            return [[{"label": "negative", "score": .8}, {"label": "neutral", "score": .15},
                     {"label": "positive", "score": .05}]]

    classifier = Classifier()
    classifier.tokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None, get_num_threads=lambda: 2))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *_, **__: classifier))
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(detect=lambda _: "en"))
    rows = pd.DataFrame([{"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank", "pillar": "G",
                          "title": f"RBI fined HDFC Bank for regulatory breach case {i}", "publisher": "Mint",
                          "url": f"https://livemint.com/story/{i}"} for i in range(58)])
    monkeypatch.setattr(esg_engine, "scan_news_for_company", lambda *_, **__: (rows, pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(esg_engine, "retrieve_full_article", lambda url, _: {
        "url": url, "article_text": "", "retrieval_state": "blocked", "retrieval_reason": "HTTP 403"})
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.get("file_uploader")[0].upload("annual.pdf", b"mock annual PDF", "application/pdf").run(timeout=30)
    mapping = {"annual": [{"Ticker": "HDFCBANK.NS", "Company name": "HDFC Bank"}], "brsr": []}
    _submit(at, mapping)
    assert not at.exception
    news = at.session_state["iif_esg_analysis"]["news"]
    assert news["headline_scored"].all() and news["scoring_eligible"].all()
    ranking = next(item.value for item in at.dataframe if "negative_news_penalty" in item.value.columns
                   and "overall_score_0_21" in item.value.columns and "pillar" not in item.value.columns)
    assert ranking.iloc[0]["negative_news_penalty"] == 2
    assert ranking.iloc[0]["weighted_negative_news_candidates"] == 29
    assert ranking.iloc[0]["headline_only_scored_count"] == 58
    assert any(item.value == "4. Company ranking" for item in at.subheader)
