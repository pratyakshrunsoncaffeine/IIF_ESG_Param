import pandas as pd
import pytest

from esg.documents import DocumentMappingError, group_documents
from esg_engine import calculate_document_scores
from esg.streamlit_adapter import SCORE_COLUMNS, records_to_score_evidence


def annual(name, ticker, content=b"annual", company="Example Ltd", aliases="Example"):
    return {"file_name": name, "ticker": ticker, "content": content,
            "company_name": company, "aliases": aliases, "fiscal_year": "FY2025"}


def test_annual_and_optional_brsr_group_per_ticker_and_score_unique_topics_once():
    groups = group_documents(
        [annual("annual.pdf", "ABC.NS")],
        [{"file_name": "brsr.pdf", "ticker": "ABC.NS", "content": b"brsr",
          "company_name": "Example", "fiscal_year": "FY2025"}],
    )
    assert len(groups) == 1
    assert [doc["document_type"] for doc in groups[0]["documents"]] == ["annual_report", "brsr"]
    assert all(doc["fiscal_year"] == "FY2025" for doc in groups[0]["documents"])

    records = [
        {"pillar": "Environmental", "topic": "Climate Change & GHG Emissions", "page_number": 4,
         "source_file": "annual.pdf", "matched_sentence": "Scope 1 fell 10% in FY2025 against an audited baseline.",
         "matched_keywords": ["Scope 1"], "claim_type": "MEASURED_RESULT", "value": 10,
         "reporting_period": "FY2025", "direction": "DECREASE"},
        {"pillar": "Environmental", "topic": "Climate Change & GHG Emissions", "page_number": 8,
         "source_file": "brsr.pdf", "matched_sentence": "Scope 1 fell 10% in FY2025 against an audited baseline.",
         "matched_keywords": ["Scope 1"], "claim_type": "MEASURED_RESULT", "value": 10,
         "reporting_period": "FY2025", "direction": "DECREASE"},
    ]
    evidence_rows = []
    for record, doc_type, filename in zip(records, ("annual_report", "brsr"), ("annual.pdf", "brsr.pdf")):
        evidence_rows += records_to_score_evidence([record], "ABC.NS", "Example Ltd", filename, doc_type, "FY2025")
    evidence = pd.DataFrame(evidence_rows, columns=SCORE_COLUMNS)
    score = calculate_document_scores(evidence, groups).query("pillar == 'E'").iloc[0]
    assert score["topics_found"] == 1
    assert len(evidence) == 2
    assert set(evidence["document_type"]) == {"annual_report", "brsr"}
    assert set(evidence["source_file"]) == {"annual.pdf", "brsr.pdf"}


def test_annual_only_remains_a_single_document_company_group():
    groups = group_documents([annual("annual.pdf", "ABC.NS")])
    assert len(groups) == 1
    assert len(groups[0]["documents"]) == 1
    assert groups[0]["documents"][0]["document_type"] == "annual_report"


def test_multiple_companies_and_documents_are_grouped_by_ticker():
    groups = group_documents(
        [annual("a.pdf", "AAA.NS", b"a", "Alpha Ltd", "Alpha"),
         annual("b.pdf", "BBB.NS", b"b", "Beta Ltd", "Beta")],
        [{"file_name": "a-brsr.pdf", "ticker": "AAA.NS", "content": b"a-brsr", "company_name": "Alpha"},
         {"file_name": "b-brsr.pdf", "ticker": "BBB.NS", "content": b"b-brsr", "company_name": "Beta"}],
    )
    assert {group["ticker"]: len(group["documents"]) for group in groups} == {"AAA.NS": 2, "BBB.NS": 2}


@pytest.mark.parametrize("brsr, message", [
    ({"file_name": "brsr.pdf", "ticker": "", "content": b"b"}, "Assign a ticker"),
    ({"file_name": "brsr.pdf", "ticker": "OTHER", "content": b"b"}, "no uploaded annual report"),
    ({"file_name": "brsr.pdf", "ticker": "ABC.NS", "content": b"b", "company_name": "Wrong Co"}, "does not match"),
    ({"file_name": "brsr.pdf", "ticker": "ABC.NS", "content": b"b", "company_name": "Example", "fiscal_year": "FY2024"}, "conflicts"),
    ({"file_name": "brsr.pdf", "ticker": "ABC.NS", "content": b"annual", "company_name": "Example"}, "identical PDF content"),
])
def test_rejects_unassigned_mismatched_conflicting_or_duplicate_brsr(brsr, message):
    with pytest.raises(DocumentMappingError, match=message):
        group_documents([annual("annual.pdf", "ABC.NS")], [brsr])


def test_rejects_duplicate_names_and_alias_collision_between_tickers():
    with pytest.raises(DocumentMappingError, match="Duplicate filenames"):
        group_documents([annual("same.pdf", "AAA"), annual("same.pdf", "BBB", b"b")])
    with pytest.raises(DocumentMappingError, match="shared"):
        group_documents([annual("a.pdf", "AAA", b"a", "Alpha Ltd", "Shared"),
                         annual("b.pdf", "BBB", b"b", "Beta Ltd", "Shared")])
