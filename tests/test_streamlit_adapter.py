import json
from pathlib import Path
from unittest.mock import patch

from config import DEFAULTS
from esg.dictionary_loader import dictionary_summary, load_dictionary
from esg.keyword_matcher import KeywordMatcher
from esg import pipeline
from esg.streamlit_adapter import records_to_score_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_full_dictionary_is_shared_with_scoring_taxonomy():
    dictionary = load_dictionary(ROOT / "data" / "normalized_dictionary.json")
    summary = dictionary_summary(dictionary)
    assert [summary[p]["topics"] for p in ("Environmental", "Social", "Governance")] == [25, 17, 40]
    assert [summary[p]["keywords"] for p in ("Environmental", "Social", "Governance")] == [991, 789, 1485]
    taxonomy = json.loads((ROOT / "esg_taxonomy.json").read_text(encoding="utf-8"))
    assert taxonomy["environment_keywords"] == dictionary["Environmental"]
    assert taxonomy["social_keywords"] == dictionary["Social"]
    assert taxonomy["governance_keywords"] == dictionary["Governance"]


def test_full_report_statement_maps_to_score_evidence():
    dictionary = load_dictionary(ROOT / "data" / "normalized_dictionary.json")
    matcher = KeywordMatcher(dictionary)
    hits = matcher.find("Scope 1 emissions fell 12% in FY2025 after an audited baseline review.")
    assert ("Environmental", "Climate Change & GHG Emissions") in hits
    record = {
        "pillar": "Environmental", "topic": "Climate Change & GHG Emissions",
        "page_number": 4, "matched_keywords": hits[("Environmental", "Climate Change & GHG Emissions")],
        "matched_sentence": "Scope 1 emissions fell 12% in FY2025 after an audited baseline review.",
        "claim_type": "MEASURED_RESULT", "value": 12.0, "unit": "%",
        "reporting_period": "FY2025", "direction": "DECREASE",
    }
    row = records_to_score_evidence([record], "ABC.NS", "Example Ltd", "annual.pdf")[0]
    assert row["pillar"] == "E"
    assert row["page"] == 4
    assert row["numeric_signal"] and row["period_signal"] and row["progress_signal"]
    assert row["traceability_signal"]


def test_uploaded_pdf_bytes_keep_report_name_company_and_page_status():
    dictionary = load_dictionary(ROOT / "data" / "normalized_dictionary.json")
    matcher = KeywordMatcher(dictionary)
    sentence = "Scope 1 emissions fell 12% in FY2025 after an audited baseline review."
    page = {
        "company": "temporary name", "source_file": "uploaded_report.pdf",
        "page_number": 1, "raw_text": sentence, "cleaned_text": sentence,
        "ocr_used": False,
    }

    class FakeTableDocument:
        pages = []

        def close(self):
            pass

    page_status, errors = [], []
    with patch.object(pipeline.pdfplumber, "open", return_value=FakeTableDocument()), \
            patch.object(pipeline, "extract_pages", return_value=iter([page])):
        records = pipeline.process_pdf_bytes(
            b"test-pdf-bytes", "annual_report.pdf", "Example Ltd", matcher,
            DEFAULTS, ocr=True, errors=errors, page_status=page_status,
        )

    assert records
    assert records[0]["source_file"] == "annual_report.pdf"
    assert records[0]["company"] == "Example Ltd"
    assert page_status == [{"source_file": "annual_report.pdf", "page_number": 1,
                            "characters": len(sentence), "ocr_used": False}]
    assert not errors


def test_table_extraction_is_opt_in_and_reads_numeric_table_rows():
    dictionary = load_dictionary(ROOT / "data" / "normalized_dictionary.json")
    matcher = KeywordMatcher(dictionary)
    page = {"company": "temporary", "source_file": "uploaded_report.pdf", "page_number": 1,
            "raw_text": "Ordinary report page.", "cleaned_text": "Ordinary report page.", "ocr_used": False}

    class FakePdfPage:
        def extract_tables(self):
            return [[ ["Metric", "FY2025"], ["Scope 1 emissions", "12%"] ]]

    class FakeTableDocument:
        pages = [FakePdfPage()]

        def close(self):
            pass

    # The safe default never opens a second PDF parser for table extraction.
    with patch.object(pipeline.pdfplumber, "open") as open_pdf, \
            patch.object(pipeline, "extract_pages", return_value=iter([page])):
        default_records = pipeline.process_pdf_bytes(
            b"test-pdf-bytes", "annual.pdf", "Example Ltd", matcher, DEFAULTS,
            ocr=False, errors=[], page_status=[],
        )
    open_pdf.assert_not_called()
    assert default_records == []

    with patch.object(pipeline.pdfplumber, "open", return_value=FakeTableDocument()) as open_pdf, \
            patch.object(pipeline, "extract_pages", return_value=iter([page])):
        table_records = pipeline.process_pdf_bytes(
            b"test-pdf-bytes", "annual.pdf", "Example Ltd", matcher, DEFAULTS,
            ocr=False, errors=[], page_status=[], extract_tables=True,
        )
    open_pdf.assert_called_once()
    assert any(record.get("table_index") == 1 and record.get("value") == 12.0 for record in table_records)
