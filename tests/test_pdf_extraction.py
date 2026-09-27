from pathlib import Path
import fitz

from esg.pdf_extractor import extract_pages
from esg.keyword_matcher import KeywordMatcher
from esg.pipeline import text_records, table_records


def test_embedded_text_and_page_numbers(tmp_path):
    path = tmp_path / "Example_Annual_Report.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Scope 1 emissions were 125,400 tCO2e in FY2025.")
    doc.new_page()
    doc.save(path)
    doc.close()
    errors = []
    pages = list(extract_pages(path, errors=errors))
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1 and not pages[0]["ocr_used"]
    assert "125,400" in pages[0]["raw_text"]
    assert pages[1]["page_number"] == 2
    matcher = KeywordMatcher({"Environmental": {"Climate": ["scope 1 emissions"]}})
    rows = text_records(pages[0], matcher)
    assert rows[0]["source_file"] == path.name
    assert rows[0]["value"] == 125400
    assert rows[0]["matched_sentence"].startswith("Scope 1")
    assert "raw_text" not in rows[0] and "cleaned_text" not in rows[0]


def test_table_row_with_period_headers():
    class FakePage:
        def extract_tables(self):
            return [[['Metric', 'FY2024', 'FY2025'], ['Scope 1 emissions', '140,200', '125,400']]]
    page = {"company": "Example", "source_file": "example.pdf", "page_number": 8,
            "raw_text": "Scope 1 emissions 140,200 125,400", "cleaned_text": "Scope 1 emissions 140,200 125,400",
            "ocr_used": False}
    matcher = KeywordMatcher({"Environmental": {"Climate": ["scope 1 emissions"]}})
    rows = table_records(page, FakePage(), matcher, [])
    assert [(r["value"], r["reporting_period"]) for r in rows] == [(140200, "FY2024"), (125400, "FY2025")]
    assert all(r["table_index"] == 1 and r["raw_table_data"] for r in rows)
