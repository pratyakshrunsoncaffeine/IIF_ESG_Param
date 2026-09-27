from pathlib import Path
from esg.dictionary_loader import dictionary_summary, load_dictionary

DICTIONARY = Path(__file__).resolve().parents[1] / "data" / "ESG_KEYWORDS.rtf"


def test_complete_attached_dictionary():
    data = load_dictionary(DICTIONARY)
    summary = dictionary_summary(data)
    assert summary["Environmental"]["topics"] == 25
    assert summary["Social"]["topics"] == 17
    assert summary["Governance"]["topics"] == 40
    assert summary["Total"]["unique_phrases"] > 500
    assert "scope 1 emissions" in data["Environmental"]["Climate Change & GHG Emissions"]
    assert all(data[p][topic] for p in data for topic in data[p])


def test_bad_dictionary_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"Environmental": {}}')
    import pytest
    with pytest.raises(ValueError):
        load_dictionary(path)
