from esg.keyword_matcher import KeywordMatcher
from esg.deduplicator import deduplicate

DICTIONARY = {"Environmental": {"Climate": ["scope 1", "scope 1 emissions", "ghg emissions", "net zero", "EV", "CO"]},
              "Social": {"Workforce": ["training"]},
              "Governance": {"Board": ["board", "independent directors"]}}


def test_phrases_hyphens_line_breaks_and_case():
    m = KeywordMatcher(DICTIONARY)
    hit = m.find("SCOPE 1 emissions and greenhouse results support net-zero plans. GHG\nemissions fell.")
    words = hit[("Environmental", "Climate")]
    assert "scope 1 emissions" in words
    assert "net zero" in words
    assert "ghg emissions" in words
    assert "scope 1" not in words  # longest overlapping phrase wins


def test_acronym_boundaries_and_false_positives():
    m = KeywordMatcher(DICTIONARY)
    assert not m.find("Every company approved a cover letter.")  # EV and CO are not substrings
    assert not m.find("The board approved the invoice.")
    assert ("Governance", "Board") in m.find("The board consists of 8 directors, including 5 independent directors.")
    assert ("Environmental", "Climate") in m.find("EV charging and CO emissions were tracked.")


def test_dedup_merges_keywords():
    common = {"source_file": "a.pdf", "page_number": 3, "pillar": "Environmental", "topic": "Climate",
              "matched_sentence": "Scope 1 emissions fell.", "table_index": None,
              "reporting_period": None, "confidence": 0.8}
    rows = [{**common, "matched_keywords": ["scope 1 emissions"]},
            {**common, "matched_keywords": ["ghg emissions"]}]
    result = deduplicate(rows)
    assert len(result) == 1
    assert set(result[0]["matched_keywords"]) == {"scope 1 emissions", "ghg emissions"}
