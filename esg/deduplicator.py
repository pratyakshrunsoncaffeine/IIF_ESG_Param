"""Merge repeated evidence without losing its matched dictionary phrases."""
from .text_cleaner import normalize_for_match


def deduplicate(records: list[dict]) -> list[dict]:
    seen = {}
    for record in records:
        key = (record["source_file"], record["page_number"], record["pillar"],
               record["topic"], normalize_for_match(record["matched_sentence"]),
               record.get("table_index"), record.get("reporting_period"))
        if key not in seen:
            seen[key] = record
            continue
        old = seen[key]
        old["matched_keywords"] = list(dict.fromkeys(old["matched_keywords"] + record["matched_keywords"]))
        if record["confidence"] > old["confidence"]:
            record["matched_keywords"] = old["matched_keywords"]
            seen[key] = record
    return list(seen.values())
