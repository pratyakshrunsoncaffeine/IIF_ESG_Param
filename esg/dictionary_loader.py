"""Safely read the six literal structures in the supplied RTF dictionary."""
from __future__ import annotations

import ast
import json
from pathlib import Path

PILLARS = {
    "Environmental": "environment",
    "Social": "social",
    "Governance": "governance",
}


def load_dictionary(path: str | Path) -> dict[str, dict[str, list[str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ESG dictionary not found: {path}")
    if path.suffix.lower() == ".json":
        result = json.loads(path.read_text(encoding="utf-8"))
        _validate_hierarchy(result)
        return result
    if path.suffix.lower() != ".rtf":
        raise ValueError("Dictionary must be an .rtf or normalized .json file")
    from striprtf.striprtf import rtf_to_text

    plain = rtf_to_text(path.read_text(encoding="utf-8", errors="replace"))
    try:
        tree = ast.parse(plain)
    except SyntaxError as exc:
        raise ValueError(f"RTF dictionary is not valid literal Python data: {exc}") from exc
    structures = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in {f"{stem}_{part}" for stem in PILLARS.values() for part in ("sectors", "keywords")}:
            try:
                structures[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(f"{name} must contain only literal data") from exc
    result = {}
    for pillar, stem in PILLARS.items():
        sectors = structures.get(f"{stem}_sectors")
        keywords = structures.get(f"{stem}_keywords")
        if not isinstance(sectors, list) or not isinstance(keywords, dict):
            raise ValueError(f"Missing or invalid {stem}_sectors / {stem}_keywords")
        if len(sectors) != len(set(sectors)) or set(sectors) != set(keywords):
            raise ValueError(f"{pillar} sector list and keyword topics differ")
        result[pillar] = {topic: list(dict.fromkeys(keywords[topic])) for topic in sectors}
    _validate_hierarchy(result)
    return result


def _validate_hierarchy(data: dict) -> None:
    if set(data) != set(PILLARS):
        raise ValueError("Dictionary must contain Environmental, Social and Governance pillars")
    for pillar, topics in data.items():
        if not isinstance(topics, dict) or not topics:
            raise ValueError(f"{pillar} has no topics")
        for topic, words in topics.items():
            if not isinstance(topic, str) or not topic.strip() or not isinstance(words, list) or not words:
                raise ValueError(f"Invalid topic or empty keywords in {pillar}: {topic}")
            if any(not isinstance(w, str) or not w.strip() for w in words):
                raise ValueError(f"Invalid keyword in {pillar}: {topic}")


def dictionary_summary(data: dict) -> dict:
    summary = {pillar: {"topics": len(topics), "keywords": sum(len(v) for v in topics.values())}
               for pillar, topics in data.items()}
    summary["Total"] = {"unique_phrases": len({w.casefold().strip() for topics in data.values()
                                                for words in topics.values() for w in words})}
    return summary


def save_normalized_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
