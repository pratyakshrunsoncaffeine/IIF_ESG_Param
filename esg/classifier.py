"""Deterministic, one-label classification with explicit precedence."""
import re


def classify(sentence: str, metrics: dict) -> str:
    s = sentence.casefold()
    if re.search(r"\b(?:fined?|penalt(?:y|ies)|fatalit(?:y|ies)|death|accident|spill|"
                 r"violation|breach|sanction|lawsuit)\b", s):
        return "ADVERSE_EVENT"
    if re.search(r"\b(?:aim|target|goal|commit|pledge|plan|intend|aspiration|ambition|"
                 r"will\s+(?:reduce|achieve)|by\s+20\d{2})\b", s):
        return "TARGET"
    if re.search(r"\b(?:risk|exposure|vulnerability|scenario analysis|physical risk|transition risk)\b", s):
        return "RISK_DISCLOSURE"
    if re.search(r"\b(?:compli\w*|regulat\w*|statutor\w*|certifi\w*)\b", s):
        return "COMPLIANCE"
    if re.search(r"\b(?:policy|code of conduct|procedure|guideline)\b", s):
        return "POLICY"
    if re.search(r"\b(?:install\w*|launch\w*|implement\w*|introduc\w*|establish\w*|invest\w*)\b", s):
        return "INITIATIVE"
    if metrics.get("value") is not None and re.search(r"\b(?:reduc\w*|decreas\w*|increas\w*|"
                                                      r"represent\w*|compris\w*|consist\w*|"
                                                      r"total\w*|achiev\w*|report\w*|was|were|is|are)\b", s):
        return "MEASURED_RESULT"
    if metrics.get("value") is not None:
        return "MEASURED_RESULT"
    return "GENERAL_STATEMENT" if len(s.split()) >= 4 else "UNKNOWN"
