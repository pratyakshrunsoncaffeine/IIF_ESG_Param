"""One compiled, phrase-aware matcher for the entire supplied dictionary."""
from __future__ import annotations

import re
from collections import defaultdict
from .text_cleaner import normalize_for_match

# These are relevance filters, not added dictionary keywords.
GENERIC = {"energy", "water", "board", "risk", "waste", "salary", "training",
           "management", "policy", "employees", "workers", "community", "health",
           "safety", "compliance", "audit", "reporting", "electricity", "environment"}
ESG_CUES = re.compile(r"\b(?:emissions?|renewable|directors?|independen\w*|workforce|"
                      r"sustainab\w*|environmental|occupational|pollution|diversity|"
                      r"fatalit\w*|greenhouse|carbon|climate|recycl\w*|ethics|"
                      r"governance|csr|ghg|scope\s+[123])\b", re.I)
ESG_UNITS = re.compile(r"\b(?:tco2e|co2e|mwh|gwh|kwh|mw|gw|kl|ml|m3|"
                       r"tonnes?|fatalities|incidents|directors?|employees|workers)\b|%", re.I)


class KeywordMatcher:
    def __init__(self, dictionary: dict):
        self.lookup = defaultdict(list)
        for pillar, topics in dictionary.items():
            for topic, words in topics.items():
                for word in words:
                    key = normalize_for_match(word)
                    if key and (pillar, topic, word) not in self.lookup[key]:
                        self.lookup[key].append((pillar, topic, word))
        phrases = sorted(self.lookup, key=lambda x: (-len(x.split()), -len(x), x))
        # Boundaries prevent EV matching "every" and CO matching "company".
        choices = [re.escape(p).replace(r"\ ", r"\s+") for p in phrases]
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(choices) + r")(?!\w)", re.I)

    def find(self, sentence: str) -> dict[tuple[str, str], list[str]]:
        normalized = normalize_for_match(sentence)
        found = defaultdict(list)
        for match in self.pattern.finditer(normalized):
            key = re.sub(r"\s+", " ", match.group().casefold())
            for pillar, topic, word in self.lookup[key]:
                if word not in found[(pillar, topic)]:
                    found[(pillar, topic)].append(word)
        for key in list(found):
            words = found[key]
            specific = [w for w in words if normalize_for_match(w) not in GENERIC]
            if specific:
                found[key] = specific + [w for w in words if w not in specific]
            elif not self._generic_supported(sentence, words):
                del found[key]
        return dict(found)

    @staticmethod
    def _generic_supported(sentence: str, words: list[str]) -> bool:
        # A lone generic word needs ESG context, a relevant unit, or a second match.
        return (len({normalize_for_match(w) for w in words}) > 1
                or bool(ESG_CUES.search(sentence)) or bool(ESG_UNITS.search(sentence)))
