"""Conservative, source-bound ESG quantities, periods, targets and trends."""
from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
UNIT = (r"MtCO2e|kgCO2e|tCO2e|CO2e|metric\s+tonnes?|tonnes?|"
        r"GWh|MWh|kWh|GW|MW|KL|ML|m3|m³|%|crore|lakh|million|billion|"
        r"employees?|workers?|directors?|hours?|fatalities|incidents?|LTIFR")
QUANTITY = re.compile(rf"(?<![\w.])(?P<currency>₹|INR\s*)?\s*(?P<number>{NUMBER})\s*"
                      rf"(?:(?:independent|women|female|male)\s+)?(?P<unit>{UNIT})(?!\w)", re.I)
PERIOD = re.compile(r"\b(?:FY\s?20\d{2}|CY\s?20\d{2}|(?:financial|fiscal|calendar)\s+year\s+20\d{2}|20\d{2}[–-]\d{2,4}|20\d{2})\b", re.I)
TARGET_CUE = re.compile(r"\b(?:aims?|targets?|goal|commits?|pledges?|plans?|intends?|"
                        r"by\s+20\d{2}|aspiration|ambition)\b", re.I)
BASELINE = re.compile(r"(?:from|against|relative to|compared with|versus)\s+(?:a\s+)?(?:FY\s*)?(20\d{2})\s+baseline|baseline\s+(?:year\s+)?(?:FY\s*)?(20\d{2})", re.I)
UP = re.compile(r"\b(?:increased?|grew|growth|rose|improved?|higher)\b", re.I)
DOWN = re.compile(r"\b(?:decreased?|reduced?|declined?|fell|lower|cut)\b", re.I)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    start: int
    end: int
    text: str


def quantities(sentence: str) -> list[Quantity]:
    out = []
    for m in QUANTITY.finditer(sentence):
        unit = (m.group("unit") or "").strip()
        currency = (m.group("currency") or "").strip()
        # A bare year or page count is never a metric.
        if not unit and not currency:
            continue
        raw = m.group("number").replace(",", "")
        out.append(Quantity(float(raw), f"{currency} {unit}".strip(), m.start(), m.end(), m.group().strip()))
    return out


def periods(sentence: str) -> list[str]:
    return [m.group().strip() for m in PERIOD.finditer(sentence)]


def nearest_period(sentence: str, position: int) -> str | None:
    matches = list(PERIOD.finditer(sentence))
    if not matches:
        return None
    return min(matches, key=lambda m: abs((m.start() + m.end()) / 2 - position)).group().strip()


def direction(sentence: str) -> str | None:
    if DOWN.search(sentence):
        return "DECREASE"
    if UP.search(sentence):
        return "INCREASE"
    return None


def extract_metrics(sentence: str, primary_keyword: str = "") -> dict:
    nums = quantities(sentence)
    result = {"value": None, "unit": None, "reporting_period": None,
              "previous_value": None, "previous_period": None,
              "current_value": None, "current_period": None,
              "target_metric": None, "target_value": None, "target_year": None,
              "baseline_year": None, "direction": direction(sentence)}
    if nums:
        chosen = nums[0]
        if primary_keyword:
            tokens = re.findall(r"\w+", primary_keyword)
            phrase = re.search(r"\b" + r"[\W_]+".join(map(re.escape, tokens)) + r"\b", sentence, re.I) if tokens else None
            if phrase:
                chosen = min(nums, key=lambda q: max(phrase.start() - q.end, q.start - phrase.end(), 0))
        result.update(value=chosen.value, unit=chosen.unit,
                      reporting_period=nearest_period(sentence, chosen.end))
    else:
        found = periods(sentence)
        result["reporting_period"] = found[0] if found else None

    is_target = bool(TARGET_CUE.search(sentence)) and bool(re.search(r"\b(?:aim|target|goal|commit|pledge|plan|intend|achiev|by\s+20\d{2})", sentence, re.I))
    if is_target:
        by = re.search(r"\bby\s+(20\d{2})\b", sentence, re.I)
        baseline = BASELINE.search(sentence)
        result["target_year"] = int(by.group(1)) if by else None
        result["baseline_year"] = int(next(g for g in baseline.groups() if g)) if baseline else None
        metric_phrase = re.search(r"\b(?:reduce|increase|achieve|reach|cut)\s+(.+?)\s+by\s+(?:\d|20\d{2})", sentence, re.I)
        result["target_metric"] = metric_phrase.group(1).strip() if metric_phrase else (primary_keyword or None)
        # A target deadline or baseline is not a reporting period.
        explicit_period = re.search(r"\b(?:FY\s?20\d{2}|CY\s?20\d{2}|(?:financial|fiscal|calendar)\s+year\s+20\d{2}|20\d{2}[–-]\d{2,4})\b", sentence, re.I)
        result["reporting_period"] = explicit_period.group() if explicit_period else None
        if nums:
            # A percentage or same-sentence quantity is retained with its source statement.
            target = next((q for q in nums if q.unit == "%"), nums[0])
            result["target_value"] = target.text
            result["value"] = None
            result["unit"] = None

    # Only infer a trend when two quantities and explicit comparative language coexist.
    if len(nums) >= 2 and result["direction"]:
        pair = nums[:2]
        between = sentence[pair[0].end:pair[1].start].lower()
        prefix = sentence[:pair[0].start].lower()
        if re.search(r"\bfrom\b", prefix) and re.search(r"\bto\b", between):
            previous, current = pair
        elif re.search(r"\bto\b", prefix) and re.search(r"\bfrom\b", between):
            current, previous = pair
        else:
            previous = current = None
        if current and previous:
            result.update(value=current.value, unit=current.unit,
                          reporting_period=nearest_period(sentence, current.end),
                          current_value=current.value,
                          current_period=nearest_period(sentence, current.end),
                          previous_value=previous.value,
                          previous_period=nearest_period(sentence, previous.end))
    return result
