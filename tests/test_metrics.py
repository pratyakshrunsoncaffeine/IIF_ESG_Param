from esg.metric_extractor import extract_metrics, quantities
from esg.classifier import classify


def test_number_and_unit_extraction():
    sentence = "Emissions were 125,400 tCO2e; capacity was 45 MW; water was 450 KL; women were 38%."
    found = quantities(sentence)
    assert [(q.value, q.unit) for q in found] == [(125400, "tCO2e"), (45, "MW"), (450, "KL"), (38, "%")]


def test_trend_and_periods():
    sentence = "Scope 1 emissions declined from 150,000 tCO2e in FY2024 to 130,000 tCO2e in FY2025."
    m = extract_metrics(sentence, "scope 1 emissions")
    assert m["previous_value"] == 150000
    assert m["current_value"] == m["value"] == 130000
    assert m["previous_period"] == "FY2024"
    assert m["current_period"] == "FY2025"
    assert m["direction"] == "DECREASE"
    assert classify(sentence, m) == "MEASURED_RESULT"


def test_reverse_trend():
    sentence = "Scope 1 emissions decreased to 125,400 tCO2e in FY2025 from 140,200 tCO2e in FY2024."
    m = extract_metrics(sentence, "scope 1 emissions")
    assert (m["previous_value"], m["current_value"]) == (140200, 125400)


def test_target_and_baseline():
    sentence = "We aim to reduce Scope 1 and Scope 2 emissions by 50% by 2030 from a 2020 baseline."
    m = extract_metrics(sentence, "scope 1")
    assert m["target_value"] == "50%"
    assert m["target_year"] == 2030
    assert m["baseline_year"] == 2020
    assert m["target_metric"] == "Scope 1 and Scope 2 emissions"
    assert m["reporting_period"] is None
    assert classify(sentence, m) == "TARGET"


def test_target_without_metric_and_other_examples():
    s = "The company aims to achieve net zero emissions by 2040."
    m = extract_metrics(s, "net zero")
    assert m["target_year"] == 2040 and m["value"] is None
    assert classify(s, m) == "TARGET"
    assert extract_metrics("Renewable electricity represented 42% of total electricity consumption.")["value"] == 42
    assert extract_metrics("Women represent 38% of the workforce.")["unit"] == "%"
    assert extract_metrics("The board consists of 10 directors, six of whom are independent.")["value"] == 10
    assert extract_metrics("The board comprises 8 directors, including 5 independent directors.",
                           "independent directors")["value"] == 5
