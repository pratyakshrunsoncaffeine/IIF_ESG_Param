from types import SimpleNamespace
import sys
import json

import pandas as pd
import pytest

from esg_engine import classify_news, apply_incidents_and_rank, _eligible_unique_news
from esg.database import save_evidence_batch, read_table


class Tokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class NegativeClassifier:
    tokenizer = Tokenizer()

    def __call__(self, texts, **kwargs):
        return [[{"label": "negative", "score": .8}, {"label": "neutral", "score": .15},
                 {"label": "positive", "score": .05}]]


def candidate(i=0, **changes):
    return dict({"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank", "aliases": [],
                 "pillar": "G", "publisher": "Mint", "url": f"https://livemint.com/story/{i}",
                 "title": f"RBI fined HDFC Bank for regulatory breach in case {i} - Mint",
                 "published_at": "2026-08-10", "source_start": "2025-10-09", "source_end": "2026-10-09"}, **changes)


def blocked(url, _domains):
    return {"url": url, "article_text": "", "retrieval_state": "blocked", "retrieval_reason": "HTTP 403"}


def document_scores():
    return pd.DataFrame([{"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank", "pillar": p,
                          "document_evidence_score_0_7": 7.0} for p in "ESG"])


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(detect=lambda _: "en"))


def test_blocked_article_scores_headline_and_retains_retrieval_audit():
    news = classify_news(pd.DataFrame([candidate()]), NegativeClassifier(), article_fetcher=blocked)
    row = news.iloc[0]
    assert row.scoring_eligible and row.headline_scored and not row.full_text_scored
    assert not row.full_article_available and row.retrieval_state == "blocked"
    assert row.retrieval_reason == "HTTP 403"
    assert row.model_input_source == "headline only"
    assert row.scoring_text.endswith("case 0")  # outlet suffix removed
    assert row.finbert_negative_probability == .8 and row.effective_negative_probability == .4
    assert row.negative_evidence_weight == .5 and row.needs_analyst_review
    assert len(_eligible_unique_news(news)) == 1


@pytest.mark.parametrize("state", ["failed", "paywalled", "too_short", "truncated", "blocked"])
def test_all_unavailable_body_states_use_headline_without_scoring_teaser(state):
    result = classify_news(pd.DataFrame([candidate()]), NegativeClassifier(), article_fetcher=lambda url, _: {
        "url": url, "article_text": "A partial teaser must never be scored", "retrieval_state": state,
        "retrieval_reason": state})
    assert result.iloc[0].headline_scored and result.iloc[0].scoring_eligible
    assert "teaser" not in result.iloc[0].scoring_text


def test_unresolved_google_rss_can_use_approved_source_headline():
    row = candidate(url="https://news.google.com/rss/articles/id", discovery_source="Google News RSS", domain="Mint")
    news = classify_news(pd.DataFrame([row]), NegativeClassifier(), article_fetcher=blocked)
    assert news.iloc[0].scoring_eligible
    assert len(_eligible_unique_news(news)) == 1
    news["domain"] = "Unapproved outlet"
    assert _eligible_unique_news(news).empty


@pytest.mark.parametrize("changes", [
    {"title": "Nuvama raises HDFC Bank share price target"},
    {"title": "RBI fined SBI for regulatory compliance breaches"},
    {"title": ""}, {"published_at": "2020-01-01"},
    {"url": "https://unapproved.example/story"},
])
def test_irrelevant_stale_missing_or_unapproved_headlines_are_rejected(changes):
    news = classify_news(pd.DataFrame([candidate(**changes)]), NegativeClassifier(), article_fetcher=blocked)
    assert not news.iloc[0].scoring_eligible
    assert _eligible_unique_news(news).empty


def test_full_article_negativity_has_full_weight_and_preferred_over_headline():
    rows = [candidate(url="https://reuters.com/story/1", publisher="Reuters"), candidate()]
    body = "RBI fined HDFC Bank for a regulatory breach. " + "Complete public reporting context. " * 40

    def fetch(url, domains):
        if "reuters" in url:
            return blocked(url, domains)
        return {"url": url, "article_text": body, "retrieval_state": "retrieved"}

    news = classify_news(pd.DataFrame(rows), NegativeClassifier(), article_fetcher=fetch)
    assert not news.iloc[0].scoring_eligible and news.iloc[0].duplicate_of == rows[1]["url"]
    assert news.iloc[1].full_text_scored and news.iloc[1].negative_evidence_weight == 1
    assert news.iloc[1].effective_negative_probability == .8
    assert len(_eligible_unique_news(news)) == 1


def test_irrelevant_full_article_does_not_fall_back_to_misleading_headline():
    news = classify_news(pd.DataFrame([candidate()]), NegativeClassifier(), article_fetcher=lambda url, _: {
        "url": url, "article_text": "SBI was fined by RBI for a regulatory breach. " + "Market commentary. " * 50,
        "retrieval_state": "retrieved"})
    assert not news.iloc[0].scoring_eligible and not news.iloc[0].headline_scored


def test_headline_weights_reduce_both_news_component_and_whole_number_penalty():
    news = classify_news(pd.DataFrame([candidate(i) for i in range(58)]), NegativeClassifier(), article_fetcher=blocked)
    ranking, detail = apply_incidents_and_rank(document_scores(), pd.DataFrame(), news)
    g = detail.set_index("pillar").loc["G"]
    assert g.news_target_met and g.headline_only_scored_count == 58
    assert g.news_average_negative_probability == pytest.approx(.8)
    assert g.news_average_effective_negative_probability == pytest.approx(.4)
    assert g.news_risk_score_0_7 == 4.2
    assert ranking.iloc[0].negative_news_candidates == 58
    assert ranking.iloc[0].weighted_negative_news_candidates == 29
    assert ranking.iloc[0].negative_news_penalty == 2  # round half-up(29 * .5 / 7)

    full = news.copy()
    full["headline_scored"] = False
    full["full_text_scored"] = True
    full["full_article_available"] = True
    full["retrieval_state"] = "retrieved"
    full["model_input_source"] = "full article"
    full["article_text"] = [" ".join(f"case{i}detail{j}" for j in range(150)) for i in range(58)]
    # Recompute weights from evidence type, rather than trusting stored weight.
    ranking, detail = apply_incidents_and_rank(document_scores(), pd.DataFrame(), full)
    assert detail.set_index("pillar").loc["G", "news_risk_score_0_7"] == 1.4
    assert ranking.iloc[0].negative_news_penalty == 4
    assert ranking.iloc[0].weighted_negative_news_candidates == 58

    mixed = pd.concat([full.iloc[:30], news.iloc[30:]], ignore_index=True)
    ranking, detail = apply_incidents_and_rank(document_scores(), pd.DataFrame(), mixed)
    assert ranking.iloc[0].negative_news_candidates == 58
    assert ranking.iloc[0].weighted_negative_news_candidates == 44
    assert ranking.iloc[0].negative_news_penalty == 3
    assert ranking.iloc[0].full_article_scored_count == 30
    assert ranking.iloc[0].headline_only_scored_count == 28


def test_no_model_does_not_invent_headline_probability():
    news = classify_news(pd.DataFrame([candidate()]), None, article_fetcher=blocked)
    assert not news.iloc[0].headline_scored and not news.iloc[0].scoring_eligible
    assert pd.isna(news.iloc[0].finbert_negative_probability)
    assert "unavailable" in news.iloc[0].scoring_error


def test_non_english_headline_stays_out(monkeypatch):
    monkeypatch.setitem(sys.modules, "langdetect", SimpleNamespace(detect=lambda _: "hi"))
    news = classify_news(pd.DataFrame([candidate()]), NegativeClassifier(), article_fetcher=blocked)
    assert not news.iloc[0].scoring_eligible
    assert _eligible_unique_news(news).empty


def test_headline_fallback_database_roundtrip(tmp_path):
    news = classify_news(pd.DataFrame([candidate()]), NegativeClassifier(), article_fetcher=blocked)
    path = tmp_path / "news.sqlite"
    save_evidence_batch("headline-run", news, pd.DataFrame(), [], path=path)
    restored = json.loads(read_table("news_articles", path=path).iloc[0].payload_json)
    assert restored["headline_scored"]
    assert restored["negative_evidence_weight"] == .5
    assert restored["effective_negative_probability"] == .4
