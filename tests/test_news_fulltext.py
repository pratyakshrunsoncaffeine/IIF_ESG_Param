from types import SimpleNamespace

import pandas as pd
import pytest

import esg_engine
from esg.news import _html_has_explicit_paywall, bodies_near_duplicate, chunk_entire_article, relevance_and_pillar, resolve_google_news_url, score_whole_article


class WordTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class TailSensitiveClassifier:
    tokenizer = WordTokenizer()

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self, texts, **kwargs):
        self.calls += 1
        if self.fail_on == self.calls:
            raise RuntimeError("synthetic classifier error")
        negative = "TAIL_MARKER" in texts[0]
        return [[
            {"label": "positive", "score": 0.05 if negative else 0.95},
            {"label": "neutral", "score": 0.05},
            {"label": "negative", "score": 0.90 if negative else 0.0},
        ]]


def company():
    return {"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank Limited", "aliases": ["HDFC Bank", "HDFC"]}


def long_text(event, tail=""):
    return (
        "HDFC Bank Limited operates banking services and serves customers across India. "
        + "The company has branches and employees, and this report describes its normal operations. " * 90
        + event + " " + tail + " " + "The company responded to questions from reporters. " * 30
    )


@pytest.mark.parametrize("event,pillar", [
    ("The RBI fined HDFC Bank Limited for a regulatory fine after a compliance breach.", "G"),
    ("HDFC Bank Limited was investigated after customers suffered customer harm and a data breach.", "S"),
    ("HDFC Bank Limited was fined for an illegal discharge and environmental violation.", "E"),
    ("HDFC Bank Limited workers reported unsafe working conditions after a workplace death.", "S"),
])
def test_target_linked_material_esg_conduct_is_accepted(event, pillar):
    accepted, decision, reason, actual_pillar, evidence = relevance_and_pillar(
        long_text(event), company(), "HDFCBANK.NS", "G",
        esg_engine.NEWS_EVENT_TERMS,
    )
    assert accepted and decision == "accepted", reason
    assert actual_pillar == pillar
    assert "HDFC Bank Limited" in evidence


def test_nuvama_hdfc_market_article_is_rejected():
    text = long_text("Nuvama reiterated its buy call on HDFC Bank shares and raised its target price.")
    accepted, decision, reason, _, _ = relevance_and_pillar(text, company(), "HDFCBANK.NS", "G", esg_engine.NEWS_EVENT_TERMS)
    assert not accepted
    assert decision == "rejected"
    assert "material adverse ESG conduct" in reason


def test_unrelated_actor_adverse_event_near_incidental_target_is_rejected():
    text = (
        "HDFC Bank Limited shares were flat in morning trading. ABC Chemicals was fined after "
        "an illegal discharge contaminated a river and harmed nearby residents. "
        + "The report continued with ordinary market commentary. " * 100
    )
    accepted, _, reason, _, _ = relevance_and_pillar(text, company(), "HDFCBANK.NS", "E", esg_engine.NEWS_EVENT_TERMS)
    assert not accepted, reason


@pytest.mark.parametrize("event", [
    "HDFC Bank Limited provided a price view while Nuvama was fined for fraud.",
    "HDFC Bank Limited financed Gamma Industries, which was fined for illegal discharge.",
    "SEBI investigated Gamma Securities while HDFC Bank Limited provided market commentary.",
    "HDFC Bank Limited was unaffected by Nuvama's fraud investigation.",
])
def test_nearby_adverse_actor_does_not_transfer_conduct_to_target(event):
    text = long_text(event)
    accepted, _, reason, _, _ = relevance_and_pillar(text, company(), "HDFCBANK.NS", "G", esg_engine.NEWS_EVENT_TERMS)
    assert not accepted, reason


def test_direct_regulator_action_on_target_is_attributed():
    text = long_text("RBI imposed a penalty on HDFC Bank Limited for KYC non-compliance.")
    accepted, decision, reason, pillar, evidence = relevance_and_pillar(
        text, company(), "HDFCBANK.NS", "S", esg_engine.NEWS_EVENT_TERMS,
    )
    assert accepted and decision == "accepted", reason
    assert pillar == "G"
    assert "penalty on HDFC Bank Limited" in evidence


def test_full_body_dedup_checks_distinct_tails_beyond_15000_characters():
    prefix = "Repeated syndicated reporting context. " * 500
    assert len(prefix) > 15000
    assert not bodies_near_duplicate(prefix + "Unique tail about a customer harm event.", prefix + "Unique tail about a pollution event.")
    assert bodies_near_duplicate(prefix + "The same complete article tail.", prefix + "The same complete article tail.")


def test_google_news_decoder_uses_bounded_current_api_and_audits_message():
    calls = []

    def decoder(url, timeout):
        calls.append((url, timeout))
        return {"success": True, "decoded_url": "https://www.livemint.com/story", "message": "decoded"}

    direct, message = resolve_google_news_url("https://news.google.com/rss/articles/example", ("livemint.com",), decoder)
    assert direct == "https://www.livemint.com/story"
    assert message == "decoded"
    assert calls == [("https://news.google.com/rss/articles/example", 15)]

    failed, reason = resolve_google_news_url("https://news.google.com/rss/articles/example", ("livemint.com",),
                                            lambda *_args, **_kwargs: {"success": False, "message": "could not decode"})
    assert not failed
    assert "could not decode" in reason


def test_generic_subscribe_navigation_is_not_a_paywall_but_locked_article_is():
    free_article = "<header>Subscribe to our newsletter</header><script src='/paywall.js'></script><article><p>Public full article</p></article>"
    locked_article = "<article><div class='premium-only paywall'>Subscribe to read</div></article>"
    structured_locked = '<script type="application/ld+json">{"@type":"NewsArticle","isAccessibleForFree":false}</script>'
    assert not _html_has_explicit_paywall(free_article)
    assert _html_has_explicit_paywall(locked_article)
    assert _html_has_explicit_paywall(structured_locked)


def test_long_article_tail_is_detected_and_all_chunks_are_scored(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langdetect", SimpleNamespace(detect=lambda _text: "en"))
    body = long_text("HDFC Bank Limited workers faced unsafe working conditions after a fatality. TAIL_MARKER")
    assert len(body.split()) > 512
    row = {
        "ticker": "HDFCBANK.NS", "company_name": "HDFC Bank Limited", "aliases": ["HDFC Bank", "HDFC"],
        "pillar": "S", "queried_pillars": "S", "publisher": "Mint", "url": "https://news.google.com/rss/articles/test",
        "title": "HDFC Bank incident", "language": "en",
    }
    fetcher = lambda _url, _domains: {
        "url": "https://www.livemint.com/companies/news/article-1.html", "article_text": body,
        "retrieval_state": "retrieved", "retrieval_reason": "test full text",
    }
    result = esg_engine.classify_news(pd.DataFrame([row]), TailSensitiveClassifier(), article_fetcher=fetcher)
    article = result.iloc[0]
    assert article["full_article_available"]
    assert article["url"].startswith("https://www.livemint.com/")
    assert article["pillar"] == "S"
    assert article["scoring_eligible"]
    assert article["full_text_scored"]
    assert article["chunks_scored"] == article["chunk_count"] > 1
    assert article["token_count"] > 512
    # The only negative signal sits in the article tail, so it contributes to
    # the whole-document weighted average without being lost to truncation.
    assert article["finbert_negative_probability"] > 0.15


@pytest.mark.parametrize("token_count", [478, 500, 510, 988, 1000])
def test_chunk_weights_cover_each_token_exactly_once(token_count):
    chunks, counted, weights = chunk_entire_article(" ".join(f"token{i}" for i in range(token_count)), WordTokenizer())
    assert chunks
    assert counted == token_count
    assert sum(weights) == token_count


def test_tokenizer_failure_does_not_fall_back_to_whitespace_chunks():
    class BrokenTokenizer:
        model_max_length = 512

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("broken wordpiece tokenizer")

    result = score_whole_article("Some complete article text", SimpleNamespace(tokenizer=BrokenTokenizer()))
    assert not result["full_text_scored"]
    assert result["chunk_count"] == 0
    assert "broken wordpiece tokenizer" in result["scoring_error"]


def test_failed_chunk_does_not_mark_article_as_full_text_scored(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langdetect", SimpleNamespace(detect=lambda _text: "en"))
    body = long_text("HDFC Bank Limited was fined for an environmental violation.")
    row = {
        "ticker": "HDFCBANK.NS", "company_name": "HDFC Bank Limited", "aliases": ["HDFC Bank", "HDFC"],
        "pillar": "E", "publisher": "Mint", "url": "https://livemint.com/story", "title": "HDFC Bank fined",
    }
    result = esg_engine.classify_news(
        pd.DataFrame([row]), TailSensitiveClassifier(fail_on=2),
        article_fetcher=lambda *_args: {"url": "https://livemint.com/story", "article_text": body,
                                        "retrieval_state": "retrieved", "retrieval_reason": "test"},
    ).iloc[0]
    assert result["chunk_count"] > 1
    assert result["chunks_scored"] == 1
    assert not result["full_text_scored"]
    assert not result["scoring_eligible"]
    assert "synthetic classifier error" in result["scoring_error"]


def test_failed_retrieval_does_not_count_even_with_probability():
    scores = pd.DataFrame([{"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank", "pillar": pillar,
                            "document_evidence_score_0_7": 7.0, "report_name": "annual.pdf"}
                           for pillar in ("E", "S", "G")])
    news = pd.DataFrame([{
        "ticker": "HDFCBANK.NS", "pillar": "E", "url": "https://livemint.com/story",
        "article_text": "failed response " * 80, "retrieval_state": "blocked", "full_article_available": False,
        "relevance_decision": "not evaluated", "scoring_eligible": False, "detected_language": "en",
        "full_text_scored": False, "publication_window_valid": False, "chunk_count": 1, "chunks_scored": 1,
        "finbert_negative_probability": 0.99,
    }])
    _, detail = esg_engine.apply_incidents_and_rank(scores, pd.DataFrame(), news)
    e = detail.set_index("pillar").loc["E"]
    assert e["news_articles_found"] == 0
    assert e["news_risk_score_0_7"] == 3.5
    assert e["negative_news_candidates"] == 0


def test_published_date_outside_search_window_is_audited_but_never_scored(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langdetect", SimpleNamespace(detect=lambda _text: "en"))
    classifier = TailSensitiveClassifier()
    row = {
        "ticker": "HDFCBANK.NS", "company_name": "HDFC Bank Limited", "aliases": ["HDFC Bank"],
        "pillar": "G", "publisher": "Mint", "url": "https://livemint.com/story",
        "title": "HDFC Bank RBI penalty", "published_at": "2017-01-01T00:00:00Z",
        "source_start": "2025-10-09", "source_end": "2026-10-09",
    }
    body = long_text("RBI imposed a penalty on HDFC Bank Limited for KYC non-compliance.")
    result = esg_engine.classify_news(
        pd.DataFrame([row]), classifier,
        article_fetcher=lambda *_args: {"url": "https://livemint.com/story", "article_text": body,
                                        "retrieval_state": "retrieved", "retrieval_reason": "test"},
    ).iloc[0]
    assert result["full_article_available"]
    assert not result["publication_window_valid"]
    assert result["relevance_decision"] == "not evaluated"
    assert not result["scoring_eligible"]
    assert classifier.calls == 0


def test_coverage_and_penalty_count_each_eligible_story_once():
    scores = pd.DataFrame([{"ticker": "HDFCBANK.NS", "company_name": "HDFC Bank", "pillar": pillar,
                            "document_evidence_score_0_7": 7.0, "report_name": "annual.pdf"}
                           for pillar in ("E", "S", "G")])
    rows = []
    for i in range(7):
        article_body = f"Unique complete environmental misconduct article {i}. " + " ".join(f"case{i}evidence{word}" for word in range(140))
        rows.append({
            "ticker": "HDFCBANK.NS", "pillar": "E", "url": f"https://livemint.com/story/{i}",
            "article_text": article_body, "retrieval_state": "retrieved", "full_article_available": True,
            "relevance_decision": "accepted", "scoring_eligible": True, "detected_language": "en",
            "full_text_scored": True, "publication_window_valid": True, "chunk_count": 1, "chunks_scored": 1,
            "finbert_negative_probability": 0.8,
        })
    # Syndicated near-identical body under another publisher must not add a
    # second coverage article or a second negative candidate.
    syndicated = dict(rows[0], url="https://reuters.com/story/1")
    rows.append(syndicated)
    ranking, detail = esg_engine.apply_incidents_and_rank(scores, pd.DataFrame(), pd.DataFrame(rows))
    assert detail.set_index("pillar").loc["E", "news_articles_found"] == 7
    assert ranking.iloc[0]["negative_news_candidates"] == 7
    assert ranking.iloc[0]["negative_news_penalty"] == 1
    assert ranking.iloc[0]["overall_score_0_21"] == max(0, ranking.iloc[0]["overall_score_before_negative_news_penalty"] - 1)
