import pandas as pd

from esg_engine import apply_incidents_and_rank, NEWS_PILLAR_TARGETS


def test_news_has_larger_weight_and_shortfall_is_neutral():
    scores = pd.DataFrame([
        {"ticker": "ABC.NS", "company_name": "ABC", "pillar": pillar,
         "document_evidence_score_0_7": 7.0, "report_name": "annual.pdf"}
        for pillar in ("E", "S", "G")
    ])
    news_rows = []
    for pillar, target in NEWS_PILLAR_TARGETS.items():
        found = target if pillar == "E" else 0
        news_rows.extend({"ticker": "ABC.NS", "pillar": pillar, "canonical_url": f"https://news.example/{pillar}/{i}", "finbert_negative_probability": 0.0} for i in range(found))
    news = pd.DataFrame(news_rows)
    ranking, detail = apply_incidents_and_rank(scores, pd.DataFrame(), news)
    e_score = detail.set_index("pillar").loc["E"]
    g_score = detail.set_index("pillar").loc["G"]

    assert e_score["news_target_met"]
    assert e_score["news_weighted_component"] == 4.9  # 70% of 7
    assert not g_score["news_target_met"]
    assert g_score["news_risk_score_0_7"] == 3.5  # no-coverage neutral, not clean-news
    assert ranking.iloc[0]["news_targets_met"] is False or not ranking.iloc[0]["news_targets_met"]

    news.loc[news["pillar"].eq("E"), "finbert_negative_probability"] = 0.5
    _, negative_detail = apply_incidents_and_rank(scores, pd.DataFrame(), news)
    assert negative_detail.set_index("pillar").loc["E", "news_risk_score_0_7"] == 3.5
