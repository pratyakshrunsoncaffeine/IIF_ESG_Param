import pandas as pd

from esg.database import database_bytes, database_summary, read_table, save_evidence_batch, save_score_batch


def test_database_stores_news_report_rows_and_scores(tmp_path):
    path = tmp_path / "test.sqlite3"
    news = pd.DataFrame([{
        "ticker": "ABC.NS", "pillar": "G", "canonical_url": "https://business-standard.com/story",
        "title": "Governance event", "finbert_negative_probability": float("nan"),
    }])
    evidence = pd.DataFrame([{
        "ticker": "ABC.NS", "pillar": "G", "report_name": "annual.pdf", "page": 4,
        "topic": "Board", "evidence_excerpt": "Board disclosure",
    }])
    dataset = [{
        "ticker": "ABC.NS", "pillar": "Governance", "source_file": "annual.pdf",
        "page_number": 4, "topic": "Board", "keyword": "board", "matched_sentence": "Board structure disclosed",
    }]
    save_evidence_batch("run-1", news, evidence, dataset, path)
    scores = pd.DataFrame([{
        "ticker": "ABC.NS", "pillar": "G", "provisional_score_0_7": 3.8,
        "news_articles_found": 1, "news_article_target": 50,
    }])
    ranks = pd.DataFrame([{"ticker": "ABC.NS", "overall_score_0_21": 10.0, "esg_rank": 1}])
    incidents = pd.DataFrame([{"ticker": "ABC.NS", "pillar": "G", "summary": "Confirmed event", "source_urls": "https://business-standard.com/story"}])
    save_score_batch("run-1", ranks, scores, incidents=incidents, path=path)

    summary = database_summary(path)
    assert summary["counts"] == {
        "news_articles": 1, "report_evidence": 1, "esg_dataset": 1, "score_history": 1, "analyst_incidents": 1,
    }
    assert read_table("news_articles", path).iloc[0]["ticker"] == "ABC.NS"
    assert read_table("score_history", path).iloc[0]["run_id"] == "run-1"
    assert database_bytes(path).startswith(b"SQLite format 3")
