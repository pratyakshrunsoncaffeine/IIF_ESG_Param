"""SQLite store for discovered news, extracted ESG evidence, and latest scores."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "iif_esg_database.sqlite3"


def database_path() -> Path:
    return Path(os.environ.get("IIF_ESG_DB_PATH", str(DEFAULT_DB_PATH)))


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    target = Path(path) if path else database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_database(path: str | Path | None = None) -> None:
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                ticker TEXT NOT NULL,
                pillar TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (ticker, pillar, canonical_url)
            );
            CREATE TABLE IF NOT EXISTS report_evidence (
                evidence_key TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                pillar TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS esg_dataset (
                dataset_key TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                pillar TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS score_history (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                pillar TEXT NOT NULL,
                run_at_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, ticker, pillar)
            );
            CREATE TABLE IF NOT EXISTS analyst_incidents (
                run_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                pillar TEXT NOT NULL,
                summary TEXT NOT NULL,
                confirmed_at_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, ticker, pillar, summary)
            );
            CREATE INDEX IF NOT EXISTS idx_news_ticker_pillar ON news_articles(ticker, pillar);
            CREATE INDEX IF NOT EXISTS idx_dataset_ticker_pillar ON esg_dataset(ticker, pillar);
            CREATE INDEX IF NOT EXISTS idx_scores_run ON score_history(run_id, ticker);
            """
        )


def _json(record: dict[str, Any]) -> str:
    cleaned = {}
    for key, value in record.items():
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                value = None
        except (TypeError, ValueError):
            pass
        cleaned[str(key)] = value
    return json.dumps(cleaned, ensure_ascii=False, default=str, allow_nan=False)


def _stable_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    return "|".join(str(record.get(field, "") or "") for field in fields)


def save_evidence_batch(
    run_id: str,
    news: pd.DataFrame,
    evidence: pd.DataFrame,
    dataset_records: list[dict[str, Any]],
    path: str | Path | None = None,
) -> None:
    initialize_database(path)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        for record in news.to_dict(orient="records") if not news.empty else []:
            ticker = str(record.get("ticker", "")).upper()
            pillar = str(record.get("pillar", ""))
            url = str(record.get("canonical_url") or record.get("url") or "")
            if not ticker or not pillar or not url:
                continue
            conn.execute(
                """INSERT INTO news_articles(ticker,pillar,canonical_url,first_seen_utc,last_seen_utc,payload_json)
                VALUES(?,?,?,?,?,?) ON CONFLICT(ticker,pillar,canonical_url) DO UPDATE SET
                last_seen_utc=excluded.last_seen_utc,payload_json=excluded.payload_json""",
                (ticker, pillar, url, now, now, _json(record)),
            )
        for record in evidence.to_dict(orient="records") if not evidence.empty else []:
            ticker = str(record.get("ticker", "")).upper()
            pillar = str(record.get("pillar", ""))
            key = _stable_key(record, ("ticker", "report_name", "page", "pillar", "topic", "evidence_excerpt"))
            if not ticker:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO report_evidence(evidence_key,ticker,pillar,payload_json,last_seen_utc) VALUES(?,?,?,?,?)",
                (key, ticker, pillar, _json(record), now),
            )
        for record in dataset_records:
            ticker = str(record.get("ticker", "")).upper()
            pillar = str(record.get("pillar", ""))
            key = _stable_key(record, ("ticker", "source_file", "page_number", "pillar", "topic", "keyword", "matched_sentence"))
            if not ticker:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO esg_dataset(dataset_key,ticker,pillar,payload_json,last_seen_utc) VALUES(?,?,?,?,?)",
                (key, ticker, pillar, _json(record), now),
            )


def save_score_batch(
    run_id: str,
    rankings: pd.DataFrame,
    pillar_detail: pd.DataFrame,
    incidents: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> None:
    initialize_database(path)
    now = datetime.now(timezone.utc).isoformat()
    rows = pillar_detail.to_dict(orient="records") if not pillar_detail.empty else []
    ranking_map = {
        str(row.get("ticker", "")).upper(): row
        for row in rankings.to_dict(orient="records") if not rankings.empty
    }
    with _connect(path) as conn:
        for row in rows:
            ticker, pillar = str(row.get("ticker", "")).upper(), str(row.get("pillar", ""))
            if not ticker or not pillar:
                continue
            payload = {**row, "ranking": ranking_map.get(ticker, {})}
            conn.execute(
                "INSERT OR REPLACE INTO score_history(run_id,ticker,pillar,run_at_utc,payload_json) VALUES(?,?,?,?,?)",
                (run_id, ticker, pillar, now, _json(payload)),
            )
        if incidents is not None and not incidents.empty:
            for item in incidents.to_dict(orient="records"):
                ticker = str(item.get("ticker", "")).upper()
                pillar = str(item.get("pillar", ""))
                summary = str(item.get("summary", "")).strip()
                if ticker and pillar and summary:
                    conn.execute(
                        "INSERT OR REPLACE INTO analyst_incidents(run_id,ticker,pillar,summary,confirmed_at_utc,payload_json) VALUES(?,?,?,?,?,?)",
                        (run_id, ticker, pillar, summary, now, _json(item)),
                    )


def database_summary(path: str | Path | None = None) -> dict[str, Any]:
    initialize_database(path)
    with _connect(path) as conn:
        counts = {}
        for table in ("news_articles", "report_evidence", "esg_dataset", "score_history", "analyst_incidents"):
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        latest = conn.execute("SELECT MAX(run_at_utc) FROM score_history").fetchone()[0]
        companies = [row[0] for row in conn.execute("SELECT DISTINCT ticker FROM news_articles ORDER BY ticker")]
    return {"counts": counts, "latest_score_at": latest, "companies": companies}


def read_table(table: str, path: str | Path | None = None) -> pd.DataFrame:
    allowed = {"news_articles", "report_evidence", "esg_dataset", "score_history", "analyst_incidents"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    initialize_database(path)
    with _connect(path) as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        headers = [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
    return pd.DataFrame(rows, columns=headers)


def database_bytes(path: str | Path | None = None) -> bytes:
    initialize_database(path)
    target = Path(path) if path else database_path()
    with _connect(path) as conn:
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    return target.read_bytes()
