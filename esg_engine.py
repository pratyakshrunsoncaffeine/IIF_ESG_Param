from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import pymupdf
from esg.dictionary_loader import load_dictionary
from esg.news import bodies_near_duplicate, retrieve_full_article, relevance_and_pillar, score_whole_article

ROOT = Path(__file__).resolve().parent
TAXONOMY = json.loads((ROOT / "esg_taxonomy.json").read_text(encoding="utf-8"))
ESG_DICTIONARY = load_dictionary(ROOT / "data" / "normalized_dictionary.json")
TAXONOMY["environment_keywords"] = ESG_DICTIONARY["Environmental"]
TAXONOMY["social_keywords"] = ESG_DICTIONARY["Social"]
TAXONOMY["governance_keywords"] = ESG_DICTIONARY["Governance"]
PILLAR_KEYWORDS = {
    "E": ESG_DICTIONARY["Environmental"],
    "S": ESG_DICTIONARY["Social"],
    "G": ESG_DICTIONARY["Governance"],
}
PILLAR_NAMES = {"E": "Environment", "S": "Social", "G": "Governance"}
NEWS_EVENT_TERMS = TAXONOMY["news_event_terms"]
NEWS_LOOKBACK_DAYS = 365
NEWS_SOURCE_WINDOW_DAYS = 365
NEGATIVE_REVIEW_THRESHOLD = 0.70
NEWS_PILLAR_TARGETS = {"E": 20, "S": 20, "G": 50}
NEWS_SCORE_WEIGHT = 0.70
REPORT_SCORE_WEIGHT = 0.30
NEGATIVE_NEWS_PENALTY_PER_7 = 0.5

# Only these Indian business-news publishers are admitted to the adverse-news
# candidate set. Reuters and PTI are separate publisher groups; syndication or
# republication of one story should still be treated as one underlying source.
APPROVED_NEWS_PUBLISHERS = {
    "The Economic Times": ("economictimes.indiatimes.com", "economictimes.com"),
    "Business Standard": ("business-standard.com",),
    "Mint": ("livemint.com",),
    "Moneycontrol": ("moneycontrol.com",),
    "Reuters": ("reuters.com",),
    "PTI": ("ptinews.com", "pti.in"),
}
NEWS_SOURCE_DOMAINS = tuple(domain for domains in APPROVED_NEWS_PUBLISHERS.values() for domain in domains)
FINAL_OFFICIAL_STATUSES = {
    "final regulator order or sanction",
    "final court judgment",
    "final statutory authority decision",
}
OFFICIAL_RECORD_DOMAINS = ("gov.in", "nic.in", "rbi.org.in")

NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:\d[\d,]*(?:\.\d+)?%?|\d+(?:\.\d+)?\s*(?:tonnes?|tons?|tCO2e|MWh|GWh|kWh|litres?|kilolitres?|employees?|beneficiaries))(?!\w)", re.I
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}(?:\s*[-/]\s*(?:19|20)?\d{2})?\b")
PROGRESS_PATTERN = re.compile(r"\b(target|goal|reduced|reduction|improved|increased|decreased|year[- ]on[- ]year|baseline|progress|achieved|aims to|committed to)\b", re.I)
TRACE_PATTERN = re.compile(r"\b(scope|baseline|intensity|per employee|per unit|per tonne|assured|assurance|verified|audited|boundary|coverage)\b", re.I)


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    pages = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        ocr_used = False
        issue = ""
        if len(re.sub(r"\s+", "", text)) < 60:
            try:
                ocr_page = page.get_textpage_ocr(language="eng", dpi=220, full=True)
                text = page.get_text("text", textpage=ocr_page) or text
                ocr_used = True
            except Exception as exc:
                issue = f"OCR unavailable or failed: {type(exc).__name__}: {exc}"
        pages.append({"page": page_number, "text": text, "characters": len(text), "ocr_used": ocr_used, "extraction_issue": issue})
    doc.close()
    if not any(page["text"].strip() for page in pages):
        raise ValueError("No usable text was extracted. For scanned reports, install Tesseract OCR and retry.")
    return pages


def collect_report_evidence(pages: list[dict], ticker: str, company_name: str, report_name: str) -> pd.DataFrame:
    records = []
    for pillar, topics in PILLAR_KEYWORDS.items():
        for topic, terms in topics.items():
            safe_terms = sorted({str(term).strip() for term in terms if str(term).strip()}, key=len, reverse=True)
            if not safe_terms:
                continue
            pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in safe_terms) + r")(?!\w)", re.I)
            for page in pages:
                matches = list(pattern.finditer(page["text"]))
                if not matches:
                    continue
                contexts = []
                for match in matches[:25]:
                    start, end = max(0, match.start() - 180), min(len(page["text"]), match.end() + 260)
                    contexts.append(page["text"][start:end].replace("\n", " "))
                context = " ".join(contexts)
                records.append({
                    "ticker": ticker, "company_name": company_name or ticker, "report_name": report_name,
                    "pillar": pillar, "pillar_name": PILLAR_NAMES[pillar], "topic": topic,
                    "page": page["page"], "keyword_hits": len(matches),
                    "numeric_signal": bool(NUMBER_PATTERN.search(context)),
                    "period_signal": bool(YEAR_PATTERN.search(context)),
                    "progress_signal": bool(PROGRESS_PATTERN.search(context)),
                    "traceability_signal": bool(TRACE_PATTERN.search(context)),
                    "evidence_excerpt": contexts[0][:420],
                })
    columns = ["ticker", "company_name", "report_name", "pillar", "pillar_name", "topic", "page", "keyword_hits", "numeric_signal", "period_signal", "progress_signal", "traceability_signal", "evidence_excerpt"]
    return pd.DataFrame(records, columns=columns)


def calculate_document_scores(evidence: pd.DataFrame, companies: list[dict]) -> pd.DataFrame:
    rows = []
    for company in companies:
        for pillar, topic_map in PILLAR_KEYWORDS.items():
            topics = list(topic_map)
            p = evidence[(evidence["ticker"] == company["ticker"]) & (evidence["pillar"] == pillar)] if not evidence.empty else pd.DataFrame()
            grouped = p.groupby("topic").agg(
                numeric_signal=("numeric_signal", "max"),
                progress_signal=("progress_signal", "max"),
                traceability_signal=("traceability_signal", "max"),
            ) if not p.empty else pd.DataFrame()
            present = set(grouped.index) if not grouped.empty else set()
            numeric = set(grouped.index[grouped["numeric_signal"]]) if not grouped.empty else set()
            progress = set(grouped.index[grouped["progress_signal"]]) if not grouped.empty else set()
            traceable = set(grouped.index[grouped["traceability_signal"] & grouped["numeric_signal"]]) if not grouped.empty else set()
            denominator = max(len(topics), 1)
            base_score = min(7.0, 4.0 * len(numeric) / denominator + 2.0 * len(progress) / denominator + (len(traceable) / max(len(numeric), 1) if numeric else 0.0))
            rows.append({
                "ticker": company["ticker"], "company_name": company["company_name"] or company["ticker"],
                "report_name": company["report_name"], "pillar": pillar, "pillar_name": PILLAR_NAMES[pillar],
                "document_evidence_score_0_7": round(base_score, 2),
                "topics_found": len(present), "topics_in_dictionary": denominator,
                "topic_coverage_pct": round(100 * len(present) / denominator, 1),
                "quantified_topics": len(numeric), "progress_topics": len(progress), "traceable_topics": len(traceable),
                "score_status": "PROVISIONAL: report evidence coverage only",
            })
    return pd.DataFrame(rows)


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    clean_query = [(k, v) for k, v in parse_qsl(parsed.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))]
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower().removeprefix("www."), parsed.path.rstrip("/"), "", urlencode(clean_query), ""))


def publisher_group(value: str) -> str:
    """Map an article host or Google News source label to an approved publisher."""
    raw = str(value or "").strip()
    host = urlparse(raw if "://" in raw else f"https://{raw}").netloc.lower().removeprefix("www.")
    for publisher, domains in APPROVED_NEWS_PUBLISHERS.items():
        if any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return publisher
    normalized = re.sub(r"[^a-z]", "", raw.lower())
    aliases = {
        "theeconomictimes": "The Economic Times", "economictimes": "The Economic Times",
        "businessstandard": "Business Standard", "mint": "Mint", "livemint": "Mint",
        "moneycontrol": "Moneycontrol", "reuters": "Reuters",
        "presstrustofindia": "PTI", "pti": "PTI",
    }
    return aliases.get(normalized, "")


def count_approved_publishers(source_urls: str) -> int:
    """Count distinct approved publisher groups from one direct article URL per line."""
    groups = set()
    for url in re.split(r"[\n;]+", str(source_urls or "")):
        url = url.strip()
        if url:
            group = publisher_group(url)
            if group:
                groups.add(group)
    return len(groups)


def is_official_record_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    host = parsed.netloc.lower().removeprefix("www.")
    return parsed.scheme == "https" and any(host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_RECORD_DOMAINS)


def _news_entities(company: dict) -> list[str]:
    aliases = company.get("aliases", [])
    if isinstance(aliases, str):
        aliases = re.split(r"[;,|]+", aliases)
    if not isinstance(aliases, (list, tuple, set)):
        aliases = []
    values = ([company["company_name"]] if company.get("company_name") else []) + list(aliases)
    ticker = re.sub(r"\.(?:NS|BO)$", "", str(company.get("ticker", "")), flags=re.I)
    if ticker:
        values.append(ticker)
    return list(dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip())) or [company["ticker"]]


def _deduplicate_news(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    # Discovery can find the same story through multiple terms, entities,
    # sources and pillar searches. Keep one row per company + discovered URL,
    # while retaining every queried pillar for the later body-based decision.
    keys = [column for column in ("ticker", "canonical_url") if column in frame]
    if not keys:
        return frame.reset_index(drop=True)
    rows = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].copy()
        row["queried_pillars"] = ",".join(dict.fromkeys(str(value) for value in group.get("pillar", []) if str(value)))
        row["query_entities"] = ", ".join(dict.fromkeys(str(value) for value in group.get("query_entity", []) if str(value)))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def scan_news_for_company(company: dict, scan_end: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scan_end = scan_end or date.today()
    requested_start = scan_end - timedelta(days=NEWS_LOOKBACK_DAYS)
    source_start = max(requested_start, scan_end - timedelta(days=NEWS_SOURCE_WINDOW_DAYS - 1))
    entities = _news_entities(company)
    articles, errors, coverage = [], [], []
    domain_block = " OR ".join(f"domain:{domain}" for domain in NEWS_SOURCE_DOMAINS)
    site_block = " OR ".join(f"site:{domain}" for domain in NEWS_SOURCE_DOMAINS)
    session = requests.Session()
    session.headers.update({"User-Agent": "IIF-ESG-Research/0.1"})
    for pillar, terms in NEWS_EVENT_TERMS.items():
        target = NEWS_PILLAR_TARGETS.get(pillar, 20)
        # Narrow term batches improve recall because RSS returns only a small
        # number of results per query. The target is a search goal, not a
        # guarantee: publisher access and the search indexes control coverage.
        batch_size = 10 if target <= 20 else 4
        term_batches = [terms[i:i + batch_size] for i in range(0, len(terms), batch_size)]
        before_count = len(articles)
        for entity in entities:
            for batch in term_batches:
                term_block = " OR ".join(f'"{term}"' if " " in term else term for term in batch)
                params = {
                    "query": f'"{entity}" ({term_block}) ({domain_block}) sourcecountry:IN',
                    "mode": "artlist", "format": "json", "maxrecords": 250,
                    "startdatetime": source_start.strftime("%Y%m%d000000"),
                    "enddatetime": scan_end.strftime("%Y%m%d235959"), "sort": "datedesc",
                }
                try:
                    response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=25)
                    response.raise_for_status()
                    for item in response.json().get("articles", []):
                        url, title = item.get("url"), item.get("title", "").strip()
                        if url and title:
                            host = (item.get("domain") or urlparse(url).netloc).lower().removeprefix("www.")
                            publisher = publisher_group(host)
                            if publisher:
                                articles.append({"ticker": company["ticker"], "company_name": company.get("company_name", ""), "aliases": company.get("aliases", []), "pillar": pillar, "query_entity": entity,
                                    "title": title, "url": url, "canonical_url": canonical_url(url), "domain": host, "publisher": publisher,
                                    "published_at": item.get("seendate", ""), "language": item.get("language", ""), "source_country": item.get("sourcecountry", ""), "discovery_source": "GDELT",
                                    "source_start": source_start.isoformat(), "source_end": scan_end.isoformat()})
                except Exception as exc:
                    errors.append({"ticker": company["ticker"], "source": "GDELT", "pillar": pillar, "query": entity, "error": f"{type(exc).__name__}: {str(exc)[:250]}"})
                query = f'"{entity}" ({term_block}) ({site_block}) after:{source_start.isoformat()} before:{(scan_end + timedelta(days=1)).isoformat()}'
                url = "https://news.google.com/rss/search?" + urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
                try:
                    response = session.get(url, timeout=25)
                    response.raise_for_status()
                    root = ET.fromstring(response.content)
                    for item in root.findall(".//item"):
                        title, link = item.findtext("title", default="").strip(), item.findtext("link", default="").strip()
                        if not title or not link:
                            continue
                        source = item.find("source")
                        source_name = source.text.strip() if source is not None and source.text else ""
                        publisher = publisher_group(source_name) or publisher_group(link)
                        if not publisher:
                            continue
                        try:
                            published = parsedate_to_datetime(item.findtext("pubDate", default="")).isoformat()
                        except Exception:
                            published = item.findtext("pubDate", default="")
                        articles.append({"ticker": company["ticker"], "company_name": company.get("company_name", ""), "aliases": company.get("aliases", []), "pillar": pillar, "query_entity": entity,
                            "title": title, "url": link, "canonical_url": canonical_url(link), "domain": source_name or urlparse(link).netloc, "publisher": publisher,
                            "published_at": published, "language": "", "source_country": "India", "discovery_source": "Google News RSS",
                            "source_start": source_start.isoformat(), "source_end": scan_end.isoformat()})
                except Exception as exc:
                    errors.append({"ticker": company["ticker"], "source": "Google News RSS", "pillar": pillar, "query": entity, "error": f"{type(exc).__name__}: {str(exc)[:250]}"})
                time.sleep(0.05)
        coverage.append({"ticker": company["ticker"], "pillar": pillar, "target_articles": target,
            "candidates_found": max(0, len(_deduplicate_news(pd.DataFrame(articles[before_count:])))),
            "source": "GDELT + Google News RSS", "requested_start": requested_start.isoformat(), "source_start": source_start.isoformat(), "source_end": scan_end.isoformat(),
            "coverage_complete": False, "coverage_note": "Search attempted approved Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI publishers in narrow ESG-event query batches. The 365-day window and target count are best-effort, not guaranteed by publisher/search indexes."})
    return _deduplicate_news(pd.DataFrame(articles)), pd.DataFrame(errors), pd.DataFrame(coverage)


def classify_news(candidates: pd.DataFrame, classifier, article_fetcher=None, progress_callback=None) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    result["article_text"] = ""
    defaults = {
        "retrieval_state": "pending", "retrieval_reason": "", "full_article_available": False,
        "google_news_resolution": "",
        "relevance_decision": "pending", "relevance_reason": "", "scoring_eligible": False,
        "attribution_evidence": "",
        "detected_language": "unknown", "finbert_label": "not scored", "finbert_confidence": float("nan"),
        "finbert_negative_probability": float("nan"), "needs_analyst_review": False,
        "publication_window_valid": False, "publication_window_reason": "Publication window has not been checked",
        "chunk_count": 0, "chunks_scored": 0, "token_count": 0, "full_text_scored": False,
        "scoring_error": "", "duplicate_of": "",
    }
    for column, value in defaults.items():
        result[column] = value
    result["model_input_source"] = "none: no headline fallback"
    result["fetch_status"] = "retrieval pending"
    total = len(result)
    domains_for = dict(APPROVED_NEWS_PUBLISHERS)

    def progress(event, **details):
        if progress_callback:
            try:
                progress_callback({"event": event, "total": total, **details})
            except Exception:
                pass

    for processed, idx in enumerate(result.index, start=1):
        row = result.loc[idx]
        publisher = str(row.get("publisher", ""))
        domains = domains_for.get(publisher, ())
        progress("retrieval_started", ticker=row.get("ticker", ""), title=row.get("title", ""), processed=processed - 1)
        try:
            if article_fetcher is None:
                retrieval = retrieve_full_article(str(row.get("url", "")), domains)
            else:
                retrieval = article_fetcher(str(row.get("url", "")), domains)
            if not isinstance(retrieval, dict):
                raise TypeError("article_fetcher must return a retrieval metadata dictionary")
        except Exception as exc:
            retrieval = {"url": row.get("url", ""), "article_text": "", "retrieval_state": "failed", "retrieval_reason": f"{type(exc).__name__}: {exc}"}
        direct_url = str(retrieval.get("url") or row.get("url", ""))
        body = str(retrieval.get("article_text") or "")
        state = str(retrieval.get("retrieval_state", "failed"))
        reason = str(retrieval.get("retrieval_reason", ""))
        if state == "retrieved" and (not publisher_group(direct_url) or publisher_group(direct_url) != publisher):
            state, reason, body = "rejected_redirect", "Final URL is not an approved direct publisher URL", ""
        available = state == "retrieved" and len(body.strip()) >= 500
        if state == "retrieved" and not available:
            state, reason = "too_short", "Extracted article body is shorter than the 500-character minimum"
        result.at[idx, "url"] = direct_url
        result.at[idx, "canonical_url"] = canonical_url(direct_url)
        result.at[idx, "article_text"] = body
        result.at[idx, "retrieval_state"] = state
        result.at[idx, "retrieval_reason"] = reason
        result.at[idx, "google_news_resolution"] = str(retrieval.get("google_news_resolution", ""))
        result.at[idx, "full_article_available"] = available
        result.at[idx, "fetch_status"] = reason
        result.at[idx, "model_input_source"] = "full article" if available else "none: no headline fallback"
        progress("retrieval_complete", ticker=row.get("ticker", ""), processed=processed,
                 retrieval_state=state, full_article_available=available)
        if not available:
            result.at[idx, "relevance_decision"] = "not evaluated"
            result.at[idx, "relevance_reason"] = "Full article retrieval did not produce a usable body"
            continue

        window_start, window_end = row.get("source_start"), row.get("source_end")
        if pd.isna(window_start) or pd.isna(window_end):
            window_valid, window_reason = True, "No discovery date bounds were supplied"
        else:
            published = pd.to_datetime(row.get("published_at", ""), errors="coerce", utc=True)
            try:
                start_date = pd.to_datetime(window_start, errors="raise", utc=True).date()
                end_date = pd.to_datetime(window_end, errors="raise", utc=True).date()
            except Exception:
                start_date, end_date = None, None
            window_valid = bool(
                pd.notna(published) and start_date is not None and end_date is not None
                and start_date <= published.date() <= end_date
            )
            window_reason = "Published date is inside the requested search window" if window_valid else "Publication date is missing, invalid or outside the requested search window"
        result.at[idx, "publication_window_valid"] = window_valid
        result.at[idx, "publication_window_reason"] = window_reason
        if not window_valid:
            result.at[idx, "relevance_decision"] = "not evaluated"
            result.at[idx, "relevance_reason"] = window_reason
            continue

        company = {"company_name": str(row.get("company_name", "")), "aliases": row.get("aliases", [])}
        if isinstance(company["aliases"], str):
            company["aliases"] = [value.strip() for value in company["aliases"].split("|") if value.strip()]
        queried = str(row.get("queried_pillars", row.get("pillar", ""))).split(",")
        queried_pillar = next((value.strip() for value in queried if value.strip() in PILLAR_NAMES), "")
        relevant, decision, relevance_reason, material_pillar, attribution_evidence = relevance_and_pillar(
            body, company, str(row.get("ticker", "")), queried_pillar, NEWS_EVENT_TERMS,
        )
        result.at[idx, "relevance_decision"] = decision
        result.at[idx, "relevance_reason"] = relevance_reason
        result.at[idx, "attribution_evidence"] = attribution_evidence
        if not relevant:
            continue
        result.at[idx, "pillar"] = material_pillar
        try:
            from langdetect import detect
            language = detect(body[:10000])
        except Exception:
            language = "unknown"
        result.at[idx, "detected_language"] = language
        if language != "en":
            result.at[idx, "relevance_reason"] = f"{relevance_reason}; full article language is {language}, and FinBERT scoring is English-only"
            continue
        scored = score_whole_article(
            body, classifier,
            progress_callback=lambda event_details: progress(
                "chunk_scored", ticker=row.get("ticker", ""), processed=processed, **event_details,
            ),
        )
        for column, value in scored.items():
            result.at[idx, column] = value
        result.at[idx, "needs_analyst_review"] = bool(
            scored.get("full_text_scored") and scored.get("finbert_negative_probability", 0.0) >= NEGATIVE_REVIEW_THRESHOLD
        )
        eligible = bool(scored.get("full_text_scored") and scored.get("chunks_scored") == scored.get("chunk_count") and publisher_group(direct_url) == publisher)
        result.at[idx, "scoring_eligible"] = eligible
        if not eligible and scored.get("scoring_error"):
            result.at[idx, "relevance_reason"] = f"{relevance_reason}; full-text scoring failed: {scored['scoring_error']}"
        progress("article_scored", ticker=row.get("ticker", ""), processed=processed,
                 chunk_count=scored.get("chunk_count", 0), chunks_scored=scored.get("chunks_scored", 0),
                 full_text_scored=scored.get("full_text_scored", False), scoring_eligible=eligible)

    # Syndicated copies can arrive from multiple approved publishers. Keep all
    # records for audit, but only the first full-text copy can enter scoring.
    processed_bodies = []
    for idx in result.index[result["scoring_eligible"]].tolist():
        ticker = str(result.at[idx, "ticker"])
        body = str(result.at[idx, "article_text"])
        duplicate = next((prior for prior in processed_bodies if prior[0] == ticker and bodies_near_duplicate(prior[2], body)), None)
        if duplicate:
            result.at[idx, "scoring_eligible"] = False
            result.at[idx, "duplicate_of"] = duplicate[1]
            result.at[idx, "relevance_reason"] = "Syndicated or near-identical full article already counted from " + duplicate[1]
        else:
            processed_bodies.append((ticker, str(result.at[idx, "url"]), body))
    return result


def _eligible_unique_news(news: pd.DataFrame) -> pd.DataFrame:
    """Fail closed: only verified, full-text, fully scored articles enter metrics."""
    required = {
        "ticker", "pillar", "url", "article_text", "retrieval_state", "full_article_available",
        "relevance_decision", "scoring_eligible", "detected_language", "full_text_scored",
        "publication_window_valid", "chunk_count", "chunks_scored", "finbert_negative_probability",
    }
    if news.empty or not required.issubset(news.columns):
        return pd.DataFrame(columns=list(required))
    rows = news.copy()
    probabilities = pd.to_numeric(rows["finbert_negative_probability"], errors="coerce")
    lengths = rows["article_text"].fillna("").astype(str).str.strip().str.len()
    truthy = lambda column: rows[column].map(lambda value: str(value).strip().lower() == "true")
    mask = (
        truthy("scoring_eligible")
        & truthy("full_article_available")
        & truthy("full_text_scored")
        & truthy("publication_window_valid")
        & rows["retrieval_state"].eq("retrieved")
        & rows["relevance_decision"].eq("accepted")
        & rows["detected_language"].eq("en")
        & pd.to_numeric(rows["chunk_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(rows["chunks_scored"], errors="coerce").eq(pd.to_numeric(rows["chunk_count"], errors="coerce"))
        & lengths.ge(500)
        & probabilities.between(0.0, 1.0, inclusive="both")
    )
    rows = rows.loc[mask].copy()
    rows["finbert_negative_probability"] = probabilities.loc[rows.index]
    rows = rows[rows.apply(lambda row: publisher_group(str(row["url"])) in APPROVED_NEWS_PUBLISHERS, axis=1)]
    kept = []
    seen_urls = set()
    for idx, row in rows.iterrows():
        url = canonical_url(str(row["url"]))
        ticker = str(row["ticker"])
        if (ticker, url) in seen_urls:
            continue
        if any(str(rows.loc[prior, "ticker"]) == ticker and bodies_near_duplicate(str(rows.loc[prior, "article_text"]), str(row["article_text"])) for prior in kept):
            continue
        seen_urls.add((ticker, url))
        kept.append(idx)
    return rows.loc[kept].copy()


def apply_incidents_and_rank(document_scores: pd.DataFrame, incidents: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pillars = list(PILLAR_NAMES)
    penalties = {(ticker, pillar): 0.0 for ticker in document_scores["ticker"].unique() for pillar in pillars}
    if not incidents.empty:
        for _, item in incidents.iterrows():
            ticker, pillar = str(item.get("ticker", "")).strip().upper(), item.get("pillar", "")
            if not ticker and not pillar and not str(item.get("summary", "")).strip():
                continue
            key = (ticker, pillar)
            if key not in penalties:
                raise ValueError(f"Incident has an unknown ticker or pillar: {key}")
            if not bool(item.get("analyst_confirmed", False)):
                continue
            if bool(item.get("already_in_metric", False)):
                continue
            sources = count_approved_publishers(item.get("source_urls", ""))
            status = str(item.get("status", "")).strip().lower()
            official_url = str(item.get("official_record_url", "")).strip()
            valid_official_url = is_official_record_url(official_url)
            if sources < 5 and not (status in FINAL_OFFICIAL_STATUSES and valid_official_url):
                raise ValueError(f"Incident for {ticker} needs direct URLs from five distinct approved publishers, or a URL to a final regulator order or sanction, final court judgment, or final statutory authority decision.")
            severity = float(item.get("severity_points", 0) or 0)
            if severity not in {0.5, 1.0, 2.0}:
                raise ValueError("Use a severity adjustment of 0.5, 1.0 or 2.0 points.")
            penalties[key] += severity
    detail = document_scores.copy()
    detail["approved_incident_penalty"] = [penalties[(r.ticker, r.pillar)] for r in detail.itertuples()]
    usable_news = _eligible_unique_news(news)
    coverage_counts = usable_news.groupby(["ticker", "pillar"]).size().to_dict() if not usable_news.empty else {}
    detail["news_articles_found"] = [int(coverage_counts.get((r.ticker, r.pillar), 0)) for r in detail.itertuples()]
    detail["news_article_target"] = detail["pillar"].map(NEWS_PILLAR_TARGETS).fillna(20).astype(int)
    detail["news_target_met"] = detail["news_articles_found"] >= detail["news_article_target"]
    # FinBERT sentiment contributes to the news-heavy provisional score, but
    # incomplete coverage or missing model outputs are neutral, never clean.
    if not usable_news.empty and "finbert_negative_probability" in usable_news:
        probabilities = pd.to_numeric(usable_news["finbert_negative_probability"], errors="coerce")
        scored_news = usable_news.loc[probabilities.notna(), ["ticker", "pillar"]].copy()
        scored_news["negative_probability"] = probabilities[probabilities.notna()]
        mean_negative = scored_news.groupby(["ticker", "pillar"])["negative_probability"].mean().to_dict()
    else:
        mean_negative = {}
    detail["news_average_negative_probability"] = [mean_negative.get((r.ticker, r.pillar), float("nan")) for r in detail.itertuples()]
    detail["news_component_basis"] = [
        "FinBERT mean negative probability" if bool(row.news_target_met) and pd.notna(row.news_average_negative_probability)
        else "neutral: article target or model coverage shortfall"
        for row in detail.itertuples()
    ]
    news_baseline = [
        7.0 * (1.0 - float(row.news_average_negative_probability))
        if bool(row.news_target_met) and pd.notna(row.news_average_negative_probability)
        else 3.5
        for row in detail.itertuples()
    ]
    # Analyst-confirmed incident severity is a minimum penalty. Taking the
    # lower score avoids counting the same event twice on top of sentiment.
    detail["news_risk_score_0_7"] = [
        round(max(0.0, min(base, 7.0 - penalty)), 2)
        for base, penalty in zip(news_baseline, detail["approved_incident_penalty"])
    ]
    detail["report_weighted_component"] = (detail["document_evidence_score_0_7"] * REPORT_SCORE_WEIGHT).round(2)
    detail["news_weighted_component"] = (detail["news_risk_score_0_7"] * NEWS_SCORE_WEIGHT).round(2)
    detail["provisional_score_0_7"] = (detail["report_weighted_component"] + detail["news_weighted_component"]).round(2)
    neg = usable_news[pd.to_numeric(usable_news.get("finbert_negative_probability", pd.Series(index=usable_news.index, dtype=float)), errors="coerce") >= NEGATIVE_REVIEW_THRESHOLD].groupby(["ticker", "pillar"]).size().to_dict() if not usable_news.empty else {}
    detail["negative_news_candidates"] = [int(neg.get((r.ticker, r.pillar), 0)) for r in detail.itertuples()]
    ranking = detail.groupby(["ticker", "company_name"], as_index=False).agg(
        overall_score_before_negative_news_penalty=("provisional_score_0_7", "sum"),
        negative_news_candidates=("negative_news_candidates", "sum"),
        news_articles_found=("news_articles_found", "sum"),
        news_article_targets=("news_article_target", "sum"),
        news_targets_met=("news_target_met", "all"),
    )
    # Discrete whole-company deduction: 0.5 points per seven eligible
    # negative articles, rounded half-up, kept separate for auditability.
    ranking["negative_news_penalty"] = ranking["negative_news_candidates"].map(
        lambda count: math.floor((int(count) * NEGATIVE_NEWS_PENALTY_PER_7 / 7) + 0.5)
    )
    ranking["overall_score_before_negative_news_penalty"] = ranking["overall_score_before_negative_news_penalty"].round(2)
    ranking["overall_score_0_21"] = (
        ranking["overall_score_before_negative_news_penalty"] - ranking["negative_news_penalty"]
    ).clip(lower=0).round(2)
    ranking["esg_rank"] = ranking["overall_score_0_21"].rank(method="min", ascending=False).astype(int)
    pivot = detail.pivot_table(index="ticker", columns="pillar", values="provisional_score_0_7", aggfunc="first").reset_index()
    for pillar in pillars:
        if pillar not in pivot:
            pivot[pillar] = 0.0
    ranking = ranking.merge(pivot, on="ticker", how="left").rename(columns={"E": "E_score_0_7", "S": "S_score_0_7", "G": "G_score_0_7"})
    detail = detail.merge(ranking[["ticker", "negative_news_penalty", "overall_score_before_negative_news_penalty", "overall_score_0_21", "esg_rank"]], on="ticker", how="left")
    return ranking.sort_values(["esg_rank", "ticker"]).reset_index(drop=True), detail.sort_values(["esg_rank", "pillar"]).reset_index(drop=True)
