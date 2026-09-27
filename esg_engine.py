from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests
import pymupdf

ROOT = Path(__file__).resolve().parent
TAXONOMY = json.loads((ROOT / "esg_taxonomy.json").read_text(encoding="utf-8"))
PILLAR_KEYWORDS = {
    "E": TAXONOMY["environment_keywords"],
    "S": TAXONOMY["social_keywords"],
    "G": TAXONOMY["governance_keywords"],
}
PILLAR_NAMES = {"E": "Environment", "S": "Social", "G": "Governance"}
NEWS_EVENT_TERMS = TAXONOMY["news_event_terms"]
NEWS_LOOKBACK_DAYS = 365
NEWS_SOURCE_WINDOW_DAYS = 90
NEGATIVE_REVIEW_THRESHOLD = 0.70
MAX_ARTICLE_FETCHES_PER_COMPANY = 20

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
    values = ([company["company_name"]] if company.get("company_name") else [company["ticker"]]) + company.get("aliases", [])
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _deduplicate_news(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["ticker", "pillar", "canonical_url"]).reset_index(drop=True)
    kept = []
    for idx, row in frame.sort_values("published_at", ascending=False).iterrows():
        duplicate = any(row["ticker"] == frame.loc[prior, "ticker"] and row["pillar"] == frame.loc[prior, "pillar"] and row.get("publisher", "") == frame.loc[prior].get("publisher", "") and SequenceMatcher(None, row["title"].lower(), frame.loc[prior, "title"].lower()).ratio() >= 0.94 for prior in kept)
        if not duplicate:
            kept.append(idx)
    return frame.loc[kept].reset_index(drop=True)


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
        term_block = " OR ".join(f'"{term}"' if " " in term else term for term in terms[:18])
        for entity in entities:
            params = {
                "query": f'"{entity}" ({term_block}) ({domain_block})', "mode": "artlist", "format": "json", "maxrecords": 250,
                "startdatetime": source_start.strftime("%Y%m%d000000"), "enddatetime": scan_end.strftime("%Y%m%d235959"), "sort": "datedesc",
            }
            try:
                response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=25)
                response.raise_for_status()
                for item in response.json().get("articles", []):
                    url, title = item.get("url"), item.get("title", "").strip()
                    if url and title:
                        host = (item.get("domain") or urlparse(url).netloc).lower().removeprefix("www.")
                        publisher = publisher_group(host)
                        if not publisher:
                            continue
                        articles.append({"ticker": company["ticker"], "company_name": company.get("company_name", ""), "pillar": pillar, "query_entity": entity,
                            "title": title, "url": url, "canonical_url": canonical_url(url), "domain": host, "publisher": publisher,
                            "published_at": item.get("seendate", ""), "language": item.get("language", ""), "source_country": item.get("sourcecountry", ""), "discovery_source": "GDELT"})
            except Exception as exc:
                errors.append({"ticker": company["ticker"], "source": "GDELT", "pillar": pillar, "query": entity, "error": f"{type(exc).__name__}: {str(exc)[:250]}"})
            time.sleep(0.15)
    coverage.append({"ticker": company["ticker"], "source": "GDELT", "requested_start": requested_start.isoformat(), "source_start": source_start.isoformat(), "source_end": scan_end.isoformat(),
        "coverage_complete": source_start <= requested_start, "coverage_note": "Approved publishers only: Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI. Recent article-list window only; older requested dates are not covered"})

    for pillar, terms in NEWS_EVENT_TERMS.items():
        term_block = " OR ".join(f'"{term}"' if " " in term else term for term in terms[:12])
        for entity in entities:
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
                    articles.append({"ticker": company["ticker"], "company_name": company.get("company_name", ""), "pillar": pillar, "query_entity": entity,
                        "title": title, "url": link, "canonical_url": canonical_url(link), "domain": source_name or urlparse(link).netloc, "publisher": publisher,
                        "published_at": published, "language": "", "source_country": "India", "discovery_source": "Google News RSS"})
            except Exception as exc:
                errors.append({"ticker": company["ticker"], "source": "Google News RSS", "pillar": pillar, "query": entity, "error": f"{type(exc).__name__}: {str(exc)[:250]}"})
            time.sleep(0.15)
    coverage.append({"ticker": company["ticker"], "source": "Google News RSS", "requested_start": requested_start.isoformat(), "source_start": source_start.isoformat(), "source_end": scan_end.isoformat(),
        "coverage_complete": False, "coverage_note": "Best-effort RSS results from approved publishers only; result cap, feed format and historical coverage are not guaranteed"})
    return _deduplicate_news(pd.DataFrame(articles)), pd.DataFrame(errors), pd.DataFrame(coverage)


def classify_news(candidates: pd.DataFrame, classifier) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    result["article_text"] = ""
    result["fetch_status"] = "not fetched: per-company prototype cap"
    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    for _, group in result.groupby("ticker"):
        for idx in list(group.index[:MAX_ARTICLE_FETCHES_PER_COMPANY]):
            if trafilatura is None:
                result.loc[idx, "fetch_status"] = "trafilatura is not installed"
                continue
            try:
                downloaded = trafilatura.fetch_url(result.loc[idx, "url"])
                body = trafilatura.extract(downloaded, include_comments=False, include_tables=False, favor_recall=True) if downloaded else ""
                if body:
                    result.loc[idx, "article_text"] = body[:10000]
                    result.loc[idx, "fetch_status"] = "article text retrieved"
                else:
                    result.loc[idx, "fetch_status"] = "publisher article unavailable; headline used"
            except Exception as exc:
                result.loc[idx, "fetch_status"] = f"fetch failed: {type(exc).__name__}"
            time.sleep(0.1)
    result["model_input_source"] = result["article_text"].map(lambda text: "article text" if text else "headline only")
    detected = []
    from langdetect import detect, LangDetectException
    for _, row in result.iterrows():
        text = str(row["article_text"] or row["title"])
        declared = str(row.get("language", "")).lower()
        if declared in {"english", "en"}:
            detected.append("en")
            continue
        try:
            detected.append(detect(text[:1200]))
        except LangDetectException:
            detected.append("unknown")
    result["detected_language"] = detected
    result["finbert_label"] = "not scored: language not identified as English"
    result["finbert_confidence"] = float("nan")
    result["finbert_negative_probability"] = float("nan")
    result["needs_analyst_review"] = True
    english_indices = list(result.index[result["detected_language"].eq("en")])
    if english_indices:
        texts = [str(result.loc[idx, "article_text"] or result.loc[idx, "title"])[:4000] for idx in english_indices]
        predictions = classifier(texts, top_k=None, truncation=True, max_length=512, batch_size=8)
        for idx, prediction in zip(english_indices, predictions):
            scores = {item["label"].lower(): float(item["score"]) for item in prediction}
            label = max(scores, key=scores.get)
            result.loc[idx, "finbert_label"] = label
            result.loc[idx, "finbert_confidence"] = scores[label]
            result.loc[idx, "finbert_negative_probability"] = scores.get("negative", 0.0)
            result.loc[idx, "needs_analyst_review"] = scores.get("negative", 0.0) >= NEGATIVE_REVIEW_THRESHOLD
    return result


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
    detail["provisional_score_0_7"] = (detail["document_evidence_score_0_7"] - detail["approved_incident_penalty"]).clip(lower=0).round(2)
    if not news.empty and "finbert_negative_probability" in news:
        neg = news[news["finbert_negative_probability"] >= NEGATIVE_REVIEW_THRESHOLD].groupby("ticker").size()
    else:
        neg = pd.Series(dtype="int64")
    detail["negative_news_candidates"] = detail["ticker"].map(neg).fillna(0).astype(int)
    ranking = detail.groupby(["ticker", "company_name"], as_index=False).agg(
        overall_score_0_21=("provisional_score_0_7", "sum"),
        negative_news_candidates=("negative_news_candidates", "max"),
    )
    ranking["overall_score_0_21"] = ranking["overall_score_0_21"].round(2)
    ranking["esg_rank"] = ranking["overall_score_0_21"].rank(method="min", ascending=False).astype(int)
    pivot = detail.pivot_table(index="ticker", columns="pillar", values="provisional_score_0_7", aggfunc="first").reset_index()
    for pillar in pillars:
        if pillar not in pivot:
            pivot[pillar] = 0.0
    ranking = ranking.merge(pivot, on="ticker", how="left").rename(columns={"E": "E_score_0_7", "S": "S_score_0_7", "G": "G_score_0_7"})
    detail = detail.merge(ranking[["ticker", "overall_score_0_21", "esg_rank"]], on="ticker", how="left")
    return ranking.sort_values(["esg_rank", "ticker"]).reset_index(drop=True), detail.sort_values(["esg_rank", "pillar"]).reset_index(drop=True)
