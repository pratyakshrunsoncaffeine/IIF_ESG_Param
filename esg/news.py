"""Full-text news retrieval, company relevance and whole-article FinBERT helpers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from urllib.parse import urlparse

import requests

MIN_FULL_ARTICLE_CHARACTERS = 500
FINBERT_MAX_TOKENS = 512
FINBERT_OVERLAP_TOKENS = 32

_ADVERSE_CONDUCT = re.compile(
    r"\b(?:fined?|penalt(?:y|ies)|sanction(?:ed)?|violat\w*|breach\w*|"
    r"investigat\w*|probe\w*|alleg\w*|accus\w*|charged|charges|lawsuit|"
    r"litigation|fraud|corrupt\w*|brib\w*|spill|pollut\w*|contaminat\w*|"
    r"fatalit\w*|death|injur\w*|unsafe|recall\w*|harass\w*|discriminat\w*|"
    r"unpaid wages|wage theft|child labour|forced labour|data breach|"
    r"customer harm|consumer harm|worker harm|environmental damage)\b", re.I,
)
_MARKET_ONLY = re.compile(
    r"\b(?:shares?|stock|scrip|equity|market|trading|brokerage|analyst|"
    r"price target|buy call|sell call|upgrade|downgrade|recommendation|"
    r"rating|target price|Nuvama|Novama)\b", re.I,
)
_PAYWALL = re.compile(
    r"(?:subscribe\s+to\s+(?:read|continue|unlock)|subscribe\s+for\s+full\s+access|"
    r"premium\s+(?:article|story|content)|sign\s+in\s+to\s+read|unlock\s+this\s+story|"
    r"you(?:'|’)ve\s+reached\s+(?:your\s+)?(?:article|reading)\s+limit|"
    r"register\s+to\s+continue\s+reading|paywall)", re.I,
)
_EXPLICIT_LOCKED_MARKUP = re.compile(
    r"(?:class|id|data-access|data-paywall)[\w\s\-_=\"']{0,100}"
    r"(?:paywall|subscription-wall|subscriber-only|metered-content|premium-only|locked-content)", re.I,
)


def _html_has_explicit_paywall(html: str) -> bool:
    """Require article-scoped lock metadata/markup; generic subscribe UI is ignored."""
    scripts = re.findall(r"<script\b[^>]*type=[\"']application/ld\+json[\"'][^>]*>([\s\S]*?)</script\s*>", html, re.I)
    pending = list(scripts)
    while pending:
        try:
            data = json.loads(pending.pop())
        except (TypeError, ValueError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                raw_type = item.get("@type", [])
                types = raw_type if isinstance(raw_type, list) else [raw_type]
                if any(str(value).lower() in {"article", "newsarticle"} for value in types) and str(item.get("isAccessibleForFree", "")).lower() == "false":
                    return True
                stack.extend(value for value in item.values() if isinstance(value, (dict, list)))
    if re.search(r'<meta[^>]+(?:article:content_tier|content_tier)[^>]+content=["\'](?:metered|premium|paid|subscriber)["\']', html, re.I):
        return True
    scoped = re.findall(r"<(?:article|main)\b[^>]*>[\s\S]{0,250000}?</(?:article|main)\s*>", html[:500000], re.I)
    return any(_EXPLICIT_LOCKED_MARKUP.search(fragment) for fragment in scoped)


def _host(url: str) -> str:
    return urlparse(str(url or "")).netloc.lower().removeprefix("www.").split(":")[0]


def is_approved_url(url: str, publisher_domains: tuple[str, ...]) -> bool:
    host = _host(url)
    return bool(host and any(host == domain or host.endswith("." + domain) for domain in publisher_domains))


def resolve_google_news_url(url: str, publisher_domains: tuple[str, ...], decoder=None) -> tuple[str, str]:
    """Resolve modern Google News links with a bounded decoder call."""
    try:
        if decoder is None:
            from googlenewsdecoder import gnewsdecoder
            decoder = gnewsdecoder
        result = decoder(url, timeout=15)
    except Exception as exc:
        return "", f"Google News direct-link resolution failed: {type(exc).__name__}: {exc}"
    if not isinstance(result, dict):
        return "", "Google News decoder returned an unexpected response"
    message = str(result.get("message", "")).strip()
    direct = str(result.get("decoded_url", "")).strip()
    # googlenewsdecoder 0.2.x uses `success`; earlier/alternate builds used
    # `status`. Accept the explicit legacy key only when the current one is absent.
    succeeded = result.get("success") is True if "success" in result else result.get("status") is True
    if not succeeded or not direct.startswith("http"):
        return "", message or "Google News decoder did not resolve the article"
    if not is_approved_url(direct, publisher_domains):
        return "", f"Decoded URL is outside the approved publisher domains: {message}".strip()
    return direct, message or "Google News link decoded to approved publisher URL"


def retrieve_full_article(url: str, publisher_domains: tuple[str, ...], session=None) -> dict:
    """Follow public redirects, extract the full available body and retain retrieval failures."""
    try:
        import trafilatura
    except ImportError:
        return {"url": url, "article_text": "", "retrieval_state": "unavailable", "retrieval_reason": "trafilatura is not installed"}
    client = session or requests.Session()
    try:
        requested_url = url
        resolution_note = ""
        if _host(url) in {"news.google.com", "news.google.co.uk", "news.google.ca"}:
            requested_url, decoder_reason = resolve_google_news_url(url, publisher_domains)
            if not requested_url:
                return {"url": url, "article_text": "", "retrieval_state": "unresolved_google_news", "retrieval_reason": decoder_reason}
            resolution_note = decoder_reason
        response = client.get(requested_url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; IIF-ESG-Research/1.0)"})
        direct_url = response.url or url
        if not is_approved_url(direct_url, publisher_domains):
            return {"url": direct_url, "article_text": "", "retrieval_state": "rejected_redirect", "retrieval_reason": "Resolved URL is outside the approved publisher domains"}
        if response.status_code in (401, 403, 402):
            return {"url": direct_url, "article_text": "", "retrieval_state": "blocked", "retrieval_reason": f"Publisher returned HTTP {response.status_code}"}
        response.raise_for_status()
        html = response.text or ""
        if _html_has_explicit_paywall(html):
            return {"url": direct_url, "article_text": "", "retrieval_state": "paywalled", "retrieval_reason": "Article-scoped markup identifies locked or metered content"}
        body = trafilatura.extract(html, url=direct_url, include_comments=False, include_tables=False,
                                   favor_recall=True, deduplicate=True) or ""
        body = body.strip()
        if _PAYWALL.search(body[-3000:]):
            return {"url": direct_url, "article_text": body, "retrieval_state": "paywalled", "retrieval_reason": "Extracted body ends with a subscription or paywall prompt"}
        if len(body) < MIN_FULL_ARTICLE_CHARACTERS:
            return {"url": direct_url, "article_text": body, "retrieval_state": "too_short", "retrieval_reason": f"Extracted body has {len(body)} characters; at least {MIN_FULL_ARTICLE_CHARACTERS} are required"}
        if re.search(r"(?:\[\.\.\.]|…|\bcontinued\s+on\b|\bread\s+more)\s*$", body[-400:], re.I):
            return {"url": direct_url, "article_text": body, "retrieval_state": "truncated", "retrieval_reason": "Extracted body appears to end with a truncation marker"}
        return {"url": direct_url, "article_text": body, "retrieval_state": "retrieved",
                "retrieval_reason": (resolution_note + "; " if resolution_note else "") + "Full public article body extracted",
                "google_news_resolution": resolution_note}
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        return {"url": getattr(getattr(exc, "response", None), "url", url), "article_text": "", "retrieval_state": "blocked" if status in (401, 402, 403) else "failed", "retrieval_reason": f"HTTP retrieval failed: {status}"}
    except Exception as exc:
        return {"url": url, "article_text": "", "retrieval_state": "failed", "retrieval_reason": f"{type(exc).__name__}: {exc}"}


def _patterns(company: dict, ticker: str) -> list[tuple[str, re.Pattern]]:
    values = [company.get("company_name", ""), *company.get("aliases", [])]
    # Exchange suffixes are not company evidence; retain only a sufficiently
    # specific ticker symbol as an exact token candidate.
    bare_ticker = re.sub(r"\.(?:NS|BO)$", "", ticker, flags=re.I)
    if len(re.sub(r"\W", "", bare_ticker)) >= 3:
        values.append(bare_ticker)
    result = []
    for raw in values:
        value = str(raw or "").strip()
        if len(value) < 3:
            continue
        pattern = re.compile(r"(?<![\w])" + re.escape(value) + r"(?![\w])", re.I)
        if not any(existing.pattern == pattern.pattern for _, existing in result):
            result.append((value, pattern))
    return result


def relevance_and_pillar(article_text: str, company: dict, ticker: str, queried_pillar: str,
                         event_terms: dict[str, list[str]]) -> tuple[bool, str, str, str, str]:
    """Require exact target mention, direct adverse conduct and a body-supported pillar."""
    text = str(article_text or "")
    entity_matches = []
    for label, pattern in _patterns(company, ticker):
        entity_matches.extend((m.start(), m.end(), label) for m in pattern.finditer(text))
    if not entity_matches:
        return False, "rejected", "Target company name, alias or exact ticker token is absent from article body", "", ""
    conduct_matches = list(_ADVERSE_CONDUCT.finditer(text))
    if not conduct_matches:
        return False, "rejected", "Article body contains no material adverse ESG conduct", "", ""
    # Check sentence-sized context around each target and conduct mention. This
    # keeps unrelated actors' adverse events from qualifying on incidental name hits.
    spans = list(re.finditer(r"[^.!?\n]+(?:[.!?]|$)", text))
    linked = []
    for start, end, alias in entity_matches:
        sentence = next((m.group(0) for m in spans if m.start() <= start < m.end()), text[max(0, start - 160):min(len(text), end + 220)])
        sentence_start = next((m.start() for m in spans if m.start() <= start < m.end()), start)
        sentence_entity_start = start - sentence_start
        sentence_conducts = list(_ADVERSE_CONDUCT.finditer(sentence))
        alias_rx = re.escape(alias)
        linked_conduct = any(_attributed_conduct(sentence, sentence_entity_start, end - sentence_start, match) for match in sentence_conducts)
        unaffected_target = re.search(alias_rx + r".{0,60}\b(?:unaffected|not\s+(?:fined|involved|implicated)|cleared)\b", sentence, re.I)
        if linked_conduct and not unaffected_target:
            linked.append((start, end, sentence, alias))
    if not linked:
        if _MARKET_ONLY.search(text) and not any(_ADVERSE_CONDUCT.search(text[max(0, a - 350):b + 350]) for a, b, _ in entity_matches):
            reason = "Market or broker commentary mentions the target without linked ESG conduct"
        else:
            reason = "Adverse event is not linked to the target company in its article context"
        return False, "rejected", reason, "", ""

    supported = []
    material_patterns = {
        "E": r"\b(?:environment\w*|pollut\w*|spill|illegal discharge|contaminat\w*|emissions?|deforest\w*|biodiversity|toxic)\b",
        "S": r"\b(?:worker\w*|employee\w*|labou?r|customer\w*|consumer\w*|privacy|data breach|human rights|harass\w*|discriminat\w*|fatalit\w*|injur\w*|unsafe|wages?)\b",
        "G": r"\b(?:regulator\w*|RBI|SEBI|court|authority|brib\w*|corrupt\w*|fraud|accounting|fine|penalt\w*|sanction\w*|charges?|market manipulation|money laundering)\b",
    }
    for pillar, terms in event_terms.items():
        for start, end, context, _alias in linked:
            # Pillar support must occur in the same sentence that linked the
            # company to the adverse conduct, not merely in a nearby article.
            window = context
            taxonomy_match = any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", window, re.I) for term in terms)
            if _ADVERSE_CONDUCT.search(window) and (taxonomy_match or re.search(material_patterns[pillar], window, re.I)):
                supported.append(pillar)
                break
    if not supported:
        return False, "rejected", "Target-linked conduct is not supported by an ESG event category in the article text", "", ""
    # One underlying article receives one deterministic pillar for scoring.
    # The query pillar is only a tie-break hint after text validation.
    pillar = queried_pillar if queried_pillar in supported else next(p for p in ("E", "S", "G") if p in supported)
    evidence = max(linked, key=lambda item: (len(item[2]), -abs(item[0])))[2].strip()
    return True, "accepted", f"Target-linked conduct supported by article context: {evidence[:420]}", pillar, evidence[:600]


def _attributed_conduct(sentence: str, entity_start: int, entity_end: int, conduct: re.Match) -> bool:
    """Require a direct subject/object link with no intervening clause or actor."""
    if abs(conduct.start() - entity_start) > 220 and abs(conduct.end() - entity_start) > 220:
        return False
    if conduct.start() >= entity_end:
        bridge = sentence[entity_end:conduct.start()]
        if len(bridge) > 150 or re.search(r"[.;!?]|,\s*(?:which|while|whereas|but|and|as|after)\b|\b(?:while|whereas|but|although|despite|unaffected|not\s+involved)\b", bridge, re.I):
            return False
        return not re.search(r"\b(?:financed|funded|commented|reported|said|noted|cited)\b.{0,90}\b(?:which|that|who)\b", bridge, re.I)
    if conduct.end() <= entity_start:
        bridge = sentence[conduct.end():entity_start]
        if len(bridge) > 120 or re.search(r"[.;!?]|,\s*(?:while|whereas|but|as|after)\b|\b(?:while|whereas|but|unaffected|not\s+involved)\b", bridge, re.I):
            return False
        return bool(re.fullmatch(r"[\s,;:()'’\-]*(?:(?:to|against|on|at|of|for|with|by|the|a|an|was|were|is|are|has|had|been|imposed|issued|levied|fined|penalized|sanctioned|investigated|charged|accused|ordered)\s*)*", bridge, re.I))
    return False


def chunk_entire_article(text: str, tokenizer=None) -> tuple[list[str], int, list[int]]:
    if tokenizer is None:
        raise ValueError("Tokenizer is required for full-article scoring")
    if getattr(tokenizer, "_tokenizer", None) is not None:
        # Bypass the Transformers wrapper's length warning/truncation policy;
        # the backend returns the full untruncated token sequence directly.
        ids = tokenizer._tokenizer.encode(text, add_special_tokens=False).ids
    else:
        ids = tokenizer.encode(text, add_special_tokens=False)
    max_len = getattr(tokenizer, "model_max_length", FINBERT_MAX_TOKENS)
    if not isinstance(max_len, int) or max_len > 100000:
        max_len = FINBERT_MAX_TOKENS
    window = max(8, min(FINBERT_MAX_TOKENS - 2, max_len - 2))
    step = max(1, window - FINBERT_OVERLAP_TOKENS)
    chunks, counts = [], []
    start, covered_until = 0, 0
    while start < len(ids):
        end = min(len(ids), start + window)
        while end > start:
            chunk_ids = ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if getattr(tokenizer, "_tokenizer", None) is not None:
                encoded_length = len(tokenizer._tokenizer.encode(chunk_text, add_special_tokens=True).ids)
            else:
                try:
                    encoded_length = len(tokenizer.encode(chunk_text, add_special_tokens=True))
                except TypeError:
                    encoded_length = len(tokenizer.encode(chunk_text)) + 2
            if encoded_length <= FINBERT_MAX_TOKENS:
                break
            end -= 1
        if end <= start:
            raise ValueError("Tokenizer could not produce a safe window within the FinBERT limit")
        chunks.append(chunk_text)
        counts.append(end - covered_until)
        covered_until = end
        if end >= len(ids):
            break
        start = max(start + 1, end - FINBERT_OVERLAP_TOKENS)
    if sum(counts) != len(ids):
        raise ValueError("Chunk windows did not account for every article token exactly once")
    return chunks, len(ids), counts


def score_whole_article(text: str, classifier, progress_callback=None) -> dict:
    """Score every bounded tokenizer window; average by newly covered token count.

    The 32-token overlap preserves local context at boundaries, but overlap
    tokens are not counted twice in the deterministic document-level average.
    """
    tokenizer = getattr(classifier, "tokenizer", None)
    if tokenizer is None:
        return {"chunk_count": 0, "chunks_scored": 0, "token_count": 0, "full_text_scored": False,
                "finbert_label": "not scored", "finbert_confidence": float("nan"),
                "finbert_negative_probability": float("nan"),
                "scoring_error": "Classifier tokenizer is unavailable; full text cannot be safely windowed"}
    try:
        chunks, token_count, token_weights = chunk_entire_article(text, tokenizer)
    except Exception as exc:
        return {"chunk_count": 0, "chunks_scored": 0, "token_count": 0, "full_text_scored": False,
                "finbert_label": "not scored", "finbert_confidence": float("nan"),
                "finbert_negative_probability": float("nan"),
                "scoring_error": f"Tokenizer windowing failed: {type(exc).__name__}: {exc}"}
    result = {"chunk_count": len(chunks), "chunks_scored": 0, "token_count": token_count,
              "full_text_scored": False, "finbert_label": "not scored", "finbert_confidence": float("nan"),
              "finbert_negative_probability": float("nan"), "scoring_error": ""}
    if not chunks:
        result["scoring_error"] = "Tokenizer produced no article tokens"
        return result
    try:
        all_scores = []
        # Sequential calls bound memory on CPU and make failure atomic per article.
        for chunk_index, chunk in enumerate(chunks, start=1):
            raw = classifier([chunk], top_k=None, truncation=False, max_length=FINBERT_MAX_TOKENS, batch_size=1)
            prediction = raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
            if isinstance(prediction, dict):
                prediction = [prediction]
            if not prediction or not isinstance(prediction, list):
                raise ValueError("Classifier returned no probability labels for a chunk")
            scores = {str(item["label"]).lower(): float(item["score"]) for item in prediction}
            if set(scores) != {"positive", "neutral", "negative"}:
                raise ValueError("Classifier must return positive, neutral and negative probabilities for every chunk")
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in scores.values()):
                raise ValueError("Classifier returned a non-finite or out-of-range probability")
            if not math.isclose(sum(scores.values()), 1.0, rel_tol=0.0, abs_tol=0.02):
                raise ValueError("Classifier probabilities do not sum to one within 0.02")
            all_scores.append(scores)
            if progress_callback:
                try:
                    progress_callback({"chunk_index": chunk_index, "chunk_count": len(chunks),
                                       "chunks_scored": chunk_index, "token_count": token_count})
                except Exception:
                    pass
        labels = set().union(*(scores.keys() for scores in all_scores))
        total_weight = sum(token_weights)
        averaged = {label: sum(scores.get(label, 0.0) * weight for scores, weight in zip(all_scores, token_weights)) / total_weight for label in labels}
        best = max(averaged, key=averaged.get)
        result.update({"chunks_scored": len(chunks), "full_text_scored": True, "finbert_label": best,
                       "finbert_confidence": averaged[best], "finbert_negative_probability": averaged.get("negative", 0.0)})
    except Exception as exc:
        result["chunks_scored"] = len(all_scores)
        result["scoring_error"] = f"{type(exc).__name__}: {exc}"
    return result


def body_fingerprint(text: str) -> str:
    normalized = re.sub(r"\W+", " ", str(text).lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def bodies_near_duplicate(left: str, right: str) -> bool:
    a = re.sub(r"\W+", " ", str(left).lower()).strip()
    b = re.sub(r"\W+", " ", str(right).lower()).strip()
    if not a or not b:
        return False
    if body_fingerprint(a) == body_fingerprint(b):
        return True
    if min(len(a), len(b)) >= 800 and min(len(a), len(b)) / max(len(a), len(b)) >= 0.78:
        a_words, b_words = a.split(), b.split()
        width = 5
        a_grams = {" ".join(a_words[i:i + width]) for i in range(max(0, len(a_words) - width + 1))}
        b_grams = {" ".join(b_words[i:i + width]) for i in range(max(0, len(b_words) - width + 1))}
        union = a_grams | b_grams
        return bool(union) and len(a_grams & b_grams) / len(union) >= 0.90
    return False
