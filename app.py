from __future__ import annotations

import importlib
import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _load_esg_engine():
    """Retry transient module-cache races while Community Cloud refreshes code."""
    for attempt in range(6):
        try:
            return importlib.import_module("esg_engine")
        except KeyError as exc:
            # Python 3.14 may expose Streamlit's concurrent sys.modules cleanup
            # as KeyError when a module is being imported during a code refresh.
            missing_module = str(exc.args[0]) if exc.args else ""
            if missing_module != "esg_engine" and not missing_module.startswith("esg."):
                raise
            if attempt == 5:
                raise
            importlib.invalidate_caches()
            time.sleep(0.25 * (attempt + 1))


_engine = _load_esg_engine()
NEGATIVE_REVIEW_THRESHOLD = _engine.NEGATIVE_REVIEW_THRESHOLD
TAXONOMY = _engine.TAXONOMY
ESG_DICTIONARY = _engine.ESG_DICTIONARY
apply_incidents_and_rank = _engine.apply_incidents_and_rank
calculate_document_scores = _engine.calculate_document_scores
classify_news = _engine.classify_news
scan_news_for_company = _engine.scan_news_for_company
NEWS_PILLAR_TARGETS = _engine.NEWS_PILLAR_TARGETS
NEWS_SCORE_WEIGHT = _engine.NEWS_SCORE_WEIGHT
REPORT_SCORE_WEIGHT = _engine.REPORT_SCORE_WEIGHT

from config import DEFAULTS
from esg.database import database_bytes, database_summary, read_table, save_evidence_batch, save_score_batch
from esg.exporter import export_results
from esg.keyword_matcher import KeywordMatcher
from esg.pipeline import process_pdf_bytes
from esg.streamlit_adapter import SCORE_COLUMNS, records_to_score_evidence
from esg.documents import DocumentMappingError, group_documents

st.set_page_config(page_title="IIF ESG Parameterization", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_finbert():
    import torch
    from transformers import pipeline
    torch.set_num_threads(min(2, torch.get_num_threads()))
    return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=-1)


@st.cache_resource(show_spinner="Preparing the full ESG keyword dictionary...")
def load_esg_matcher():
    return KeywordMatcher(ESG_DICTIONARY)


def dataset_downloads(records: list[dict], errors: list[dict]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="iif_esg_exports_") as temp_dir:
        paths = export_results(records, errors, temp_dir)
        return {name: path.read_bytes() for name, path in paths.items()}


def session_database_path():
    """Keep each browser session's evidence store isolated from other users."""
    key = "iif_esg_database_path"
    if key not in st.session_state:
        session_folder = tempfile.mkdtemp(prefix="iif_esg_session_")
        st.session_state[key] = str(Path(session_folder) / "iif_esg_database.sqlite3")
    return st.session_state[key]


DB_PATH = session_database_path()


st.title("IIF ESG Parameterization Framework")
st.caption("Upload complete annual reports without cropping pages, review topic evidence and adverse-news candidates, then compare provisional company scores.")

with st.expander("How this prototype scores companies", expanded=False):
    st.write("Each E, S and G score runs from 0 to 7. The provisional pillar score weights news at 70% and annual-report/BRSR evidence at 30%. The news targets are 20 Environmental, 20 Social and 50 Governance full-text-scored English articles per company. FinBERT negative probabilities affect the provisional news component when coverage targets are met; a shortfall receives a neutral component, not a clean-news score. A separate company-level negative-news penalty deducts 0.5 points per seven negative candidates, rounded half-up to a whole point. Sentiment is a review signal, not proof. Analyst-confirmed incidents can impose a severity penalty. The overall ranking is out of 21. The scan admits only The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI.")
    st.write("To confirm a news-reported incident, enter direct article URLs from five distinct approved publishers. Count each publisher once and exclude syndicated or repeated copies of the same report. The only one-document exception is an HTTPS link on a .gov.in, .nic.in or rbi.org.in domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not qualify as final findings.")
    st.warning(TAXONOMY.get("note", "Review the topic dictionaries before relying on comparisons."))

st.subheader("1. Upload reports")
with st.expander("Full-report extraction options", expanded=False):
    use_ocr = st.checkbox(
        "Use OCR for pages with little or no embedded text",
        value=False,
        key="use_ocr",
        help="Reads the whole uploaded report. OCR is applied only to pages with little embedded text and may take longer.",
    )
    use_tables = st.checkbox(
        "Extract numeric data from report tables",
        value=False,
        key="use_tables",
        help="Table extraction can be resource intensive on large PDFs. Enable only when needed.",
    )
    st.caption("Each ESG dataset row retains its matched statement, nearby context, page number, detected metrics, claim type, and extraction confidence. Tables are extracted alongside page text.")
annual_uploads = st.file_uploader(
    "Annual reports: upload one PDF per company. You can upload several reports at once.",
    type=["pdf"],
    accept_multiple_files=True,
    key="annual_report_uploads",
)
brsr_uploads = st.file_uploader(
    "Optional standalone SEBI BRSR PDFs (attach each to a company below)",
    type=["pdf"], accept_multiple_files=True, key="optional_brsr_uploads",
)

if not annual_uploads:
    st.info("Upload one or more annual reports to enter ticker details and start a comparison. BRSR uploads are optional.")
    st.stop()

annual_names = [file.name for file in annual_uploads]
brsr_names = [file.name for file in brsr_uploads]
annual_mapping_key = "annual_company_mapping_" + hashlib.sha256(json.dumps(annual_names).encode()).hexdigest()[:12]
brsr_mapping_key = "brsr_company_mapping_" + hashlib.sha256(json.dumps([annual_names, brsr_names]).encode()).hexdigest()[:12]
mapping_defaults = pd.DataFrame({
    "PDF file": annual_names,
    "Ticker": [""] * len(annual_names),
    "Company name": [""] * len(annual_names),
    "Aliases (separate with ;)": [""] * len(annual_names),
    "Fiscal year (optional)": [""] * len(annual_names),
})
st.write("Enter one ticker per annual report. Use the same optional fiscal year for all documents attached to that ticker. Company name and aliases improve news matching.")
annual_mapping_defaults = mapping_defaults.rename(columns={"Fiscal year (optional)": "Fiscal year"})
annual_mapping = st.data_editor(
    annual_mapping_defaults,
    key=annual_mapping_key,
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "PDF file": st.column_config.TextColumn(disabled=True),
        "Ticker": st.column_config.TextColumn(required=True, help="Use an exchange-qualified ticker, such as ABC.NS."),
        "Company name": st.column_config.TextColumn(),
        "Aliases (separate with ;)": st.column_config.TextColumn(),
        "Fiscal year": st.column_config.TextColumn(help="Optional, for example FY2025."),
    },
)
brsr_mapping = pd.DataFrame(columns=["PDF file", "Ticker", "Company name", "Fiscal year"])
if brsr_uploads:
    st.caption("Assign each BRSR to a ticker from the annual-report table. If you enter a company name, it must match that company's annual-report name or alias. A blank fiscal year inherits the annual-report year. The app uses your mapping and does not independently verify the issuer inside the PDF.")
    known_tickers = sorted({str(value).strip().upper() for value in annual_mapping["Ticker"].fillna("") if str(value).strip()})
    brsr_mapping = st.data_editor(
        pd.DataFrame({"PDF file": brsr_names, "Ticker": [""] * len(brsr_names),
                      "Company name": [""] * len(brsr_names), "Fiscal year": [""] * len(brsr_names)}),
        key=brsr_mapping_key, hide_index=True, num_rows="fixed", use_container_width=True,
        column_config={"PDF file": st.column_config.TextColumn(disabled=True),
                       "Ticker": st.column_config.SelectboxColumn(options=known_tickers, required=True),
                       "Company name": st.column_config.TextColumn(),
                       "Fiscal year": st.column_config.TextColumn()},
    )

with st.expander("News scan options", expanded=False):
    run_news = st.checkbox("Search GDELT and Google News RSS, then classify English candidates with FinBERT", value=True, key="run_news")
    st.caption("Per-company targets: 20 Environmental, 20 Social and 50 Governance accepted full-text-scored English articles. Search indexes and publisher access can return fewer; the results show counts and shortfalls. Queries request up to 365 days, but historical coverage is not guaranteed. FinBERT model files download the first time the scan runs.")
run = st.button("Analyse reports and rank companies", type="primary", use_container_width=True)
annual_by_name = {file.name: file for file in annual_uploads}
brsr_by_name = {file.name: file for file in brsr_uploads}
analysis_signature = hashlib.sha256(json.dumps({
    "annual": [(f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in annual_uploads],
    "brsr": [(f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in brsr_uploads],
    "annual_mapping": annual_mapping.to_dict(orient="records"),
    "brsr_mapping": brsr_mapping.to_dict(orient="records"),
    "news": bool(st.session_state.get("run_news", True)), "ocr": bool(st.session_state.get("use_ocr", False)),
    "tables": bool(st.session_state.get("use_tables", False)),
}, sort_keys=True, default=str).encode()).hexdigest()
if run:
    annual_rows = [{"file_name": row["PDF file"], "ticker": row.get("Ticker"),
                    "company_name": row.get("Company name"), "aliases": row.get("Aliases (separate with ;)"),
                    "fiscal_year": row.get("Fiscal year"), "content": annual_by_name[row["PDF file"]].getvalue()}
                   for row in annual_mapping.to_dict(orient="records")]
    brsr_rows = [{"file_name": row["PDF file"], "ticker": row.get("Ticker"),
                  "company_name": row.get("Company name"), "fiscal_year": row.get("Fiscal year"),
                  "content": brsr_by_name[row["PDF file"]].getvalue()}
                 for row in brsr_mapping.to_dict(orient="records")]
    try:
        companies = group_documents(annual_rows, brsr_rows)
    except DocumentMappingError as exc:
        st.error(str(exc))
        st.stop()

    evidence_frames, page_rows, error_rows, coverage_frames, news_frames = [], [], [], [], []
    dataset_records, dataset_errors = [], []
    run_id = uuid.uuid4().hex
    esg_matcher = load_esg_matcher()
    status = st.status("Processing uploaded reports and news…", expanded=True)
    for company in companies:
        display_company = company["company_name"] or company["ticker"]
        annual_readable = False
        for document in company["documents"]:
            status.write(f"Extracting {document['source_file']} ({document['document_type']}) for {company['ticker']}")
            extractor_errors, extracted_pages = [], []
            try:
                records = process_pdf_bytes(document["content"], document["source_file"], display_company,
                    esg_matcher, DEFAULTS, ocr=use_ocr, extract_tables=use_tables,
                    errors=extractor_errors, page_status=extracted_pages)
                for record in records:
                    record.update(ticker=company["ticker"], document_type=document["document_type"],
                                  fiscal_year=document["fiscal_year"])
                dataset_records.extend(records)
                dataset_errors.extend({"ticker": company["ticker"], "document_type": document["document_type"], **row} for row in extractor_errors)
                evidence_rows = records_to_score_evidence(records, company["ticker"], display_company,
                    document["source_file"], document["document_type"], document["fiscal_year"])
                evidence_frames.append(pd.DataFrame(evidence_rows, columns=SCORE_COLUMNS))
                for page in extracted_pages:
                    page_rows.append({"ticker": company["ticker"], "document_type": document["document_type"],
                                      "fiscal_year": document["fiscal_year"], "report_name": document["source_file"], **page})
                status.write(f"Finished {document['source_file']}: {len(extracted_pages)} pages checked, {len(records)} evidence rows, {len(extractor_errors)} extraction issues")
                if document["document_type"] == "annual_report" and any(int(page.get("characters", 0) or 0) > 0 for page in extracted_pages):
                    annual_readable = True
                if not extracted_pages:
                    error_rows.extend({"ticker": company["ticker"], "document_type": document["document_type"],
                        "stage": row.get("error_type", "PDF extraction"),
                        "error": row.get("message", "PDF could not be read")} for row in extractor_errors)
            except Exception as exc:
                error_rows.append({"ticker": company["ticker"], "document_type": document["document_type"],
                                   "stage": "PDF extraction", "error": f"{type(exc).__name__}: {exc}"})
        company["_annual_readable"] = annual_readable
        if run_news:
            status.write(f"Searching GDELT and Google News RSS for {company['ticker']}")
            try:
                news, errors, coverage = scan_news_for_company(company, date.today())
                if not news.empty:
                    # Carry the selected identity terms through retrieval so full-text
                    # relevance screening can bind adverse conduct to this issuer.
                    news["aliases"] = [company.get("aliases", []) for _ in range(len(news))]
                    news_frames.append(news)
                if not errors.empty:
                    error_rows.extend(errors.to_dict(orient="records"))
                coverage_frames.append(coverage)
            except Exception as exc:
                error_rows.append({"ticker": company["ticker"], "stage": "News search", "error": f"{type(exc).__name__}: {exc}"})

    successful_tickers = {company["ticker"] for company in companies if company.get("_annual_readable")}
    successful_companies = [company for company in companies if company["ticker"] in successful_tickers]
    for company in successful_companies:
        company.pop("_annual_readable", None)
    if not successful_companies:
        status.update(label="No reports could be processed", state="error", expanded=True)
        st.error("No PDF text was extracted. Check the files and install Tesseract OCR for scanned reports.")
        if error_rows:
            st.dataframe(pd.DataFrame(error_rows), hide_index=True, use_container_width=True)
        st.stop()
    evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
    news = pd.concat(news_frames, ignore_index=True) if news_frames else pd.DataFrame()
    if run_news and not news.empty:
        status.write("Classifying candidate news with FinBERT")
        classifier = None
        try:
            classifier = load_finbert()
        except Exception as exc:
            error_rows.append({"ticker": "ALL", "stage": "FinBERT", "error": f"{type(exc).__name__}: {exc}"})
        news_progress_bar = status.progress(0, text="Retrieving and screening full articles")
        def news_progress(event):
            step = event.get("event", "news")
            done, total = event.get("processed", 0), event.get("total", 0)
            title = str(event.get("title", ""))[:100]
            chunks = event.get("chunk_count")
            chunk_note = f"; {event.get('chunks_scored', 0)}/{chunks} chunks scored" if chunks is not None else ""
            progress = min(1.0, done / total) if total else 0.0
            news_progress_bar.progress(progress, text=f"News {step.replace('_', ' ')} {done}/{total}{chunk_note}: {title}")
        try:
            # Even without model weights, fetch and screen every candidate so
            # rejected, unreadable and unscored articles remain in the audit.
            news = classify_news(news, classifier, progress_callback=news_progress)
        except Exception as exc:
            error_rows.append({"ticker": "ALL", "stage": "News article screening", "error": f"{type(exc).__name__}: {exc}"})
            news["finbert_label"] = "not scored: FinBERT unavailable"
            news["finbert_negative_probability"] = float("nan")
            news["scoring_eligible"] = False
            news["needs_analyst_review"] = True
    document_scores = calculate_document_scores(evidence, successful_companies)
    save_evidence_batch(run_id, news, evidence, dataset_records, DB_PATH)
    st.session_state["iif_esg_analysis"] = {
        "run_id": run_id,
        "companies": successful_companies,
        "evidence": evidence,
        "document_scores": document_scores,
        "news": news,
        "errors": pd.DataFrame(error_rows),
        "dataset_records": dataset_records,
        "dataset": pd.DataFrame(dataset_records),
        "dataset_errors": dataset_errors,
        "dataset_files": dataset_downloads(dataset_records, dataset_errors) if (dataset_records or dataset_errors) else {},
        "coverage": pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame(),
        "pages": pd.DataFrame(page_rows),
        "input_signature": analysis_signature,
    }
    status.update(label="Analysis complete", state="complete", expanded=False)

analysis = st.session_state.get("iif_esg_analysis")
if analysis and analysis.get("input_signature") != analysis_signature:
    analysis = None
if analysis:
    st.subheader("2. Review coverage and news")
    if not analysis["coverage"].empty:
        st.markdown("**News source coverage**")
        st.dataframe(analysis["coverage"], hide_index=True, use_container_width=True)
    if run_news:
        news_data = analysis["news"]
        st.markdown("**Accepted news coverage (full text scored, English, and eligible)**")
        accepted = news_data[news_data.get("scoring_eligible", pd.Series(False, index=news_data.index)).fillna(False).astype(bool)] if not news_data.empty else news_data
        discovered = news_data.groupby(["ticker", "pillar"]).size().to_dict() if not news_data.empty else {}
        accepted_counts = accepted.groupby(["ticker", "pillar"]).size().to_dict() if not accepted.empty else {}
        eligible_rows = []
        for company in analysis["companies"]:
            for pillar in ("E", "S", "G"):
                count = int(accepted_counts.get((company["ticker"], pillar), 0))
                target = int(NEWS_PILLAR_TARGETS[pillar])
                eligible_rows.append({"ticker": company["ticker"], "pillar": pillar,
                                      "discovered_candidates": int(discovered.get((company["ticker"], pillar), 0)),
                                      "accepted_full_text_scored": count, "target": target,
                                      "shortfall": max(0, target - count)})
        st.dataframe(pd.DataFrame(eligible_rows), hide_index=True, use_container_width=True)
        display_columns = [col for col in ["ticker", "pillar", "title", "publisher", "domain", "published_at", "discovery_source", "detected_language", "retrieval_state", "retrieval_reason", "full_article_available", "relevance_decision", "relevance_reason", "attribution_evidence", "scoring_eligible", "duplicate_of", "scoring_error", "full_text_scored", "chunk_count", "chunks_scored", "fetch_status", "model_input_source", "finbert_label", "finbert_negative_probability", "needs_analyst_review", "url"] if col in news_data.columns]
        if news_data.empty:
            st.info("No news candidates returned. Review the coverage and query errors before interpreting this as no adverse news.")
        elif "scoring_eligible" in news_data:
            rejected = news_data[~news_data["scoring_eligible"].fillna(False).astype(bool)]
            if not accepted.empty:
                st.dataframe(accepted[display_columns].sort_values(["ticker", "pillar", "finbert_negative_probability"], ascending=[True, True, False], na_position="last"), hide_index=True, use_container_width=True)
            st.markdown(f"**Rejected, unreadable, or unscored news audit ({len(rejected)})**")
            if not rejected.empty:
                st.dataframe(rejected[display_columns], hide_index=True, use_container_width=True)
    if not analysis["errors"].empty:
        with st.expander(f"Processing and search issues ({len(analysis['errors'])})"):
            st.dataframe(analysis["errors"], hide_index=True, use_container_width=True)

    st.subheader("3. Record analyst-confirmed incidents")
    st.caption("Enter one row per underlying event, not per article. Separate direct article URLs with semicolons in Source URLs. The app counts distinct approved publisher groups and applies one rule: five of six listed publishers, or an HTTPS URL on a .gov.in, .nic.in or rbi.org.in domain to a final regulator order or sanction, final court judgment, or final statutory authority decision.")
    incident_columns = ["ticker", "pillar", "severity_points", "status", "source_urls", "official_record_url", "summary", "already_in_metric", "analyst_confirmed"]
    incident_defaults = pd.DataFrame(columns=incident_columns)
    if "iif_esg_incidents" not in st.session_state:
        st.session_state["iif_esg_incidents"] = incident_defaults
    incidents = st.data_editor(
        st.session_state["iif_esg_incidents"],
        key="iif_esg_incident_editor",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "ticker": st.column_config.SelectboxColumn(options=[company["ticker"] for company in analysis["companies"]], required=True),
            "pillar": st.column_config.SelectboxColumn(options=["E", "S", "G"], required=True),
            "severity_points": st.column_config.SelectboxColumn(options=[0.5, 1.0, 2.0]),
            "status": st.column_config.SelectboxColumn(options=["news reports", "final regulator order or sanction", "final court judgment", "final statutory authority decision"], required=True),
            "source_urls": st.column_config.TextColumn(help="Separate direct article URLs with semicolons. Only the approved six publisher groups count, and each group counts once."),
            "official_record_url": st.column_config.TextColumn(help="Link directly to the final official decision document. Required for the one-document exception."),
            "summary": st.column_config.TextColumn(),
            "already_in_metric": st.column_config.CheckboxColumn(),
            "analyst_confirmed": st.column_config.CheckboxColumn(help="Only confirmed rows can change scores."),
        },
    )
    st.session_state["iif_esg_incidents"] = incidents
    # Ignore unused blank editor rows, but enforce validation for populated entries.
    populated = incidents[incidents["summary"].fillna("").astype(str).str.strip().ne("")].copy() if not incidents.empty else incidents
    if not populated.empty:
        populated = populated[populated["analyst_confirmed"].fillna(False).astype(bool)]
    try:
        rankings, pillar_detail = apply_incidents_and_rank(analysis["document_scores"], populated, analysis["news"])
        save_score_batch(analysis["run_id"], rankings, pillar_detail, incidents=populated, path=DB_PATH)
        st.subheader("4. Company ranking")
        st.dataframe(rankings, hide_index=True, use_container_width=True)
        st.bar_chart(rankings.set_index("ticker")["overall_score_0_21"], y_label="Provisional score out of 21")
        st.markdown("**Pillar score detail**")
        st.dataframe(pillar_detail, hide_index=True, use_container_width=True)
        st.caption("Score formula: 70% FinBERT news signal and 30% annual-report/BRSR evidence. The separate negative-news penalty deducts 0.5 points per seven negative candidates, rounded half-up to a whole point; the final score cannot fall below zero. A pillar below its full-text-scored English article target receives a neutral news component and is flagged; it is not treated as having no controversy. FinBERT sentiment is provisional and never proves an incident. Analyst-confirmed incidents that satisfy the source corroboration rule can impose a severity penalty without double-counting the same sentiment.")
        st.download_button("Download company ranking CSV", rankings.to_csv(index=False).encode("utf-8-sig"), "iif_esg_company_ranking.csv", "text/csv")
        st.download_button("Download pillar score detail CSV", pillar_detail.to_csv(index=False).encode("utf-8-sig"), "iif_esg_pillar_scores.csv", "text/csv")
    except Exception as exc:
        st.error(f"Incident table needs attention: {exc}")

    with st.expander("Report evidence"):
        if analysis["evidence"].empty:
            st.info("No dictionary matches were found in the extracted report text.")
        else:
            st.dataframe(analysis["evidence"], hide_index=True, use_container_width=True)
        if not analysis["pages"].empty:
            st.markdown("**Page extraction status**")
            st.dataframe(analysis["pages"], hide_index=True, use_container_width=True)
        st.download_button("Download report evidence CSV", analysis["evidence"].to_csv(index=False).encode("utf-8-sig"), "iif_esg_report_evidence.csv", "text/csv")
    st.subheader("5. Backend ESG dataset")
    st.caption("One row per matched statement and ESG topic, with page reference, neighboring context, claim type, extracted values, table trace, OCR status, and confidence. Check the source page before treating an extracted statement or value as verified.")
    if analysis["dataset"].empty:
        st.info("No ESG dictionary matches were extracted. The uploaded pages were still checked; inspect the page status and extraction issues above.")
    else:
        visible_dataset_columns = [column for column in [
            "ticker", "company", "source_file", "page_number", "pillar", "topic",
            "document_type", "fiscal_year",
            "keyword", "claim_type", "matched_sentence", "value", "unit",
            "reporting_period", "target_value", "target_year", "baseline_year",
            "direction", "confidence", "ocr_used", "table_index",
        ] if column in analysis["dataset"].columns]
        st.dataframe(analysis["dataset"][visible_dataset_columns], hide_index=True, use_container_width=True)
    if analysis["dataset_errors"]:
        with st.expander(f"Dataset extraction issues ({len(analysis['dataset_errors'])})"):
            st.dataframe(pd.DataFrame(analysis["dataset_errors"]), hide_index=True, use_container_width=True)
    if analysis["dataset_files"]:
        downloads = analysis["dataset_files"]
        if "xlsx" in downloads:
            st.download_button("Download ESG dataset Excel workbook", downloads["xlsx"], "iif_esg_backend_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Excel export is temporarily unavailable while the workbook dependency loads. CSV and JSON downloads are ready.")
        st.download_button("Download ESG dataset CSV", downloads["csv"], "iif_esg_backend_dataset.csv", "text/csv")
        st.download_button("Download ESG dataset JSON", downloads["json"], "iif_esg_backend_dataset.json", "application/json")
    if not analysis["news"].empty:
        st.download_button("Download news candidates CSV", analysis["news"].to_csv(index=False).encode("utf-8-sig"), "iif_esg_news_candidates.csv", "text/csv")

    st.subheader("6. ESG evidence database")
    db_summary = database_summary(DB_PATH)
    count_cols = st.columns(5)
    count_labels = [("news_articles", "News articles"), ("report_evidence", "Report evidence rows"), ("esg_dataset", "ESG dataset rows"), ("score_history", "Stored pillar scores"), ("analyst_incidents", "Confirmed incidents")]
    for col, (key, label) in zip(count_cols, count_labels):
        col.metric(label, db_summary["counts"][key])
    st.caption("This browser session has its own isolated SQLite database for deduplicated news, report evidence, ESG dataset rows, analyst-confirmed incidents, and score history. Other users cannot see this session's database. Download it for durable storage or backup; session files can be removed when Streamlit Cloud restarts or the session expires.")
    database_view = st.selectbox("Database table", ["News articles", "Report evidence", "ESG dataset", "Score history", "Analyst-confirmed incidents"], key="database_table_view")
    table_names = {"News articles": "news_articles", "Report evidence": "report_evidence", "ESG dataset": "esg_dataset", "Score history": "score_history", "Analyst-confirmed incidents": "analyst_incidents"}
    stored_rows = read_table(table_names[database_view], DB_PATH)
    if not stored_rows.empty:
        st.dataframe(stored_rows, hide_index=True, use_container_width=True)
    else:
        st.info("This database table is empty until a report analysis is saved.")
    st.download_button("Download SQLite ESG database", database_bytes(DB_PATH), "iif_esg_database.sqlite3", "application/vnd.sqlite3")

st.divider()
st.caption("Prototype for analyst review. The report extractor uses 25 Environmental, 17 Social, and 40 Governance topics. Scores remain provisional evidence-coverage measures, not verified ESG performance. Uploaded reports are processed by this app; avoid uploading confidential documents to any hosted service unless your organisation has approved that use.")
