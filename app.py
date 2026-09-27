from __future__ import annotations

import importlib
import tempfile
from datetime import date

import pandas as pd
import streamlit as st


def _load_esg_engine():
    """Retry once if Streamlit's source watcher evicts the module during import."""
    try:
        return importlib.import_module("esg_engine")
    except KeyError as exc:
        # Streamlit can clear local modules from sys.modules while syncing a GitHub
        # update. Python 3.14 may surface that race as KeyError instead of retrying.
        missing_module = str(exc.args[0]) if exc.args else ""
        if missing_module != "esg_engine" and not missing_module.startswith("esg."):
            raise
        importlib.invalidate_caches()
        return importlib.import_module("esg_engine")


_engine = _load_esg_engine()
NEGATIVE_REVIEW_THRESHOLD = _engine.NEGATIVE_REVIEW_THRESHOLD
TAXONOMY = _engine.TAXONOMY
ESG_DICTIONARY = _engine.ESG_DICTIONARY
apply_incidents_and_rank = _engine.apply_incidents_and_rank
calculate_document_scores = _engine.calculate_document_scores
classify_news = _engine.classify_news
scan_news_for_company = _engine.scan_news_for_company

from config import DEFAULTS
from esg.exporter import export_results
from esg.keyword_matcher import KeywordMatcher
from esg.pipeline import process_pdf_bytes
from esg.streamlit_adapter import SCORE_COLUMNS, records_to_score_evidence

st.set_page_config(page_title="IIF ESG Parameterization", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=-1)


@st.cache_resource(show_spinner="Preparing the full ESG keyword dictionary...")
def load_esg_matcher():
    return KeywordMatcher(ESG_DICTIONARY)


def dataset_downloads(records: list[dict], errors: list[dict]) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="iif_esg_exports_") as temp_dir:
        paths = export_results(records, errors, temp_dir)
        return {name: path.read_bytes() for name, path in paths.items()}


st.title("IIF ESG Parameterization Framework")
st.caption("Upload complete annual reports without cropping pages, review topic evidence and adverse-news candidates, then compare provisional company scores.")

with st.expander("How this prototype scores companies", expanded=False):
    st.write("Each E, S and G score runs from 0 to 7 and measures evidence found in the uploaded report. The overall ranking is out of 21. Negative news is surfaced for analyst review and does not reduce a score automatically. The news scan admits only The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI.")
    st.write("To confirm a news-reported incident, enter direct article URLs from five distinct approved publishers. Count each publisher once and exclude syndicated or repeated copies of the same report. The only one-document exception is an HTTPS link on a .gov.in, .nic.in or rbi.org.in domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not qualify as final findings.")
    st.warning(TAXONOMY.get("note", "Review the topic dictionaries before relying on comparisons."))

st.subheader("1. Upload reports")
with st.expander("Full-report extraction options", expanded=False):
    use_ocr = st.checkbox(
        "Use OCR for pages with little or no embedded text",
        value=True,
        help="Reads the whole uploaded report. OCR is applied only to pages with little embedded text and may take longer.",
    )
    st.caption("Each ESG dataset row retains its matched statement, nearby context, page number, detected metrics, claim type, and extraction confidence. Tables are extracted alongside page text.")
uploads = st.file_uploader(
    "Select one PDF per company. You can upload several reports at once.",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload one or more PDFs to enter ticker details and start a comparison.")
    st.stop()

names = [file.name for file in uploads]
if len(names) != len(set(names)):
    st.error("The uploaded files contain duplicate filenames. Rename them and upload again so each ticker can be mapped correctly.")
    st.stop()

mapping_defaults = pd.DataFrame({
    "PDF file": names,
    "Ticker": [""] * len(names),
    "Company name": [""] * len(names),
    "Aliases (separate with ;)": [""] * len(names),
})
st.write("Enter one ticker for each report. Company name and aliases improve news matching.")
mapping = st.data_editor(
    mapping_defaults,
    key="company_mapping",
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "PDF file": st.column_config.TextColumn(disabled=True),
        "Ticker": st.column_config.TextColumn(required=True, help="Use an exchange-qualified ticker, such as ABC.NS."),
        "Company name": st.column_config.TextColumn(),
        "Aliases (separate with ;)": st.column_config.TextColumn(),
    },
)

with st.expander("News scan options", expanded=False):
    run_news = st.checkbox("Search GDELT and Google News RSS, then classify English candidates with FinBERT", value=True)
    st.caption("News search is best-effort and currently covers a recent window of up to 90 days. It is not a complete 12-month archive. FinBERT model files download the first time the scan runs.")

run = st.button("Analyse reports and rank companies", type="primary", use_container_width=True)
if run:
    tickers = mapping["Ticker"].fillna("").astype(str).str.strip().str.upper()
    if (tickers == "").any():
        st.error("Enter a ticker for every PDF before starting.")
        st.stop()
    if tickers.duplicated().any():
        st.error("Each ticker must appear only once. Use one report per company in a comparison.")
        st.stop()

    companies = []
    upload_by_name = {file.name: file for file in uploads}
    for row in mapping.to_dict(orient="records"):
        ticker = str(row["Ticker"]).strip().upper()
        alias_col = next((key for key in row if key.startswith("Aliases")), "")
        aliases = [item.strip() for item in str(row.get(alias_col, "") or "").split(";") if item.strip()]
        companies.append({
            "ticker": ticker,
            "company_name": str(row.get("Company name", "") or "").strip(),
            "aliases": aliases,
            "report_name": row["PDF file"],
            "content": upload_by_name[row["PDF file"]].getvalue(),
        })

    evidence_frames, page_rows, error_rows, coverage_frames, news_frames = [], [], [], [], []
    dataset_records, dataset_errors = [], []
    esg_matcher = load_esg_matcher()
    status = st.status("Processing uploaded reports and news…", expanded=True)
    for company in companies:
        status.write(f"Extracting {company['report_name']} for {company['ticker']}")
        extractor_errors, extracted_pages = [], []
        try:
            display_company = company["company_name"] or company["ticker"]
            records = process_pdf_bytes(
                company["content"], company["report_name"], display_company,
                esg_matcher, DEFAULTS, ocr=use_ocr, errors=extractor_errors,
                page_status=extracted_pages,
            )
            for record in records:
                record["ticker"] = company["ticker"]
            dataset_records.extend(records)
            dataset_errors.extend({"ticker": company["ticker"], **row} for row in extractor_errors)
            evidence_rows = records_to_score_evidence(
                records, company["ticker"], display_company, company["report_name"]
            )
            evidence_frames.append(pd.DataFrame(evidence_rows, columns=SCORE_COLUMNS))
            for page in extracted_pages:
                page_rows.append({"ticker": company["ticker"], "report_name": company["report_name"], **page})
            if not extracted_pages:
                error_rows.extend({
                    "ticker": company["ticker"],
                    "stage": row.get("error_type", "PDF extraction"),
                    "error": row.get("message", "PDF could not be read"),
                } for row in extractor_errors)
        except Exception as exc:
            error_rows.append({"ticker": company["ticker"], "stage": "PDF extraction", "error": f"{type(exc).__name__}: {exc}"})
            evidence_frames.append(pd.DataFrame())
            continue
        if run_news:
            status.write(f"Searching GDELT and Google News RSS for {company['ticker']}")
            try:
                news, errors, coverage = scan_news_for_company(company, date.today())
                if not news.empty:
                    news_frames.append(news)
                if not errors.empty:
                    error_rows.extend(errors.to_dict(orient="records"))
                coverage_frames.append(coverage)
            except Exception as exc:
                error_rows.append({"ticker": company["ticker"], "stage": "News search", "error": f"{type(exc).__name__}: {exc}"})

    successful_tickers = {row["ticker"] for row in page_rows}
    successful_companies = [company for company in companies if company["ticker"] in successful_tickers]
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
        try:
            classifier = load_finbert()
            news = classify_news(news, classifier)
        except Exception as exc:
            error_rows.append({"ticker": "ALL", "stage": "FinBERT", "error": f"{type(exc).__name__}: {exc}"})
            news["finbert_label"] = "not scored: FinBERT unavailable"
            news["finbert_negative_probability"] = float("nan")
            news["needs_analyst_review"] = True
    document_scores = calculate_document_scores(evidence, successful_companies)
    st.session_state["iif_esg_analysis"] = {
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
    }
    status.update(label="Analysis complete", state="complete", expanded=False)

analysis = st.session_state.get("iif_esg_analysis")
if analysis:
    st.subheader("2. Review coverage and news")
    if not analysis["coverage"].empty:
        st.markdown("**News source coverage**")
        st.dataframe(analysis["coverage"], hide_index=True, use_container_width=True)
    if not analysis["news"].empty:
        display_columns = [col for col in ["ticker", "pillar", "title", "publisher", "domain", "published_at", "discovery_source", "detected_language", "finbert_label", "finbert_negative_probability", "needs_analyst_review", "url"] if col in analysis["news"].columns]
        st.dataframe(analysis["news"][display_columns].sort_values(["ticker", "finbert_negative_probability"], ascending=[True, False], na_position="last"), hide_index=True, use_container_width=True)
    elif run_news:
        st.info("No news candidates returned. Review the coverage and query errors before interpreting this as no adverse news.")
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
        st.subheader("4. Company ranking")
        st.dataframe(rankings, hide_index=True, use_container_width=True)
        st.bar_chart(rankings.set_index("ticker")["overall_score_0_21"], y_label="Provisional score out of 21")
        st.markdown("**Pillar score detail**")
        st.dataframe(pillar_detail, hide_index=True, use_container_width=True)
        st.caption("Higher scores reflect more report evidence under this prototype. They do not establish better ESG outcomes.")
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
        st.download_button("Download ESG dataset Excel workbook", downloads["xlsx"], "iif_esg_backend_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download ESG dataset CSV", downloads["csv"], "iif_esg_backend_dataset.csv", "text/csv")
        st.download_button("Download ESG dataset JSON", downloads["json"], "iif_esg_backend_dataset.json", "application/json")
    if not analysis["news"].empty:
        st.download_button("Download news candidates CSV", analysis["news"].to_csv(index=False).encode("utf-8-sig"), "iif_esg_news_candidates.csv", "text/csv")

st.divider()
st.caption("Prototype for analyst review. The report extractor uses 25 Environmental, 17 Social, and 40 Governance topics. Scores remain provisional evidence-coverage measures, not verified ESG performance. Uploaded reports are processed by this app; avoid uploading confidential documents to any hosted service unless your organisation has approved that use.")
