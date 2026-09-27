from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from esg_engine import (
    NEGATIVE_REVIEW_THRESHOLD,
    TAXONOMY,
    apply_incidents_and_rank,
    calculate_document_scores,
    classify_news,
    collect_report_evidence,
    extract_pdf_pages,
    scan_news_for_company,
)

st.set_page_config(page_title="IIF ESG Parameterization", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=-1)
st.title("IIF ESG Parameterization Framework")
st.caption("Upload reports, review topic evidence and adverse-news candidates, then compare provisional company scores.")

with st.expander("How this prototype scores companies", expanded=False):
    st.write("Each E, S and G score runs from 0 to 7 and measures evidence found in the uploaded report. The overall ranking is out of 21. Negative news is surfaced for analyst review and does not reduce a score automatically. The news scan admits only The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI.")
    st.write("To confirm a news-reported incident, enter direct article URLs from five distinct approved publishers. Count each publisher once and exclude syndicated or repeated copies of the same report. The only one-document exception is an HTTPS link on a .gov.in, .nic.in or rbi.org.in domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not qualify as final findings.")
    st.warning(TAXONOMY.get("note", "Review the topic dictionaries before relying on comparisons."))

st.subheader("1. Upload reports")
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
    status = st.status("Processing uploaded reports and news…", expanded=True)
    for company in companies:
        status.write(f"Extracting {company['report_name']} for {company['ticker']}")
        try:
            pages = extract_pdf_pages(company["content"])
            evidence_frames.append(collect_report_evidence(pages, company["ticker"], company["company_name"], company["report_name"]))
            for page in pages:
                page_rows.append({"ticker": company["ticker"], "report_name": company["report_name"], **{key: value for key, value in page.items() if key != "text"}})
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
    st.caption("Enter one row per underlying event, not per article. Paste one direct article URL per line in Source URLs. The app counts distinct approved publisher groups and applies one rule: five of six listed publishers, or an HTTPS URL on a .gov.in, .nic.in or rbi.org.in domain to a final regulator order or sanction, final court judgment, or final statutory authority decision.")
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
            "source_urls": st.column_config.TextColumn(help="One direct article URL per line. Only the approved six publisher groups count, and each group counts once."),
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
    if not analysis["news"].empty:
        st.download_button("Download news candidates CSV", analysis["news"].to_csv(index=False).encode("utf-8-sig"), "iif_esg_news_candidates.csv", "text/csv")

st.divider()
st.caption("Prototype for analyst review. Complete the truncated Environment and Governance dictionaries and validate sector weights before investment use. Uploaded reports are processed by this app; avoid uploading confidential documents to any hosted service unless your organisation has approved that use.")
