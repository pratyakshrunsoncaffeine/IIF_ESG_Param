# IIF ESG Parameterization Framework

A Streamlit app and Jupyter Notebook workflow for reviewing ESG evidence across several companies. Map one annual report to each company and optionally attach a standalone BRSR PDF to that same company. The app searches GDELT and Google News RSS for candidates from an approved Indian business-news publisher list and ranks only articles that pass full-text relevance and scoring checks.

## Run locally

1. Install Python 3.10 or newer.
2. Install Python packages with `pip install -r requirements.txt`.
3. Install Tesseract OCR only if you need OCR for scanned pages.
4. Start the app from this folder with `streamlit run app.py`.
5. Upload annual reports, map each to one exchange-qualified ticker, optionally map BRSR PDFs to the same tickers, and select **Analyse reports and rank companies**.
6. Review extracted report evidence and news audit rows. Full-text news coverage and eligible negative candidates affect provisional scores; analyst-confirmed incidents can add the separate severity penalty when they pass corroboration.

## Full-report ESG dataset

Upload complete searchable annual reports and, optionally, standalone BRSR PDFs. Each company needs one annual report; a matching BRSR adds evidence to that company's scorecard and does not create a second company or a second news search. If you enter a company name for a BRSR, it must match the annual-report company name or one of its aliases. The app validates ticker, provided name, year, filename, and duplicate-content consistency across the grouped documents.

The app reads embedded PDF text by default. OCR is off by default and can be enabled for pages with little embedded text; install Tesseract only when using that option. Numeric table extraction is also off by default and can be enabled when needed because it can use more memory on large reports. Table rows appear in the dataset only when table extraction is enabled.

The extractor matches the complete supplied dictionary of **25 Environmental, 17 Social and 40 Governance topics**. It exports a backend dataset with one row per matched statement and topic, including document type, fiscal year, source file and page, neighboring context, matched terms, claim type, quantities, units, reporting periods, targets, trends, optional table data, OCR status and an evidence confidence hint. Download the dataset as Excel, CSV or JSON. The Excel workbook includes pillar sheets, a topic summary and extraction errors.

Confidence is a rule-based evidence-quality hint, not a calibrated probability or ESG score. The dataset supports analyst review; it does not prove a company claim or validate a reported outcome. See [`ESG_DATASET_README.md`](ESG_DATASET_README.md) for the extractor field definitions and limitations.

The local notebook workflow is available as `ESG_company_screener.ipynb`. It uses the same grouping, extraction, scoring and export modules as the app. Add local PDF paths and mappings in its configuration cell; BRSR entries are optional. News search and FinBERT are opt-in in the notebook.

## Streamlit Community Cloud

Deploy `app.py` from the root of this repository and keep the repository and app private unless the IIF explicitly approves public access. Streamlit Cloud uses `requirements.txt` for Python libraries and `packages.txt` for Linux dependencies such as Tesseract. Private-repository deployment requires connecting Streamlit Community Cloud to the GitHub repository.

## Scoring and review

Each E, S and G pillar score runs from 0 to 7. The provisional score weights the FinBERT news signal at **70%** and annual-report/BRSR evidence at **30%**. News contributes its mean negative probability only when accepted, English full articles meet the target and every model chunk was scored successfully. A coverage or model shortfall receives a neutral news component. The per-company targets are **20 Environmental, 20 Social and 50 Governance articles**. Unavailable, rejected, duplicate, non-English, or partially scored candidates do not count toward coverage or scoring. The company-level negative-news deduction is 0.5 points per seven eligible negative candidates, rounded half-up to a whole point. The overall ranking is out of 21 before that deduction and is floored at zero. Analyst-confirmed incidents can impose a severity penalty on the relevant pillar, capped so a score is not reduced twice for the same signal. This is a provisional analyst-review ranking, not a verified ESG performance rating; FinBERT sentiment does not prove misconduct.

The approved publisher groups are The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI. Headlines are used only to discover articles; scoring requires a retrievable full body, target-company-linked adverse conduct, an ESG event category supported by the body, English language, and successful scoring of the complete article in model-sized chunks. There is no headline fallback. Syndicated or near-identical copies count once. A news-reported incident requires direct article URLs from five distinct approved publisher groups; each group counts once. The sole one-document exception is an HTTPS URL on a `.gov.in`, `.nic.in` or `rbi.org.in` domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not meet this exception.

## News coverage

GDELT and the India edition of Google News RSS are discovery sources, restricted to the approved publisher list above. Searches are issued in narrow event-term batches over a requested 365-day window. Actual historical coverage, item limits and outlet access vary by index. The app reports accepted full-text counts and shortfalls separately from rejected, unreadable and unscored audit rows. Verify source identity, direct article URL, date, company identity, allegation status, outcome, company response and whether coverage is syndicated before recording an incident. Publisher inclusion is a source-quality screen, not proof that an allegation is true.

## Evidence database

The app writes discovered articles, extracted report evidence, structured ESG dataset rows, analyst-confirmed incidents and pillar score history to a SQLite database and exposes the tables in the app. Each browser session has an isolated database, so one user's uploaded report evidence is not shown to another user. Download the SQLite file for backup or durable storage. Community Cloud does not guarantee that local files persist across restarts or session expiry, so connect a managed database before treating the hosted SQLite file as the permanent system of record. Database files are never committed to the repository.

## Limitations

Equal topic weights can mis-rank companies across sectors. Validate the complete dictionaries, extraction quality, and scoring thresholds against manually reviewed reports and news before using rankings for investment decisions. OCR and complex table extraction can be imperfect; check extracted values against the cited page. The notebook is a local front end to the shared pipeline; older saved notebooks may not reflect current behavior.

Uploaded PDFs are processed by the hosted app and should not be uploaded unless hosting and data handling have been approved by the IIF. The app does not commit uploaded PDFs or results to the repository. Extracted evidence and score data are stored in the SQLite database and remain available to download from the app while the file is retained by the host.
