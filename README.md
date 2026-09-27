# IIF ESG Parameterization Framework

A Streamlit app and Jupyter Notebook prototype for reviewing ESG evidence across several companies. Upload one PDF per company, enter tickers in the report mapping table, search GDELT and Google News RSS for adverse coverage from a defined Indian business-news publisher list, classify English candidates with FinBERT, and compare provisional scores.

## Run locally

1. Install Python 3.10 or newer.
2. Install Python packages with `pip install -r requirements.txt`.
3. Install Tesseract OCR and make it available on your system path for scanned reports.
4. Start the app from this folder with `streamlit run app.py`.
5. Upload PDFs, enter one exchange-qualified ticker per file, and select **Analyse reports and rank companies**.
6. Review the extracted ESG dataset, news and report evidence. Only analyst-confirmed incidents that satisfy the corroboration rule can lower scores.

## Full-report ESG dataset

Upload complete annual, sustainability, ESG, BRSR or governance PDFs. The app processes each page and its tables without requiring users to crop pages. OCR is enabled by default for pages with little embedded text and can be turned off in the extraction options. Tesseract OCR is installed in Streamlit Cloud through `packages.txt`.

The extractor matches the complete supplied dictionary of **25 Environmental, 17 Social and 40 Governance topics**. It exports a backend dataset with one row per matched statement and topic, including the source file and page, neighboring context, matched terms, claim type, quantities, units, reporting periods, targets, trends, table data, OCR status and an evidence confidence hint. Download the dataset as Excel, CSV or JSON. The Excel workbook includes pillar sheets, a topic summary and extraction errors.

Confidence is a rule-based evidence-quality hint, not a calibrated probability or ESG score. The dataset supports analyst review; it does not prove a company claim or validate a reported outcome. See [`ESG_DATASET_README.md`](ESG_DATASET_README.md) for the extractor field definitions and limitations.

The notebook prototype is available as `ESG_company_screener.ipynb`.

## Streamlit Community Cloud

Deploy `app.py` from the root of this repository and keep the repository and app private unless the IIF explicitly approves public access. Streamlit Cloud uses `requirements.txt` for Python libraries and `packages.txt` for Linux dependencies such as Tesseract. Private-repository deployment requires connecting Streamlit Community Cloud to the GitHub repository.

## Scoring and review

Each E, S and G pillar score runs from 0 to 7. The provisional score weights the FinBERT news signal at **70%** and annual-report evidence at **30%**. The news component uses the mean negative probability when article and model coverage targets are met. A shortfall or unavailable FinBERT output receives a neutral component. Analyst-confirmed incidents can impose a severity penalty, capped to avoid counting the same signal twice. The overall company ranking sums the three adjusted pillar scores for a maximum of 21. News targets per company are **20 Environmental, 20 Social and 50 Governance articles**. Counts and shortfalls appear in the coverage table. These counts are retrieval targets, not guarantees, because the search indexes and publisher access determine available results. This is a provisional analyst-review ranking, not a verified ESG performance rating. FinBERT sentiment is not proof that misconduct occurred. The approved publisher groups are The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI. A news-reported incident requires direct article URLs from five distinct approved publisher groups; each group counts once, and syndicated or repeated copies of the same report must not be counted as independent evidence. The sole one-document exception is an HTTPS URL on a `.gov.in`, `.nic.in` or `rbi.org.in` domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not meet this exception.

## News coverage

GDELT and the India edition of Google News RSS are discovery sources, restricted to the approved publisher list above. Searches are issued in narrow event-term batches over a requested 365-day window. Actual historical coverage, item limits and outlet access vary by index. The app reports retrieved article counts, shortfalls, and search errors by pillar. Verify source identity, direct article URL, date, company identity, allegation status, outcome, company response and whether coverage is syndicated before recording an incident. Publisher inclusion is a source-quality screen, not proof that an allegation is true.

## Evidence database

The app writes discovered articles, extracted report evidence, structured ESG dataset rows, analyst-confirmed incidents and pillar score history to a SQLite database and exposes the tables in the app. Each browser session has an isolated database, so one user's uploaded report evidence is not shown to another user. Download the SQLite file for backup or durable storage. Community Cloud does not guarantee that local files persist across restarts or session expiry, so connect a managed database before treating the hosted SQLite file as the permanent system of record. Database files are never committed to the repository.

## Limitations

Equal topic weights can mis-rank companies across sectors. Validate the complete dictionaries, extraction quality, and scoring thresholds against manually reviewed reports and news before using rankings for investment decisions. OCR and complex table extraction can be imperfect; check extracted values against the cited page.

Uploaded PDFs are processed by the hosted app and should not be uploaded unless hosting and data handling have been approved by the IIF. The app does not commit uploaded PDFs or results to the repository. Extracted evidence and score data are stored in the SQLite database and remain available to download from the app while the file is retained by the host.
