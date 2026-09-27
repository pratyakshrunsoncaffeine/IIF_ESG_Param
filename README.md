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

Each E, S and G report-evidence score runs from 0 to 7. The overall company ranking sums the three adjusted pillar scores for a maximum of 21. This is a prototype disclosure-evidence ranking, not a verified ESG performance rating. FinBERT negative sentiment is a review flag, not proof of misconduct and does not automatically reduce a score. The approved publisher groups are The Economic Times, Business Standard, Mint, Moneycontrol, Reuters and PTI. A news-reported incident requires direct article URLs from five distinct approved publisher groups; each group counts once, and syndicated or repeated copies of the same report must not be counted as independent evidence. The sole one-document exception is an HTTPS URL on a `.gov.in`, `.nic.in` or `rbi.org.in` domain to a final regulator order or sanction, final court judgment, or final statutory authority decision. Preliminary notices, allegations, company statements and ordinary filings do not meet this exception.

## News coverage

GDELT and the India edition of Google News RSS are discovery sources, restricted to the approved publisher list above. GDELT queries are limited to India-origin coverage. The prototype covers a recent search window of up to 90 days and reports source coverage and errors. It is not a complete 12-month news archive. Google News RSS is best effort and its feed format and history can change. Verify source identity, direct article URL, date, company identity, allegation status, outcome, company response and whether coverage is syndicated before recording an incident. Publisher inclusion is a source-quality screen, not proof that an allegation is true.

## Limitations

Equal topic weights can mis-rank companies across sectors. Validate the complete dictionaries, extraction quality, and scoring thresholds against manually reviewed reports and news before using rankings for investment decisions. OCR and complex table extraction can be imperfect; check extracted values against the cited page.

Uploaded PDFs are processed by the hosted app and should not be uploaded unless the hosting and data handling have been approved by the IIF. The app does not commit uploaded PDFs or results to the repository. Generated evidence and score data are available to download as CSV during the session.
