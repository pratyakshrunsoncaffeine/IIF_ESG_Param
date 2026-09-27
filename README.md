# IIF ESG Parameterization Framework

A Streamlit app and Jupyter Notebook prototype for reviewing ESG evidence across several companies. Upload one PDF per company, enter tickers in the report mapping table, search GDELT and Google News RSS for candidate adverse coverage, classify English candidates with FinBERT, and compare provisional scores.

## Run locally

1. Install Python 3.10 or newer.
2. Install Python packages with `pip install -r requirements.txt`.
3. Install Tesseract OCR and make it available on your system path for scanned reports.
4. Start the app from this folder with `streamlit run app.py`.
5. Upload PDFs, enter one exchange-qualified ticker per file, and select **Analyse reports and rank companies**.
6. Review news and report evidence. Only analyst-confirmed incidents that satisfy the corroboration rule can lower scores.

The notebook prototype is available as `ESG_company_screener.ipynb`.

## Streamlit Community Cloud

Deploy `app.py` from the root of this repository and keep the repository and app private unless the IIF explicitly approves public access. Streamlit Cloud uses `requirements.txt` for Python libraries and `packages.txt` for Linux dependencies such as Tesseract. Private-repository deployment requires connecting Streamlit Community Cloud to the GitHub repository.

## Scoring and review

Each E, S and G report-evidence score runs from 0 to 7. The overall company ranking sums the three adjusted pillar scores for a maximum of 21. This is a prototype disclosure-evidence ranking, not a verified ESG performance rating. FinBERT negative sentiment is a review flag, not proof of misconduct and does not automatically reduce a score. Analysts must confirm incidents and provide five independent sources or a final regulator finding, court judgment or official sanction.

## News coverage

GDELT and Google News RSS are discovery sources. The prototype covers a recent search window of up to 90 days and reports source coverage and errors. It is not a complete 12-month news archive. Google News RSS is best effort and its feed format and history can change. Verify source quality, article date, company identity, allegation status, outcomes, company response and duplicates before recording an incident.

## Limitations

The supplied Environmental and Governance keyword attachments were truncated. Complete and validate those dictionaries before comparing companies. Equal topic weights can mis-rank companies across sectors. Validate the framework against manually reviewed reports and news before using its rankings for investment decisions.

Uploaded PDFs are processed by the hosted app and should not be uploaded unless the hosting and data handling have been approved by the IIF. The app does not commit uploaded PDFs or results to the repository. Generated evidence and score data are available to download as CSV during the session.
