# ESG PDF Extractor

This Python 3.9+ project extracts traceable ESG statements and nearby quantitative data from annual, sustainability, ESG, BRSR, and governance PDFs. It uses **the supplied `data/ESG_KEYWORDS.rtf`** as its only default word dictionary. It does not score companies or use a paid service or LLM.

## What is included

```text
esg_pdf_extractor/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── config.example.json
├── data/
│   ├── ESG_KEYWORDS.rtf
│   └── pdfs/                 ← put company PDFs here
├── esg/
│   ├── dictionary_loader.py
│   ├── pdf_extractor.py
│   ├── text_cleaner.py
│   ├── keyword_matcher.py
│   ├── context_extractor.py
│   ├── metric_extractor.py
│   ├── classifier.py
│   ├── deduplicator.py
│   ├── pipeline.py
│   └── exporter.py
├── output/                   ← generated files
└── tests/
```

The supplied RTF contains six literal Python structures: `environment_sectors`, `environment_keywords`, `social_sectors`, `social_keywords`, `governance_sectors`, and `governance_keywords`. The loader strips RTF formatting and reads **literal values only** with `ast.literal_eval`; it does not execute document code. It validates that sector names and dictionary topics match. On this copy it loads:

| Pillar | Topics | Keyword entries |
|---|---:|---:|
| Environmental | 25 | 991 |
| Social | 17 | 789 |
| Governance | 40 | 1,485 |

There are **3,061 unique phrases** across all pillars. The loader writes the complete normalized hierarchy to `output/normalized_dictionary.json` on each run. No handful of keywords is substituted for the source dictionary. You can edit the RTF and rerun; malformed structures cause a visible error and an error CSV.

## Install

Open a terminal in this project folder. A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. PyMuPDF extracts embedded PDF text; pdfplumber attempts tables; openpyxl writes Excel; striprtf reads the dictionary; pytest runs the tests.

## Run

Copy company `.pdf` files into **`data/pdfs/`**, then run from the project root:

```bash
python main.py --pdf-dir data/pdfs --dictionary data/ESG_KEYWORDS.rtf --output-dir output
```

For one PDF:

```bash
python main.py --pdf data/pdfs/company.pdf --dictionary data/ESG_KEYWORDS.rtf --output-dir output
```

Options: `--ocr` enables OCR fallback; `--config config.example.json` loads tunable settings. The defaults live in `config.py`. PDFs are processed one at a time, pages once each, and matching uses one precompiled regular expression. A failed PDF is logged and the next PDF continues.

## Pipeline and outputs

1. PyMuPDF reads each page's embedded text and preserves its raw and cleaned forms.
2. Phrases are matched case insensitively after punctuation, hyphen, whitespace and line-break normalization. Word boundaries prevent acronym substring matches. A lone broad word such as “board” needs supporting ESG context.
3. Each sentence is retained with its previous and next sentences, source filename, company name (PDF title or filename), and one-based page number.
4. Same-sentence quantities and units, reporting periods, targets, baseline years, and explicit from/to trends are extracted. Years alone are not treated as measured values.
5. Deterministic rules label claims as `MEASURED_RESULT`, `TARGET`, `POLICY`, `INITIATIVE`, `RISK_DISCLOSURE`, `ADVERSE_EVENT`, `COMPLIANCE`, `GENERAL_STATEMENT`, or `UNKNOWN`. One label is chosen with adverse events and targets taking precedence.
6. pdfplumber attempts page tables. Numeric cells are used only when their row label matches the dictionary and a column header is preserved. Ambiguous/malformed rows are skipped or logged, without invented relationships.
7. Duplicate matches for the same sentence, topic, source and page are merged. The most specific matched phrase is the primary keyword and all detected phrases stay in `matched_keywords`.

Files in `output/`:

- `esg_results.csv` — flat analyst-friendly records, including source text.
- `esg_results.json` — structured records with arrays and nulls.
- `esg_results.xlsx` — `All_ESG_Data`, the three pillar sheets, `ESG_Summary`, and `Extraction_Errors`.
- `extraction_errors.csv` — failures with source filename and page where available.
- `normalized_dictionary.json` — the complete parsed source dictionary.

The Excel summary groups by company, pillar and topic and counts matches, measured results, targets, adverse events and source pages. Each quantitative record retains `matched_sentence`, source filename and page. Table records also retain table index, full row, headers and raw table data.

### Confidence

The transparent heuristic starts at 0.52. It adds up to 0.10 for a multiword primary phrase, 0.10 for several matched phrases, 0.10 for a same-statement quantity, 0.07 for a recognized unit and 0.06 for a period. It subtracts 0.16 for OCR, 0.17 for table extraction and 0.12 for a generic primary word. It is clamped to 0–1. This is an evidence-quality hint, **not** a company score or calibrated probability. Adjust `minimum_confidence` in a JSON config to filter low-confidence records.

### OCR

Embedded text is always attempted first. OCR runs only for pages with fewer than the configured number of alphanumeric embedded characters **and only when `--ocr` is passed**. Install optional Python packages with `pip install pytesseract Pillow`, then install the Tesseract executable on your system (`brew install tesseract` on macOS; `sudo apt install tesseract-ocr` on Ubuntu). Without OCR dependencies, normal PDF extraction still works; OCR failures appear in `extraction_errors.csv`.

### Limits and analyst review

PDF text order and table structure vary by publisher. A recognized quantity in the same sentence is evidence-linked but may still refer to a different measure in a complex sentence; check the original statement and page. A table cell without a clear numeric value is skipped. The tool does not infer missing years, units or values, and does not assume a publication year is a reporting year. Publication-quality validation of extracted facts remains an analyst task.

## Test and troubleshoot

```bash
python -m pytest -q
```

- **No records:** Confirm the PDF has selectable text or use `--ocr` for scanned pages. Inspect `output/extraction_errors.csv`.
- **Dictionary parsing failure:** Keep all six source structures as literal lists/dictionaries in the RTF. The error CSV records the cause.
- **Encrypted PDF:** Unlock it before processing; the application logs it and continues.
- **Tesseract missing:** Omit `--ocr`, or install the executable and optional packages.
- **Table rows missing:** Inspect the source PDF; badly extracted tables are intentionally not converted into uncertain metrics.
