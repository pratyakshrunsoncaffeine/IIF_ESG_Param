# Upgrade validation

Validation run on 9 October 2026. These checks exercise extraction, grouping, news eligibility, scoring, exports, persistence, and the Streamlit upload path. They do not establish an ESG rating or validate every extracted claim.

## Automated and application checks

- The repository test suite passed: **58 tests passed** with Python 3.12 and the project `.venv`.
- The AppTest acceptance runner completed an annual-report-only run and a grouped two-company run with one optional HDFC BRSR. Dataset rows retained document type and fiscal year. The stubbed news scanner ran once for each grouped ticker, including only once for the company with two PDFs.
- Generated searchable PDF fixtures confirmed that annual and BRSR rows can retain the same ESG topic and distinct source filenames while that repeated topic contributes once to the company evidence score. CSV, Excel, SQLite dataset rows, and score history preserved the source metadata.
- News fixtures confirmed that an unrelated HDFC/Nuvama market article does not count toward coverage, change the ranking or incident penalty; a target-linked regulatory/customer-harm article was evaluated from its full body, including text beyond 5,000 characters and the first model window. Retrieval failure and a classifier interruption after an earlier successful chunk did not produce an eligible partial score.
- A real FinBERT CPU smoke test scored a long article across all **6 of 6 chunks** (2,793 tokens); model output was finite. This verifies inference plumbing and whole-text chunk coverage, not model accuracy.
- A single live Google News RSS sample resolved to an approved Business Standard direct URL and yielded a 3,979-character body. The body concerned a different bank, and the HDFC relevance check rejected it because it contained no HDFC company mention. This was one RSS query, not a broad news coverage check.

## Local report smoke

The supplied searchable HDFC FY2025–26 report was processed without OCR or table extraction: **629 pages**, **7,825 ESG dataset rows**, and **one extraction issue**. The TCS FY2025–26 report was processed with the same settings: **360 pages**, **4,094 rows**, and **no extraction errors**. The HDFC extraction issue remains visible for analyst review; this check did not manually validate each extracted row. The source PDFs stayed local and were not uploaded to a hosted app.

## Limits

The generated-PDF and news tests use deterministic fixtures and fake retrieval/classification where indicated. The live RSS sample checks direct-link decoding and article-body extraction for one item only. The real-model check measures successful inference and chunk coverage, not sentiment correctness. Real-report smoke used embedded text with OCR and numeric table extraction disabled. Review extracted statements against cited PDF pages before relying on them.

## Reproduce

Run the local app-path acceptance checks:

```powershell
.venv\Scripts\python.exe tools\acceptance_runner.py
```

Add `--real-finbert`, `--live-news`, or `--real-reports <PDF paths...>` for the optional local model, one-item RSS, or searchable-report smoke checks. `--live-news` makes one Google News RSS request and retrieves at most one article body.
