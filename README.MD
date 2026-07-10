# CredTech — Corporate Credit Deterioration Platform

Ranks S&P 500 companies by the probability their credit profile deteriorates over
the next year, blends that with news sentiment into a single **NexScore**, and
serves it on a FastAPI + Next.js dashboard.

> **What this predicts, precisely.** Not corporate default. The S&P 500 is
> survivorship-filtered — index membership is conditional on *not* being in
> distress, and companies that fail are removed before they fail — so the
> unconditional default rate in this universe is effectively zero. The model
> instead ranks companies by **relative credit deterioration**: whose Altman
> Z-Score falls into the worst quintile of the cross-section over the next four
> quarters. That is the question a credit desk actually asks of investment-grade
> names, and it is one this data can support.

---

## Architecture

```
build_structured_features.py   yfinance -> ratios -> weekly panel -> PostgreSQL
            |
            v
credit_risk_pipeline.py         forward Z-target -> LightGBM -> calibrated P(deterioration) + SHAP
            |
run.py                          credit score x news sentiment -> NexScore -> PostgreSQL
            |
backend/main.py (FastAPI)       REST over PostgreSQL
            |
frontend-next/ (Next.js)        leaderboard, company detail, compare
```

- **Database:** PostgreSQL
- **ML:** LightGBM, scikit-learn, SHAP
- **NLP:** RoBERTa (`rahilv/news-sentiment-analysis-roberta`), Gemini for summaries
- **API:** FastAPI + psycopg2
- **UI:** Next.js + React

---

## Known data limitation (read before running)

yfinance currently returns only **~5 quarters** of fundamentals per ticker, all
from 2025 onward. After the 45-day reporting lag, usable history spans roughly
**four months**. Consequences:

- A one-year forward horizon produces **zero** trainable rows on the current
  panels. A 90-day horizon produces ~2,000 rows across ~50 tickers — enough to
  run end-to-end, not enough to trust.
- The shipped `worker/data_out/*.parquet` files are **daily** (3,650 rows,
  weekends included). The builder now writes **weekly** (`W-FRI`).
  **Regenerate the panels** before relying on any number.

Run the diagnostic first, always:

```bash
python credit_risk_pipeline.py --diagnose
```

It prints fundamental coverage, ratio null-rates, missing columns, and the target
base rate. If it warns, downstream metrics are not trustworthy yet.

---

## What changed from the original submission

The original code and its documentation had diverged. Corrections, all verified
against the real parquet data:

| Area | Was | Now |
|---|---|---|
| **Target label** | `eps<0 & coverage<1 & ni<0` -> **all zeros** (coverage column never existed); model trained on one class, every company scored identically | Forward 1y Altman Z, ranked cross-sectionally; 20% base rate by construction |
| **Z-Score flows** | quarterly Sales/EBIT in an annual formula -> 83% of rows "distressed", Apple included | flows annualized (trailing 4 quarters) |
| **`debt_to_equity`** | silently absent — builder searched for a yfinance column that was renamed | canonical `COL` map; builder aborts loudly if a required ratio can't be built |
| **Train window** | hardcoded 2015-2020, which contained **no** fundamentals | derived from the data; refuses to train on an empty split |
| **Panel frequency** | docstring said weekly, shipped daily | actually weekly |
| **Entity leakage** | same 300 tickers in train and test | 60 tickers held out entirely; reports seen-firm vs unseen-firm AUC and the gap |
| **Reporting lag** | none — used filings before they were public | fundamentals lagged 45 days |
| **Metrics** | row-level AUC on ~13x-redundant weekly rows | company-quarter level, 5-seed mean +/- std, vs a 2-feature baseline |
| **Calibration** | `cv='prefit'` (removed in sklearn >=1.6) | version-safe `FrozenEstimator` path |
| **NexScore** | static 60/40 | sentiment weight scales with article count |
| **Update mode** | plain averaging | EMA, alpha=0.7 |
| **Sentiment dedup / decay** | described, absent | SequenceMatcher dedup + exponential recency decay; model cached as singleton |
| **`/stats` sentiment** | `s*100` (disagreed with every other route) | one shared `to_display_scale` |
| **`/company/{t}/risk-drivers`** | documented, missing | added; SHAP now persisted to `companies.risk_drivers` |
| **`PARQUET_GLOB`** | pointed at project root; matched nothing | `worker/data_out/` |
| **DB connections** | leaked on error | context manager closes in `finally` |
| **requirements.txt** | listed streamlit, tweepy, plotly, sqlalchemy — none imported | trimmed to what's used |

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`.env` in the project root:

```env
DATABASE_URL=postgresql://user:password@host/db
GEMINI_API_KEY=...
NEWS_KEY=...
```

---

## Run

```bash
# 1. Build the panel (regenerate — shipped parquets are stale/daily)
cd worker
python build_structured_features.py --years 10 --outdir data_out

# 2. Inspect the data before trusting the model
cd ..
python credit_risk_pipeline.py --diagnose

# 3. Evaluate (5-seed, company-quarter, with baseline and memorization gap)
python credit_risk_pipeline.py --report
python credit_risk_pipeline.py --report --target distress   # see the broken label for contrast

# 4. Full pipeline -> PostgreSQL
cd worker
python run.py --mode init

# 5. API
cd ../backend
uvicorn main:app --port 8000

# 6. Dashboard
cd ../frontend-next
npm install && npm run dev        # http://localhost:3000
```

---

## Honest evaluation posture

On the current data the model demonstrates the pipeline works but is
**statistically underpowered** (few positive firms, a four-month label window).
The evaluation is built to *show* that rather than hide it:

- **Company-quarter AUC on unseen firms**, not row-level AUC on redundant rows.
- **Mean +/- std over 5 seeds** — a single seed on this holdout is a coin flip.
- **Memorization gap** = seen-firm AUC minus unseen-firm AUC. A large gap means
  the model leans on firm identity.
- **A two-feature logistic baseline.** If LightGBM can't beat leverage + coverage,
  that's the finding.

To make this genuinely predictive: a data source with real filing history (SEC
EDGAR, or Stooq for a full 10 years), or a universe with real defaults (high-yield
issuers, or point-in-time S&P constituents including delistings), where the base
rate is 4-6% instead of under 1%.
