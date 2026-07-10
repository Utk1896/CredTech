"""
run.py
======
Orchestrator: credit model -> sentiment -> NexScore -> PostgreSQL.

WHAT CHANGED AND WHY
--------------------
Five divergences between the documentation and the shipped code:

1. PARQUET_GLOB pointed at the project root. The panels live in worker/data_out/.
   The parquet fallback therefore never fired; if PostgreSQL was empty the
   pipeline silently scored nothing.

2. NexScore used a STATIC 0.6/0.4 split. The design documents describe a weight
   that scales with news volume, so one stray article cannot swing a score. Now:
       confidence  = min(num_articles / 10, 1.0)
       w_sentiment = 0.4 * confidence
   A company with two articles gives sentiment 8% weight, not 40%.

3. Update mode AVERAGED old and new ((a+b)/2). The docs describe an EMA with
   alpha = 0.7. Averaging is an EMA with alpha = 0.5 and no memory of how many
   times it has run -- after ten runs the first observation still carries 1/1024
   weight, which is not what "averaging" implies to a reader. Now an explicit EMA.

4. get_credit_scores() returned {ticker: float}; it now returns a dict with
   risk_drivers. Both shapes are handled here so an old model file cannot crash
   the worker.

5. SHAP risk drivers were computed but never persisted. The dashboard promises a
   "Risk Drivers" panel. They are now written to companies.risk_drivers (JSONB)
   and injected into the Gemini analyst-note prompt, which is what makes the note
   grounded rather than a restatement of the score.

Run:
    python run.py --mode init      # wipe, seed queue, process
    python run.py --mode update    # EMA against existing scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unstructured import compute_sentiment_score          # noqa: E402
from credit_risk_pipeline import get_credit_scores        # noqa: E402

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NEWS_API_KEY = os.getenv("NEWS_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The panels are written by worker/build_structured_features.py --outdir data_out,
# i.e. worker/data_out/. The old value (PROJECT_ROOT / "panel_*.parquet") matched
# nothing.
PANEL_GLOB = str(Path(__file__).resolve().parent / "data_out" / "panel_*.parquet")

# --- NexScore weights (documented values, now actually used) ---
MAX_SENTIMENT_WEIGHT = 0.40    # ceiling, reached at >= 10 articles
CONFIDENCE_ARTICLES = 10       # articles needed for full confidence
EMA_ALPHA = 0.70               # weight on the new observation in update mode

BATCH_SIZE = 5                 # 5 tickers x ~2 Gemini calls = 10 <= 15 RPM
BATCH_SLEEP_SECONDS = 60

_gemini = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"WARNING: Gemini init failed ({e}); analyst notes will be templated.")
else:
    print("WARNING: GEMINI_API_KEY not set; analyst notes will be templated.")


# ======================================================================
# SCORING
# ======================================================================

def get_grade(nexscore: float) -> str:
    for cutoff, grade in [(90, "AAA"), (80, "AA"), (70, "A"),
                          (60, "BBB"), (50, "BB"), (40, "B")]:
        if nexscore >= cutoff:
            return grade
    return "CCC"


def compute_nexscore(default_probability: float, sentiment_score: float,
                     num_articles: int) -> tuple[float, float, float, float]:
    """Composite 0-100 with a volume-scaled sentiment weight.

    Returns (nexscore, creditworthiness, sentiment_display, w_sentiment).

    The sentiment weight is conditional on how much news actually exists. With two
    articles the effective weight is 0.4 * 0.2 = 8%, so a single hostile headline
    cannot drag a company two grades. This is what the design docs describe; the
    old code hardcoded 0.4 regardless.
    """
    creditworthiness = (1.0 - default_probability) * 100.0
    sentiment_display = ((sentiment_score + 1.0) / 2.0) * 100.0      # [-1,1] -> [0,100]

    confidence = min(num_articles / CONFIDENCE_ARTICLES, 1.0)
    w_sentiment = MAX_SENTIMENT_WEIGHT * confidence
    w_credit = 1.0 - w_sentiment

    nexscore = w_credit * creditworthiness + w_sentiment * sentiment_display
    return (round(nexscore, 1), round(creditworthiness, 1),
            round(sentiment_display, 1), round(w_sentiment, 3))


def ema(new: float, old: float | None, alpha: float = EMA_ALPHA) -> float:
    """new*alpha + old*(1-alpha). Absorbs most of the new signal, keeps context."""
    return new if old is None else alpha * new + (1 - alpha) * old


# ======================================================================
# GEMINI
# ======================================================================

def generate_analyst_note(company: str, ticker: str, nexscore: float, grade: str,
                          creditworthiness: float, sentiment_display: float,
                          risk_drivers: list[dict]) -> str:
    """Two sentences, grounded in the SHAP drivers.

    Injecting the drivers is what makes the LLM a formatting layer over real
    numbers rather than a paraphrase of the score. The old prompt passed no
    drivers, so the note could only restate what the reader already saw.
    """
    fallback = (f"{company} ({ticker}) carries a NexScore of {nexscore}/100 (grade {grade}), "
                f"reflecting creditworthiness of {creditworthiness}/100 and market "
                f"sentiment of {sentiment_display}/100.")
    if _gemini is None:
        return fallback

    drivers = "; ".join(f"{d['feature']} {d['direction']} (impact {d['impact']:+.3f})"
                        for d in risk_drivers[:3]) or "no dominant driver identified"

    prompt = (
        f"You are a credit analyst. Write exactly 2 sentences (max 60 words) on "
        f"{company} ({ticker}).\n"
        f"NexScore {nexscore}/100 (grade {grade}); creditworthiness {creditworthiness}/100; "
        f"market sentiment {sentiment_display}/100.\n"
        f"Model risk drivers: {drivers}.\n"
        f"Reference the drivers specifically. Be professional. No preamble."
    )
    try:
        return _gemini.models.generate_content(
            model="gemini-1.5-flash", contents=prompt).text.strip()
    except Exception as e:
        print(f"    Gemini failed for {ticker}: {e}")
        return fallback


# ======================================================================
# DATABASE
# ======================================================================

def get_db_connection():
    if not DATABASE_URL:
        print("Error: DATABASE_URL not set")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None


SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT,
    credit_score FLOAT,
    sentiment_score FLOAT,
    sentiment_summary TEXT,
    nexscore FLOAT,
    grade TEXT,
    analyst_note TEXT,
    risk_drivers JSONB,
    sector TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id) ON DELETE CASCADE,
    credit_score FLOAT,
    sentiment_score FLOAT,
    nexscore FLOAT,
    grade TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS news_headlines (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id) ON DELETE CASCADE,
    run_id TEXT,
    title TEXT,
    source TEXT,
    url TEXT,
    sentiment_label TEXT,
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS market_features (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    features JSONB,
    UNIQUE(ticker, date)
);
CREATE TABLE IF NOT EXISTS pipeline_queue (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scores_company    ON scores(company_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_headlines_company ON news_headlines(company_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_queue_status      ON pipeline_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_features_ticker   ON market_features(ticker, date);
"""


def apply_schema(conn) -> None:
    """Idempotent. Also adds risk_drivers to pre-existing databases."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("ALTER TABLE companies ADD COLUMN IF NOT EXISTS risk_drivers JSONB")
        cur.execute("ALTER TABLE companies ADD COLUMN IF NOT EXISTS nexscore FLOAT")
        cur.execute("ALTER TABLE companies ADD COLUMN IF NOT EXISTS grade TEXT")
        cur.execute("ALTER TABLE companies ADD COLUMN IF NOT EXISTS analyst_note TEXT")
        cur.execute("ALTER TABLE scores ADD COLUMN IF NOT EXISTS nexscore FLOAT")
        cur.execute("ALTER TABLE scores ADD COLUMN IF NOT EXISTS grade TEXT")
    conn.commit()
    print("Schema applied.")


def recreate_database(conn) -> bool:
    print("Recreating tables ...")
    try:
        with conn.cursor() as cur:
            for t in ["news_headlines", "scores", "companies",
                      "market_features", "pipeline_queue"]:
                cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            cur.execute(SCHEMA)
        conn.commit()
        seed_queue(conn)
        return True
    except Exception as e:
        print(f"Error recreating DB: {e}")
        conn.rollback()
        return False


def seed_queue(conn) -> None:
    try:
        import io
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        tickers = pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()[:300]
        with conn.cursor() as cur:
            execute_batch(cur, "INSERT INTO pipeline_queue (ticker) VALUES (%s) "
                               "ON CONFLICT DO NOTHING", [(t,) for t in tickers])
        conn.commit()
        print(f"Seeded pipeline_queue with {len(tickers)} tickers.")
    except Exception as e:
        print(f"Failed to seed queue: {e}")
        conn.rollback()


def claim_batch(conn, batch_size: int = BATCH_SIZE) -> tuple[list[int], list[str]]:
    """FOR UPDATE SKIP LOCKED: two workers never claim the same rows."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker FROM pipeline_queue
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT %s FOR UPDATE SKIP LOCKED
            """, (batch_size,))
            rows = cur.fetchall()
            if not rows:
                return [], []
            ids = [r[0] for r in rows]
            cur.execute("UPDATE pipeline_queue SET status='processing', updated_at=NOW() "
                        "WHERE id = ANY(%s)", (ids,))
        conn.commit()
        return ids, [r[1] for r in rows]
    except Exception as e:
        print(f"queue claim failed: {e}")
        conn.rollback()
        return [], []


def mark_batch(conn, ids: list[int], status: str) -> None:
    if not ids:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE pipeline_queue SET status=%s, updated_at=NOW() "
                        "WHERE id = ANY(%s)", (status, ids))
        conn.commit()
    except Exception as e:
        print(f"queue mark failed: {e}")
        conn.rollback()


def upsert_company(conn, ticker, name, credit, sentiment, summary,
                   nexscore, grade, note, drivers, is_update) -> int | None:
    """One row per ticker. In update mode, EMA against the stored values."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, credit_score, sentiment_score FROM companies "
                        "WHERE ticker = %s", (ticker,))
            row = cur.fetchone()

            if row and is_update:
                cid, old_credit, old_sent = row
                credit = ema(credit, old_credit)
                sentiment = ema(sentiment, old_sent)
                print(f"    EMA: credit -> {credit:.4f}, sentiment -> {sentiment:+.4f}")

            if row:
                cid = row[0]
                cur.execute("""
                    UPDATE companies SET name=%s, credit_score=%s, sentiment_score=%s,
                        sentiment_summary=%s, nexscore=%s, grade=%s, analyst_note=%s,
                        risk_drivers=%s
                    WHERE id=%s
                """, (name, credit, sentiment, summary, nexscore, grade, note,
                      json.dumps(drivers), cid))
            else:
                cur.execute("""
                    INSERT INTO companies (ticker, name, credit_score, sentiment_score,
                        sentiment_summary, nexscore, grade, analyst_note, risk_drivers)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """, (ticker, name, credit, sentiment, summary, nexscore, grade, note,
                      json.dumps(drivers)))
                cid = cur.fetchone()[0]
        conn.commit()
        return cid
    except Exception as e:
        print(f"    upsert failed for {ticker}: {e}")
        conn.rollback()
        return None


def insert_score(conn, cid, credit, sentiment, nexscore, grade) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO scores (company_id, credit_score, sentiment_score,
                           nexscore, grade, updated_at) VALUES (%s,%s,%s,%s,%s,%s)""",
                        (cid, credit, sentiment, nexscore, grade, datetime.now()))
        conn.commit()
    except Exception as e:
        print(f"    score insert failed: {e}")
        conn.rollback()


def insert_headlines(conn, cid, run_id, headlines) -> None:
    if not headlines:
        return
    try:
        with conn.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO news_headlines (company_id, run_id, title, source, url,
                                            sentiment_label, sentiment_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, [(cid, run_id, h.get("title", ""), h.get("source", ""), h.get("url", ""),
                   h.get("sentiment_label", "neutral"), h.get("sentiment_score", 0.0))
                  for h in headlines])
        conn.commit()
    except Exception as e:
        print(f"    headline insert failed: {e}")
        conn.rollback()


# ======================================================================
# PIPELINE
# ======================================================================

def _normalize_credit_scores(raw: dict) -> dict[str, dict]:
    """Accept both the old {ticker: float} and the new {ticker: {...}} shapes."""
    out = {}
    for t, v in (raw or {}).items():
        if isinstance(v, dict):
            out[t] = v
        else:
            out[t] = {"default_probability": float(v), "risk_drivers": [], "as_of": None}
    return out


COMPANY_NAMES = {
    "AAPL": "Apple Inc", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc",
    "AMZN": "Amazon.com Inc", "META": "Meta Platforms Inc", "TSLA": "Tesla Inc",
    "JPM": "JPMorgan Chase & Co", "BAC": "Bank of America Corp",
    "GS": "Goldman Sachs Group Inc", "V": "Visa Inc", "MA": "Mastercard Inc",
    "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation",
    "PG": "Procter & Gamble Co", "KO": "Coca-Cola Company", "PEP": "PepsiCo Inc",
    "WMT": "Walmart Inc", "UNH": "UnitedHealth Group Inc", "JNJ": "Johnson & Johnson",
    "NVDA": "NVIDIA Corporation",
}


def process_data(start_date=None, end_date=None, is_update: bool = False) -> None:
    print("=" * 66)
    print(f"CredTech pipeline  |  mode = {'update (EMA)' if is_update else 'init'}")
    print("=" * 66)

    run_id = str(uuid.uuid4())
    print(f"run_id: {run_id}")

    conn = get_db_connection()
    if not conn:
        return

    if is_update:
        apply_schema(conn)
    else:
        if not recreate_database(conn):
            conn.close()
            return

    # ---- Step 1: credit model (once, for all tickers) ----
    print("\n[1] credit model ...")
    try:
        credit = _normalize_credit_scores(get_credit_scores(panel_glob=PANEL_GLOB))
    except Exception as e:
        print(f"    credit pipeline failed: {e}")
        credit = {}

    if not credit:
        print("    !! No credit scores. Every ticker will fall back to 0.5 (neutral),")
        print("    !! which means the dashboard shows a constant. Run:")
        print("    !!   python credit_risk_pipeline.py --diagnose")
        print("    !! Do not ship a dashboard in this state.")
    else:
        probs = [v["default_probability"] for v in credit.values()]
        print(f"    scored {len(credit)} tickers | range [{min(probs):.3f}, {max(probs):.3f}]")
        if max(probs) - min(probs) < 1e-6:
            print("    !! All probabilities identical -- the model learned nothing.")
            print("    !! Almost certainly a single-class target. Run --diagnose.")

    # ---- Step 2+3: sentiment, blend, persist, batch by batch ----
    while True:
        ids, tickers = claim_batch(conn)
        if not tickers:
            print("\nQueue empty. Done.")
            break

        print(f"\n[2] sentiment for {tickers} ...")
        names = {t: COMPANY_NAMES.get(t, t) for t in tickers}

        try:
            sentiment = compute_sentiment_score(tickers, names, NEWS_API_KEY,
                                                start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"    sentiment failed: {e}")
            mark_batch(conn, ids, "pending")     # release, do not strand
            break

        print("\n[3] NexScore + analyst notes + DB write ...")
        for _, row in sentiment.iterrows():
            ticker = row["ticker"]
            c = credit.get(ticker, {"default_probability": 0.5, "risk_drivers": []})
            dp = c["default_probability"]
            drivers = c.get("risk_drivers", [])

            n_articles = int(row.get("num_articles", 0))
            s = float(row["sentiment_score"])

            nexscore, creditworthiness, sentiment_display, w_sent = compute_nexscore(
                dp, s, n_articles)
            grade = get_grade(nexscore)

            print(f"\n  {ticker} ({row['company_name']})")
            print(f"    default_prob {dp:.4f} -> creditworthiness {creditworthiness}/100")
            print(f"    sentiment {s:+.4f} over {n_articles} articles "
                  f"-> display {sentiment_display}/100 (weight {w_sent:.2f})")
            print(f"    NexScore {nexscore}/100  grade {grade}")

            note = generate_analyst_note(row["company_name"], ticker, nexscore, grade,
                                         creditworthiness, sentiment_display, drivers)

            cid = upsert_company(conn, ticker, row["company_name"], dp, s,
                                 row.get("sentiment_summary", ""), nexscore, grade,
                                 note, drivers, is_update)
            if cid:
                insert_score(conn, cid, dp, s, nexscore, grade)
                insert_headlines(conn, cid, run_id, row.get("top_headlines", []))
                print(f"    saved (company_id={cid})")

        mark_batch(conn, ids, "done")
        print(f"\nBatch done. Sleeping {BATCH_SLEEP_SECONDS}s for Gemini rate limits.")
        time.sleep(BATCH_SLEEP_SECONDS)

    conn.close()


def get_latest_scores() -> pd.DataFrame | None:
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT c.ticker, c.name, c.credit_score, c.sentiment_score,
                       c.nexscore, c.grade, s.updated_at
                FROM companies c
                LEFT JOIN scores s ON s.id = (
                    SELECT MAX(id) FROM scores WHERE company_id = c.id)
                ORDER BY c.nexscore DESC NULLS LAST
            """)
            return pd.DataFrame(cur.fetchall())
    except Exception as e:
        print(f"Error retrieving scores: {e}")
        return None
    finally:
        conn.close()


def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad date {s}; use YYYY-MM-DD")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CredTech pipeline runner")
    ap.add_argument("--start-date", type=parse_date)
    ap.add_argument("--end-date", type=parse_date)
    ap.add_argument("--mode", choices=["init", "update"], default="init")
    args = ap.parse_args()

    if args.start_date and args.end_date and args.start_date >= args.end_date:
        raise SystemExit("start-date must precede end-date")

    process_data(args.start_date, args.end_date, args.mode == "update")

    print("\n--- latest scores ---")
    latest = get_latest_scores()
    if latest is not None and not latest.empty:
        latest["creditworthiness"] = ((1 - latest["credit_score"]) * 100).round(1)
        # Same [-1,1] -> [0,100] map the API and dashboard use. The old code used
        # sentiment*100 here and (s+1)/2*100 in the API: two scales, one number.
        latest["sentiment_display"] = (((latest["sentiment_score"] + 1) / 2) * 100).round(1)
        print(latest[["ticker", "nexscore", "grade", "creditworthiness",
                      "sentiment_display", "updated_at"]].to_string(index=False))
    else:
        print("No scores in DB.")
