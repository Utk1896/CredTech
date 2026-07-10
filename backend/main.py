"""
backend/main.py
===============
Thin REST layer over PostgreSQL.

WHAT CHANGED AND WHY
--------------------
1. SENTIMENT SCALE WAS INCONSISTENT.  /companies, /leaderboard and
   /company/{t}/nexscore all mapped raw sentiment [-1,1] to [0,100] via
   (s + 1) / 2 * 100.  /stats used  s * 100.  Same number, two scales: a raw
   sentiment of -0.4 rendered as 30.0 on the leaderboard and -40.0 in the
   summary. Both now use one function, `to_display_scale`.

2. /company/{ticker}/risk-drivers DID NOT EXIST.  The design docs describe it and
   the dashboard promises a "Risk Drivers" panel. run.py now persists SHAP output
   to companies.risk_drivers (JSONB); this serves it.

3. CONNECTIONS LEAKED ON ERROR.  `with get_conn() as conn` commits or rolls back
   the transaction but does NOT close the socket -- that is psycopg2's documented
   behaviour and a common surprise. Under load the pool exhausts. Now a
   contextmanager that closes in `finally`.

4. NO N+1 IN /leaderboard.  Rank is computed in SQL with a window function rather
   than by enumerating a Python list, so pagination past offset 0 no longer
   restarts the rank at 1. The old code numbered page two's first row "1".
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="CredTech API", version="3.0")

# NOTE: allow_origins=["*"] is fine for a local dashboard. In production, pin it
# to the deployed frontend origin -- a wildcard plus credentials is rejected by
# browsers anyway, and without credentials it invites abuse of the endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@contextmanager
def get_conn():
    """Commit/rollback AND close.

    `with psycopg2.connect(...) as conn` handles the transaction, not the socket.
    Wrapping it means the connection actually goes back to the OS on the way out.
    """
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ======================================================================
# SCALES -- one definition, used everywhere
# ======================================================================

def to_creditworthiness(default_probability: float | None) -> float | None:
    """Model outputs P(deterioration) in [0,1]. Higher is worse. Invert for display."""
    if default_probability is None:
        return None
    return round((1.0 - float(default_probability)) * 100.0, 1)


def to_display_scale(sentiment: float | None) -> float | None:
    """Raw sentiment lives in [-1, 1]. Dashboard shows [0, 100].

    This is the ONLY place that map is written. /stats previously used s * 100,
    which sent negative sentiment below zero and disagreed with every other route.
    """
    if sentiment is None:
        return None
    return round(((float(sentiment) + 1.0) / 2.0) * 100.0, 1)


# ======================================================================
# ROUTES
# ======================================================================

@app.get("/")
def root():
    return {"message": "CredTech API v3", "docs": "/docs"}


@app.get("/companies")
def get_companies(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, ticker, name, credit_score, sentiment_score, nexscore, grade
            FROM companies ORDER BY ticker LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()

    return [{**dict(r),
             "creditworthiness_display": to_creditworthiness(r["credit_score"]),
             "sentiment_display": to_display_scale(r["sentiment_score"])}
            for r in rows]


@app.get("/scores")
def get_scores(limit: int = Query(20, ge=1, le=200)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT c.ticker, s.credit_score, s.sentiment_score, s.nexscore,
                   s.grade, s.updated_at
            FROM scores s JOIN companies c ON s.company_id = c.id
            ORDER BY s.updated_at DESC LIMIT %s
        """, (limit,))
        return [dict(r) for r in cur.fetchall()]


@app.get("/leaderboard")
def get_leaderboard(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Rank comes from SQL, not from enumerate().

    The old implementation numbered rows 1..N inside the Python loop, so page two
    (offset=100) also started at rank 1. RANK() OVER is computed across the whole
    table before the LIMIT applies.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM (
                SELECT ticker, name, nexscore, grade, credit_score, sentiment_score,
                       RANK() OVER (ORDER BY nexscore DESC NULLS LAST) AS rank
                FROM companies
            ) ranked
            ORDER BY rank LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()

    return [{"rank": r["rank"], "ticker": r["ticker"], "name": r["name"],
             "nexscore": r["nexscore"], "grade": r["grade"],
             "credit_display": to_creditworthiness(r["credit_score"]),
             "sentiment_display": to_display_scale(r["sentiment_score"])}
            for r in rows]


@app.get("/company/{ticker}/history")
def get_company_history(ticker: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, name, credit_score, sentiment_score, sentiment_summary,
                       nexscore, grade, analyst_note
                FROM companies WHERE ticker = %s
            """, (ticker.upper(),))
            company = cur.fetchone()
        if not company:
            raise HTTPException(404, f"Ticker '{ticker}' not found")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT credit_score, sentiment_score, nexscore, grade, updated_at
                FROM scores WHERE company_id = %s ORDER BY updated_at DESC
            """, (company["id"],))
            history = cur.fetchall()

    return {"company_info": dict(company),
            "score_history": [dict(h) for h in history],
            "total_records": len(history)}


@app.get("/company/{ticker}/nexscore")
def get_company_nexscore(ticker: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT c.ticker, c.name, c.nexscore, c.grade, c.analyst_note,
                   c.credit_score, c.sentiment_score,
                   (SELECT MAX(updated_at) FROM scores WHERE company_id = c.id) AS updated_at
            FROM companies c WHERE c.ticker = %s
        """, (ticker.upper(),))
        r = cur.fetchone()

    if not r:
        raise HTTPException(404, f"Ticker '{ticker}' not found")

    return {"ticker": r["ticker"], "name": r["name"], "nexscore": r["nexscore"],
            "grade": r["grade"], "analyst_note": r["analyst_note"],
            "credit_display": to_creditworthiness(r["credit_score"]),
            "sentiment_display": to_display_scale(r["sentiment_score"]),
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None}


@app.get("/company/{ticker}/news")
def get_company_news(ticker: str, limit: int = Query(5, ge=1, le=20)):
    """Most impactful headlines from the latest pipeline run."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM companies WHERE ticker = %s", (ticker.upper(),))
        company = cur.fetchone()
        if not company:
            raise HTTPException(404, f"Ticker '{ticker}' not found")

        cur.execute("""
            SELECT title, source, url, sentiment_label, sentiment_score, created_at, run_id
            FROM news_headlines
            WHERE company_id = %s AND run_id = (
                SELECT run_id FROM news_headlines WHERE company_id = %s
                ORDER BY created_at DESC LIMIT 1)
            ORDER BY ABS(sentiment_score) DESC LIMIT %s
        """, (company["id"], company["id"], limit))
        rows = cur.fetchall()

    return {"ticker": ticker.upper(),
            "run_id": rows[0]["run_id"] if rows else None,
            "headlines": [dict(r) for r in rows]}


@app.get("/company/{ticker}/risk-drivers")
def get_company_risk_drivers(ticker: str):
    """SHAP feature attributions for this ticker's most recent scoring.

    This route is referenced in the design docs and by the dashboard, and did not
    exist. run.py now writes companies.risk_drivers (JSONB); this reads it.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT ticker, name, risk_drivers, nexscore, grade "
                    "FROM companies WHERE ticker = %s", (ticker.upper(),))
        r = cur.fetchone()

    if not r:
        raise HTTPException(404, f"Ticker '{ticker}' not found")

    return {"ticker": r["ticker"], "name": r["name"],
            "nexscore": r["nexscore"], "grade": r["grade"],
            "risk_drivers": r["risk_drivers"] or []}


@app.get("/stats")
def get_stats():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS total_companies,
                   AVG(credit_score) AS avg_credit, MIN(credit_score) AS min_credit,
                   MAX(credit_score) AS max_credit,
                   AVG(sentiment_score) AS avg_sent, MIN(sentiment_score) AS min_sent,
                   MAX(sentiment_score) AS max_sent,
                   AVG(nexscore) AS avg_nex, MIN(nexscore) AS min_nex,
                   MAX(nexscore) AS max_nex
            FROM companies
        """)
        s = cur.fetchone()

    def r4(v):
        return round(float(v), 4) if v is not None else None

    return {
        "total_companies": s["total_companies"],
        "credit": {"raw_avg": r4(s["avg_credit"]), "raw_min": r4(s["min_credit"]),
                   "raw_max": r4(s["max_credit"]),
                   "display_avg": to_creditworthiness(s["avg_credit"])},
        # was: avg_sentiment * 100 -- a different scale from every other route
        "sentiment": {"raw_avg": r4(s["avg_sent"]), "raw_min": r4(s["min_sent"]),
                      "raw_max": r4(s["max_sent"]),
                      "display_avg": to_display_scale(s["avg_sent"])},
        "nexscore": {"avg": r4(s["avg_nex"]), "min": r4(s["min_nex"]),
                     "max": r4(s["max_nex"])},
    }


@app.get("/health")
def health():
    """Cheap liveness probe; also confirms the DB is reachable."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM companies")
            n = cur.fetchone()["n"]
        return {"status": "ok", "companies": n}
    except Exception as e:
        raise HTTPException(503, f"database unreachable: {e}")
