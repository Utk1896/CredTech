"""
unstructured.py
===============
News collection, deduplication, sentiment scoring, and recency weighting.

WHAT CHANGED AND WHY
--------------------
Three things the documentation described and the code did not do:

1. DEDUPLICATION.  Financial news is heavily syndicated -- the same Reuters wire
   appears on twenty sites. Without dedup, one story is counted twenty times and
   the mean sentiment is whatever the wire said. Now: pairwise SequenceMatcher on
   titles, drop anything >85% similar to an already-accepted article.

2. EXPONENTIAL RECENCY DECAY.  The old code took a flat mean, so a 28-day-old
   article counted as much as this morning's earnings miss. Now: w = e^(-0.1 * days_old),
   giving a ~7-day half-life, and the score is the weighted mean.

3. MODEL SINGLETON.  `pipeline(...)` was called once per batch of five tickers,
   reloading ~500MB of RoBERTa weights every time. Now cached at module level.

Also fixed: the error branch wrote a key called `news_sentiment` that nothing
downstream reads, so a failed ticker silently produced a row with no
`sentiment_score`. It now matches the success schema exactly.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# --- tunables (documented values, now actually used) ---
DEDUP_THRESHOLD = 0.85     # titles more similar than this are the same story
DECAY_LAMBDA = 0.10        # w = e^(-lambda * days_old); half-life ~= 6.9 days
MAX_ARTICLES = 50
SENTIMENT_MODEL = "rahilv/news-sentiment-analysis-roberta"

# RoBERTa is ~500MB. Loading it per batch was the single slowest thing in the
# worker. Module-level cache; the process is long-lived.
_SENTIMENT_PIPELINE = None


def _get_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        from transformers import pipeline
        print(f"[unstructured] loading {SENTIMENT_MODEL} (once) ...")
        _SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                                       truncation=True, max_length=512)
    return _SENTIMENT_PIPELINE


# ======================================================================
# COLLECTION
# ======================================================================

def get_company_news(ticker: str, tag_to_comp: dict, news_api_key: str | None,
                     start_date=None, end_date=None) -> pd.DataFrame:
    """NewsAPI, falling back to yfinance. Returns tidy article rows."""
    company = tag_to_comp.get(ticker, ticker)
    query = f'"{company}" OR {ticker}'
    rows = []

    if news_api_key:
        try:
            params = {"q": query, "language": "en", "sortBy": "publishedAt",
                      "pageSize": MAX_ARTICLES, "apiKey": news_api_key}
            if start_date and end_date:
                params["from"] = start_date.strftime("%Y-%m-%d")
                params["to"] = end_date.strftime("%Y-%m-%d")

            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)
            if r.status_code == 200:
                for a in r.json().get("articles", []):
                    if not a.get("title") or not a.get("description"):
                        continue
                    rows.append({
                        "source": (a.get("source") or {}).get("name", "Unknown"),
                        "title": a["title"],
                        "description": a["description"],
                        "url": a.get("url", ""),
                        "published_at": a.get("publishedAt", ""),
                        "text": f"{a['title']}\n{a['description']}",
                    })
            else:
                print(f"[unstructured] NewsAPI {r.status_code} for {ticker}")
        except Exception as e:
            print(f"[unstructured] NewsAPI error for {ticker}: {e}")

    if not rows:
        try:
            import yfinance as yf
            for a in (yf.Ticker(ticker).news or []):
                content = a.get("content", a)
                title = content.get("title", "")
                if not title:
                    continue
                summary = content.get("summary", title)
                rows.append({
                    "source": (content.get("provider") or {}).get("displayName", "Unknown"),
                    "title": title,
                    "description": summary,
                    "url": content.get("canonicalUrl", {}).get("url", ""),
                    "published_at": content.get("pubDate", ""),
                    "text": f"{title}\n{summary}",
                })
        except Exception as e:
            print(f"[unstructured] yfinance fallback failed for {ticker}: {e}")

    df = pd.DataFrame(rows)
    print(f"[unstructured] {ticker}: {len(df)} raw articles")
    return df


# ======================================================================
# DEDUPLICATION
# ======================================================================

def deduplicate_articles(df: pd.DataFrame, threshold: float = DEDUP_THRESHOLD) -> pd.DataFrame:
    """Greedy pairwise title similarity.

    O(n^2) on titles, but n <= 50, so it costs microseconds. Without this, a
    syndicated Reuters story appearing on twenty sites contributes twenty
    identical sentiment scores and the weighted mean becomes whatever the wire
    said. This was described in the design docs and absent from the code.
    """
    if df.empty:
        return df

    kept: list[int] = []
    for i, title in enumerate(df["title"].fillna("")):
        t = title.lower().strip()
        if any(SequenceMatcher(None, t, df["title"].iloc[k].lower().strip()).ratio() > threshold
               for k in kept):
            continue
        kept.append(i)

    out = df.iloc[kept].reset_index(drop=True)
    if len(out) < len(df):
        print(f"[unstructured] dedup: {len(df)} -> {len(out)} articles")
    return out


# ======================================================================
# SENTIMENT
# ======================================================================

def label_to_numeric(label: str, score: float) -> float:
    """Model label + confidence -> a signed value in [-1, 1]."""
    l = str(label).lower().strip()
    if l in ("bullish", "positive", "label_2"):
        return float(score)
    if l in ("bearish", "negative", "label_0"):
        return -float(score)
    return 0.0


def _days_old(published_at) -> float:
    if not published_at:
        return 0.0
    try:
        ts = pd.to_datetime(published_at, utc=True, errors="coerce")
        if pd.isna(ts):
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preds = _get_pipeline()(df["text"].tolist())
    df = df.copy()
    df["sentiment_label"] = [p["label"] for p in preds]
    df["sentiment_confidence"] = [p["score"] for p in preds]
    df["numeric_sentiment"] = [label_to_numeric(p["label"], p["score"]) for p in preds]
    df["days_old"] = df["published_at"].apply(_days_old)
    df["recency_weight"] = np.exp(-DECAY_LAMBDA * df["days_old"])
    return df


def weighted_sentiment(df: pd.DataFrame) -> float:
    """Exponentially recency-weighted mean.

    A flat mean treats a four-week-old analyst note the same as this morning's
    earnings miss. w = e^(-0.1 * days) gives a ~7-day half-life.
    """
    if df.empty:
        return 0.0
    w = df["recency_weight"].to_numpy()
    if w.sum() <= 0:
        return float(df["numeric_sentiment"].mean())
    return float(np.average(df["numeric_sentiment"].to_numpy(), weights=w))


def get_top_headlines(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Most impactful by |sentiment|, ordered."""
    if df.empty:
        return []
    top = df.reindex(df["numeric_sentiment"].abs().sort_values(ascending=False).index).head(n)
    return [{"title": r.get("title", ""),
             "source": r.get("source", "Unknown"),
             "url": r.get("url", ""),
             "sentiment_label": r.get("sentiment_label", "neutral"),
             "sentiment_score": float(r.get("numeric_sentiment", 0.0))}
            for _, r in top.iterrows()]


# ======================================================================
# GEMINI SUMMARY
# ======================================================================

def generate_sentiment_summary(company_name: str, score: float,
                               positives: list[str], negatives: list[str]) -> str:
    if not GOOGLE_API_KEY:
        return f"Sentiment for {company_name}: {score:+.3f} (no API key; summary skipped)."
    try:
        from google import genai
        prompt = (
            f"Summarize market sentiment for {company_name}.\n"
            f"Weighted sentiment score: {score:+.3f} (range -1 to +1).\n\n"
            f"Positive headlines:\n" + ("\n".join(positives) or "none") + "\n\n"
            f"Negative headlines:\n" + ("\n".join(negatives) or "none") + "\n\n"
            "Write 2-3 short paragraphs: overall assessment, key themes, "
            "potential market implications. Be objective and specific."
        )
        client = genai.Client(api_key=GOOGLE_API_KEY)
        return client.models.generate_content(model="gemma-3-27b-it", contents=prompt).text
    except Exception as e:
        return f"Summary unavailable ({e})."


# ======================================================================
# ENTRY POINT
# ======================================================================

def compute_sentiment_score(tickers, tag_to_comp, news_api_key,
                            bearer_token=None, start_date=None, end_date=None) -> pd.DataFrame:
    """One row per ticker.

    Columns: ticker, company_name, sentiment_score, sentiment_summary,
             top_headlines, num_articles

    num_articles is new and REQUIRED: run.py uses it to scale the sentiment weight
    in the NexScore. Without it, a company with one article gets the same 40%
    sentiment weight as one with fifty.

    The error branch now emits the same schema as the success branch. The old code
    wrote a `news_sentiment` key that nothing read, so failures produced rows the
    orchestrator could not interpret.
    """
    results = []

    for ticker in tickers:
        try:
            print(f"\n[unstructured] === {ticker} ===")
            df = get_company_news(ticker, tag_to_comp, news_api_key, start_date, end_date)
            df = deduplicate_articles(df)
            df = analyze_sentiment(df)

            score = weighted_sentiment(df)
            n = len(df)

            if n:
                ordered = df.sort_values("numeric_sentiment", ascending=False)
                pos = ordered["title"].head(5).tolist()
                neg = ordered["title"].tail(5).tolist()
            else:
                pos = neg = []

            results.append({
                "ticker": ticker,
                "company_name": tag_to_comp.get(ticker, ticker),
                "sentiment_score": score,
                "sentiment_summary": generate_sentiment_summary(
                    tag_to_comp.get(ticker, ticker), score, pos, neg),
                "top_headlines": get_top_headlines(df, 5),
                "num_articles": n,
            })
            print(f"[unstructured] {ticker}: weighted sentiment {score:+.4f} over {n} articles")

        except Exception as e:
            print(f"[unstructured] {ticker} failed: {e}")
            results.append({
                "ticker": ticker,
                "company_name": tag_to_comp.get(ticker, ticker),
                "sentiment_score": 0.0,
                "sentiment_summary": f"Processing error: {e}",
                "top_headlines": [],
                "num_articles": 0,          # forces sentiment weight to zero downstream
            })

    return pd.DataFrame(results)
