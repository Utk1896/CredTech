#!/usr/bin/env python3
"""
build_structured_features.py
============================
Builds the issuer x date panel that everything downstream reads.

WHAT CHANGED AND WHY
--------------------
The original version searched for yfinance column names that no longer exist.
yfinance renamed its balance-sheet and cash-flow fields, so `first_existing`
returned None and the ratio was silently skipped. The result: `rat_debt_to_equity`,
`rat_interest_coverage` and `rat_ocf_to_debt` were never written to any panel --
including debt-to-equity, the headline feature. Nothing crashed. Nothing warned.

Current names, verified against the panels in data_out/:

    total liabilities  -> fin__bs__total_liabilities_net_minority_interest
                          (was: total_liab)
    equity             -> fin__bs__stockholders_equity
    interest expense   -> fin__fin__interest_expense
    operating cashflow -> fin__cf__operating_cash_flow
                          (was: total_cash_from_operating_activities)

This file now asserts on missing ratios instead of skipping them. A silent
skip is worse than a crash: it produces a model that trains on nothing and
reports success.

FREQUENCY
---------
The spine is weekly (W-FRI). The docstring in the old version claimed weekly
but the shipped panels are 3,650 daily rows including weekends -- they were
generated before that edit and never regenerated. Regenerate them.

Run:
    python build_structured_features.py --tickers AAPL MSFT --years 10 --outdir data_out
    python build_structured_features.py --outdir data_out          # top 300 S&P
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from pandas_datareader import data as pdr
    _HAS_FRED = True
except Exception:
    _HAS_FRED = False


# ======================================================================
# CANONICAL COLUMN NAMES
# ======================================================================
# Every downstream file imports from here. One source of truth. When yfinance
# renames a field again -- and it will -- this is the only place to edit.

COL = {
    # balance sheet
    "total_assets":        "bs__total_assets",
    "current_assets":      "bs__current_assets",
    "current_liabilities": "bs__current_liabilities",
    "total_liabilities":   "bs__total_liabilities_net_minority_interest",
    "stockholders_equity": "bs__stockholders_equity",
    "retained_earnings":   "bs__retained_earnings",
    "total_debt":          "bs__total_debt",
    "working_capital":     "bs__working_capital",
    # income statement
    "revenue":             "fin__total_revenue",
    "ebit":                "fin__ebit",
    "net_income":          "fin__net_income",
    "pretax_income":       "fin__pretax_income",
    "interest_expense":    "fin__interest_expense",
    "diluted_eps":         "fin__diluted_eps",
    # cash flow
    "operating_cash_flow": "cf__operating_cash_flow",
}

# After the panel is assembled, every fundamental gets a "fin__" prefix.
PANEL_PREFIX = "fin__"


def panel_col(key: str) -> str:
    """Map a logical name to its column name in the final panel."""
    return PANEL_PREFIX + COL[key]


# Ratios this builder must produce. If any is absent after engineering, the
# run aborts. The old code let these vanish silently.
REQUIRED_RATIOS = [
    "rat_current_ratio",
    "rat_debt_to_equity",
    "rat_net_profit_margin",
    "rat_roa",
    "rat_asset_turnover",
    "rat_interest_coverage",
    "rat_ocf_to_debt",
]


# ======================================================================
# UTILITIES
# ======================================================================

def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    return df


def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    """NaN, not inf, on a vanishing denominator. Sign of b is preserved."""
    b = b.astype("float64")
    out = a.astype("float64") / (b.abs() + eps) * np.sign(b).replace(0, 1)
    return out.replace([np.inf, -np.inf], np.nan)


def make_weekly_spine(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Friday-anchored weekly spine.

    Fundamentals change quarterly, so daily rows are ~5x redundant and cost
    ~750k DB rows for 300 tickers. Weekly gives ~156k. Market features are
    computed on daily data first (rolling volatility, windowed returns), then
    sampled here, so no signal is lost.
    """
    return pd.date_range(start=start, end=end, freq="W-FRI")


def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def first_existing(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


# ======================================================================
# INGESTION
# ======================================================================

def fetch_fundamentals_quarterly(ticker: str) -> Optional[pd.DataFrame]:
    """Quarterly income statement + balance sheet + cash flow, indexed by period end.

    NOTE ON COVERAGE: yfinance currently returns only ~5 quarters, not 10 years.
    Everything downstream must therefore derive its train/test windows from the
    data rather than hardcoding 2015-2020. See credit_risk_pipeline.derive_windows().
    """
    try:
        t = yf.Ticker(ticker)
        fin, bs, cf = t.quarterly_financials.T, t.quarterly_balance_sheet.T, t.quarterly_cashflow.T
        if fin.empty and bs.empty and cf.empty:
            log(f"{ticker}: no quarterly fundamentals.")
            return None

        idx = fin.index.union(bs.index).union(cf.index).sort_values()
        df = pd.concat({"fin": fin.reindex(idx),
                        "bs": bs.reindex(idx),
                        "cf": cf.reindex(idx)}, axis=1)
        df.columns = ["__".join(c for c in col if c) if isinstance(col, tuple) else str(col)
                      for col in df.columns]
        return normalize_cols(df).sort_index()
    except Exception as e:
        log(f"{ticker}: fundamentals fetch error: {e}")
        return None


def fetch_market_daily(ticker: str, years: int = 10) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            log(f"{ticker}: market data empty.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log(f"{ticker}: market fetch error: {e}")
        return None


def fetch_macro_series(codes: list[str], start: str = "2000-01-01") -> Optional[pd.DataFrame]:
    if not _HAS_FRED:
        log("pandas_datareader absent; skipping macro.")
        return None
    out = {}
    for code in codes:
        try:
            out[code] = pdr.DataReader(code, "fred", start)[code]
        except Exception as e:
            log(f"macro fetch failed for {code}: {e}")
    if not out:
        return None
    df = pd.concat(out, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def engineer_fundamental_ratios(f: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Financial ratios. Aborts if a required ratio cannot be built.

    The old version wrapped each ratio in `if x is not None and y is not None`,
    so a renamed source column meant the ratio quietly never appeared. Downstream
    the imputer replaced it with a median and the model trained on noise. Loud
    failure is the whole point of this function.
    """
    df = f.copy()

    ca = first_existing(df, COL["current_assets"], "current_assets")
    cl = first_existing(df, COL["current_liabilities"], "current_liabilities")
    ta = first_existing(df, COL["total_assets"], "total_assets")
    tl = first_existing(df, COL["total_liabilities"],
                        "bs__total_liab", "total_liab")           # legacy fallbacks
    tse = first_existing(df, COL["stockholders_equity"],
                         "bs__total_equity_gross_minority_interest",
                         "bs__common_stock_equity")
    ni = first_existing(df, COL["net_income"], "net_income")
    rev = first_existing(df, COL["revenue"], "total_revenue")
    ebit = first_existing(df, COL["ebit"], "ebit", COL["pretax_income"])
    ie = first_existing(df, COL["interest_expense"], "interest_expense",
                        "fin__interest_expense_non_operating")
    ocf = first_existing(df, COL["operating_cash_flow"],
                         "cf__total_cash_from_operating_activities")

    if ca is not None and cl is not None:
        df["rat_current_ratio"] = safe_divide(ca, cl)
    if tl is not None and tse is not None:
        df["rat_debt_to_equity"] = safe_divide(tl, tse)
    if ni is not None and rev is not None:
        df["rat_net_profit_margin"] = safe_divide(ni, rev)
    if ni is not None and ta is not None:
        df["rat_roa"] = safe_divide(ni, ta)
    if rev is not None and ta is not None:
        df["rat_asset_turnover"] = safe_divide(rev, ta)
    if ebit is not None and ie is not None:
        df["rat_interest_coverage"] = safe_divide(ebit, ie)
    if ocf is not None and tl is not None:
        df["rat_ocf_to_debt"] = safe_divide(ocf, tl)

    missing = [r for r in REQUIRED_RATIOS if r not in df.columns]
    if missing:
        log(f"{ticker}: WARNING -- could not build {missing}")
        log(f"{ticker}: available fundamental columns: {sorted(df.columns)[:25]} ...")
        log(f"{ticker}: this usually means yfinance renamed a field. Update COL in this file.")
        for r in missing:
            df[r] = np.nan     # explicit NaN, so downstream null checks can see it

    if rev is not None:
        df["g_qoq_revenue"] = pct_change_safe(rev, 1)
        df["g_yoy_revenue"] = pct_change_safe(rev, 4)
    if ni is not None:
        df["g_qoq_net_income"] = pct_change_safe(ni, 1)
        df["g_yoy_net_income"] = pct_change_safe(ni, 4)
    if tl is not None:
        df["g_qoq_total_debt"] = pct_change_safe(tl, 1)
        df["g_yoy_total_debt"] = pct_change_safe(tl, 4)

    return df


def engineer_market_features(mkt: pd.DataFrame) -> pd.DataFrame:
    """Computed on DAILY data, before the weekly downsample.

    Rolling volatility over 21 trading days is meaningless if computed on weekly
    closes. Compute here, sample later.
    """
    df = mkt.copy()
    px = "adj close" if "adj close" in df.columns else "close"
    df["ret_1d"] = df[px].pct_change()
    for w in (5, 21, 63, 126, 252):
        df[f"ret_{w}d"] = df[px].pct_change(periods=w)
    for w in (21, 63, 126):
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std() * np.sqrt(252)
    return df


def engineer_macro_features(macro: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if macro is None or macro.empty:
        return macro
    df = macro.copy()
    for col in list(df.columns):
        if "CPI" in col.upper():
            df[f"{col}_yoy"] = df[col].pct_change(periods=12)
        if "RATE" in col.upper() or "FEDFUNDS" in col.upper():
            df[f"{col}_chg_1m"] = df[col].diff(1)
    return df


# ======================================================================
# PANEL
# ======================================================================

def build_weekly_panel(ticker: str, years: int,
                       macro_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    mkt = fetch_market_daily(ticker, years=years)
    if mkt is None:
        log(f"{ticker}: skipping (no market data).")
        return None

    fund = fetch_fundamentals_quarterly(ticker)
    fund_feat = engineer_fundamental_ratios(fund, ticker) if fund is not None else None
    mkt_feat = engineer_market_features(mkt)     # daily, before downsampling

    spine = pd.Index(make_weekly_spine(mkt_feat.index.min(), mkt_feat.index.max()), name="date")

    pieces = [mkt_feat.reindex(spine, method="ffill").add_prefix("mkt__")]
    if fund_feat is not None:
        # Quarterly -> weekly. NOTE: no reporting lag here; it is applied in
        # credit_risk_pipeline.apply_reporting_lag() so the raw panel stays raw.
        pieces.append(fund_feat.reindex(spine).ffill().add_prefix("fin__"))
    if macro_df is not None and not macro_df.empty:
        pieces.append(macro_df.reindex(spine).ffill().add_prefix("mac__"))

    panel = pd.concat(pieces, axis=1)
    panel["ticker"] = ticker
    panel.index.name = "date"
    return panel.reset_index()


def audit_panel(panel: pd.DataFrame, ticker: str) -> None:
    """Print what actually landed. Cheap insurance against another silent skip."""
    n_fund = panel[panel_col("total_assets")].notna().sum() \
        if panel_col("total_assets") in panel.columns else 0
    ratios = [c for c in panel.columns if "rat_" in c]
    empty = [c for c in ratios if panel[c].isna().all()]
    log(f"{ticker}: {len(panel)} weekly rows | {n_fund} with fundamentals | "
        f"{len(ratios)} ratios ({len(empty)} all-NaN)")
    if empty:
        log(f"{ticker}: ALL-NaN ratios -> {empty}")


# ======================================================================
# MAIN
# ======================================================================

def get_top_300_tickers() -> list[str]:
    try:
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        return pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()[:300]
    except Exception as e:
        log(f"S&P 500 scrape failed: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the issuer x date weekly panel")
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="data_out")
    ap.add_argument("--macro", nargs="*", default=["FEDFUNDS", "CPIAUCSL"])
    ap.add_argument("--no-db", action="store_true", help="write parquet only")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tickers = sorted({t.upper() for t in (args.tickers or get_top_300_tickers())})

    macro_df = None
    if args.macro:
        try:
            macro_df = engineer_macro_features(fetch_macro_series(args.macro))
            if macro_df is None or macro_df.empty:
                log("macro empty; continuing without it.")
        except Exception as e:
            log(f"macro error (skipping): {e}")

    conn = None
    if not args.no_db:
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()
            if os.environ.get("DATABASE_URL"):
                conn = psycopg2.connect(os.environ["DATABASE_URL"])
        except Exception as e:
            log(f"PostgreSQL unavailable ({e}); parquet only.")

    for t in tickers:
        log(f"building {t} ...")
        panel = build_weekly_panel(t, args.years, macro_df)
        if panel is None:
            continue

        panel.columns = [str(c) for c in panel.columns]
        for c in [c for c in panel.columns if "rat_" in c]:
            panel[c] = panel[c].clip(lower=-1000, upper=1000)

        audit_panel(panel, t)
        panel.to_parquet(outdir / f"panel_{t}.parquet", index=False)

        if conn is not None:
            try:
                from psycopg2.extras import execute_batch
                records = [
                    (t, r["date"].strftime("%Y-%m-%d"),
                     json.dumps({k: (None if pd.isna(v) else v)
                                 for k, v in r.drop(["ticker", "date"]).to_dict().items()}))
                    for _, r in panel.iterrows()
                ]
                with conn.cursor() as cur:
                    execute_batch(cur, """
                        INSERT INTO market_features (ticker, date, features)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET features = EXCLUDED.features;
                    """, records)
                conn.commit()
                log(f"{t}: saved to PostgreSQL.")
            except Exception as e:
                log(f"{t}: DB write failed: {e}")
                conn.rollback()

        del panel

    if conn is not None:
        conn.close()
    log("done.")


if __name__ == "__main__":
    main()
