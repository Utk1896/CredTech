"""
credit_risk_pipeline.py
=======================
Cross-sectional credit deterioration model for large-cap equities.

WHAT THIS PREDICTS
------------------
Given a company's fundamentals and market signals as of week t, rank it against
its peers on the probability that its Altman Z-Score deteriorates into the worst
quintile of the cross-section over the following four quarters.

FOUR THINGS THE OLD VERSION GOT WRONG
-------------------------------------
1. THE TARGET WAS ALL ZEROS.
   The proxy was  (eps_growth < 0) & (interest_coverage < 1) & (net_income < 0).
   `rat_interest_coverage` never existed in the panels -- yfinance renamed the
   source column and the builder skipped the ratio silently -- so the middle
   clause compared NaN < 1.0, which is False for every row. Positive rate: 0.000.
   LightGBM trained on a single class, roc_auc_score raised, the bare `except`
   swallowed it, and predict_proba returned a constant. Every company on the
   dashboard received an identical score. Verify with --diagnose.

2. QUARTERLY FLOWS IN AN ANNUAL FORMULA.
   Altman calibrated Sales/TA and EBIT/TA on ANNUAL figures. yfinance returns
   quarterly. Dividing one quarter of sales by total assets understates those
   terms roughly 4x, dragging Z down by ~1.0-1.5 for every company. On the
   shipped panels, 83% of rows fell below the 1.81 "distress" threshold --
   Apple included. Flows are now annualized (trailing four quarters where
   available, else 4x the latest quarter).

3. HARDCODED 2015-2020 TRAIN WINDOW.
   yfinance currently returns ~5 quarters of fundamentals, all from 2025 onward.
   The train window contained no fundamental data whatsoever. Windows are now
   derived from the data (derive_windows) and the run aborts if either split is
   empty rather than proceeding on medians.

4. ABSOLUTE DISTRESS IS UNLEARNABLE ON THIS UNIVERSE.
   The S&P 500 is survivorship-filtered: index membership is conditional on not
   being in distress, and firms that fail are removed before they fail. Even with
   the annualization fixed, the threshold either fires on nearly everything or
   nearly nothing depending on sector mix -- banks, REITs and utilities have
   balance sheets Altman was never calibrated for. The default target is
   therefore RELATIVE: worst quintile of forward Z-change, ranked within each
   date. Base rate is exactly 20% by construction.

   Pass --target distress to reproduce the absolute label and see the diagnostics
   for yourself. That comparison is the honest thing to show an interviewer.

METHODOLOGICAL COMMITMENTS
--------------------------
- Fundamentals lagged REPORT_LAG_DAYS (45) for the filing delay. A Q4 balance
  sheet is not public on Dec 31. Market data is not lagged; it was observable.
- Z-Score input ratios excluded from features (asset_turnover == Sales/TA,
  roa ~ EBIT/TA). Feeding them back lets the model reconstruct the label.
- Two held-out test sets: seen firms / unseen years, and unseen firms / unseen
  years. Their difference estimates memorization of firm identity.
- Nothing is fit on test: imputer, cross-sectional medians, calibrator, all train.
- Metrics at company-quarter level. Weekly rows are ~13x redundant.
- A two-feature logistic baseline is always reported.
- N_SEEDS grouped splits, mean +/- std. One seed is noise.

USAGE
-----
    python credit_risk_pipeline.py --diagnose            # inspect data, no training
    python credit_risk_pipeline.py --report              # evaluation
    python credit_risk_pipeline.py --target distress     # the broken label, for comparison
    python credit_risk_pipeline.py                       # evaluate + score

    from credit_risk_pipeline import get_credit_scores
    scores = get_credit_scores()
    # {"AAPL": {"default_probability": 0.31, "risk_drivers": [...], "as_of": "..."}}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TargetMode = Literal["deterioration", "distress"]


# ======================================================================
# CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class Config:
    date_col: str = "date"
    ticker_col: str = "ticker"
    target_col: str = "target"

    target_mode: TargetMode = "deterioration"
    deterioration_quantile: float = 0.20
    distress_threshold: float = 1.81
    min_cross_section: int = 30       # dates with fewer peers cannot support a rank

    horizon_years: int = 1
    horizon_tolerance_days: int = 45
    report_lag_days: int = 45
    val_fraction: float = 0.20
    test_fraction: float = 0.35       # tail of the timeline reserved for test

    holdout_frac: float = 0.20
    n_seeds: int = 5

    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 31              # lowered: effective sample is ~7k, not 150k
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    random_state: int = 42

    model_dir: str = "models"
    top_k_drivers: int = 3


CFG = Config()

PANEL_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "worker", "data_out", "panel_*.parquet")

# Terms inside Altman Z, or close enough to reconstruct one.
Z_LEAKAGE = frozenset({
    "fin_rat_asset_turnover",   # == Sales / TA, literally a Z term
    "fin_rat_roa",              # == NI / TA ~ EBIT / TA
    "rat_asset_turnover", "rat_roa",
    "z_score", "z_score_fwd", "z_change", "z_change_rank",
})

META_COLS = frozenset({
    "date", "ticker", "year", "quarter",
    "target", "z_score", "z_score_fwd", "z_change", "z_change_rank",
    "credit_risk_score", "split",
})


# ======================================================================
# UTILITIES
# ======================================================================

def _log(m: str) -> None:
    print(f"[credit_pipeline] {m}", flush=True)


def safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-9) -> pd.Series:
    den = den.astype("float64")
    out = num.astype("float64") / (den.abs() + eps) * np.sign(den).replace(0, 1)
    return out.replace([np.inf, -np.inf], np.nan)


def _clean(c: Any) -> str:
    s = str(c)
    for a, b in [("(", ""), (")", ""), ("'", ""), (",", "_"), (" ", ""), ("__", "_")]:
        s = s.replace(a, b)
    return s.lower()


def _find(cols: set[str], *cands: str) -> str | None:
    for c in cands:
        if c in cols:
            return c
    for c in cands:                       # substring fallback
        hits = [x for x in cols if x.endswith(c)]
        if hits:
            return hits[0]
    return None


# ======================================================================
# STAGE 1 -- LOAD
# ======================================================================

def load_panel(db_url: str | None = None, panel_glob: str | None = None) -> pd.DataFrame:
    """PostgreSQL first, parquet fallback. Either way, one tidy frame."""
    db_url = db_url or os.environ.get("DATABASE_URL")
    data = pd.DataFrame()

    if db_url:
        try:
            import psycopg2
            with psycopg2.connect(db_url) as conn:
                raw = pd.read_sql("SELECT ticker, date, features FROM market_features", conn)
            if not raw.empty:
                parsed = [json.loads(x) if isinstance(x, str) else (x or {})
                          for x in raw["features"]]
                data = pd.concat([raw[["ticker", "date"]], pd.DataFrame(parsed)], axis=1)
                _log(f"loaded {len(data):,} rows from PostgreSQL")
        except Exception as e:
            _log(f"PostgreSQL unavailable ({e}); falling back to parquet")

    if data.empty:
        files = sorted(glob.glob(panel_glob or PANEL_GLOB))
        if not files:
            raise RuntimeError(f"no data in DB and no parquet at {panel_glob or PANEL_GLOB}")
        frames = []
        for f in files:
            df = pd.read_parquet(f)
            if "ticker" not in df.columns:
                df["ticker"] = os.path.basename(f).replace("panel_", "").split(".")[0]
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        _log(f"loaded {len(data):,} rows from {len(files)} parquet files")

    data.columns = [_clean(c) for c in data.columns]
    # Parquet stores datetime64[ms]; pandas date arithmetic produces [us].
    # merge_asof refuses to join across resolutions, so normalize once, here.
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    data = (data.dropna(subset=["date", "ticker"])
                .sort_values(["ticker", "date"])
                .reset_index(drop=True))
    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.to_period("Q")

    # 300 tickers x 3.6k rows x 170 float64 cols is ~1.5GB. float32 halves it and
    # costs nothing: these are financial ratios, not physics.
    f64 = data.select_dtypes("float64").columns
    if len(f64):
        data[f64] = data[f64].astype("float32")

    _log(f"{data.ticker.nunique()} tickers | {data.date.min():%Y-%m} to {data.date.max():%Y-%m}")
    _log(f"memory: {data.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return data


# ======================================================================
# STAGE 2 -- DIAGNOSTICS  (run this before trusting anything)
# ======================================================================

def diagnose(data: pd.DataFrame) -> None:
    """Everything that determines whether a downstream metric is real."""
    print("\n" + "=" * 78)
    print("DATA DIAGNOSTICS")
    print("=" * 78)

    cols = set(data.columns)
    ta = _find(cols, "fin_bs_total_assets")

    print(f"\n  rows           : {len(data):,}")
    print(f"  tickers        : {data.ticker.nunique()}")
    print(f"  market span    : {data.date.min():%Y-%m-%d} to {data.date.max():%Y-%m-%d}")

    if ta:
        fund = data[data[ta].notna()]
        if len(fund):
            print(f"  fundamentals   : {fund.date.min():%Y-%m-%d} to {fund.date.max():%Y-%m-%d}")
            per_tk = fund.groupby("ticker")[ta].nunique()
            print(f"  distinct quarters per ticker: median {per_tk.median():.0f}, max {per_tk.max()}")
            if per_tk.median() < 12:
                print("\n  !! yfinance is returning very few quarters of fundamentals.")
                print("  !! Any hardcoded 2015-2020 train window contains NO fundamental data.")
                print("  !! Windows must be derived from the data. This pipeline does that.")
        else:
            print("  fundamentals   : NONE FOUND")
    else:
        print("  fundamentals   : total_assets column absent")

    print("\n  ratio coverage:")
    ratios = sorted(c for c in data.columns if "rat_" in c)
    if not ratios:
        print("    NO RATIO COLUMNS AT ALL -- rerun build_structured_features.py")
    for r in ratios:
        nn = data[r].notna().mean()
        flag = "  <-- ALL NaN" if nn == 0 else ""
        print(f"    {r:38s} {nn:6.1%} non-null{flag}")

    for want in ["debt_to_equity", "interest_coverage", "ocf_to_debt"]:
        if not any(want in c for c in data.columns):
            print(f"    MISSING ENTIRELY: rat_{want}")
            print(f"      -> yfinance renamed the source column. Fix COL in build_structured_features.py")

    print("=" * 78 + "\n")


# ======================================================================
# STAGE 3 -- POINT-IN-TIME CORRECTNESS
# ======================================================================

def apply_reporting_lag(data: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Shift fundamentals forward by the filing delay.

    yfinance stamps a filing with the period END date (Dec 31), but a 10-Q is
    filed ~40 days later and a 10-K ~60. Joining to Dec 31 means the model reads
    numbers that were not public until mid-February. Market columns are not
    lagged; prices were observable in real time.

    This is the standard reason backtests beat live trading.
    """
    fin_cols = [c for c in data.columns
                if c.startswith(("fin_", "mac_")) and c not in META_COLS]
    if not fin_cols:
        _log("WARNING: no fundamental columns; reporting lag NOT applied.")
        return data

    frames = []
    for _, g in data.groupby("ticker", sort=False):
        g = g.sort_values("date")
        lagged = g[["date"] + fin_cols].copy()
        lagged["date"] = (lagged["date"] + pd.Timedelta(days=cfg.report_lag_days)) \
            .astype("datetime64[ns]")
        frames.append(pd.merge_asof(
            g.drop(columns=fin_cols).sort_values("date"),
            lagged.sort_values("date"),
            on="date", direction="backward",   # only what has already been filed
        ))

    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    _log(f"applied {cfg.report_lag_days}d reporting lag to {len(fin_cols)} columns")
    return out.reset_index(drop=True)


# ======================================================================
# STAGE 4 -- TARGET
# ======================================================================

def _annualize(flow: pd.Series, ticker: pd.Series) -> pd.Series:
    """Quarterly flow -> annual run-rate.

    Altman calibrated Sales/TA and EBIT/TA on ANNUAL figures. yfinance gives
    quarterly. Using them raw understates both terms ~4x and drags Z below the
    distress threshold for healthy companies -- on the shipped panels, 83% of
    rows including Apple's.

    Trailing four DISTINCT quarters where available; else 4x the latest. Because
    the weekly spine forward-fills, we deduplicate on value before summing,
    otherwise the same quarter is counted thirteen times.
    """
    out = pd.Series(np.nan, index=flow.index, dtype="float64")
    for _, idx in flow.groupby(ticker).groups.items():
        s = flow.loc[idx]
        changed = s.ne(s.shift())
        q = s[changed]                                 # one row per distinct quarter
        if len(q) >= 4:
            ttm = q.rolling(4).sum()
            out.loc[idx] = ttm.reindex(s.index, method="ffill")
        else:
            out.loc[idx] = s * 4.0
    return out


def compute_z_score(data: pd.DataFrame) -> pd.Series:
    """Altman Z (1968), less the market-value-of-equity term.

        Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 1.0(Sales/TA)

    The MVE/TL term is dropped deliberately: it is a market quantity, and the
    point of the exercise is to test whether OUR market features add information
    beyond fundamentals. A market term inside the label makes that test circular.

    EBIT and Sales are annualized. See _annualize().
    """
    cols = set(data.columns)

    def col(*names: str) -> pd.Series:
        n = _find(cols, *names)
        return data[n] if n else pd.Series(np.nan, index=data.index)

    ta = col("fin_bs_total_assets")
    ca = col("fin_bs_current_assets")
    cl = col("fin_bs_current_liabilities")
    re = col("fin_bs_retained_earnings")
    ebit_q = col("fin_fin_ebit", "fin_fin_operating_income", "fin_fin_pretax_income")
    sales_q = col("fin_fin_total_revenue")

    ebit = _annualize(ebit_q, data["ticker"])
    sales = _annualize(sales_q, data["ticker"])

    return (1.2 * safe_divide(ca - cl, ta)
            + 1.4 * safe_divide(re, ta)
            + 3.3 * safe_divide(ebit, ta)
            + 1.0 * safe_divide(sales, ta))


def _attach_forward_z(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Match each row to the Z observed one year later, ON DATE.

    Not shift(-52): that assumes exactly 52 rows per ticker-year. Real spines have
    gaps (halts, mid-sample IPOs, delistings), so a positional shift silently pairs
    week t with a label 11 or 14 months out and never warns you. merge_asof on the
    date column is gap-invariant; the tolerance caps how far forward it may reach
    before giving up and returning NaN.
    """
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    data["z_score"] = compute_z_score(data)

    fut = data[["ticker", "date", "z_score"]].copy()
    fut["date"] = (fut["date"] - pd.DateOffset(years=cfg.horizon_years)) \
        .astype("datetime64[ns]")
    fut = fut.rename(columns={"z_score": "z_score_fwd"})

    return pd.merge_asof(
        data.sort_values("date"), fut.sort_values("date"),
        on="date", by="ticker", direction="forward",
        tolerance=pd.Timedelta(days=cfg.horizon_tolerance_days),
    ).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_target(data: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    data = _attach_forward_z(data, cfg)

    if cfg.target_mode == "distress":
        data[cfg.target_col] = np.where(
            data["z_score_fwd"].isna(), np.nan,
            (data["z_score_fwd"] < cfg.distress_threshold).astype(float))
        _target_report(data, cfg, "distress")
        return data

    # Worst quantile of forward 1y Z-change, ranked within each date's cross-section.
    data["z_change"] = data["z_score_fwd"] - data["z_score"]

    counts = data.groupby("date")["z_change"].transform("count")
    rankable = counts >= cfg.min_cross_section

    data["z_change_rank"] = np.nan
    data.loc[rankable, "z_change_rank"] = (
        data.loc[rankable].groupby("date")["z_change"].rank(pct=True, method="average"))

    data[cfg.target_col] = np.where(
        data["z_change_rank"].isna(), np.nan,
        (data["z_change_rank"] <= cfg.deterioration_quantile).astype(float))

    _target_report(data, cfg, "deterioration")
    return data


def _target_report(data: pd.DataFrame, cfg: Config, mode: str) -> None:
    lab = data[cfg.target_col].notna()
    n = int(lab.sum())

    _log(f"target mode = {mode}")
    _log(f"labelled rows: {n:,} (unlabelled {len(data) - n:,}, kept for inference)")

    if n == 0:
        _log("!" * 74)
        _log("! ZERO LABELLED ROWS. Nothing can be trained.")
        _log("! Run --diagnose. The usual cause is that fundamentals span fewer")
        _log("! than horizon_years, so no row has a forward Z to match against.")
        _log("!" * 74)
        return

    rate = data.loc[lab, cfg.target_col].mean()
    ever = data.loc[lab].groupby("ticker")[cfg.target_col].max()
    _log(f"row-level positive rate: {rate:.4f}")
    _log(f"companies EVER positive: {int(ever.sum())} / {ever.size}")

    if rate == 0.0 or rate == 1.0:
        _log("!" * 74)
        _log(f"! DEGENERATE TARGET: positive rate is exactly {rate:.0f}.")
        _log("! The model will train on one class. AUC is undefined.")
        _log("! predict_proba will return a constant and every company will score")
        _log("! identically. This is what the ORIGINAL pipeline did, silently.")
        _log("!" * 74)
    elif mode == "distress" and (rate > 0.60 or rate < 0.02):
        _log("!" * 74)
        _log(f"! Absolute-distress rate is {rate:.1%}. This threshold is not")
        _log("! informative on this universe. Altman was calibrated on 1960s")
        _log("! manufacturers; the S&P 500 contains banks, REITs and utilities")
        _log("! whose balance sheets it was never meant to score. Use")
        _log("! --target deterioration.")
        _log("!" * 74)


# ======================================================================
# STAGE 5 -- FEATURES
# ======================================================================

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Own-history deterioration signals. Cross-sectional normalization happens
    later, in fit_sector_medians(), because it needs the train split.
    """
    data = data.sort_values(["ticker", "date"]).copy()
    g = data.groupby("ticker", sort=False)
    cols = set(data.columns)

    base_names = [_find(cols, "fin_rat_debt_to_equity"),
                  _find(cols, "fin_rat_interest_coverage"),
                  _find(cols, "mkt_vol_21d_mkt", "mkt_vol_21d")]
    base_names = [b for b in base_names if b]

    new = {}
    for base in base_names:
        roll = g[base].rolling(window=8, min_periods=4)
        mu = roll.mean().reset_index(level=0, drop=True)
        sd = roll.std().reset_index(level=0, drop=True)
        new[f"zroll_{base}"] = safe_divide(data[base] - mu, sd)
        new[f"chg4q_{base}"] = g[base].diff(4)

    if new:
        data = pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)
    return data.reset_index(drop=True)


def median_cols(data: pd.DataFrame) -> list[str]:
    cols = set(data.columns)
    return [c for c in [_find(cols, "fin_rat_debt_to_equity"),
                        _find(cols, "fin_rat_interest_coverage"),
                        _find(cols, "mkt_vol_21d_mkt", "mkt_vol_21d")] if c]


def fit_sector_medians(train: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    """Per-year cross-sectional medians, computed on TRAIN ONLY.

    Fitting these on the full panel leaks the distribution of future years and
    the presence of the held-out companies into the test set. Both are silent.
    Both inflate the score.
    """
    return {c: train.groupby("year")[c].median() for c in cols if c in train.columns}


def apply_sector_medians(df: pd.DataFrame, med: dict[str, pd.Series]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c, m in med.items():
        if c in df.columns:
            df[f"rel_{c}"] = df[c] - df["year"].map(m).fillna(m.median())
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns
             if c not in META_COLS and c not in Z_LEAKAGE
             and pd.api.types.is_numeric_dtype(df[c])
             and df[c].notna().any()]          # an all-NaN column teaches nothing
    dropped = sorted(set(df.columns) & Z_LEAKAGE)
    if dropped:
        _log(f"excluded {len(dropped)} Z-derived features: {dropped}")
    return sorted(feats)


# ======================================================================
# STAGE 6 -- SPLITS
# ======================================================================

@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test_time: pd.DataFrame      # seen companies,   unseen years
    test_entity: pd.DataFrame    # unseen companies, unseen years
    holdout_tickers: set[str]


def derive_windows(data: pd.DataFrame, cfg: Config = CFG) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Train/test cut from the DATA, not from a constant.

    The old pipeline hardcoded train=2015-2020, test=2021-2025. With yfinance
    now returning ~5 quarters of fundamentals, all from 2025, the train window
    was empty of fundamentals and the model learned from imputed medians. Deriving
    the cut means the code cannot silently train on nothing.
    """
    lab = data[data[cfg.target_col].notna()]
    if lab.empty:
        raise RuntimeError("no labelled rows; cannot derive train/test windows")

    dates = lab["date"].sort_values()
    cut = dates.quantile(1 - cfg.test_fraction)
    _log(f"derived split at {cut:%Y-%m-%d} "
         f"(train <= cut, test > cut; labelled span "
         f"{dates.min():%Y-%m} to {dates.max():%Y-%m})")
    return dates.min(), cut


def make_splits(data: pd.DataFrame, seed: int, cfg: Config = CFG) -> Splits:
    """
                       <= cut              > cut
        80% tickers  |  TRAIN + VAL   |  TEST_TIME
        20% tickers  |  (discarded)   |  TEST_ENTITY

    Holdout tickers are removed from training entirely, including their pre-cut
    rows. That costs data. It is the only way TEST_ENTITY means anything: a
    company seen in training is a company that can be memorized.
    """
    _, cut = derive_windows(data, cfg)
    lab = data[data[cfg.target_col].notna()].copy()

    tickers = np.array(sorted(lab["ticker"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(tickers)
    holdout = set(tickers[:max(1, int(len(tickers) * cfg.holdout_frac))])

    pre, post = lab["date"] <= cut, lab["date"] > cut
    is_hold = lab["ticker"].isin(holdout)

    trainval = lab[pre & ~is_hold].sort_values("date")
    v = int(len(trainval) * (1 - cfg.val_fraction))

    return Splits(
        train=trainval.iloc[:v],
        val=trainval.iloc[v:],              # chronological tail, never random
        test_time=lab[post & ~is_hold],
        test_entity=lab[post & is_hold],
        holdout_tickers=holdout,
    )


# ======================================================================
# STAGE 7 -- METRICS
# ======================================================================

def collapse_to_company_quarter(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """One row per (ticker, quarter).

    Weekly rows inside a quarter share identical fundamentals and an identical
    label. Scoring all of them counts each real observation ~13 times: the point
    estimate barely moves, but the apparent confidence interval shrinks by roughly
    sqrt(13). Both levels are reported; quote this one.
    """
    if df.empty:
        return df
    return df.sort_values("date").groupby(["ticker", "quarter"], as_index=False).last()


def evaluate(y: np.ndarray, p: np.ndarray, label: str) -> dict[str, float]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        _log(f"{label}: single class present; metrics undefined.")
        return {"auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "logloss": np.nan,
                "n": len(y), "pos_rate": float(y.mean()) if len(y) else np.nan}
    return {"auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "n": int(len(y)),
            "pos_rate": float(y.mean())}


# ======================================================================
# STAGE 8 -- MODELS
# ======================================================================

def fit_baseline(X_tr, y_tr, X_te) -> np.ndarray:
    """Two-feature logistic regression: leverage and interest coverage.

    The bar the booster must clear. If sixty engineered features and a gradient
    booster cannot beat two ratios and a linear model, the honest conclusion is
    that the engineering added nothing -- and reporting that is worth more than
    any amount of tuning.
    """
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(X_tr), y_tr)
    return lr.predict_proba(sc.transform(X_te))[:, 1]


def fit_lgbm(sp: Splits, feats: list[str], cfg: Config = CFG):
    """LightGBM + early stopping + class weighting + isotonic calibration.

    Raises on a single-class training set instead of continuing. The original
    code let LightGBM fit one class, let roc_auc_score raise, swallowed it in a
    bare except, and shipped a constant to the dashboard.

    The calibrator is fit on VALIDATION with cv='prefit' -- not on train, where
    the model has memorized and the map would be a no-op, and not on test, which
    is fitting on test.
    """
    import lightgbm as lgb

    if sp.train.empty or sp.val.empty:
        raise RuntimeError("empty train or validation split")

    ytr = sp.train[cfg.target_col].astype(int)
    yva = sp.val[cfg.target_col].astype(int)
    if ytr.nunique() < 2:
        raise RuntimeError(
            f"training target has a single class (positive rate {ytr.mean():.4f}). "
            "Run --diagnose. This is the failure mode the original pipeline hid.")
    if yva.nunique() < 2:
        raise RuntimeError("validation target has a single class; cannot early-stop or calibrate")

    imp = SimpleImputer(strategy="median").fit(sp.train[feats])
    Xtr, Xva = imp.transform(sp.train[feats]), imp.transform(sp.val[feats])

    n_pos = max(int(ytr.sum()), 1)
    spw = (len(ytr) - n_pos) / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample, subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        scale_pos_weight=spw, random_state=cfg.random_state, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])

    cal = _calibrate_prefit(model, Xva, yva)
    _log(f"trained {model.best_iteration_}/{cfg.n_estimators} trees, scale_pos_weight={spw:.2f}")
    return model, cal, imp


def _calibrate_prefit(model, Xva, yva):
    """Isotonic calibration of an already-fitted model, across sklearn versions.

    `CalibratedClassifierCV(cv="prefit")` was deprecated in sklearn 1.2 and REMOVED
    in 1.6 -- it now raises InvalidParameterError. Most tutorials (and the earlier
    version of this file) still use it. The replacement is FrozenEstimator.
    """
    try:                                     # sklearn >= 1.6
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(Xva, yva)
    except ImportError:                      # sklearn < 1.6
        return CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(Xva, yva)


# ======================================================================
# STAGE 9 -- EXPLAINABILITY
# ======================================================================

def shap_drivers(model, imp, df: pd.DataFrame, feats: list[str],
                 cfg: Config = CFG) -> dict[str, list[dict]]:
    """Top-k SHAP contributions per ticker, on its most recent row.

    Computed on the UNCALIBRATED model: isotonic is a monotone post-hoc map, so it
    preserves the ranking of contributions but breaks TreeExplainer's additivity
    guarantee. Explain the tree, then calibrate the output.
    """
    import shap

    latest = df.sort_values("date").groupby("ticker", as_index=False).last()
    vals = shap.TreeExplainer(model).shap_values(imp.transform(latest[feats]))
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    return {t: [{"feature": feats[j],
                 "impact": round(float(vals[i, j]), 4),
                 "direction": "increases risk" if vals[i, j] > 0 else "decreases risk"}
                for j in np.argsort(np.abs(vals[i]))[::-1][:cfg.top_k_drivers]]
            for i, t in enumerate(latest["ticker"])}


# ======================================================================
# STAGE 10 -- EVALUATION
# ======================================================================

def run_evaluation(data: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    rows = []
    mcols = median_cols(data)

    for seed in range(cfg.n_seeds):
        sp = make_splits(data, seed, cfg)
        med = fit_sector_medians(sp.train, mcols)
        sp = Splits(*[apply_sector_medians(d, med)
                      for d in (sp.train, sp.val, sp.test_time, sp.test_entity)],
                    sp.holdout_tickers)

        feats = select_features(sp.train)
        try:
            model, cal, imp = fit_lgbm(sp, feats, cfg)
        except RuntimeError as e:
            _log(f"seed {seed}: {e}")
            continue

        base = [f for f in feats if "debt_to_equity" in f or "interest_coverage" in f][:2]
        bidx = [feats.index(f) for f in base]

        for name, te in [("test_time", sp.test_time), ("test_entity", sp.test_entity)]:
            if te.empty:
                continue
            for level, frame in [("row", te), ("company_quarter", collapse_to_company_quarter(te, cfg))]:
                if frame.empty:
                    continue
                y = frame[cfg.target_col].astype(int).values
                X = imp.transform(frame[feats])

                m = evaluate(y, cal.predict_proba(X)[:, 1], f"{name}/{level}")
                m.update(seed=seed, split=name, level=level, model="lightgbm")
                rows.append(m)

                if base and level == "company_quarter":
                    pb = fit_baseline(imp.transform(sp.train[feats])[:, bidx],
                                      sp.train[cfg.target_col].astype(int), X[:, bidx])
                    mb = evaluate(y, pb, f"{name}/baseline")
                    mb.update(seed=seed, split=name, level=level, model="logreg_2feat")
                    rows.append(mb)

    return pd.DataFrame(rows)


def print_report(res: pd.DataFrame, cfg: Config = CFG) -> None:
    if res.empty:
        print("\nNo evaluation rows produced. Run --diagnose.\n")
        return

    print("\n" + "=" * 82)
    print(f"EVALUATION  --  target={cfg.target_mode}  --  mean +/- std over {cfg.n_seeds} seeds")
    print("=" * 82)

    agg = (res.groupby(["split", "level", "model"])
              .agg(auc=("auc", "mean"), sd=("auc", "std"), pr=("pr_auc", "mean"),
                   brier=("brier", "mean"), n=("n", "mean"), pos=("pos_rate", "mean"))
              .reset_index())

    for _, r in agg.iterrows():
        sd = 0.0 if pd.isna(r.sd) else r.sd
        print(f"  {r.split:<12} {r.level:<16} {r.model:<14} "
              f"AUC {r.auc:.3f} +/- {sd:.3f}   PR {r.pr:.3f}   "
              f"Brier {r.brier:.3f}   n={int(r.n):,}  pos={r.pos:.3f}")

    def get(split, model="lightgbm"):
        m = agg[(agg.split == split) & (agg.level == "company_quarter") & (agg.model == model)]
        return float(m.auc.iloc[0]) if len(m) else np.nan

    t, e, b = get("test_time"), get("test_entity"), get("test_entity", "logreg_2feat")

    print("\n" + "-" * 82)
    print("INTERPRETATION  --  quote these, not the row-level numbers")
    print("-" * 82)
    if not (np.isnan(t) or np.isnan(e)):
        print(f"  Memorization gap (time - entity) : {t - e:+.3f}")
        print("     large positive -> leans on firm identity, not general rules")
    if not (np.isnan(e) or np.isnan(b)):
        lift = e - b
        print(f"  Lift over 2-feature logistic     : {lift:+.3f}")
        if lift < 0.02:
            print("     -> engineering added little. Report this. It IS the finding.")
    print("=" * 82 + "\n")


# ======================================================================
# STAGE 11 -- PRODUCTION SCORING
# ======================================================================

def get_credit_scores(db_url: str | None = None, panel_glob: str | None = None,
                      cfg: Config = CFG) -> dict[str, dict]:
    """Train on all history; score every ticker's most recent week.

    Returns {ticker: {default_probability, risk_drivers, as_of, target}}.

    NOTE: the old signature returned {ticker: float}. run.py has been updated to
    read the dict. Both shapes are handled there for safety.

    Asymmetry with run_evaluation() is deliberate: here we train on the full
    universe, holdout included. The holdout exists to ESTIMATE generalization,
    not to be permanently sacrificed. The honest AUC has already been measured;
    there is no reason to leave data unused when producing live scores.
    """
    try:
        data = engineer_features(build_target(
            apply_reporting_lag(load_panel(db_url, panel_glob), cfg), cfg))
    except Exception as e:
        _log(f"data stage failed: {e}")
        return {}

    lab = data[data[cfg.target_col].notna()].sort_values("date")
    if lab.empty:
        _log("no labelled rows; cannot train. Run --diagnose.")
        return {}

    v = int(len(lab) * (1 - cfg.val_fraction))
    train, val = lab.iloc[:v], lab.iloc[v:]

    med = fit_sector_medians(train, median_cols(data))
    train, val = apply_sector_medians(train, med), apply_sector_medians(val, med)
    inference = apply_sector_medians(data, med)      # includes the unlabelled tail

    feats = select_features(train)
    try:
        model, cal, imp = fit_lgbm(Splits(train, val, pd.DataFrame(), pd.DataFrame(), set()),
                                   feats, cfg)
    except Exception as e:
        _log(f"training failed: {e}")
        return {}

    # The rows that matter: latest week per ticker. Their forward label is NaN by
    # construction -- the future has not happened yet. That is the point.
    latest = inference.sort_values("date").groupby("ticker", as_index=False).last()
    probs = cal.predict_proba(imp.transform(latest[feats]))[:, 1]

    try:
        drivers = shap_drivers(model, imp, inference, feats, cfg)
    except Exception as e:
        _log(f"SHAP failed ({e}); returning probabilities without drivers.")
        drivers = {}

    os.makedirs(cfg.model_dir, exist_ok=True)
    try:
        import joblib
        joblib.dump({"model": model, "calibrator": cal, "imputer": imp,
                     "features": feats, "sector_medians": med, "config": asdict(cfg)},
                    os.path.join(cfg.model_dir, f"credit_model_{datetime.now():%Y%m%d_%H%M}.joblib"))
    except Exception as e:
        _log(f"persistence failed: {e}")

    desc = ("1y forward credit deterioration (worst quintile of Z-Score change)"
            if cfg.target_mode == "deterioration"
            else f"1y forward Altman distress (Z < {cfg.distress_threshold})")

    out = {t: {"default_probability": round(float(p), 6),
               "as_of": str(latest.loc[latest.ticker == t, "date"].iloc[0].date()),
               "target": desc,
               "risk_drivers": drivers.get(t, [])}
           for t, p in zip(latest["ticker"], probs)}

    _log(f"scored {len(out)} tickers | prob range "
         f"[{min(v['default_probability'] for v in out.values()):.3f}, "
         f"{max(v['default_probability'] for v in out.values()):.3f}]")
    return out


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-sectional credit deterioration model")
    ap.add_argument("--target", choices=["deterioration", "distress"], default="deterioration",
                    help="'distress' reproduces the absolute label, for comparison")
    ap.add_argument("--diagnose", action="store_true", help="inspect the data and exit")
    ap.add_argument("--report", action="store_true", help="evaluation only, no scoring")
    ap.add_argument("--seeds", type=int, default=CFG.n_seeds)
    ap.add_argument("--panel-glob", type=str, default=None)
    args = ap.parse_args()

    cfg = Config(target_mode=args.target, n_seeds=args.seeds)
    raw = load_panel(panel_glob=args.panel_glob)

    if args.diagnose:
        diagnose(raw)
        d = build_target(apply_reporting_lag(raw, cfg), cfg)
        return

    data = engineer_features(build_target(apply_reporting_lag(raw, cfg), cfg))

    res = run_evaluation(data, cfg)
    print_report(res, cfg)
    if not res.empty:
        res.to_csv(f"evaluation_{cfg.target_mode}.csv", index=False)
        _log(f"wrote evaluation_{cfg.target_mode}.csv")

    if not args.report:
        scores = get_credit_scores(panel_glob=args.panel_glob, cfg=cfg)
        if scores:
            pd.DataFrame([{"ticker": t,
                           "default_probability": v["default_probability"],
                           "creditworthiness": round((1 - v["default_probability"]) * 100, 1),
                           "as_of": v["as_of"]}
                          for t, v in scores.items()]) \
              .sort_values("default_probability", ascending=False) \
              .to_csv("credit_scores_latest.csv", index=False)
            _log("wrote credit_scores_latest.csv")


if __name__ == "__main__":
    main()
