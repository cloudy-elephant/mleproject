#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly monitoring for two models over Gold-layer monthly snapshots.

Part 1: Logistic Regression (joblib) under model_bank/logreg/*.joblib
Part 2: CatBoost (cbm) under model_bank/catboost/*.cbm

Data: datamart/gold/gold_YYYY_MM_DD.csv (inclusive range: 2024-05-01 .. 2024-09-01)

Usage (defaults work if your repo layout matches):
    python monitor.py

Optional environment overrides:
    DATA_DIR=/path/to/datamart  MODEL_BANK_DIR=/path/to/model_bank  python monitor.py

Outputs: prints per-month metrics for each model and a final summary table.
"""
from __future__ import annotations
import os
import sys
import re
import glob
import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(
    os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))
).resolve()
if not BASE_DIR.exists():
    BASE_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

# === data and model path ===
DATA_DIR = BASE_DIR / "datamart" / "gold"
MODEL_BANK_DIR = BASE_DIR / "model_bank"

LABEL_COL_CANDIDATES = ["label", "churn", "is_churn", "target"]
ID_COL_CANDIDATES = [
    "Customer_ID", "customerID", "customer_id",
    "CUSTOMER_ID", "cust_id"
]
SNAPSHOT_COL_CANDIDATES = ["snapshot_date", "snapdate", "SNAPSHOT_DATE"]

# Months to monitor (inclusive; filenames use YYYY_MM_DD)
START_DATE = date(2024, 5, 1)
END_DATE   = date(2024, 9, 1)

THRESHOLD = 0.5  # for converting prob -> class
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------------------------
# Utilities
# ----------------------------
_date_pat = re.compile(r"gold_(\d{4})_(\d{2})_(\d{2})\.csv$")

def parse_date_from_filename(p: str) -> Optional[date]:
    m = _date_pat.search(os.path.basename(p))
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    try:
        return date(y, mth, d)
    except ValueError:
        return None


def find_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: try a binary-ish column
    for c in df.columns:
        if set(pd.unique(df[c].dropna())) <= {0, 1} and df[c].dtype != object:
            return c
    return None


def find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Kolmogorov–Smirnov between positive and negative score distributions
    # Compute empirical CDFs at unique score points
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    probs_sorted = y_prob[order]
    # cumulative positives/negatives
    pos = (y_sorted == 1).astype(int)
    neg = (y_sorted == 0).astype(int)
    cum_pos = np.cumsum(pos) / max(1, pos.sum())
    cum_neg = np.cumsum(neg) / max(1, neg.sum())
    return float(np.max(np.abs(cum_pos - cum_neg)))


def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = THRESHOLD) -> Dict[str, float]:
    metrics = {}
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)

    # Degenerate cases
    unique = np.unique(y_true)
    if unique.size == 1:
        # Only accuracy/ll/brier on constant true labels make limited sense
        metrics.update({
            "auc": np.nan,
            "pr_auc": np.nan,
            "ks": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        })
    else:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc"] = np.nan
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["pr_auc"] = np.nan
        try:
            metrics["ks"] = ks_statistic(y_true, y_prob)
        except Exception:
            metrics["ks"] = np.nan
        y_pred = (y_prob >= thr).astype(int)
        try:
            metrics["f1"] = f1_score(y_true, y_pred)
        except Exception:
            metrics["f1"] = np.nan
        try:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics["precision"] = np.nan
        try:
            metrics["recall"] = recall_score(y_true, y_pred)
        except Exception:
            metrics["recall"] = np.nan

    # Always try these
    y_pred = (y_prob >= thr).astype(int)
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["accuracy"] = np.nan
    try:
        metrics["brier"] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics["brier"] = np.nan
    try:
        # log_loss needs probs clipped
        metrics["logloss"] = log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7))
    except Exception:
        metrics["logloss"] = np.nan

    metrics["pred_mean"] = float(np.mean(y_prob))
    metrics["pred_std"] = float(np.std(y_prob))
    metrics["positives"] = int(np.sum(y_true==1))
    metrics["negatives"] = int(np.sum(y_true==0))
    metrics["n"] = int(len(y_true))
    return metrics


def load_monthly_frames(data_dir: str, start: date, end: date) -> List[Tuple[date, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "gold_*.csv")))
    out: List[Tuple[date, pd.DataFrame]] = []
    for p in paths:
        d = parse_date_from_filename(p)
        if d is None:
            continue
        if start <= d <= end:
            df = pd.read_csv(p)
            out.append((d, df))
    # sort by date
    out.sort(key=lambda x: x[0])
    return out


# ----------------------------
# PART 1: Logistic Regression
# ----------------------------
from joblib import load as joblib_load

# Optional: detect sklearn Pipeline type (for readability only)
try:
    from sklearn.pipeline import Pipeline as _SkPipeline  # type: ignore
except Exception:
    _SkPipeline = tuple()  # falsy fallback

def run_logreg_monitor():
    print("\n========== PART 1: Logistic Regression monitor ==========")
    # Load model (you can keep multiple artifacts; we pick the first *.joblib)
    logreg_dir = os.path.join(MODEL_BANK_DIR, "logreg")
    joblib_files = sorted(glob.glob(os.path.join(logreg_dir, "*.joblib")))
    if not joblib_files:
        raise FileNotFoundError(f"No joblib model found in {logreg_dir}")
    # Prefer a file that looks like a classifier, otherwise first one
    model_path = None
    for f in joblib_files:
        if re.search(r"model|clf|logreg", os.path.basename(f), re.I):
            model_path = f
            break
    if model_path is None:
        model_path = joblib_files[0]

    model = joblib_load(model_path)
    # Try to get expected feature names if available (raw input schema at fit time)
    expected_cols: Optional[List[str]] = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

    # Load monthly
    monthly = load_monthly_frames(DATA_DIR, START_DATE, END_DATE)
    if not monthly:
        raise FileNotFoundError(f"No gold_YYYY_MM_DD.csv found within {START_DATE}..{END_DATE} under {DATA_DIR}")

    summary_rows = []
    for d, df in monthly:
        month_tag = d.strftime("%Y-%m-%d")
        # Identify columns
        label_col = find_label_col(df)
        if label_col is None:
            print(f"[LogReg][{month_tag}] WARNING: label column not found; skip")
            continue
        id_col = find_first(df, ID_COL_CANDIDATES)
        snap_col = find_first(df, SNAPSHOT_COL_CANDIDATES)

        y = df[label_col].astype(int).values

        # Build X (keep DataFrame to preserve column names for ColumnTransformer/Pipeline)
        drop_cols = set([c for c in [label_col, id_col, snap_col] if c is not None])
        if expected_cols is not None:
            # Align to training-time raw schema; add missing columns as NaN (let imputers/transformers handle them)
            Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            for c in expected_cols:
                if c not in Xdf.columns:
                    Xdf[c] = np.nan
            Xdf = Xdf[expected_cols].copy()
        else:
            # No schema available: pass everything except obvious non-features
            Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

        # Predict proba — prefer passing a DataFrame (pipelines expect column names)
        try:
            y_prob = model.predict_proba(Xdf)[:, 1]
        except Exception:
            # Fallbacks for atypical estimators
            if hasattr(model, "decision_function"):
                try:
                    scores = model.decision_function(Xdf)
                except Exception:
                    scores = model.decision_function(Xdf.values)
                y_prob = 1/(1+np.exp(-scores))
            else:
                try:
                    y_prob = model.predict(Xdf).astype(float)
                except Exception:
                    y_prob = model.predict(Xdf.values).astype(float)

        met = safe_metrics(y, y_prob, THRESHOLD)
        met_row = {"model": "logreg", "month": month_tag, **met}
        summary_rows.append(met_row)

        # Pretty print
        print(f"[LogReg][{month_tag}] n={met['n']} pos={met['positives']} neg={met['negatives']} pred_mean={met['pred_mean']:.4f}")
        print(
            "    auc={auc:.4f} pr_auc={pr_auc:.4f} ks={ks:.4f} f1={f1:.4f} "
            "prec={precision:.4f} rec={recall:.4f} acc={accuracy:.4f} "
            "brier={brier:.4f} logloss={logloss:.4f}".format(**{k:(v if not isinstance(v,float) or not math.isnan(v) else float('nan')) for k,v in met.items()})
        )

    return pd.DataFrame(summary_rows)


# ----------------------------
# PART 2: CatBoost
# ----------------------------
try:
    from catboost import CatBoostClassifier
    _HAS_CB = True
except Exception:
    _HAS_CB = False


def run_catboost_monitor():
    if not _HAS_CB:
        print("\n========== PART 2: CatBoost monitor ==========")
        print("CatBoost not installed; skipping CatBoost monitoring. Please install `catboost`.\n")
        return pd.DataFrame([])

    print("\n========== PART 2: CatBoost monitor ==========")
    cb_dir = os.path.join(MODEL_BANK_DIR, "catboost")
    cb_files = sorted(glob.glob(os.path.join(cb_dir, "*.cbm")))
    if not cb_files:
        raise FileNotFoundError(f"No CatBoost .cbm model found in {cb_dir}")

    cbm_path = cb_files[0]
    model = CatBoostClassifier()
    model.load_model(cbm_path)

    # Try to retrieve feature names from model if stored
    expected_cols: Optional[List[str]] = None
    try:
        if hasattr(model, "feature_names_") and model.feature_names_:
            expected_cols = list(model.feature_names_)
    except Exception:
        expected_cols = None

    monthly = load_monthly_frames(DATA_DIR, START_DATE, END_DATE)
    if not monthly:
        raise FileNotFoundError(f"No gold_YYYY_MM_DD.csv found within {START_DATE}..{END_DATE} under {DATA_DIR}")

    summary_rows = []
    for d, df in monthly:
        month_tag = d.strftime("%Y-%m-%d")
        label_col = find_label_col(df)
        if label_col is None:
            print(f"[CatBoost][{month_tag}] WARNING: label column not found; skip")
            continue
        id_col = find_first(df, ID_COL_CANDIDATES)
        snap_col = find_first(df, SNAPSHOT_COL_CANDIDATES)

        y = df[label_col].astype(int).values

        # CatBoost can handle categorical, but since we don't know training pool meta,
        # we'll keep numeric-only unless the model exposes feature_names
        drop_cols = set([c for c in [label_col, id_col, snap_col] if c is not None])
        Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        if expected_cols is not None:
            # Align using model's feature names
            missing = [c for c in expected_cols if c not in Xdf.columns]
            for c in missing:
                # fill numeric 0 or empty string; CatBoost will treat types accordingly
                Xdf[c] = 0
            # Ensure column order
            Xdf = Xdf[expected_cols]
        else:
            # Fallback to numeric only with sorted order
            Xdf = Xdf.select_dtypes(include=[np.number]).sort_index(axis=1)

        # Predict probabilities
        try:
            y_prob = model.predict_proba(Xdf)[:, 1]
        except Exception:
            # some CatBoost versions require numpy array
            y_prob = model.predict_proba(Xdf.values)[:, 1]

        met = safe_metrics(y, y_prob, THRESHOLD)
        met_row = {"model": "catboost", "month": month_tag, **met}
        summary_rows.append(met_row)

        print(f"[CatBoost][{month_tag}] n={met['n']} pos={met['positives']} neg={met['negatives']} pred_mean={met['pred_mean']:.4f}")
        print(
            "    auc={auc:.4f} pr_auc={pr_auc:.4f} ks={ks:.4f} f1={f1:.4f} "
            "prec={precision:.4f} rec={recall:.4f} acc={accuracy:.4f} "
            "brier={brier:.4f} logloss={logloss:.4f}".format(**{k:(v if not isinstance(v,float) or not math.isnan(v) else float('nan')) for k,v in met.items()})
        )

    return pd.DataFrame(summary_rows)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"DATA_DIR={DATA_DIR}")
    print(f"MODEL_BANK_DIR={MODEL_BANK_DIR}")
    print(f"Date range: {START_DATE} .. {END_DATE}\n")

    logreg_df = run_logreg_monitor()
    cb_df = run_catboost_monitor()

    # # Show summary tables if any
    # if not logreg_df.empty:
    #     print("\n==== LogReg monthly summary ====")
    #     print(logreg_df.to_string(index=False))
    # if not cb_df.empty:
    #     print("\n==== CatBoost monthly summary ====")
    #     print(cb_df.to_string(index=False))
    #
    # # Combined summary
    # all_df = pd.concat([logreg_df, cb_df], axis=0, ignore_index=True)
    # if not all_df.empty:
    #     out_path = os.path.join(DATA_DIR, "monitor_summary_2024-05_to_2024-09.csv")
    #     all_df.to_csv(out_path, index=False)
    #     print(f"\nSaved combined summary to: {out_path}")
    # else:
    #     print("\nNo metrics produced (no data or labels found in the date range).")
