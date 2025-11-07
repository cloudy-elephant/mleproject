# -*- coding: utf-8 -*-

import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             accuracy_score, f1_score, precision_score, recall_score)

BASE_DIR = Path(
    os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))
).resolve()
if not BASE_DIR.exists():
    BASE_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

OUT_DIR = BASE_DIR / "model_bank" / "logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_mostly_numeric(series: pd.Series, thresh: float = 0.9) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    coerced = pd.to_numeric(s, errors="coerce")
    ratio = coerced.notna().mean()
    return ratio >= thresh

def to_numeric_array(X):

    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").to_numpy()
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()

def to_str_array(X):

    if isinstance(X, pd.DataFrame):
        return X.astype(str).to_numpy()
    return pd.DataFrame(X).astype(str).to_numpy()

def map_churn(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in {"1","yes","y","true","t"}: return 1
    if s in {"0","no","n","false","f"}: return 0
    try:
        f = float(s)
        return 1 if f >= 0.5 else 0
    except Exception:
        return np.nan

def _read_csv_any(p: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig","utf-8","gb18030","cp1252"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception as e:
            last = e
    raise last

def _extract_date_from_name(name: str):
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", name)
    if not m: return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"

def _normalize_month(s):
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def load_gold(gold_dir: Path) -> pd.DataFrame:
    gold = gold_dir
    if not gold.exists():
        alt = gold_dir / "feature_store"
        if alt.exists():
            gold = alt
        else:
            raise FileNotFoundError(f"catalogue missing：{gold_dir}")

    files = sorted(gold.glob("gold_*.csv"))
    if not files:
        raise FileNotFoundError(f"cannot find gold_*.csv：{gold}")

    parts = []
    for fp in files:
        df = _read_csv_any(fp)

        if "customerID" in df.columns:
            df["customerID"] = df["customerID"].astype(str).str.strip()
        else:
            continue

        if "snapdate" in df.columns:
            df["snapdate"] = pd.to_datetime(df["snapdate"], errors="coerce")
        else:
            d = _extract_date_from_name(fp.name)
            if d:
                df["snapdate"] = pd.Timestamp(d)
            else:
                df["snapdate"] = pd.NaT

        df["snapdate"] = _normalize_month(df["snapdate"])


        if "Churn" not in df.columns:
            ycols = [c for c in df.columns if c.lower() == "churn"]
            if ycols:
                df["Churn"] = df[ycols[0]]

        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map(map_churn).astype("Int64")
            df = df.dropna(subset=["Churn"]).copy()
            df["Churn"] = df["Churn"].astype(int)
        else:
            continue

        df = df.drop_duplicates(subset=["customerID","snapdate"], keep="last")
        parts.append(df)

    if not parts:
        raise RuntimeError("no available files with Churn and customerID in the gold directory.")

    all_df = pd.concat(parts, ignore_index=True)

    start, end = pd.Timestamp("2023-02-01"), pd.Timestamp("2024-04-01")
    all_df = all_df[(all_df["snapdate"] >= start) & (all_df["snapdate"] <= end)].copy()

    return all_df

def time_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("snapdate").copy()
    months = sorted(df["snapdate"].dt.to_period("M").unique())
    mon_n = 5 if len(months) >= 5 else max(1, len(months)//3)
    monitor_months = months[-mon_n:]
    remaining = months[:-mon_n]
    cut = int(len(remaining) * 0.75) if len(remaining) > 0 else 0
    train_months = remaining[:cut]
    test_months  = remaining[cut:]
    def assign(m):
        if m in monitor_months: return "monitor"
        if m in train_months:   return "train"
        return "test"
    df["split"] = df["snapdate"].dt.to_period("M").map(assign)
    return df

BAD_TOKENS = {"unknown","unk","na","n/a","null","none","nan","-","—",""}
def sanitize_tokens_to_nan(df_in: pd.DataFrame) -> pd.DataFrame:
    df2 = df_in.copy()
    protected = {"customerID","snapdate","split","Churn"}
    for c in df2.columns:
        if c in protected:
            continue
        s = (df2[c].astype(str).str.strip().str.lower()
                .str.replace("\u2014","-", regex=False)  # — → -
                .str.replace("\u2013","-", regex=False)  # – → -
                .str.replace(r"\s*/\s*","/", regex=True))# n / a → n/a
        df2.loc[s.isin(BAD_TOKENS), c] = np.nan
    return df2

def fit_and_eval(df: pd.DataFrame, outdir: Path):

    df = sanitize_tokens_to_nan(df)
    TARGET = "Churn"
    drop_cols = {"customerID", "snapdate", "split", TARGET}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]


    train_df = df[df["split"]=="train"].copy()
    test_df  = df[df["split"]=="test"].copy()
    mon_df   = df[df["split"]=="monitor"].copy()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc",  StandardScaler(with_mean=False))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh",  OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    # X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]
    # X_mon,   y_mon   = mon_df[feature_cols],   mon_df[TARGET]

    pipe.fit(X_train, y_train)
    model_path = OUT_DIR / "logreg_model.joblib"
    joblib.dump(pipe, model_path)
    print(f"✅ modle stored to：{model_path}")

    def evaluate(name, X, y, idx):
        res = {"N": int(len(y))}
        if len(y)==0 or y.nunique()<2:
            res.update({"AUC": None, "ACC": None, "F1": None, "Precision": None, "Recall": None, "CM": None})
            return res
        prob = pipe.predict_proba(X)[:,1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y, pred)
        res.update({
            "AUC": float(roc_auc_score(y, prob)),
            "ACC": float(accuracy_score(y, pred)),
            "F1":  float(f1_score(y, pred)),
            "Precision": float(precision_score(y, pred)),
            "Recall": float(recall_score(y, pred)),
            "CM": cm.tolist()
        })
        out = pd.DataFrame({"customerID": df.loc[idx, "customerID"],
                            "snapdate": df.loc[idx, "snapdate"],
                            "y": y.values, "p": prob})
        out.to_csv(outdir / f"preds_{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        print("\n" + "="*66)
        print(f"{name.upper()} Set val:")
        print(classification_report(y, pred, target_names=["No Churn","Churn"]))
        print("Accuracy:", round(res["ACC"],4))
        print("AUC-ROC:", round(res["AUC"],4))
        print("CF:", cm.tolist())
        tn, fp, fn, tp = cm.ravel()
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
        return res

    metrics = {
        "train":   evaluate("train",   X_train, y_train, X_train.index),
    #     "test":    evaluate("test",    X_test,  y_test,  X_test.index),
    #     "monitor": evaluate("monitor", X_mon,   y_mon,   X_mon.index),
    }

    oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"] if len(cat_cols) else None
    cat_feat_names = list(oh.get_feature_names_out(cat_cols)) if oh is not None else []
    feat_names = num_cols + cat_feat_names
    coef = pipe.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feat_names,
                            "coef": coef,
                            "odds_ratio": np.exp(coef)}).sort_values("odds_ratio", ascending=False)
    coef_df.to_csv(outdir / "logreg_coefficients.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    df.to_csv(outdir / "gold_merged_for_training.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics,
                   "split_sizes": {"train": int(len(train_df)), "test": int(len(test_df)), "monitor": int(len(mon_df))},
                   "n_features": {"numeric": len(num_cols), "categorical": len(cat_cols)}},
                  f, ensure_ascii=False, indent=2)
    return metrics


def _default_root() -> Path:
    root = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
    if not root.exists():
        root = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()
    return root

def _auto_gold_dir(root: Path) -> Path:
    candidates = [
        root / "datamart" / "gold" / "feature_store",
        root / "datamart" / "gold",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return candidates[0]

def _default_out_dir(root: Path) -> Path:
    return root / "model_bank" / "logreg"

def main():
    parser = argparse.ArgumentParser(description="Train churn model from GOLD monthly files.")
    parser.add_argument("--gold_dir", type=str, default=None,
                        help="GOLD catalogue path（defult：datamart/gold/feature_store or datamart/gold）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="output path（dedult：model_bank/logreg）")
    args = parser.parse_args()

    root = _default_root()

    gold_dir = Path(args.gold_dir) if args.gold_dir else _auto_gold_dir(root)
    out_dir  = Path(args.out_dir)  if args.out_dir  else _default_out_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logreg] ROOT={root}")
    print(f"[logreg] GOLD_DIR={gold_dir}")
    print(f"[logreg] OUT_DIR={out_dir}")

    df = load_gold(gold_dir)
    df = time_split(df)
    fit_and_eval(df, out_dir)

if __name__ == "__main__":
    main()


