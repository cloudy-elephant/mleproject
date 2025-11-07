# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional
import pandas as pd
import os

# === Path ===
BASE_DIR = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
if not BASE_DIR.exists():
    BASE_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

LABEL_DIR   = BASE_DIR / "datamart" / "silver" / "label_store"
FEATURE_DIR = BASE_DIR / "datamart" / "silver" / "feature_store"
OUT_DIR     = BASE_DIR / "datamart" / "gold"

OUT_DIR.mkdir(parents=True, exist_ok=True)

feature_paths = {
    "contract":    FEATURE_DIR / "contract_df_clean.csv",
    "service":     FEATURE_DIR / "service_df_clean.csv",
    "demographic": FEATURE_DIR / "demographic_df_clean.csv",
    "financial":   FEATURE_DIR / "financial_df_clean.csv",
}

# === utility function ===
def log(msg):
    print(f"[gold] {msg}")

def safe_read_csv(path: Path):
    if not path.exists():
        log(f"⚠️ cannot find file: {path.name}，skipped。")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "customerID" not in df.columns:
        log(f"⚠️ {path.name} missing customerID，skipped。")
        return pd.DataFrame()
    df["customerID"] = df["customerID"].astype(str).str.strip()
    return df

def drop_all_duplicate_customer_ids_in_gold(base_dir):

    base = Path(base_dir)
    gold_dir = base / "datamart" / "gold" / "feature_store"
    if not gold_dir.exists():
        gold_dir = base / "datamart" / "gold"
    if not gold_dir.exists():
        raise FileNotFoundError(f"cannot find gold catalogue：{gold_dir}")

    report = []
    files = sorted(gold_dir.glob("gold_*.csv"))
    if not files:
        raise FileNotFoundError(f"cannot find gold_*.csv：{gold_dir}")

    def _read_csv_any(p: Path) -> pd.DataFrame:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "gb18030", "cp1252"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err

    for f in files:
        df = _read_csv_any(f)
        if "customerID" not in df.columns:
            report.append({"file": f.name, "rows_before": len(df), "rows_after": len(df),
                           "dropped": 0, "note": "no customerID col"})
            continue


        ids = df["customerID"].astype(str).str.strip()
        dup_mask = ids.duplicated(keep=False)
        dropped = int(dup_mask.sum())

        if dropped > 0:
            df_clean = df.loc[~dup_mask].copy()

            df_clean.to_csv(f, index=False, encoding="utf-8-sig", lineterminator="\n")
            report.append({"file": f.name, "rows_before": len(df), "rows_after": len(df_clean),
                           "dropped": dropped, "note": "duplicates removed"})
        else:

            df.to_csv(f, index=False, encoding="utf-8-sig", lineterminator="\n")
            report.append({"file": f.name, "rows_before": len(df), "rows_after": len(df),
                           "dropped": 0, "note": "no duplicates"})

    # return report
def drop_unknown_rows(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    norm = df.apply(
        lambda s: (s.astype(str)
                     .str.strip()
                     .str.lower()
                     .str.replace("\u2014", "-", regex=False)  # — → -
                     .str.replace("\u2013", "-", regex=False)  # – → -
                     .str.replace(r"\s*/\s*", "/", regex=True) # n / a → n/a
                  )
    )

    bad_tokens = {"unknown", "unk", "na", "n/a", "null", "none", "nan", "-", ""}

    bad_mask = norm.apply(lambda col: col.isin(bad_tokens))
    bad_any = bad_mask.any(axis=1)

    return df.loc[~bad_any].copy()



def main():



    log(" Silver features table...")
    features = {}
    for name, path in feature_paths.items():
        df = safe_read_csv(path)
        if not df.empty:
            features[name] = df
            log(f"✅ Loaded {name} ({len(df)} rows)")
        else:
            log(f"⚠️ Skip {name}")

    label_files = sorted(LABEL_DIR.glob("lable_*.csv"))
    if not label_files:
        log(f"❌ cannot find any lable_*.csv，check catalogue：{LABEL_DIR}")
        exit(1)

    for label_path in sorted(LABEL_DIR.glob("lable_*.csv")):
        label_name = label_path.stem.replace("lable_", "")
        log(f"\n🟡 handle label file：{label_path.name}")

        label_df = pd.read_csv(label_path)
        if "customerID" not in label_df.columns:
            log(f"⚠️ {label_path.name} missing customerID，skipped。")
            continue
        label_df["customerID"] = label_df["customerID"].astype(str).str.strip()


        merged = drop_unknown_rows(label_df)

        for name, fdf in features.items():
            if fdf.empty:
                continue
            f = fdf.copy()
            if "customerID" not in f.columns:
                log(f"⚠️ features table {name} missing customerID，skipped。")
                continue
            f["customerID"] = f["customerID"].astype(str).str.strip()

            f = drop_unknown_rows(f)

            f = f.drop_duplicates(subset=["customerID"], keep="last")

            merged = merged.merge(f, on="customerID", how="inner")

        merged = drop_unknown_rows(merged)
        merged = merged.drop_duplicates(subset=["customerID"], keep="last")

        out_path = OUT_DIR / f"gold_{label_name}.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        log(f"✅ output file：{out_path}（{len(merged)} rows）")

    drop_all_duplicate_customer_ids_in_gold(BASE_DIR)

    log("\n🎉 Gold meged！")

if __name__ == "__main__":
    main()


