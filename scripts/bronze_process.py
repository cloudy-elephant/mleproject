# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import sys
import pandas as pd
import os


ROOT = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
if not ROOT.exists():
    ROOT = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

DATAMART = ROOT / "datamart"
BRONZE   = DATAMART / "bronze"
BRONZE_DIR_LABEL   = BRONZE / "label_store"
BRONZE_DIR_FEATURE = BRONZE / "feature_store"
DATA_DIR          = ROOT / "data"
BRONZE_DIR_LABEL  = ROOT / "datamart" / "bronze" / "label_store"
BRONZE_DIR_FEATURE= ROOT / "datamart" / "bronze" / "feature_store"

LABEL_FILE = "label.csv"

demographicFEATURE_FILE = "demographic_df_noisy.csv"
financialFEATURE_FILE   = "financial_df_noisy.csv"
contractFEATURE_FILE    = "contract_df_noisy.csv"
serviceFEATURE_FILE     = "service_df_noisy.csv"




def log(msg: str):
    print(f"[bronze] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_label_file(data_dir: Path) -> Path:
    # for name in LABEL_FILE:
    name = LABEL_FILE
    f = data_dir / name
    if f.exists():
        return f
    raise FileNotFoundError(
        f"cannnot find {LABEL_FILE} ，make sure the path is：{data_dir}"
    )


def load_label_df(path: Path) -> pd.DataFrame:

    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.read_csv(path)


def parse_snapdate(df: pd.DataFrame) -> pd.DataFrame:
    if "snapdate" not in df.columns:
        raise KeyError("lable/label.csv missing column 'snapdate'")

    snap = pd.to_datetime(df["snapdate"], errors="coerce")
    bad = snap.isna().sum()
    if bad > 0:
        log(f"warning：have {bad} rows snapdate parsing failed，skipped。")

    df = df.loc[~snap.isna()].copy()
    df["__snapdate_dt"] = snap.loc[~snap.isna()].dt.date  # 仅日期
    return df


def write_partitions(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    groups = df.groupby("__snapdate_dt")
    n_files = 0
    for d, g in groups:

        fname = f"lable_{d.year:04d}_{d.month:02d}_{d.day:02d}.csv"
        out_path = out_dir / fname

        g.drop(columns=["__snapdate_dt"], inplace=True)
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_files += 1
        log(f"output：{out_path}")
    log(f"output {n_files} files.")


def copy_feature_file(data_dir: Path, out_dir: Path):
    for FEATURE_FILE in [demographicFEATURE_FILE,
                         financialFEATURE_FILE,
                         contractFEATURE_FILE,
                         serviceFEATURE_FILE,
                         demographicFEATURE_FILE]:
        src = data_dir / FEATURE_FILE
        if not src.exists():
            log(f"Tip：cannot find {FEATURE_FILE}，skipped.")
            return
        ensure_dir(out_dir)
        dst = out_dir / FEATURE_FILE
        shutil.copy2(src, dst)
        log(f"feature.csv to：{dst}")

DATAMART = ROOT / "datamart"
def _safe_mkdir(p: Path):

    if p.exists() and not p.is_dir():
        # bak = p.with_name(p.name + f".{int(time.time())}.bak")
        p.replace("old_version")
        print(f"[bronze] ⚠ {p} is not a dir")
    p.mkdir(parents=True, exist_ok=True)
def bootstrap_dirs():

    for p in (BRONZE, BRONZE_DIR_LABEL, BRONZE_DIR_FEATURE):
        print(f"[bronze] ensure dir: {p}")
        # _safe_mkdir(p)
        p.mkdir(parents=True, exist_ok=True)
    print("[bronze] ✅ okay")


def main():
    # bootstrap_dirs()
    log(f"data catalogue：{DATA_DIR}")
    log(f"lable output catalogue：{BRONZE_DIR_LABEL}")
    log(f"feature output catalogue：{BRONZE_DIR_FEATURE}")
    ensure_dir(BRONZE_DIR_LABEL)
    ensure_dir(BRONZE_DIR_FEATURE)

    try:
        label_path = find_label_file(DATA_DIR)
        log(f"use label file：{label_path.name}")
    except FileNotFoundError as e:
        log(str(e))
        sys.exit(1)

    try:
        df = load_label_df(label_path)
    except Exception as e:
        log(f"load {label_path} fail：{e}")
        sys.exit(1)

    try:
        df = parse_snapdate(df)
    except Exception as e:
        log(f"parsing snapdate failed：{e}")
        sys.exit(1)

    if df.empty:
        log("data empty")
    else:
        write_partitions(df, BRONZE_DIR_LABEL)

    for dir_path in [DATA_DIR, BRONZE_DIR_FEATURE]:
        copy_feature_file(DATA_DIR, BRONZE_DIR_FEATURE)

    log("finish all。")


if __name__ == "__main__":
    main()
