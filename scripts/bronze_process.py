# -*- coding: utf-8 -*-
"""
bronze_process.py
功能：
1) data 目录下按 snapdate 拆分 lable.csv（若无则用 label.csv），输出 lable_yyyy_mm_dd.csv 到 datamart/bronze
2) 将 feature.csv 原样复制到 datamart/bronze
"""

from pathlib import Path
import shutil
import sys
import pandas as pd
import os

# # --- 固定路径（按你的要求） ---
# DATA_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject\data")
# BRONZE_DIR_LABLE = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\lable_store")
# BRONZE_DIR_FEATURE = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store")
#
# # --- 文件名设定 ---
# LABEL_CANDIDATES = ["lable.csv", "label.csv"]  # 优先使用 lable.csv（你给的文件名）
# demographicFEATURE_FILE = "demographic_df_noisy.csv"
# financialFEATURE_FILE = "financial_df_noisy.csv"
# contractFEATURE_FILE = "contract_df_noisy.csv"
# serviceFEATURE_FILE = "service_df_noisy.csv"
# 容器优先的项目根目录：/opt/airflow；本机回退到你的目录


ROOT = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
if not ROOT.exists():
    ROOT = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

DATAMART = ROOT / "datamart"
BRONZE   = DATAMART / "bronze"
BRONZE_DIR_LABEL   = BRONZE / "label_store"
BRONZE_DIR_FEATURE = BRONZE / "feature_store"
# 路径
DATA_DIR          = ROOT / "data"
BRONZE_DIR_LABEL  = ROOT / "datamart" / "bronze" / "label_store"
BRONZE_DIR_FEATURE= ROOT / "datamart" / "bronze" / "feature_store"

# 仅使用正确拼写的文件名
LABEL_FILE = "label.csv"

# 其余特征文件
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
        f"未找到 {LABEL_FILE} 中任一文件，请确认位于：{data_dir}"
    )


def load_label_df(path: Path) -> pd.DataFrame:
    # 兼容常见编码
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 若均失败，回退默认
    return pd.read_csv(path)


def parse_snapdate(df: pd.DataFrame) -> pd.DataFrame:
    if "snapdate" not in df.columns:
        raise KeyError("lable/label.csv 中缺少列 'snapdate'")

    # 尝试解析日期；保留日期部分
    snap = pd.to_datetime(df["snapdate"], errors="coerce")
    bad = snap.isna().sum()
    if bad > 0:
        log(f"警告：有 {bad} 行 snapdate 解析失败，将被跳过。")

    df = df.loc[~snap.isna()].copy()
    df["__snapdate_dt"] = snap.loc[~snap.isna()].dt.date  # 仅日期
    return df


def write_partitions(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    # 按日期分组
    groups = df.groupby("__snapdate_dt")
    n_files = 0
    for d, g in groups:
        # 文件名使用 lable_yyyy_mm_dd.csv（按你的要求用 lable 前缀）
        fname = f"lable_{d.year:04d}_{d.month:02d}_{d.day:02d}.csv"
        out_path = out_dir / fname
        # 写出时去掉我们内部的辅助列
        g.drop(columns=["__snapdate_dt"], inplace=True)
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_files += 1
        log(f"已写出：{out_path}")
    log(f"共生成 {n_files} 个分片文件。")


def copy_feature_file(data_dir: Path, out_dir: Path):
    for FEATURE_FILE in [demographicFEATURE_FILE,
                         financialFEATURE_FILE,
                         contractFEATURE_FILE,
                         serviceFEATURE_FILE,
                         demographicFEATURE_FILE]:
        src = data_dir / FEATURE_FILE
        if not src.exists():
            log(f"提示：未找到 {FEATURE_FILE}，跳过复制。")
            return
        ensure_dir(out_dir)
        dst = out_dir / FEATURE_FILE
        shutil.copy2(src, dst)
        log(f"已复制 feature.csv 到：{dst}")

DATAMART = ROOT / "datamart"
def _safe_mkdir(p: Path):
    """
    如果路径已存在但不是目录（例如是文件/链接），先备份再创建目录。
    """
    if p.exists() and not p.is_dir():
        # bak = p.with_name(p.name + f".{int(time.time())}.bak")
        p.replace("old_version")  # 原地改名，避免跨分区问题
        print(f"[bronze] ⚠ {p} 不是目录")
    p.mkdir(parents=True, exist_ok=True)
def bootstrap_dirs():
    # 逐层保证是“目录”
    for p in (BRONZE, BRONZE_DIR_LABEL, BRONZE_DIR_FEATURE):
        print(f"[bronze] ensure dir: {p}")
        # _safe_mkdir(p)
        p.mkdir(parents=True, exist_ok=True)
    print("[bronze] ✅ 目录就绪")


def main():
    # bootstrap_dirs()
    log(f"数据目录：{DATA_DIR}")
    log(f"lable输出目录：{BRONZE_DIR_LABEL}")
    log(f"feature输出目录：{BRONZE_DIR_FEATURE}")
    ensure_dir(BRONZE_DIR_LABEL)
    ensure_dir(BRONZE_DIR_FEATURE)

    # 1) 处理 lable/label.csv
    try:
        label_path = find_label_file(DATA_DIR)
        log(f"使用标签文件：{label_path.name}")
    except FileNotFoundError as e:
        log(str(e))
        sys.exit(1)

    try:
        df = load_label_df(label_path)
    except Exception as e:
        log(f"读取 {label_path} 失败：{e}")
        sys.exit(1)

    try:
        df = parse_snapdate(df)
    except Exception as e:
        log(f"解析 snapdate 失败：{e}")
        sys.exit(1)

    if df.empty:
        log("解析后数据为空，未生成任何分片文件。")
    else:
        write_partitions(df, BRONZE_DIR_LABEL)

    # 2) 复制 feature.csv（不做任何处理）
    for dir_path in [DATA_DIR, BRONZE_DIR_FEATURE]:
        copy_feature_file(DATA_DIR, BRONZE_DIR_FEATURE)

    log("全部完成。")


if __name__ == "__main__":
    main()
