# -*- coding: utf-8 -*-
"""
gold_process.py

功能：
从 silver/label_store 中读取分月的 label_xxx.csv，
与 silver/feature_store 中的 4 张特征表按 customerID 合并，
输出到 gold/feature_store。

输出文件名： gold_yyyy_mm_dd.csv
"""
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional
import pandas as pd
import os

# === 路径定义 ===
BASE_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject")

LABEL_DIR = BASE_DIR / "datamart" / "silver" / "lable_store"
FEATURE_DIR = BASE_DIR / "datamart" / "silver" / "feature_store"
OUT_DIR = BASE_DIR / "datamart" / "gold"

# 确保输出目录存在
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 四张特征表路径
feature_paths = {
    "contract": FEATURE_DIR / "contract_df_clean.csv",
    "service": FEATURE_DIR / "service_df_clean.csv",
    "demographic": FEATURE_DIR / "demographic_df_clean.csv",
    "financial": FEATURE_DIR / "financial_df_clean.csv",
}

# === 辅助函数 ===
def log(msg):
    print(f"[gold] {msg}")

def safe_read_csv(path: Path):
    if not path.exists():
        log(f"⚠️ 未找到文件: {path.name}，跳过。")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "customerID" not in df.columns:
        log(f"⚠️ {path.name} 缺少 customerID，跳过。")
        return pd.DataFrame()
    df["customerID"] = df["customerID"].astype(str).str.strip()
    return df

def drop_all_duplicate_customer_ids_in_gold(base_dir):

    base = Path(base_dir)
    gold_dir = base / "datamart" / "gold" / "feature_store"
    if not gold_dir.exists():
        gold_dir = base / "datamart" / "gold"
    if not gold_dir.exists():
        raise FileNotFoundError(f"未找到 gold 目录：{gold_dir}")

    report = []
    files = sorted(gold_dir.glob("gold_*.csv"))
    if not files:
        raise FileNotFoundError(f"目录中未找到 gold_*.csv：{gold_dir}")

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

        # 标准化ID
        ids = df["customerID"].astype(str).str.strip()
        dup_mask = ids.duplicated(keep=False)  # True 表示该ID在本文件出现≥2次
        dropped = int(dup_mask.sum())

        if dropped > 0:
            df_clean = df.loc[~dup_mask].copy()
            # 原地覆盖写回，避免Windows多空行
            df_clean.to_csv(f, index=False, encoding="utf-8-sig", lineterminator="\n")
            report.append({"file": f.name, "rows_before": len(df), "rows_after": len(df_clean),
                           "dropped": dropped, "note": "duplicates removed"})
        else:
            # 也统一写一下，保证换行格式一致（可注释）
            df.to_csv(f, index=False, encoding="utf-8-sig", lineterminator="\n")
            report.append({"file": f.name, "rows_before": len(df), "rows_after": len(df),
                           "dropped": 0, "note": "no duplicates"})

    # return report
def drop_unknown_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除任意列等值为以下脏值的行（大小写不敏感；前后空格/中划线等已归一）：
      unknown, unk, na, n/a, null, none, nan, -, —, 空串
    注意：是“等值”匹配，不会误伤 'unknownable' 之类。
    """
    if df.empty:
        return df

    # 统一到小写+去首尾空白+把中文/长破折号等归一成 '-'
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

    # 任意列命中上述“等值”脏值就判为坏行
    bad_mask = norm.apply(lambda col: col.isin(bad_tokens))
    bad_any = bad_mask.any(axis=1)

    return df.loc[~bad_any].copy()



def main():


    # === 加载所有特征表 ===
    log("加载 Silver 特征表...")
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
        log(f"❌ 未找到任何 lable_*.csv 文件，请检查目录：{LABEL_DIR}")
        exit(1)

    for label_path in sorted(LABEL_DIR.glob("lable_*.csv")):
        label_name = label_path.stem.replace("lable_", "")
        log(f"\n🟡 处理标签文件：{label_path.name}")

        # 读取 & 标准化
        label_df = pd.read_csv(label_path)
        if "customerID" not in label_df.columns:
            log(f"⚠️ {label_path.name} 缺少 customerID，跳过。")
            continue
        label_df["customerID"] = label_df["customerID"].astype(str).str.strip()

        # ① 先清 label 的 unknown 行
        merged = drop_unknown_rows(label_df)

        # ② 逐个特征表：先清 unknown，再按 ID 去重，最后合并
        for name, fdf in features.items():
            if fdf.empty:
                continue
            f = fdf.copy()
            if "customerID" not in f.columns:
                log(f"⚠️ 特征表 {name} 缺少 customerID，跳过该表。")
                continue
            f["customerID"] = f["customerID"].astype(str).str.strip()

            # 清 unknown 行
            f = drop_unknown_rows(f)
            # 避免一对多膨胀：相同 ID 只保留最后一条（或改 'first'）
            f = f.drop_duplicates(subset=["customerID"], keep="last")

            merged = merged.merge(f, on="customerID", how="inner")

        # ③ 合并完成再兜底清一次 unknown，并按 ID 去重
        merged = drop_unknown_rows(merged)
        merged = merged.drop_duplicates(subset=["customerID"], keep="last")

        out_path = OUT_DIR / f"gold_{label_name}.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        log(f"✅ 输出文件：{out_path}（{len(merged)} 行）")

    drop_all_duplicate_customer_ids_in_gold(BASE_DIR)

    log("\n🎉 Gold 层特征合并完成！")

if __name__ == "__main__":
    main()


