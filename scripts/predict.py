import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from datetime import datetime
from pathlib import Path
import os

# =========================================================
# ✅ predict.py
# 从 gold feature store 读取特征数据
# 加载已训练模型 + scaler，生成每月预测结果
# =========================================================

default_datamart = (
    Path(__file__).resolve().parents[1] / "datamart/gold/feature_store"
    if os.name == "nt"
    else Path("/opt/airflow/datamart")
)
FEATURE_DIR = Path(os.getenv("DATA_DIR", default_datamart))

PRED_DIR    = "./datamart/gold/predictions"

os.makedirs(PRED_DIR, exist_ok=True)

# 优先环境变量，否则尝试本地项目结构
default_model_path = (
    Path(__file__).resolve().parents[1] / "model_bank"
    if os.name == "nt"  # Windows
    else Path("/opt/airflow/model_bank")
)
MODEL_BANK_DIR = Path(os.getenv("MODEL_BANK_DIR", default_model_path))


def _mb(*parts) -> str:
    """拼出 model_bank 下的绝对路径，并做存在性检查给出友好报错。"""
    p = (MODEL_BANK_DIR.joinpath(*parts)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"missing artifact: {p} "
                                f"(MODEL_BANK_DIR={MODEL_BANK_DIR.as_posix()})")
    return p.as_posix()


# =========================================================
# 🔹 工具函数
# =========================================================
def norm_id(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()

def to_ymd_str(s):
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.strftime("%Y-%m-%d")

# =========================================================
# 🔹 主函数
# =========================================================
def main(snapshotdate: str):
    print(f"\n🚀 Running model inference for {snapshotdate}")

    # === 1️⃣ 加载模型和 scaler ===
    scaler_path = _mb("v1", "scaler.joblib")
    model_path = _mb("v1", "logreg_model.joblib")

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    print("✅ 模型与Scaler加载完毕")

    # === 2️⃣ 读取特征表（3个子表）===
    attr_path = os.path.join(FEATURE_DIR, "attributes_feature.parquet")
    clk_path  = os.path.join(FEATURE_DIR, "clickstream_features.parquet")
    fin_path  = os.path.join(FEATURE_DIR, "financial_feature.parquet")

    attributes = pd.read_parquet(attr_path)
    clickstream = pd.read_parquet(clk_path)
    financial  = pd.read_parquet(fin_path)

    # inner join（按 Customer_ID + snapshot_date）
    features = attributes.merge(clickstream, on=["Customer_ID", "snapshot_date"], how="inner")
    features = features.merge(financial,  on=["Customer_ID", "snapshot_date"], how="inner")

    features["Customer_ID"]   = features["Customer_ID"].map(norm_id)
    features["snapshot_date"] = to_ymd_str(features["snapshot_date"])

    print(f"✅ 特征表合并完成: {features.shape}")

    # === 3️⃣ 过滤当前月份 ===
    features = features[features["snapshot_date"] == snapshotdate]
    if features.empty:
        raise ValueError(f"⚠️ 没有找到 snapshot_date={snapshotdate} 的特征数据")

    print(f"📅 本次预测样本数: {len(features)}")

    # === 4️⃣ 模型推理 ===
    X = features.drop(columns=["Customer_ID", "snapshot_date"], errors="ignore")

    # 缺失值处理：先填0
    X = X.fillna(0)

    # 特征对齐：防止缺少训练时的列
    if hasattr(scaler, "feature_names_in_"):
        missing_cols = [c for c in scaler.feature_names_in_ if c not in X.columns]
        if missing_cols:
            X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing_cols)], axis=1)
        X = X[scaler.feature_names_in_]
        print(f"🧩 已补齐缺失特征: {len(missing_cols)} 列")

    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    preds = (probs > 0.5).astype(int)

    # === 5️⃣ 保存预测结果 ===
    out_df = features[["Customer_ID", "snapshot_date"]].copy()
    out_df["churn_prob"] = probs
    out_df["churn_pred"] = preds

    out_path = os.path.join(PRED_DIR, f"gold_pred_{snapshotdate.replace('-', '_')}.parquet")
    out_df.to_parquet(out_path, index=False)

    print(f"✅ 预测完成并保存到: {out_path}")
    print(f"🔢 正样本预测率: {out_df['churn_pred'].mean():.4f}")

# =========================================================
# 🧭 命令行接口
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monthly churn prediction")
    parser.add_argument("--snapshotdate", type=str, required=True, help="预测月份 (YYYY-MM-DD 或 YYYY_MM_DD)")
    args = parser.parse_args()

    # 修正格式
    snapshotdate = args.snapshotdate.replace("_", "-")
    try:
        datetime.strptime(snapshotdate, "%Y-%m-%d")
    except ValueError:
        raise ValueError("❌ snapshotdate 格式应为 YYYY-MM-DD 或 YYYY_MM_DD")

    main(snapshotdate)
