# monitor.py
import os, glob, re
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, brier_score_loss
)
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 配置区 =========
# default_model_path = (
#     Path(__file__).resolve().parents[1] / "model_bank"
#     if os.name == "nt"  # Windows
#     else Path("/opt/airflow/model_bank")
# )
# MODEL_BANK_DIR = Path(os.getenv("MODEL_BANK_DIR", default_model_path))
from pathlib import Path
import os

is_windows = os.name == "nt"

# 项目根：本地=仓库根；容器=/opt/airflow
BASE_DIR = Path(__file__).resolve().parents[1] if is_windows else Path("/opt/airflow")

# 模型根目录（可用环境变量覆盖）
MODEL_BANK_DIR = Path(os.getenv("MODEL_BANK_DIR", BASE_DIR / "model_bank"))

# 版本号（默认 v1，可用环境变量覆盖）
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# datamart 根目录（可用环境变量覆盖）
DATAMART_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "datamart"))

# 子目录
LABEL_DIR   = DATAMART_DIR / "gold" / "label_store"
FEATURE_DIR = DATAMART_DIR / "gold" / "feature_store"
PRED_DIR    = DATAMART_DIR / "gold" / "predictions"
OUT_DIR     = DATAMART_DIR / "gold" / "monitoring"

def _mb(*parts) -> Path:
    """
    在 model_bank/<MODEL_VERSION>/ 下拼路径；若不存在，再回退到 model_bank 根下找。
    """
    # 首选：带版本目录
    p1 = (MODEL_BANK_DIR / MODEL_VERSION).joinpath(*parts)
    if p1.exists():
        return p1.resolve()
    # 退路：不带版本（兼容老文件）
    p2 = MODEL_BANK_DIR.joinpath(*parts)
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(
        f"missing artifact; tried: {p1.as_posix()} and {p2.as_posix()} "
        f"(MODEL_BANK_DIR={MODEL_BANK_DIR.as_posix()}, MODEL_VERSION={MODEL_VERSION})"
    )


THRESHOLD = 0.5

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
# ====================================


# ---------- 函数定义 ----------

def merge_features_and_labels(
    feature_dir: str,
    label_dir: str,
    start_date: str = "2024_07_01",
    end_date: str = "2024_12_01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合并 feature_store 下三张表与 label_store 下各月标签（inner join by Customer_ID）"""

    def norm_id(x):
        if pd.isna(x):
            return x
        return str(x).strip().upper()

    def to_ymd_str(s):
        s = pd.to_datetime(s, errors="coerce")
        return s.dt.strftime("%Y-%m-%d")

    # === 读取三张特征表 ===
    attr_path = os.path.join(feature_dir, "attributes_feature.parquet")
    clk_path  = os.path.join(feature_dir, "clickstream_features.parquet")
    fin_path  = os.path.join(feature_dir, "financial_feature.parquet")

    attributes  = pd.read_parquet(attr_path)
    clickstream = pd.read_parquet(clk_path)
    financial   = pd.read_parquet(fin_path)

    features = attributes.merge(clickstream, on=["Customer_ID", "snapshot_date"], how="inner")
    features = features.merge(financial,  on=["Customer_ID", "snapshot_date"], how="inner")
    print(f"✅ features merge 完成: {features.shape}")

    features["Customer_ID"]   = features["Customer_ID"].map(norm_id)
    features["snapshot_date"] = to_ymd_str(features["snapshot_date"])

    # === 遍历 label 文件 ===
    label_files = sorted(glob.glob(os.path.join(label_dir, "gold_label_store_*.parquet")))
    summary_rows, merged_list = [], []
    # start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date.replace("_", "-"))
    end_dt = pd.to_datetime(end_date.replace("_", "-"))

    for fpath in label_files:
        base = os.path.basename(fpath)
        print(f"\n📄 正在处理: {base}")
        m = re.search(r"(\d{4})_(\d{2})_(\d{2})", base)
        if not m:
            print("⚠️ 文件名无日期，跳过")
            continue
        snapshot_date = pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        if not (start_dt <= snapshot_date <= end_dt):
            print(f"⏭️ 跳过 {snapshot_date.date()}（不在时间窗内）")
            continue

        labels = pd.read_parquet(fpath)
        if "Customer_ID" not in labels.columns:
            print("⚠️ 缺少 Customer_ID，跳过")
            continue

        labels["Customer_ID"] = labels["Customer_ID"].map(norm_id)
        labels["snapshot_date"] = snapshot_date.strftime("%Y-%m-%d")

        merged = features.merge(labels, on="Customer_ID", how="inner")
        print(f"✅ merge 后行数: {merged.shape[0]}, 列数: {merged.shape[1]}")

        merged_list.append(merged)
        summary_rows.append({
            "file": base,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "merged_rows": merged.shape[0],
            "merged_cols": merged.shape[1],
        })

    if not merged_list:
        raise RuntimeError("❌ 没有任何 label 文件成功 merge，请检查路径或时间窗口。")

    merged_all_df = pd.concat(merged_list, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    print("\n📊 各文件 merge 后的结果：")
    print(summary_df)
    return merged_all_df, summary_df


def compute_monitor_metrics(
    merged_df: pd.DataFrame,
    prob_col: str = "churn_prob",
    label_col: str = "label",
    month_col: str = "snapshot_date"
) -> pd.DataFrame:
    """计算每月监控指标（AUC、F1、KS、PSI等）"""

    def ks_stat(y_true, y_prob):
        df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
        pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
        neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
        return np.max(np.abs(pos - neg))

    def psi(actual, expected, bins=10, eps=1e-6):
        cuts = np.quantile(expected, np.linspace(0, 1, bins+1))
        cuts[0], cuts[-1] = -np.inf, np.inf
        e_cnt = np.histogram(expected, bins=cuts)[0] / (len(expected)+eps)
        a_cnt = np.histogram(actual,   bins=cuts)[0] / (len(actual)+eps)
        return np.sum((a_cnt - e_cnt) * np.log((a_cnt + eps) / (e_cnt + eps)))

    merged_df = merged_df.copy()
    merged_df[month_col] = pd.to_datetime(merged_df[month_col], errors="coerce")
    merged_df["month_str"] = merged_df[month_col].dt.to_period("M").astype(str)

    metrics, base_prob = [], None
    for month, dfm in merged_df.groupby("month_str"):
        y = dfm[label_col].values
        p = dfm[prob_col].values
        yhat = (p > THRESHOLD).astype(int)
        if len(np.unique(y)) < 2:
            print(f"⚠️ {month} 标签全为同类，跳过指标计算。")
            continue
        m = {
            "month": month,
            "n": len(dfm),
            "pos": int(y.sum()),
            "auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, yhat),
            "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat),
            "ks": ks_stat(y, p),
            "brier": brier_score_loss(y, p),
            "pd_rate": p.mean(),
        }
        if base_prob is None:
            m["psi_vs_base"] = np.nan
            base_prob = p
        else:
            m["psi_vs_base"] = psi(p, base_prob)
        metrics.append(m)

    result = pd.DataFrame(metrics)
    print("✅ 已计算各月监控指标：")
    print(result)
    return result


def load_model_and_scaler(model_dir: str):
    # scaler_path = os.path.join(model_dir, "scaler.joblib")
    scaler_path = _mb("scaler.joblib")
    model_path = _mb("logreg_model.joblib")  # 按你的实际模型文件名
    # model_path  = os.path.join(model_dir, "logreg_model.joblib")
    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)
    return scaler, model


def plot_monitor_trends(mdf: pd.DataFrame, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for col in ["auc", "pr_auc", "pd_rate", "psi_vs_base"]:
        plt.figure(figsize=(6,4))
        plt.plot(mdf["month"], mdf[col], marker="o")
        plt.title(f"{col.upper()} Trend")
        plt.xlabel("Month")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_trend.png"), dpi=150)
        plt.close()
    print(f"📈 趋势图已保存到 {out_dir}")

def credit_history_to_months(text):
    """
    把 '19 Years and 9 Months' 转成总月份数（int）。
    """
    if pd.isna(text):
        return np.nan

    text = str(text)
    # 匹配两个数字（例如 19 和 9）
    match = re.findall(r'(\d+)', text)
    if len(match) >= 2:
        years = int(match[0])
        months = int(match[1])
    elif len(match) == 1:
        years = int(match[0])
        months = 0
    else:
        return np.nan

    return years * 12 + months

def drop_useless_columns_and_onehot_coding(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["Customer_ID", "Credit_History_Age", "Name", "SSN",
               "snapshot_date_x", "snapshot_date_y", "label_def", "loan_id"]

    cat_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in id_cols
    ]

    # 删除无关列
    df = df.drop(columns=["Name", "SSN", "snapshot_date_x", "label_def", "loan_id"], errors="ignore")

    # ✅ 保留 label 的 snapshot_date（_y）
    if "snapshot_date_y" in df.columns:
        df = df.rename(columns={"snapshot_date_y": "snapshot_date"})
    else:
        # 兜底：如果没有 snapshot_date_y，保留 x
        if "snapshot_date_x" in df.columns:
            df = df.rename(columns={"snapshot_date_x": "snapshot_date"})
        else:
            df["snapshot_date"] = np.nan

    # One-hot 编码
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df





# ---------- 主流程 ----------
def main():
    # Step 1. 加载模型
    scaler, model = load_model_and_scaler(MODEL_BANK_DIR)
    print("✅ 模型与Scaler加载完毕")

    # Step 2. 合并特征与标签
    merged_all_df, summary_df = merge_features_and_labels(FEATURE_DIR, LABEL_DIR)

    # print('---------------------')
    # print(merged_all_df.columns)

    merged_all_df["Credit_History_Age_months"] = merged_all_df["Credit_History_Age"].apply(credit_history_to_months)
    merged_all_df = merged_all_df.drop(columns=["Credit_History_Age"])

    merged_all_df = drop_useless_columns_and_onehot_coding(merged_all_df)
    # print('---------------------')
    # print(merged_all_df.shape)

    # Step 3. 模型推理
    # ====== 和训练时列名对齐（关键补丁）======
    # 1) 训练时的列顺序（scikit-learn 1.0+ 会带这个属性）
    if not hasattr(scaler, "feature_names_in_"):
        raise RuntimeError(
            "当前 scaler 没有 feature_names_in_ 属性。建议在训练时用 DataFrame 拟合，"
            "或把训练时的特征列保存为 feature_list.json 并在此读取。"
        )

    expected_cols = list(scaler.feature_names_in_)

    # 2) 预测用的特征（先把不需要的剔掉）
    # 你的代码原来是：feature_cols = [c for c in merged_all_df.columns if c not in ["Customer_ID", "label"]]
    # 这里改为：以实际 one-hot 后的所有列为候选，然后按 expected 对齐
    X = merged_all_df.drop(
        columns=[c for c in ["Customer_ID", "label", "snapshot_date"] if c in merged_all_df.columns],
        errors="ignore"
    )

    # 去掉重复列（有时 get_dummies 会生成重复名）
    X = X.loc[:, ~X.columns.duplicated()]

    # 3) 缺失的训练列补 0；多余的推理列删除
    missing = [c for c in expected_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_cols]

    if missing:
        # 一次性创建所有缺失列并拼接
        missing_df = pd.DataFrame(
            {c: np.zeros(len(X), dtype=float) for c in missing},
            index=X.index
        )
        X = pd.concat([X, missing_df], axis=1)

    if extra:
        # 丢掉训练时没见过的列（避免 transform 报错）
        X = X.drop(columns=extra)

    # 4) 严格按训练时顺序重排；并确保数值类型
    X = X[expected_cols].astype(float)

    X = X.fillna(0)
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[:, 1]
    merged_all_df["churn_prob"] = prob
    merged_all_df["churn_pred"] = (prob > THRESHOLD).astype("int8")
    print("✅ 已完成预测并写入 merged_all_df")

    # Step 4. 计算监控指标
    monitor_df = compute_monitor_metrics(merged_all_df)
    monitor_df.to_csv(os.path.join(OUT_DIR, "monitor_summary.csv"), index=False)
    print(f"✅ 监控结果已保存到 {OUT_DIR}/monitor_summary.csv")

    # Step 5. 绘制趋势图
    plot_monitor_trends(monitor_df, OUT_DIR)







if __name__ == "__main__":
    main()
