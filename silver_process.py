# -*- coding: utf-8 -*-
# """
# silver_process.py
#
# 功能：
# - 从 bronze/feature_store 读取四个 noisy 表（只清洗 demographic_df_noisy 与 financial_df_noisy）
# - 输出到 silver/feature_store 目录（自动创建）
# - 保留审计字段（_raw、_was_negative、_was_outlier_... 等），便于回溯
#
# 来源路径（固定）：
# C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store\contract_df_noisy.csv
# C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store\demographic_df_noisy.csv
# C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store\financial_df_noisy.csv
# C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store\service_df_noisy.csv
#
# 目标路径（固定）：
# C:\Users\HP\Desktop\MLE\mleproject\datamart\silver\feature_store\demographic_df_clean.csv
# C:\Users\HP\Desktop\MLE\mleproject\datamart\silver\feature_store\financial_df_clean.csv
# """

from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import sys
import pandas as pd
import re


# 可选依赖（模糊匹配）。没有 rapidfuzz 时退化到 difflib。
try:
    from rapidfuzz import process as rf_process
    def fuzzy_best(val, choices):
        match, score, _ = rf_process.extractOne(val, choices)
        return match, score
except Exception:
    from difflib import SequenceMatcher
    def fuzzy_best(val, choices):
        best, best_score = None, -1.0
        for c in choices:
            s = SequenceMatcher(None, val, c).ratio()*100
            if s > best_score:
                best, best_score = c, s
        return best, best_score

# --------- 常量路径（按你的要求固定） ---------
BRONZE_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\feature_store")
SILVER_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\silver\feature_store")

PATHS = {
    "contract":    BRONZE_DIR / "contract_df_noisy.csv",
    "demographic": BRONZE_DIR / "demographic_df_noisy.csv",
    "financial":   BRONZE_DIR / "financial_df_noisy.csv",
    "service":     BRONZE_DIR / "service_df_noisy.csv",
}

OUT_FILES = {
    "demographic": SILVER_DIR / "demographic_df_clean.csv",
    "financial":   SILVER_DIR / "financial_df_clean.csv",
    "contract":    SILVER_DIR / "contract_df_clean.csv",
    "service":     SILVER_DIR / "service_df_clean.csv",
}


def log(msg: str): print(f"[silver] {msg}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_any(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if "customerID" in df.columns:
        df["customerID"] = df["customerID"].astype(str).str.strip()
    if "snapdate" in df.columns:
        df["snapdate"] = pd.to_datetime(df["snapdate"], errors="coerce")
    return df

# ------------------- Demographic 清洗 -------------------
def clean_demographic(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        basic_map = {
            # female 常见脏写
            "female":"female","f":"female","femail":"female","femole":"female",
            "cemale":"female","femade":"female","femcle":"female","fomale":"female",
            "gemale":"female","temale":"female","pemale":"female","nemale":"female","vale":"female",
            # male 常见脏写
            "male":"male","m":"male","maie":"male","mahe":"male","mace":"male","bale":"male",
            "oale":"male","make":"male","maln":"male","maze":"male",
        }
        df["gender"] = df["gender"].replace(basic_map)
        def _fzg(v):
            if pd.isna(v) or v == "": return np.nan
            m, score = fuzzy_best(v, ["female","male"])
            return m if score > 70 else np.nan
        df["gender"] = df["gender"].apply(_fzg)

    # Partner / Dependents -> 统一 Yes/No
    # def clean_yesno(col):
    #     if col not in df.columns: return
    #     df[col] = df[col].astype(str).str.strip().str.lower()
    #     manual = {"yes":"Yes","y":"Yes","yep":"Yes","yeah":"Yes","yess":"Yes",
    #               "no":"No","n":"No","nope":"No"}
    #     df[col] = df[col].replace(manual)
    #     def _f(v):
    #         if pd.isna(v) or v == "": return np.nan
    #         if v in ("yes","no"): return v.capitalize()
    #         m, score = fuzzy_best(v, ["yes","no"])
    #         return m.capitalize() if score > 70 else np.nan
    #     df[col] = df[col].apply(_f)
    #
    def clean_yesno(col):
        if col not in df.columns:
            return

        s = df[col].astype(str).str.strip().str.lower()

        # 统一去噪（末尾标点、1.0/0.0 等）
        s = (s
             .str.replace(r"[，。、,.;：:]+$", "", regex=True)
             .str.replace(r"^\s*([01])\.0\s*$", r"\1", regex=True)  # 1.0 -> 1
             .str.replace(r"\s+", " ", regex=True)
             )

        mapping = {
            # 英文/缩写
            "yes": "Yes", "y": "Yes", "yeah": "Yes", "yep": "Yes", "yess": "Yes",
            "no": "No", "n": "No", "nope": "No",

            # 数字/布尔/开关
            "1": "Yes", "true": "Yes", "t": "Yes", "on": "Yes",
            "0": "No", "false": "No", "f": "No", "off": "No",

            # 符号
            "✓": "Yes", "✔": "Yes", "✔️": "Yes",
            "✗": "No", "✘": "No",

            # 中文
            "是": "Yes", "有": "Yes", "开启": "Yes", "开": "Yes",
            "否": "No", "无": "No", "关闭": "No", "关": "No",

            # 空值
            "": "", "na": "", "none": "", "nan": ""
        }

        # 先按映射表粗清
        s = s.map(lambda v: mapping.get(v, v))

        # 再做一次兜底：把非空、非 Yes/No 的值做模糊匹配
        def _to_yesno(v):
            if v in ("Yes", "No", ""):  # 已经标准化或空
                return v
            if v is None:
                return ""
            m, score = fuzzy_best(str(v), ["Yes", "No"])
            return m if score >= 70 else ""

        s = s.map(_to_yesno)

        # 空串 -> NaN（如果希望把缺失当成 No，可改成 s.replace({"": "No"}））
        s = s.replace({"": np.nan})

        df[col] = s

    clean_yesno("Partner")
    clean_yesno("Dependents")


    # EducationLevel -> HighSchool / College / Bachelor
    if "EducationLevel" in df.columns:
        df["EducationLevel"] = df["EducationLevel"].astype(str).str.strip().str.lower()
        repl = {
            "bachelor":"Bachelor","bachclor":"Bachelor","bachelar":"Bachelor",
            "machelor":"Bachelor","rachelor":"Bachelor","oachelor":"Bachelor",
            "highschool":"HighSchool","high school":"HighSchool",
            "college":"College","collage":"College",
        }
        df["EducationLevel"] = df["EducationLevel"].replace(repl)
        choices = ["HighSchool","College","Bachelor"]
        def _fe(v):
            if pd.isna(v) or v == "": return np.nan
            m, score = fuzzy_best(v, choices)
            return m if score > 70 else np.nan
        df["EducationLevel"] = df["EducationLevel"].apply(_fe)

    # EmploymentStatus -> Employed / Retired / Self-employed
    if "EmploymentStatus" in df.columns:
        df["EmploymentStatus"] = df["EmploymentStatus"].astype(str).str.strip().str.lower()
        repl = {"employed":"Employed","retired":"Retired","self-employed":"Self-employed"}
        df["EmploymentStatus"] = df["EmploymentStatus"].replace(repl)
        choices = ["Employed","Retired","Self-employed"]
        def _fs(v):
            if pd.isna(v) or v == "": return np.nan
            m, score = fuzzy_best(v, choices)
            return m if score > 70 else np.nan
        df["EmploymentStatus"] = df["EmploymentStatus"].apply(_fs)

    # AnnualIncomeBracket -> 60–100k / 100–150k / >150k
    if "AnnualIncomeBracket" in df.columns:
        s = df["AnnualIncomeBracket"].astype(str).str.strip().str.lower()
        # 常见 OCR 错误修正
        s = s.replace({
            r"1[tfq]0":"150",
            r"6[zf]":"60",
            r"[mqv]":"0",
        }, regex=True)
        def _income(v):
            if v is None: return np.nan
            v = str(v).strip().replace("–","-").replace("—","-")
            choices = ["60-100k","100-150k",">150k"]
            m, score = fuzzy_best(v, choices)
            return m.replace("-", "–") if score > 70 else np.nan
        df["AnnualIncomeBracket"] = s.apply(_income)

    # 去掉关键键缺失
    if "customerID" in df.columns:
        df = df[df["customerID"].notna() & (df["customerID"].astype(str).str.strip()!="")]

    return df

# ------------------- Financial 清洗 -------------------
def clean_monthly_charges(df: pd.DataFrame,
                          col="MonthlyCharges",
                          keep_refund=False,
                          outlier_mode="clip",
                          iqr_k=3.0):
    df = df.copy()
    raw_col  = f"{col}_raw"
    neg_flag = f"{col}_was_negative"
    out_flag = f"{col}_was_outlier"
    refund_col = f"{col}_refund"
    paid_col   = f"{col}_paid_pos"

    if raw_col not in df.columns:
        df[raw_col] = df.get(col)

    s = (df[col].astype(str).str.strip().str.lower()
            .replace({'-':np.nan,'—':np.nan,'unknown':np.nan,
                      'nan':np.nan,'none':np.nan,'':np.nan}))
    s = pd.to_numeric(s, errors="coerce")

    neg_mask = s < 0
    df[neg_flag] = neg_mask.fillna(False)

    if keep_refund:
        df[refund_col] = s.where(neg_mask, 0.0).abs()
        df[paid_col]   = s.where(~neg_mask, 0.0)
    else:
        s = s.mask(neg_mask, 0.0)

    pos = s[(s.notna()) & (s >= 0)]
    if not pos.empty:
        q1, q3 = pos.quantile([0.25, 0.75])
        iqr = max(q3 - q1, 0.0)
        upper = q3 + iqr_k*iqr
        out_mask = s > upper
        df[out_flag] = out_mask.fillna(False)
        if outlier_mode == "clip":
            s = s.clip(upper=upper)
        elif outlier_mode == "nan":
            s = s.mask(out_mask, np.nan)
    else:
        df[out_flag] = False

    df[col] = s.round(2)
    return df

def clean_average_data_usage(df: pd.DataFrame,
                             col="AverageDataUsage",
                             outlier_mode="clip",
                             iqr_k=3.0,
                             round_ndigits=2):
    df = df.copy()
    raw_col  = f"{col}_raw"
    neg_flag = f"{col}_was_negative"
    out_flag = f"{col}_was_outlier_high"

    if raw_col not in df.columns:
        df[raw_col] = df.get(col)

    s = (df[col].astype(str).str.strip().str.lower()
            .replace({'':np.nan,' ':np.nan,'-':np.nan,'—':np.nan,
                      'unknown':np.nan,'nan':np.nan,'none':np.nan}))
    s = pd.to_numeric(s, errors="coerce")

    neg_mask = s < 0
    df[neg_flag] = neg_mask.fillna(False)
    s = s.mask(neg_mask, np.nan)

    pos = s[(s.notna()) & (s >= 0)]
    df[out_flag] = False
    if not pos.empty:
        q1, q3 = pos.quantile([0.25, 0.75])
        iqr = max(q3 - q1, 0.0)
        upper = q3 + iqr_k*iqr
        high = s > upper
        df[out_flag] = high.fillna(False)
        if outlier_mode == "clip":
            s = s.clip(upper=upper)
        elif outlier_mode == "nan":
            s = s.mask(high, np.nan)

    df[col] = s.round(round_ndigits)
    return df

def clean_app_login_frequency(df: pd.DataFrame,
                              col="AppLoginFrequency",
                              negative_policy="zero",   # "zero" | "nan"
                              round_mode="floor",       # "floor" | "round"
                              outlier_mode="clip",      # "clip"  | "nan"
                              iqr_k=3.0):
    df = df.copy()
    raw_col   = f"{col}_raw"
    neg_flag  = f"{col}_was_negative"
    frac_flag = f"{col}_was_fractional"
    out_flag  = f"{col}_was_outlier_high"

    if raw_col not in df.columns:
        df[raw_col] = df.get(col)

    s = (df[col].astype(str).str.strip().str.lower()
            .replace({'':np.nan,' ':np.nan,'-':np.nan,'—':np.nan,
                      'unknown':np.nan,'none':np.nan,'nan':np.nan}))
    s = pd.to_numeric(s, errors="coerce")

    neg_mask = s < 0
    df[neg_flag] = neg_mask.fillna(False)
    if negative_policy == "zero":
        s = s.mask(neg_mask, 0)
    elif negative_policy == "nan":
        s = s.mask(neg_mask, np.nan)
    else:
        raise ValueError("negative_policy must be 'zero' or 'nan'")

    frac_mask = s.notna() & (s % 1 != 0)
    df[frac_flag] = frac_mask.fillna(False)
    if round_mode == "floor":
        s = s.where(~frac_mask, np.floor(s))
    elif round_mode == "round":
        s = s.where(~frac_mask, np.round(s))
    else:
        raise ValueError("round_mode must be 'floor' or 'round'")

    pos = s.dropna()
    df[out_flag] = False
    if not pos.empty:
        q1, q3 = np.percentile(pos, [25, 75])
        iqr = max(q3 - q1, 0.0)
        upper = q3 + iqr_k*iqr
        high = s > upper
        df[out_flag] = high.fillna(False)
        if outlier_mode == "clip":
            s = s.where(~high, upper)
        elif outlier_mode == "nan":
            s = s.where(~high, np.nan)

    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s.isna() | (s >= 0), 0)
    df[col] = s.astype("Int64")
    return df

def clean_financial(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # 先删除不需要的列（与之前脚本一致）
    for c in ["tenure", "LatePaymentCount"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "MonthlyCharges" in df.columns:
        df = clean_monthly_charges(df, "MonthlyCharges",
                                   keep_refund=False, outlier_mode="clip", iqr_k=3.0)
    if "AverageDataUsage" in df.columns:
        df = clean_average_data_usage(df, "AverageDataUsage",
                                      outlier_mode="nan", iqr_k=3.0)
    if "AppLoginFrequency" in df.columns:
        df = clean_app_login_frequency(df, "AppLoginFrequency",
                                       negative_policy="zero", round_mode="floor",
                                       outlier_mode="nan", iqr_k=3.0)

    # 重要键
    if "customerID" in df.columns:
        df = df[df["customerID"].notna() & (df["customerID"].astype(str).str.strip()!="")]

    # 可选：保留关键列（按你之前的习惯）
    keep_cols = [c for c in [
        "MonthlyCharges", "AverageDataUsage", "AppLoginFrequency",
        "CustomerSatisfactionScore", "customerID"
    ] if c in df.columns]
    if keep_cols:
        df = df[keep_cols].copy()
    return df

def clean_contract():
    CONTRACT_PATH = PATHS["contract"]
    CONTRACT_OUT = OUT_FILES["contract"]


    print("✅ Cleaning contract_df_noisy...")
    df = pd.read_csv(CONTRACT_PATH)

    # 删除 Contract 列中为 "unknown" 的行
    if "Contract" in df.columns:
        df = df[df["Contract"].astype(str).str.lower() != "unknown"]

    # 清理 snapdate 格式（仅尝试解析，不改变值）
    if "snapdate" in df.columns:
        df["snapdate"] = pd.to_datetime(df["snapdate"], errors="coerce")

    # 删除重复行
    df = df.drop_duplicates()

    # 保存结果
    df.to_csv(CONTRACT_OUT, index=False, encoding="utf-8-sig")
    print(f"✅ Saved cleaned contract to: {CONTRACT_OUT}")


def clean_service():
    SERVICE_PATH = PATHS["service"]
    SERVICE_OUT = OUT_FILES["service"]
    print("✅ Cleaning service_df_noisy...")
    df = pd.read_csv(SERVICE_PATH)

    # 清理 snapdate 格式（仅解析，不改成月初）
    if "snapdate" in df.columns:
        df["snapdate"] = pd.to_datetime(df["snapdate"], errors="coerce")

    # 填充缺失值（非强制，可按需保留）
    df = df.fillna("")

    # 删除重复行
    df = df.drop_duplicates()

    # 保存结果
    df.to_csv(SERVICE_OUT, index=False, encoding="utf-8-sig")
    print(f"✅ Saved cleaned service to: {SERVICE_OUT}")

def copy_lable_file(data_dir: Path, out_dir: Path):

    for pat in ("lable_*.csv", "label_*.csv"):
        for f in data_dir.glob(pat):
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)
# use for lable file copy
LABLE_FILE = "lable.csv"
DATA_DIR = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\lable_store")
BRONZE_DIR_LABEL = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\bronze\lable_store")
SILVER_DIR_LABEL = Path(r"C:\Users\HP\Desktop\MLE\mleproject\datamart\silver\lable_store")



def main():

    ensure_dir(SILVER_DIR)
    ensure_dir(SILVER_DIR_LABEL)

    copy_lable_file(BRONZE_DIR_LABEL, SILVER_DIR_LABEL)

    # 只清洗 demographic 与 financial；其它两个表此阶段不处理
    demo_path = PATHS["demographic"]
    fin_path  = PATHS["financial"]

    log(f"读取：{demo_path}")
    demographic_df = read_csv_any(demo_path)
    log(f"读取：{fin_path}")
    financial_df = read_csv_any(fin_path)

    log("清洗 demographic_df_noisy ...")
    demographic_clean = clean_demographic(demographic_df)
    log("清洗 financial_df_noisy ...")
    financial_clean = clean_financial(financial_df)

    # 输出
    out_demo = OUT_FILES["demographic"]
    out_fin  = OUT_FILES["financial"]
    demographic_clean.to_csv(out_demo, index=False, encoding="utf-8-sig")
    financial_clean.to_csv(out_fin, index=False, encoding="utf-8-sig")

    clean_service()
    clean_contract()

    log("✅ 导出完成：")
    log(f" - {out_demo}")
    log(f" - {out_fin}")

if __name__ == "__main__":
    main()
