# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# import pprint
# import pyspark
# import pyspark.sql.functions as F
# from pyspark.sql import SparkSession
#
# from pyspark.sql.functions import col
# from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
#
# from utils.Bronze_Process import process_bronze_table
# from utils.Silver_Process import process_silver_table
# from utils.Silver_Process import process_silver_table_cs
# from utils.Silver_Process import process_silver_table_attributes
# from utils.Silver_Process import process_silver_table_financial
# from utils.Gold_Process import process_labels_gold_table
#
#
# # Initialize SparkSession
# spark = pyspark.sql.SparkSession.builder \
#     .appName("dev") \
#     .master("local[*]") \
#     .getOrCreate()
#
# # Set log level to ERROR to hide warnings
# spark.sparkContext.setLogLevel("ERROR")
#
# # set up config
# snapshot_date_str = "2023-01-01"
#
# start_date_str = "2023-01-01"
# end_date_str = "2024-12-01"
#
#
# # generate list of dates to process
# def generate_first_of_month_dates(start_date_str, end_date_str):
#     # Convert the date strings to datetime objects
#     start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
#     end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
#
#     # List to store the first of month dates
#     first_of_month_dates = []
#
#     # Start from the first of the month of the start_date
#     current_date = datetime(start_date.year, start_date.month, 1)
#
#     while current_date <= end_date:
#         # Append the date in yyyy-mm-dd format
#         first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
#
#         # Move to the first of the next month
#         if current_date.month == 12:
#             current_date = datetime(current_date.year + 1, 1, 1)
#         else:
#             current_date = datetime(current_date.year, current_date.month + 1, 1)
#
#     return first_of_month_dates
#
#
# dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
# print(dates_str_lst)
#
# # create bronze datalake
# bronze_lms_directory = "datamart/bronze/lms/"
#
# if not os.path.exists(bronze_lms_directory):
#     os.makedirs(bronze_lms_directory)
#
# # run bronze backfill
# for date_str in dates_str_lst:
#     process_bronze_table(date_str, bronze_lms_directory, spark)
#
# # ------------------------Silver--------------------------------------------------
#
# # silver: clean loan daily table
# silver_loan_daily_directory = "datamart/silver/loan_daily/"
#
# if not os.path.exists(silver_loan_daily_directory):
#     os.makedirs(silver_loan_daily_directory)
#
# for date_str in dates_str_lst:
#     process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
#
# # silver: clean financial table
# silver_loan_daily_directory_financial = "datamart/silver/financial/"
# process_silver_table_financial(silver_loan_daily_directory_financial, spark)
#
# # silver: clean attributed table
# silver_loan_daily_directory_attributes = "datamart/silver/attributes/"
# process_silver_table_attributes(silver_loan_daily_directory_attributes, spark)
#
# # silver: clean clickstream table
# silver_loan_daily_directory_cs = "datamart/silver/click_stream/"
# process_silver_table_cs(silver_loan_daily_directory_cs, spark)
#
# # ------------------------Gold--------------------------------------------------
# silver_loan_daily_directory_financial = "datamart/gold/feature_store/"
# process_silver_table_financial(silver_loan_daily_directory_financial, spark)
#
# silver_loan_daily_directory_attributes = "datamart/gold/feature_store/"
# process_silver_table_attributes(silver_loan_daily_directory_attributes, spark)
#
# silver_loan_daily_directory_cs = "datamart/gold/feature_store/"
# process_silver_table_cs(silver_loan_daily_directory_cs, spark)
#
# gold_label_store_directory = "/datamart/gold/label_store/"
# if not os.path.exists(gold_label_store_directory):
#     os.makedirs(gold_label_store_directory)
#
# for date_str in dates_str_lst:
#     process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
#


# scripts/main.py
import os
import argparse
from datetime import datetime, timedelta
from typing import List

import pyspark
from pyspark.sql import SparkSession

# --- 本地导入 ---
# 这些函数在你的 utils 里，保持你现有的签名：
#   process_bronze_table(date_str, bronze_lms_directory, spark)
#   process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
#   process_silver_table_financial(output_dir, spark)
#   process_silver_table_attributes(output_dir, spark)
#   process_silver_table_cs(output_dir, spark)
#   process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6)
from utils.Bronze_Process import process_bronze_table
from utils.Silver_Process import (
    process_silver_table,
    process_silver_table_financial,
    process_silver_table_attributes,
    process_silver_table_cs,
)
from utils.Gold_Process import process_labels_gold_table
from pathlib import Path
import os

# ========= 配置（可按需修改默认值） =========
DEFAULT_START = "2023-01-01"
DEFAULT_END   = "2024-12-01"

# 项目根目录 = scripts 的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许通过环境变量覆盖根路径（例如 Airflow/Docker 时可映射到 /opt/airflow/datamart）
DATAMART_ROOT = Path(os.getenv("DATAMART_DIR", PROJECT_ROOT / "datamart"))

# 下面这些替换你现在的常量
BRONZE_LMS_DIR         = str(DATAMART_ROOT / "bronze" / "lms")
SILVER_LOAN_DAILY_DIR  = str(DATAMART_ROOT / "silver" / "loan_daily")
SILVER_FINANCIAL_DIR   = str(DATAMART_ROOT / "silver" / "financial")
SILVER_ATTRIBUTES_DIR  = str(DATAMART_ROOT / "silver" / "attributes")
SILVER_CLICKSTREAM_DIR = str(DATAMART_ROOT / "silver" / "click_stream")

GOLD_FEATURE_DIR       = str(DATAMART_ROOT / "gold" / "feature_store")
GOLD_LABEL_STORE_DIR   = str(DATAMART_ROOT / "gold" / "label_store")

# ========================================


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_first_of_month_dates(start_date_str: str, end_date_str: str) -> List[str]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    dates = []
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        # 跳到下月 1 号
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)
    return dates


def run_bronze(dates: List[str], spark: SparkSession) -> None:
    print(f"🟤 [Bronze] 输出目录: {BRONZE_LMS_DIR}")
    ensure_dirs(BRONZE_LMS_DIR)
    for ds in dates:
        print(f"  → Bronze backfill @ {ds}")
        process_bronze_table(ds, BRONZE_LMS_DIR, spark)


def run_silver(dates: List[str], spark: SparkSession) -> None:
    print("⚪ [Silver] 清洗并生成中间层")
    ensure_dirs(SILVER_LOAN_DAILY_DIR, SILVER_FINANCIAL_DIR, SILVER_ATTRIBUTES_DIR, SILVER_CLICKSTREAM_DIR)

    # 1) 逐月由 Bronze → Silver loan_daily
    for ds in dates:
        print(f"  → Silver loan_daily @ {ds}")
        # 注意：按照你现有签名：process_silver_table(date_str, bronze_dir, silver_dir, spark)
        process_silver_table(ds, BRONZE_LMS_DIR, SILVER_LOAN_DAILY_DIR, spark)

    # 2) 其它三张 Silver 表（这些函数使用内部读取/聚合）
    print("  → Silver financial")
    process_silver_table_financial(SILVER_FINANCIAL_DIR, spark)

    print("  → Silver attributes")
    process_silver_table_attributes(SILVER_ATTRIBUTES_DIR, spark)

    print("  → Silver clickstream")
    process_silver_table_cs(SILVER_CLICKSTREAM_DIR, spark)


def run_gold(dates: List[str], spark: SparkSession, dpd: int = 30, mob: int = 6) -> None:
    print("🟡 [Gold] 生成特征 & 标签")
    # 你的 Silver → Gold 过程：根据你之前的脚本，这三个函数也可以直接把产物写到 gold/feature_store
    ensure_dirs(GOLD_FEATURE_DIR, GOLD_LABEL_STORE_DIR)

    # 将三类特征产出到 GOLD_FEATURE_DIR
    print("  → Gold feature_store (financial/attributes/clickstream)")
    process_silver_table_financial(GOLD_FEATURE_DIR, spark)
    process_silver_table_attributes(GOLD_FEATURE_DIR, spark)
    process_silver_table_cs(GOLD_FEATURE_DIR, spark)

    # 标签表按月生成（依赖 SILVER_LOAN_DAILY_DIR）
    for ds in dates:
        print(f"  → Gold label_store @ {ds} (dpd={dpd}, mob={mob})")
        process_labels_gold_table(
            ds,
            SILVER_LOAN_DAILY_DIR,
            GOLD_LABEL_STORE_DIR,
            spark,
            dpd=dpd,
            mob=mob,
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLE Assignment Pipeline (Bronze/Silver/Gold)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default=DEFAULT_START, help="起始月份（YYYY-MM-DD，使用每月1号）")
    p.add_argument("--end",   default=DEFAULT_END,   help="结束月份（YYYY-MM-DD，使用每月1号）")
    p.add_argument(
        "--stages",
        default="all",
        choices=["all", "bronze", "silver", "gold", "bronze_silver", "silver_gold"],
        help="选择要运行的阶段"
    )
    p.add_argument("--dpd", type=int, default=30, help="标签构造参数：days past due")
    p.add_argument("--mob", type=int, default=6,  help="标签构造参数：months on book")
    return p


def main():
    args = build_argparser().parse_args()

    dates = generate_first_of_month_dates(args.start, args.end)
    print(f"📅 将处理这些月份：{dates[0]} .. {dates[-1]} （共 {len(dates)} 个月）")

    # 初始化 Spark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("dev")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 确保基础目录存在
    ensure_dirs(
        BRONZE_LMS_DIR,
        SILVER_LOAN_DAILY_DIR, SILVER_FINANCIAL_DIR, SILVER_ATTRIBUTES_DIR, SILVER_CLICKSTREAM_DIR,
        GOLD_FEATURE_DIR, GOLD_LABEL_STORE_DIR
    )

    stages = args.stages
    if stages in ("all", "bronze", "bronze_silver"):
        run_bronze(dates, spark)

    if stages in ("all", "silver", "bronze_silver", "silver_gold"):
        run_silver(dates, spark)

    if stages in ("all", "gold", "silver_gold"):
        run_gold(dates, spark, dpd=args.dpd, mob=args.mob)

    print("✅ 全部完成。")


if __name__ == "__main__":
    main()
