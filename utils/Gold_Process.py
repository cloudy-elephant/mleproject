from pathlib import Path
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType
import pyspark.sql.functions as F
from datetime import datetime
import os

def process_labels_gold_table(snapshot_date_str: str,
                              silver_loan_daily_directory: str,
                              gold_label_store_directory: str,
                              spark,
                              dpd: int,
                              mob: int):
    """
    从 silver_loan_daily 读取某月分区，按 DPD/MOB 打标，写入 gold/label_store。
    输出文件名：gold_label_store_YYYY_MM_DD.parquet
    """
    # 解析日期 & 统一文件名片段
    ymd = snapshot_date_str.replace("-", "_")

    # === 输入：silver 路径 ===
    silver_dir  = Path(silver_loan_daily_directory)
    silver_file = silver_dir / f"silver_loan_daily_{ymd}.parquet"
    if not silver_file.exists():
        raise FileNotFoundError(f"缺少 silver 输入分区：{silver_file}")

    df = spark.read.parquet(str(silver_file))
    print("loaded from:", silver_file, "row count:", df.count())

    # 过滤 MOB
    if "mob" not in df.columns:
        raise KeyError("输入数据缺少列：mob")
    df = df.filter(col("mob") == mob)

    # 打标
    if "dpd" not in df.columns:
        raise KeyError("输入数据缺少列：dpd")
    df = (
        df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
          .withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    )

    # 保障 snapshot_date 列存在（有些管道里可能叫 snapshot_date）
    if "snapshot_date" not in df.columns:
        df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

    # 选择输出列
    need_cols = ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"输出缺少必要列：{missing}")
    df_out = df.select(*need_cols)

    # === 输出：gold/label_store 路径 ===
    out_dir  = Path(gold_label_store_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gold_label_store_{ymd}.parquet"

    # 用 Spark 写（更安全；需要覆写就用 overwrite）
    # df_out.write.mode("overwrite").parquet(str(out_path))
    df.toPandas().to_parquet(out_path,
              compression='gzip')
    print("saved to:", out_path)

    return df_out
