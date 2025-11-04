import os
import numpy as np
import pyspark
import pandas as pd
from datetime import datetime
# import pyspark.sql.functions as F
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pathlib import Path
import os

def clean_columns(df, mode="strict"):

    white_list = r"A-Za-z0-9 \-'\./&(),_"
    allowed_full = rf"^[{white_list}]+$"     # all match white list
    disallowed_any = rf"[^{white_list}]"

    out = df
    for c in out.columns:
        s = F.trim(F.col(c).cast("string"))
        s = F.when(
            s.isNull() | (s == "") | F.lower(s).isin("na", "n/a", "none", "null"),
            F.lit(None)
        ).otherwise(s)

        s = F.regexp_replace(s, r"[\x00-\x1F\x7F]", "")

        if mode == "strict":
            s = F.when(s.isNotNull() & (~s.rlike(allowed_full)), F.lit(None)).otherwise(s)
        else:  # soft
            kept = F.regexp_replace(s, disallowed_any, "")
            kept = F.trim(kept)
            s = F.when(kept == "", F.lit(None)).otherwise(kept)

        out = out.withColumn(c, s)

    return out



def resolve_assignment_csv(filename: str, base: str | Path | None = None) -> str:
    root = Path(base or os.getenv("DATA_DIR") or os.getenv("MLE_ASSIGNMENT_DIR") or "/opt/airflow")
    candidates = [
        root / "data" / "data" / filename,
        root / "data" / filename,
        root / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve().as_posix())
    raise FileNotFoundError(
        f"cannot find {filename}. Tried: {', '.join(map(str, candidates))}. "
        "Set DATA_DIR or MLE_ASSIGNMENT_DIR, or pass base=..."
    )

# def resolve_assignment_csv(filename: str, base: str | Path | None = None) -> str:
#     base = Path(base) if base else Path.cwd()
#
#     cur = base.resolve()
#     assignment_dir = None
#     for p in [cur, *cur.parents]:
#         if (p / "MLE_Assignment").is_dir():
#             assignment_dir = p / "MLE_Assignment"
#             break
#     if assignment_dir is None:
#         raise FileNotFoundError("cannot find 'Assignment'，please ensure the base parameters")
#
#     csv_path = assignment_dir / "data" / "data" / filename
#     if not csv_path.is_file():
#         raise FileNotFoundError(f"cannot find file：{csv_path}")
#
#     return csv_path.resolve().as_posix()


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """
    Process silver table for a given snapshot date.

    Args:
        snapshot_date_str: Date string in 'YYYY-MM-DD' format
        bronze_lms_directory: Bronze layer directory
        silver_loan_daily_directory: Silver layer output directory
        spark: SparkSession instance

    Returns:
        Processed DataFrame
    """
    # Parse snapshot date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load bronze data
    partition_name = f"bronze_loan_daily_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_lms_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # Clean data: enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Augment data: add month on book (MOB)
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Augment data: calculate overdue metrics
    # Calculate installments missed
    df = df.withColumn(
        "installments_missed",
        F.when(
            (col("due_amt").isNull()) | (col("due_amt") == 0),
            0  # 如果 due_amt 是 0 或 null，设置为 0
        ).otherwise(
            F.ceil(col("overdue_amt") / col("due_amt"))
        ).cast(IntegerType())
    )

    # Calculate first missed date
    df = df.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
        ).cast(DateType())
    )

    # Calculate days past due (DPD)
    df = df.withColumn(
        "dpd",
        F.when(
            col("overdue_amt") > 0.0,
            F.datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0).cast(IntegerType())
    )

    # Ensure output directory exists
    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    # Save silver table using Pandas (no Hadoop dependency required)
    partition_name = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_loan_daily_directory, partition_name)

    # Convert to Pandas and save as Parquet
    df.toPandas().to_parquet(
        filepath,
        compression='gzip',  # Compress to save space
        index=False  # Don't save index column
    )

    print(f'✓ Saved to: {filepath}')

    return df


def process_silver_table_cs(silver_loan_daily_directory, spark):
    """
    Process silver table for clickstream features.

    Args:
        silver_loan_daily_directory: Silver layer output directory
        spark: SparkSession instance

    Returns:
        Processed DataFrame
    """

    # Load csv
    filepath = resolve_assignment_csv("feature_clickstream.csv")
    # filepath = r"C:\Users\HP\Desktop\MLE\Assignment\data\data\feature_clickstream.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # Remove garbled characters --> Null
    df = clean_columns(df)

    # Clean data: enforce schema / data type
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    # empty value --> null
    def normalize_to_string_null(c):
        s = F.trim(F.col(c).cast("string"))
        return F.when(
            s.isNull() |
            (s == "") |
            (s.rlike(r"^(?i)\s*(NULL|None|NaN)\s*$")),
            lit("null")
        ).otherwise(s)

    for c in df.columns:
        df = df.withColumn(c, normalize_to_string_null(c))

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Ensure output directory exists
    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    output_filepath = os.path.join(silver_loan_daily_directory, "clickstream_features.parquet")

    # Convert to Pandas and save as Parquet
    df.toPandas().to_parquet(
        output_filepath,
        compression='gzip',  # Compress to save space
        index=False  # Don't save index column
    )

    print(f'✓ Saved to: {output_filepath}')

    return df

def process_silver_table_attributes(silver_loan_daily_directory, spark):

    # Process silver table for attributes features.

    # Load csv
    filepath = resolve_assignment_csv("features_attributes.csv")
    # filepath = r"C:\Users\HP\Desktop\MLE\Assignment\data\data\features_attributes.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # Remove garbled characters --> Null
    df = clean_columns(df)

    # Clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    # empty value --> null
    def normalize_to_string_null(c):
        s = F.trim(F.col(c).cast("string"))
        return F.when(
            s.isNull() |
            (s == "") |
            (s.rlike(r"^(?i)\s*(NULL|None|NaN)\s*$")),
            lit("null")
        ).otherwise(s)

    for c in df.columns:
        df = df.withColumn(c, normalize_to_string_null(c))

    for column, new_type in column_type_map.items():
        if isinstance(new_type, IntegerType):
            df = df.withColumn(column,
                               F.floor(
                                   F.expr(f"try_cast(regexp_replace({column}, '[,\\s]', '') as double)")
                               ).cast("int")
                               )
        else:
            df = df.withColumn(column, F.expr(f"try_cast({column} as {new_type.simpleString()})"))

    # Ensure output directory exists
    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    output_filepath = os.path.join(silver_loan_daily_directory, "attributes_feature.parquet")

    # Convert to Pandas and save as Parquet
    df.toPandas().to_parquet(
        output_filepath,
        compression='gzip',  # Compress to save space
        index=False  # Don't save index column
    )

    print(f'✓ Saved to: {output_filepath}')

    return df

def process_silver_table_financial(silver_loan_daily_directory, spark):

    # Process silver table for financial features.
    # Load csv
    filepath = resolve_assignment_csv("features_financials.csv")
    # filepath = r"C:\Users\HP\Desktop\MLE\Assignment\data\data\features_financials.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # Remove garbled characters --> Null
    df = clean_columns(df)

    # Clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # empty value --> null
    def normalize_to_string_null(c):
        s = F.trim(F.col(c).cast("string"))
        return F.when(
            s.isNull() |
            (s == "") |
            (s.rlike(r"^(?i)\s*(NULL|None|NaN)\s*$")),
            lit("null")
        ).otherwise(s)

    for c in df.columns:
        df = df.withColumn(c, normalize_to_string_null(c))

    for column, new_type in column_type_map.items():
        if isinstance(new_type, IntegerType):
            df = df.withColumn(column,
                               F.floor(
                                   F.expr(f"try_cast(regexp_replace({column}, '[,\\s]', '') as double)")
                               ).cast("int")
                               )
        else:
            df = df.withColumn(column, F.expr(f"try_cast({column} as {new_type.simpleString()})"))

    # Ensure output directory exists
    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    output_filepath = os.path.join(silver_loan_daily_directory, "financial_feature.parquet")

    # Convert to Pandas and save as Parquet
    df.toPandas().to_parquet(
        output_filepath,
        compression='gzip',  # Compress to save space
        index=False  # Don't save index column
    )

    print(f'✓ Saved to: {output_filepath}')

    return df










