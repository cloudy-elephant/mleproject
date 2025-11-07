import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


def setup_logging():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model(model_path: str, logger: logging.Logger) -> CatBoostClassifier:

    logger.info(f"loading model: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file missing: {model_path}")
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    logger.info("✓ model loaded successfully")
    return model


def load_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:

    logger.info(f"loading data: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file missing: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def preprocess_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:

    logger.info("preprocessing data...")
    
    df = df.copy()

    id_cols = ['customerID', 'id', 'ID', 'customer_id', 'CustomerId']
    for col in id_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            logger.info(f"  remove 10 columns: {col}")

    date_cols = [col for col in df.columns 
                if 'date' in col.lower() or 'time' in col.lower()]
    
    for col in date_cols:
        if col not in ['year', 'month', 'year_month']:
            try:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
                df.drop(col, axis=1, inplace=True)
                logger.info(f"extract date features: {col}")
            except:
                pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)

    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    logger.info("✓ data preprocessing completed")
    
    return df


from catboost import Pool

def make_predictions(model: CatBoostClassifier, df: pd.DataFrame,
                    logger: logging.Logger, output_path: str = None):

    logger.info("predict start...")

    # 排除不需要的列
    exclude_cols = ['year', 'month', 'year_month']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]

    # 🔹自动识别类别列：object / category / bool
    cat_cols = [c for c in X.columns if X[c].dtype == "object"
                or str(X[c].dtype) == "category"
                or X[c].dtype == "bool"]
    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
    logger.info(f"detected categorical columns {len(cat_cols)}: {cat_cols}")

    pool = Pool(data=X, cat_features=cat_idx)

    predictions = model.predict(pool)
    probabilities = model.predict_proba(pool)[:, 1]

    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })

    for id_col in ['customerID', 'CustomerID', 'customer_id']:
        if id_col in df.columns:
            results.insert(0, id_col, df[id_col])
            break

    logger.info(f"✓ predict finished: {len(results)} records")
    logger.info(f"categorical: 0={ (predictions==0).sum() }, 1={ (predictions==1).sum() }")
    logger.info(f"ave forcast ratio: {probabilities.mean():.4f}")

    # 保存结果
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"✓ prediction result stored: {output_path}")

    return results


# read data gold something issues, can only predict one month.

def _base_dir() -> Path:
    root = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
    if not root.exists():
        root = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()
    return root

def _latest_file(dir_path: Path, patterns: list[str]) -> Path | None:
    cands = []
    for pat in patterns:
        cands.extend(Path(dir_path).glob(pat))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _auto_gold_file(root: Path) -> Path | None:
    for d in [root / "datamart" / "gold" / "feature_store",
              root / "datamart" / "gold"]:
        if d.exists() and d.is_dir():
            f = _latest_file(d, ["*.csv"])
            if f:
                return f
    return None

def _default_output_path(root: Path) -> Path:
    out_dir = root / "model_bank" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"predictions_{ts}.csv"

def main():
    parser = argparse.ArgumentParser(description='CatBoost Gold Price Prediction - Inference')
    parser.add_argument('--model_path', type=str, default=None, help='model file path(.cbm)，defult:model_bank/catboost.cbm')
    parser.add_argument('--data_path',  type=str, default=None, help='predicted date(.csv)，defult: datamart/gold(/feature_store).csv')
    parser.add_argument('--output_path', type=str, default=None, help='forcast result path(.csv)，defult: model_bank/predictions/')
    args = parser.parse_args()

    logger = setup_logging()

    try:
        root = _base_dir()

        if args.model_path:
            model_path = Path(args.model_path)
        else:
            model_dir = root / "model_bank" / "catboost"
            model_path = _latest_file(model_dir, ["*.cbm"])
            if not model_path:
                raise FileNotFoundError(f"cannot fine model file：{model_dir}/*.cbm，train first or manually select --model_path")

        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = _auto_gold_file(root)
            if not data_path:
                raise FileNotFoundError(
                    f"cannot find any forcast data： {root/'datamart'/'gold'} ，"
                    f"or selected manually --data_path"
                )

        output_path = Path(args.output_path) if args.output_path else _default_output_path(root)

        logger.info("="*60)
        logger.info("CatBoost Gold Price Prediction - start inference")
        logger.info("="*60)
        logger.info(f"[auto] ROOT         = {root}")
        logger.info(f"[auto] MODEL_PATH   = {model_path}")
        logger.info(f"[auto] DATA_PATH    = {data_path}")
        logger.info(f"[auto] OUTPUT_PATH  = {output_path}")

        model = load_model(model_path, logger)

        df = load_data(data_path, logger)

        df = preprocess_data(df, logger)

        results = make_predictions(model, df, logger, output_path)

        logger.info("\n" + "="*60)
        logger.info("✅ prediction process completed!")
        logger.info("="*60)
        logger.info("\nforcast result sample (first 5 rows):")
        try:
            logger.info(f"\n{results.head()}")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"\n❌ error in prediction: {e}")
        logger.exception("error message:")
        sys.exit(1)

if __name__ == "__main__":
    main()


