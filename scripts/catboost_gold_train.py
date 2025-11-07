#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, accuracy_score, 
    precision_recall_curve
)

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


class Config:

    def __init__(self):
        # Flexible path
        base_dir = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
        if not base_dir.exists():
            base_dir = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()


        self.data_path = base_dir / "datamart" / "gold"
        self.output_dir = base_dir / "model_bank" / "catboost"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # time range
        self.train_start = (2023, 2)
        self.train_end = (2024, 4)
        self.oot_start = (2024, 5)     # OOT
        self.oot_end = (2024, 9)       # OOT

        self.test_size = 0.2
        self.random_state = 42
        
        # model config
        self.iterations = 1000
        self.learning_rate = 0.03
        self.depth = 6
        self.loss_function = 'Logloss'
        self.eval_metric = 'AUC'
        self.auto_class_weights = 'Balanced'
        self.verbose = 100
        self.early_stopping_rounds = 50

        self.target_candidates = ['Churn', 'label', 'target', 'y']

        self.plot_dpi = 300
        self.top_n_features = 30


# ==================== log setting ====================
def setup_logging(output_dir: str) -> logging.Logger:

    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class DataLoader:

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def generate_months(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:

        months = []
        year, month = start
        end_year, end_month = end
        
        while (year < end_year) or (year == end_year and month <= end_month):
            months.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return months
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        self.logger.info("="*60)
        self.logger.info("start reading gold layer data...")
        self.logger.info("="*60)

        # list of months
        train_months = self.generate_months(self.config.train_start, self.config.train_end)
        oot_months = self.generate_months(self.config.oot_start, self.config.oot_end)
        
        self.logger.info(f"\ntraining set: {self.config.train_start[0]}-{self.config.train_start[1]:02d} to "
                        f"{self.config.train_end[0]}-{self.config.train_end[1]:02d} ({len(train_months)}months)")
        self.logger.info(f"OOT: {self.config.oot_start[0]}-{self.config.oot_start[1]:02d} to "
                        f"{self.config.oot_end[0]}-{self.config.oot_end[1]:02d} ({len(oot_months)}months)")
        

        train_df = self._read_months(train_months, "training set")

        oot_df = self._read_months(oot_months, "OOT")
        
        # print message
        self.logger.info("\n" + "="*60)
        self.logger.info("loading done!")
        self.logger.info("="*60)
        self.logger.info(f"\ntraining set: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        if len(oot_df) > 0:
            self.logger.info(f"OOT:  {oot_df.shape[0]:,} rows × {oot_df.shape[1]} columns")
        else:
            self.logger.info("OOT:  no data")
        
        return train_df, oot_df
    
    def _read_months(self, months: List[Tuple[int, int]], dataset_name: str) -> pd.DataFrame:

        self.logger.info(f"\nread{dataset_name}data...")
        self.logger.info("-"*60)
        
        dfs = []
        for year, month in months:
            filename = f"gold_{year}_{month:02d}_01.csv"
            filepath = os.path.join(self.config.data_path, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['year'] = year
                    df['month'] = month
                    df['year_month'] = f"{year}-{month:02d}"
                    dfs.append(df)
                    self.logger.info(f"  ✓ {filename}: {df.shape[0]:,} records")
                except Exception as e:
                    self.logger.error(f"  ✗ {filename}: loading fail - {str(e)}")
            else:
                self.logger.warning(f"  ⚠ {filename}: missing file")
        
        if len(dfs) == 0:
            if dataset_name == "training set":
                raise ValueError(f"error: cannot find any{dataset_name}！")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"\n{dataset_name}Merge complete: {combined_df.shape[0]:,} records")
        return combined_df


# ==================== preprocess ====================
class DataPreprocessor:
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cat_features_names = []
        self.target_col = None
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("\n" + "="*60)
        self.logger.info("start preprocess...")
        self.logger.info("="*60)
        
        df = df.copy()

        self._identify_target(df)
        
        self._remove_id_columns(df)

        self._process_date_features(df)

        self._fill_missing_values(df)

        self._process_categorical_features(df)
        
        self.logger.info(f"\n✓ Done,the shape is: {df.shape}")
        
        return df
    
    def _identify_target(self, df: pd.DataFrame):

        for col in self.config.target_candidates:
            if col in df.columns:
                self.target_col = col
                self.logger.info(f"✓ target variable: {col}")

                if df[col].dtype == 'object':
                    if set(df[col].unique()).issubset({'Yes', 'No', 'yes', 'no'}):
                        df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
                        self.logger.info("✓ exchange finish (Yes/No -> 1/0)")
                
                self.logger.info(f"\nDistribution of target variables:")
                self.logger.info(f"{df[col].value_counts()}")
                self.logger.info(f"Positive sample ratio: {df[col].mean():.2%}")
                return
        
        self.logger.warning("⚠ warning: cannot find target variable!")
        self.logger.warning(f"columns: {df.columns.tolist()}")
    
    def _remove_id_columns(self, df: pd.DataFrame):

        id_cols = ['customerID', 'id', 'ID', 'customer_id', 'CustomerId']
        for col in id_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                self.logger.info(f"✓ removed: {col}")
    
    def _process_date_features(self, df: pd.DataFrame):

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
                    self.logger.info(f"✓ date features: {col}")
                except:
                    self.logger.warning(f"⚠ cannot handle features: {col}")
    
    def _fill_missing_values(self, df: pd.DataFrame):

        # numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        self.logger.info("✓ numeric missing values have been filled")
        
        # categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        self.logger.info("✓ categorical missing values have been filled")
    
    def _process_categorical_features(self, df: pd.DataFrame):

        self.logger.info("\n" + "-"*60)
        self.logger.info("handle categorical...")
        self.logger.info("-"*60)

        self.cat_features_names = list(df.select_dtypes(include=['object']).columns)

        exclude_from_cat = ['year_month']
        self.cat_features_names = [col for col in self.cat_features_names 
                                   if col not in exclude_from_cat]

        for col in self.cat_features_names:
            df[col] = df[col].astype(str)
        
        self.logger.info(f"✓ {len(self.cat_features_names)} categorical features")
        if len(self.cat_features_names) > 0:
            self.logger.info(f"categorical features: {self.cat_features_names[:10]}" +
                           ("..." if len(self.cat_features_names) > 10 else ""))
        

        numeric_features = [col for col in df.columns 
                          if col not in self.cat_features_names 
                          and col != self.target_col 
                          and col not in ['year', 'month', 'year_month']]
        
        for col in numeric_features:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].median(), inplace=True)
                    self.logger.info(f"  ✓ to {col} numerical")
                except:
                    self.logger.warning(f"  ⚠ cannot change {col}, would keep as categorical")
                    df[col] = df[col].astype(str)
                    self.cat_features_names.append(col)


# ==================== train ====================
class ModelTrainer:
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.cat_features_indices = []
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    cat_features_names: List[str]) -> Tuple:
        self.logger.info("\n" + "="*60)
        self.logger.info("training data...")
        self.logger.info("="*60)
        
        # train time split
        train_mask = (
            ((df['year'] == self.config.train_start[0]) & (df['month'] >= self.config.train_start[1])) |
            ((df['year'] > self.config.train_start[0]) & (df['year'] < self.config.train_end[0])) |
            ((df['year'] == self.config.train_end[0]) & (df['month'] <= self.config.train_end[1]))
        )
        # oot time split
        oot_mask = (
            ((df['year'] == self.config.oot_start[0]) & (df['month'] >= self.config.oot_start[1])) |
            ((df['year'] > self.config.oot_start[0]) & (df['year'] < self.config.oot_end[0])) |
            ((df['year'] == self.config.oot_end[0]) & (df['month'] <= self.config.oot_end[1]))
        )
        
        df_train_dev = df[train_mask].copy()
        df_oot = df[oot_mask].copy()
        
        self.logger.info(f"\ntime range:")
        self.logger.info(f"  train+val: {df_train_dev.shape[0]}")
        if df_oot.shape[0] > 0:
            self.logger.info(f"  OOT: {df_oot.shape[0]}")

        exclude_cols = [target_col, 'year', 'month', 'year_month']
        feature_cols = [col for col in df_train_dev.columns if col not in exclude_cols]
        
        X_train_dev = df_train_dev[feature_cols]
        y_train_dev = df_train_dev[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_dev, y_train_dev,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_train_dev
        )
        
        self.logger.info(f"\nfinal:")
        self.logger.info(f"  train: {X_train.shape}")
        self.logger.info(f"  val: {X_test.shape}")

        if df_oot.shape[0] > 0:
            X_oot = df_oot[feature_cols]
            y_oot = df_oot[target_col]
            self.logger.info(f"  OOT: {X_oot.shape}")
        else:
            X_oot, y_oot = None, None
        
        # index
        self.cat_features_indices = [i for i, col in enumerate(feature_cols) 
                                     if col in cat_features_names]
        
        self.logger.info(f"\ncategorical index: {len(self.cat_features_indices)} 个")
        
        return X_train, X_test, y_train, y_test, X_oot, y_oot
    
    def train(self, X_train, y_train, X_test, y_test):

        self.logger.info("\n" + "="*60)
        self.logger.info("train CatBoost...")
        self.logger.info("="*60)
        self.logger.info(f"model config:")
        self.logger.info(f"  - iterations: {self.config.iterations}")
        self.logger.info(f"  - learning_rate: {self.config.learning_rate}")
        self.logger.info(f"  - depth: {self.config.depth}")
        self.logger.info(f"  - cat_features: {len(self.cat_features_indices)}")
        
        self.model = CatBoostClassifier(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            loss_function=self.config.loss_function,
            eval_metric=self.config.eval_metric,
            random_seed=self.config.random_state,
            cat_features=self.cat_features_indices,
            auto_class_weights=self.config.auto_class_weights,
            verbose=self.config.verbose,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        self.logger.info("\ntraining start...")
        self.logger.info("-"*60)
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        
        self.logger.info(f"\n✓ finished！")
        self.logger.info(f"✓ Optimal number of iterations: {self.model.best_iteration_}")
        self.logger.info(f"✓ best AUC: {self.model.best_score_['validation']['AUC']:.4f}")
        
        return self.model


# ==================== val ====================
class ModelEvaluator:
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def evaluate(self, model, X_test, y_test, X_oot=None, y_oot=None):
        self.logger.info("\n" + "="*60)
        self.logger.info("model val...")
        self.logger.info("="*60)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        self.logger.info("\nval:")
        self.logger.info("="*60)
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info(f"\ncore matrix:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  AUC-ROC:  {auc_score:.4f}")
        
        results = {
            'validation': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'auc': auc_score,
                'cm': cm
            }
        }
        
        # OOT
        if X_oot is not None and len(X_oot) > 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("OOT val:")
            self.logger.info("="*60)
            
            y_oot_pred = model.predict(X_oot)
            y_oot_pred_proba = model.predict_proba(X_oot)[:, 1]
            
            self.logger.info(f"\n{classification_report(y_oot, y_oot_pred)}")
            
            accuracy_oot = accuracy_score(y_oot, y_oot_pred)
            auc_oot = roc_auc_score(y_oot, y_oot_pred_proba)
            cm_oot = confusion_matrix(y_oot, y_oot_pred)
            
            self.logger.info(f"\ncore matrix:")
            self.logger.info(f"  Accuracy: {accuracy_oot:.4f}")
            self.logger.info(f"  AUC-ROC:  {auc_oot:.4f}")
            
            # PSI
            psi_value = self._calculate_psi(y_pred_proba, y_oot_pred_proba)
            self.logger.info(f"\n(PSI): {psi_value:.4f}")
            
            if psi_value < 0.1:
                self.logger.info("  ✓ PSI < 0.1: model good")
            elif psi_value < 0.2:
                self.logger.info("  ⚠ 0.1 ≤ PSI < 0.2: need attention")
            else:
                self.logger.info("  ✗ PSI ≥ 0.2: model drifted")
            
            results['oot'] = {
                'y_pred': y_oot_pred,
                'y_pred_proba': y_oot_pred_proba,
                'accuracy': accuracy_oot,
                'auc': auc_oot,
                'cm': cm_oot,
                'psi': psi_value
            }
        
        return results
    
    @staticmethod
    def _calculate_psi(expected, actual, bins=10):

        breakpoints = np.linspace(0, 1, bins + 1)
        expected_hist = np.histogram(expected, bins=breakpoints)[0]
        actual_hist = np.histogram(actual, bins=breakpoints)[0]
        
        expected_pct = expected_hist / len(expected)
        actual_pct = actual_hist / len(actual)
        
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi


# ==================== keep result ====================
class ResultSaver:
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def save_all(self, model, X_train, y_test, results):

        self.logger.info("\n" + "="*60)
        self.logger.info("keep model and result...")
        self.logger.info("="*60)
        
        os.makedirs(self.config.output_dir, exist_ok=True)

        model_path = os.path.join(self.config.output_dir, 'catboost_gold_model.cbm')
        model.save_model(model_path)
        self.logger.info(f"✓ model keep at: {model_path}")

        self._save_feature_importance(model, X_train)

        self._save_predictions(y_test, results)

        self._save_metrics(results)

        # self._generate_plots(model, X_train, y_test, results)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("✅ all have stored!")
        self.logger.info("="*60)
    
    def _save_feature_importance(self, model, X_train):

        importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        path = os.path.join(self.config.output_dir, 'feature_importance.csv')
        importance_df.to_csv(path, index=False, encoding='utf-8-sig')
        self.logger.info(f"✓ feature importance stored: {path}")
        
        return importance_df
    
    def _save_predictions(self, y_test, results):

        val_results = results['validation']
        
        df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': val_results['y_pred'],
            'probability': val_results['y_pred_proba'],
            'dataset': 'validation'
        })
        
        if 'oot' in results:
            oot_results = results['oot']
            # Note: y_oot needs to be passed separately in real implementation
            # This is a simplified version
        
        path = os.path.join(self.config.output_dir, 'predictions.csv')
        df.to_csv(path, index=False)
        self.logger.info(f"✓ predict value stored: {path}")
    
    def _save_metrics(self, results):

        val = results['validation']
        cm = val['cm']
        
        metrics_list = [
            ['Validation', 'Accuracy', val['accuracy']],
            ['Validation', 'AUC-ROC', val['auc']],
            ['Validation', 'Precision', cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0],
            ['Validation', 'Recall', cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0],
        ]
        
        if 'oot' in results:
            oot = results['oot']
            cm_oot = oot['cm']
            metrics_list.extend([
                ['OOT', 'Accuracy', oot['accuracy']],
                ['OOT', 'AUC-ROC', oot['auc']],
                ['OOT', 'Precision', cm_oot[1,1] / (cm_oot[1,1] + cm_oot[0,1]) if (cm_oot[1,1] + cm_oot[0,1]) > 0 else 0],
                ['OOT', 'Recall', cm_oot[1,1] / (cm_oot[1,1] + cm_oot[1,0]) if (cm_oot[1,1] + cm_oot[1,0]) > 0 else 0],
            ])
        
        df = pd.DataFrame(metrics_list, columns=['Dataset', 'Metric', 'Value'])
        path = os.path.join(self.config.output_dir, 'model_metrics.csv')
        df.to_csv(path, index=False)
        self.logger.info(f"✓ val result stored: {path}")
    


def main():

    cfg = Config()
    parser = argparse.ArgumentParser(description='CatBoost Gold Price Prediction Training')
    parser.add_argument('--data_path', type=str, default=cfg.data_path,
                       help='Data folder path')
    parser.add_argument('--output_dir', type=str, default=cfg.output_dir,
                       help='Output folder path')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='CatBoost iterations number')
    parser.add_argument('--learning_rate', type=float, default=0.03,
                       help='lr')
    parser.add_argument('--depth', type=int, default=6,
                       help='tree depth')
    
    args = parser.parse_args()

    config = Config()
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.iterations = args.iterations
    config.learning_rate = args.learning_rate
    config.depth = args.depth

    logger = setup_logging(config.output_dir)
    
    try:
        logger.info("="*60)
        logger.info("CatBoost Gold Price Prediction - train start")
        logger.info("="*60)
        logger.info(f"\nconfig message:")
        logger.info(f"  data path: {config.data_path}")
        logger.info(f"  output path: {config.output_dir}")
        logger.info(f"  time range: {config.train_start} - {config.train_end}")
        logger.info(f"  OOT range:  {config.oot_start} - {config.oot_end}")

        data_loader = DataLoader(config, logger)
        train_df, oot_df = data_loader.load_data()

        if len(oot_df) > 0:
            df = pd.concat([train_df, oot_df], ignore_index=True)
        else:
            df = train_df

        preprocessor = DataPreprocessor(config, logger)
        df = preprocessor.preprocess(df)
        
        if preprocessor.target_col is None:
            raise ValueError("cannot find data！")

        trainer = ModelTrainer(config, logger)
        X_train, X_test, y_train, y_test, X_oot, y_oot = trainer.prepare_data(
            df, preprocessor.target_col, preprocessor.cat_features_names
        )

        # train model
        model = trainer.train(X_train, y_train, X_test, y_test)

        evaluator = ModelEvaluator(config, logger)
        results = evaluator.evaluate(model, X_test, y_test, X_oot, y_oot)

        saver = ResultSaver(config, logger)
        saver.save_all(model, X_train, y_test, results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ all finish！")
        logger.info("="*60)
        logger.info(f"\nfile path: {config.output_dir}")
        logger.info("\nmodel loading path:")
        logger.info("  from catboost import CatBoostClassifier")
        logger.info("  model = CatBoostClassifier()")
        logger.info(f"  model.load_model('{os.path.join(config.output_dir, 'catboost_gold_model.cbm')}')")
        
    except Exception as e:
        logger.error(f"\n❌ training progress error: {str(e)}")
        logger.exception("errror message:")
        sys.exit(1)


if __name__ == "__main__":
    main()
