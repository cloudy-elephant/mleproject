#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost Gold Price Prediction - Local Deployment Version
=========================================================
黄金价格预测模型 - 本地部署版本

使用方法:
    python catboost_gold_train.py --data_path /path/to/gold/data --output_dir ./output

作者: Auto-generated
日期: 2024
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，适合服务器环境
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


# ==================== 配置类 ====================
class Config:
    """模型训练配置"""

    def __init__(self):
        # 容器优先（/opt/airflow），本机回退到你的 Windows 路径
        base_dir = Path(os.getenv("PROJECT_ROOT", os.getenv("AIRFLOW_PROJ_DIR", "/opt/airflow"))).resolve()
        if not base_dir.exists():
            base_dir = Path(r"C:\Users\HP\Desktop\MLE\mleproject").resolve()

        # 数据与输出路径（自动创建输出目录）
        self.data_path = base_dir / "datamart" / "gold"
        self.output_dir = base_dir / "model_bank" / "catboost"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 时间范围配置
        self.train_start = (2023, 2)   # 训练集起始 (年, 月)
        self.train_end = (2024, 4)     # 训练集结束 (年, 月)
        self.oot_start = (2024, 5)     # OOT集起始 (年, 月)
        self.oot_end = (2024, 9)       # OOT集结束 (年, 月)
        
        # 数据划分配置
        self.test_size = 0.2
        self.random_state = 42
        
        # 模型配置
        self.iterations = 1000
        self.learning_rate = 0.03
        self.depth = 6
        self.loss_function = 'Logloss'
        self.eval_metric = 'AUC'
        self.auto_class_weights = 'Balanced'
        self.verbose = 100
        self.early_stopping_rounds = 50
        
        # 目标变量候选名称
        self.target_candidates = ['Churn', 'label', 'target', 'y']
        
        # 可视化配置
        self.plot_dpi = 300
        self.top_n_features = 30


# ==================== 日志配置 ====================
def setup_logging(output_dir: str) -> logging.Logger:
    """配置日志系统"""
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


# ==================== 数据加载模块 ====================
class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def generate_months(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """生成月份列表"""
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
        """
        读取黄金价格数据
        
        Returns:
            train_df: 训练集数据
            oot_df: OOT集数据
        """
        self.logger.info("="*60)
        self.logger.info("开始读取黄金价格数据...")
        self.logger.info("="*60)
        
        # 生成月份列表
        train_months = self.generate_months(self.config.train_start, self.config.train_end)
        oot_months = self.generate_months(self.config.oot_start, self.config.oot_end)
        
        self.logger.info(f"\n训练集: {self.config.train_start[0]}-{self.config.train_start[1]:02d} 至 "
                        f"{self.config.train_end[0]}-{self.config.train_end[1]:02d} (共{len(train_months)}个月)")
        self.logger.info(f"OOT集: {self.config.oot_start[0]}-{self.config.oot_start[1]:02d} 至 "
                        f"{self.config.oot_end[0]}-{self.config.oot_end[1]:02d} (共{len(oot_months)}个月)")
        
        # 读取训练集
        train_df = self._read_months(train_months, "训练集")
        
        # 读取OOT集
        oot_df = self._read_months(oot_months, "OOT集")
        
        # 打印统计信息
        self.logger.info("\n" + "="*60)
        self.logger.info("数据加载完成!")
        self.logger.info("="*60)
        self.logger.info(f"\n训练集: {train_df.shape[0]:,} 条 × {train_df.shape[1]} 列")
        if len(oot_df) > 0:
            self.logger.info(f"OOT集:  {oot_df.shape[0]:,} 条 × {oot_df.shape[1]} 列")
        else:
            self.logger.info("OOT集:  无数据")
        
        return train_df, oot_df
    
    def _read_months(self, months: List[Tuple[int, int]], dataset_name: str) -> pd.DataFrame:
        """读取指定月份的数据"""
        self.logger.info(f"\n读取{dataset_name}数据...")
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
                    self.logger.info(f"  ✓ {filename}: {df.shape[0]:,} 条记录")
                except Exception as e:
                    self.logger.error(f"  ✗ {filename}: 读取失败 - {str(e)}")
            else:
                self.logger.warning(f"  ⚠ {filename}: 文件不存在")
        
        if len(dfs) == 0:
            if dataset_name == "训练集":
                raise ValueError(f"错误: 没有找到任何{dataset_name}数据文件！")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"\n{dataset_name}合并完成: {combined_df.shape[0]:,} 条记录")
        return combined_df


# ==================== 数据预处理模块 ====================
class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cat_features_names = []
        self.target_col = None
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理主流程
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("开始数据预处理...")
        self.logger.info("="*60)
        
        df = df.copy()
        
        # 1. 识别目标变量
        self._identify_target(df)
        
        # 2. 移除ID列
        self._remove_id_columns(df)
        
        # 3. 处理日期特征
        self._process_date_features(df)
        
        # 4. 填充缺失值
        self._fill_missing_values(df)
        
        # 5. 处理类别特征（关键修复）
        self._process_categorical_features(df)
        
        self.logger.info(f"\n✓ 数据预处理完成，当前数据维度: {df.shape}")
        
        return df
    
    def _identify_target(self, df: pd.DataFrame):
        """识别目标变量"""
        for col in self.config.target_candidates:
            if col in df.columns:
                self.target_col = col
                self.logger.info(f"✓ 找到目标变量: {col}")
                
                # 转换目标变量
                if df[col].dtype == 'object':
                    if set(df[col].unique()).issubset({'Yes', 'No', 'yes', 'no'}):
                        df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
                        self.logger.info("✓ 目标变量转换完成 (Yes/No -> 1/0)")
                
                self.logger.info(f"\n目标变量分布:")
                self.logger.info(f"{df[col].value_counts()}")
                self.logger.info(f"正样本比例: {df[col].mean():.2%}")
                return
        
        self.logger.warning("⚠ 警告: 未找到目标变量列")
        self.logger.warning(f"可用的列: {df.columns.tolist()}")
    
    def _remove_id_columns(self, df: pd.DataFrame):
        """移除ID列"""
        id_cols = ['customerID', 'id', 'ID', 'customer_id', 'CustomerId']
        for col in id_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                self.logger.info(f"✓ 已移除ID列: {col}")
    
    def _process_date_features(self, df: pd.DataFrame):
        """处理日期特征"""
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
                    self.logger.info(f"✓ 日期特征提取完成: {col}")
                except:
                    self.logger.warning(f"⚠ 无法处理日期列: {col}")
    
    def _fill_missing_values(self, df: pd.DataFrame):
        """填充缺失值"""
        # 数值型
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        self.logger.info("✓ 数值型缺失值填充完成")
        
        # 类别型
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        self.logger.info("✓ 类别型缺失值填充完成")
    
    def _process_categorical_features(self, df: pd.DataFrame):
        """处理类别特征（关键修复）"""
        self.logger.info("\n" + "-"*60)
        self.logger.info("识别和处理类别特征...")
        self.logger.info("-"*60)
        
        # 1. 识别类别特征
        self.cat_features_names = list(df.select_dtypes(include=['object']).columns)
        
        # 2. 排除不应该作为特征的列
        exclude_from_cat = ['year_month']
        self.cat_features_names = [col for col in self.cat_features_names 
                                   if col not in exclude_from_cat]
        
        # 3. 关键：转换为字符串类型
        for col in self.cat_features_names:
            df[col] = df[col].astype(str)
        
        self.logger.info(f"✓ 识别到 {len(self.cat_features_names)} 个类别特征")
        if len(self.cat_features_names) > 0:
            self.logger.info(f"类别特征: {self.cat_features_names[:10]}" + 
                           ("..." if len(self.cat_features_names) > 10 else ""))
        
        # 4. 确保数值特征是数值类型
        numeric_features = [col for col in df.columns 
                          if col not in self.cat_features_names 
                          and col != self.target_col 
                          and col not in ['year', 'month', 'year_month']]
        
        for col in numeric_features:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].median(), inplace=True)
                    self.logger.info(f"  ✓ 转换 {col} 为数值类型")
                except:
                    self.logger.warning(f"  ⚠ 无法转换 {col}，将作为类别特征")
                    df[col] = df[col].astype(str)
                    self.cat_features_names.append(col)


# ==================== 模型训练模块 ====================
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.cat_features_indices = []
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    cat_features_names: List[str]) -> Tuple:
        """准备训练数据"""
        self.logger.info("\n" + "="*60)
        self.logger.info("准备训练数据...")
        self.logger.info("="*60)
        
        # 时间划分
        train_mask = (
            ((df['year'] == self.config.train_start[0]) & (df['month'] >= self.config.train_start[1])) |
            ((df['year'] > self.config.train_start[0]) & (df['year'] < self.config.train_end[0])) |
            ((df['year'] == self.config.train_end[0]) & (df['month'] <= self.config.train_end[1]))
        )
        
        oot_mask = (
            ((df['year'] == self.config.oot_start[0]) & (df['month'] >= self.config.oot_start[1])) |
            ((df['year'] > self.config.oot_start[0]) & (df['year'] < self.config.oot_end[0])) |
            ((df['year'] == self.config.oot_end[0]) & (df['month'] <= self.config.oot_end[1]))
        )
        
        df_train_dev = df[train_mask].copy()
        df_oot = df[oot_mask].copy()
        
        self.logger.info(f"\n时间范围分布:")
        self.logger.info(f"  训练+验证集: {df_train_dev.shape[0]} 条")
        if df_oot.shape[0] > 0:
            self.logger.info(f"  OOT集: {df_oot.shape[0]} 条")
        
        # 准备特征和标签
        exclude_cols = [target_col, 'year', 'month', 'year_month']
        feature_cols = [col for col in df_train_dev.columns if col not in exclude_cols]
        
        X_train_dev = df_train_dev[feature_cols]
        y_train_dev = df_train_dev[target_col]
        
        # 划分训练和验证集
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_dev, y_train_dev,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_train_dev
        )
        
        self.logger.info(f"\n最终划分:")
        self.logger.info(f"  训练集: {X_train.shape}")
        self.logger.info(f"  验证集: {X_test.shape}")
        
        # 准备OOT集
        if df_oot.shape[0] > 0:
            X_oot = df_oot[feature_cols]
            y_oot = df_oot[target_col]
            self.logger.info(f"  OOT集: {X_oot.shape}")
        else:
            X_oot, y_oot = None, None
        
        # 获取类别特征索引
        self.cat_features_indices = [i for i, col in enumerate(feature_cols) 
                                     if col in cat_features_names]
        
        self.logger.info(f"\n类别特征索引: {len(self.cat_features_indices)} 个")
        
        return X_train, X_test, y_train, y_test, X_oot, y_oot
    
    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        self.logger.info("\n" + "="*60)
        self.logger.info("训练CatBoost模型...")
        self.logger.info("="*60)
        self.logger.info(f"模型配置:")
        self.logger.info(f"  - iterations: {self.config.iterations}")
        self.logger.info(f"  - learning_rate: {self.config.learning_rate}")
        self.logger.info(f"  - depth: {self.config.depth}")
        self.logger.info(f"  - cat_features: {len(self.cat_features_indices)} 个")
        
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
        
        self.logger.info("\n开始训练...")
        self.logger.info("-"*60)
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        
        self.logger.info(f"\n✓ 模型训练完成！")
        self.logger.info(f"✓ 最佳迭代次数: {self.model.best_iteration_}")
        self.logger.info(f"✓ 最佳验证AUC: {self.model.best_score_['validation']['AUC']:.4f}")
        
        return self.model


# ==================== 模型评估模块 ====================
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def evaluate(self, model, X_test, y_test, X_oot=None, y_oot=None):
        """评估模型"""
        self.logger.info("\n" + "="*60)
        self.logger.info("模型评估...")
        self.logger.info("="*60)
        
        # 验证集评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        self.logger.info("\n验证集评估:")
        self.logger.info("="*60)
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info(f"\n核心指标:")
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
        
        # OOT集评估
        if X_oot is not None and len(X_oot) > 0:
            self.logger.info("\n" + "="*60)
            self.logger.info("OOT集评估:")
            self.logger.info("="*60)
            
            y_oot_pred = model.predict(X_oot)
            y_oot_pred_proba = model.predict_proba(X_oot)[:, 1]
            
            self.logger.info(f"\n{classification_report(y_oot, y_oot_pred)}")
            
            accuracy_oot = accuracy_score(y_oot, y_oot_pred)
            auc_oot = roc_auc_score(y_oot, y_oot_pred_proba)
            cm_oot = confusion_matrix(y_oot, y_oot_pred)
            
            self.logger.info(f"\n核心指标:")
            self.logger.info(f"  Accuracy: {accuracy_oot:.4f}")
            self.logger.info(f"  AUC-ROC:  {auc_oot:.4f}")
            
            # PSI计算
            psi_value = self._calculate_psi(y_pred_proba, y_oot_pred_proba)
            self.logger.info(f"\n模型稳定性 (PSI): {psi_value:.4f}")
            
            if psi_value < 0.1:
                self.logger.info("  ✓ PSI < 0.1: 模型稳定性良好")
            elif psi_value < 0.2:
                self.logger.info("  ⚠ 0.1 ≤ PSI < 0.2: 需要关注")
            else:
                self.logger.info("  ✗ PSI ≥ 0.2: 模型不稳定")
            
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
        """计算PSI"""
        breakpoints = np.linspace(0, 1, bins + 1)
        expected_hist = np.histogram(expected, bins=breakpoints)[0]
        actual_hist = np.histogram(actual, bins=breakpoints)[0]
        
        expected_pct = expected_hist / len(expected)
        actual_pct = actual_hist / len(actual)
        
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi


# ==================== 结果保存模块 ====================
class ResultSaver:
    """结果保存器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def save_all(self, model, X_train, y_test, results):
        """保存所有结果"""
        self.logger.info("\n" + "="*60)
        self.logger.info("保存模型和结果...")
        self.logger.info("="*60)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(self.config.output_dir, 'catboost_gold_model.cbm')
        model.save_model(model_path)
        self.logger.info(f"✓ 模型已保存: {model_path}")
        
        # 保存特征重要性
        self._save_feature_importance(model, X_train)
        
        # 保存预测结果
        self._save_predictions(y_test, results)
        
        # 保存评估指标
        self._save_metrics(results)
        
        # 生成可视化
        # self._generate_plots(model, X_train, y_test, results)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("✅ 所有结果已保存!")
        self.logger.info("="*60)
    
    def _save_feature_importance(self, model, X_train):
        """保存特征重要性"""
        importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        path = os.path.join(self.config.output_dir, 'feature_importance.csv')
        importance_df.to_csv(path, index=False, encoding='utf-8-sig')
        self.logger.info(f"✓ 特征重要性已保存: {path}")
        
        return importance_df
    
    def _save_predictions(self, y_test, results):
        """保存预测结果"""
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
        self.logger.info(f"✓ 预测结果已保存: {path}")
    
    def _save_metrics(self, results):
        """保存评估指标"""
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
        self.logger.info(f"✓ 评估指标已保存: {path}")
    
    # def _generate_plots(self, model, X_train, y_test, results):
    #     """生成可视化图表"""
    #     self.logger.info("\n生成可视化图表...")
    #
    #     # 特征重要性图
    #     importance = model.get_feature_importance()
    #     importance_df = pd.DataFrame({
    #         'feature': X_train.columns,
    #         'importance': importance
    #     }).sort_values('importance', ascending=False)
    #
    #     plt.figure(figsize=(12, 8))
    #     top_n = min(self.config.top_n_features, len(importance_df))
    #     sns.barplot(data=importance_df.head(top_n), y='feature', x='importance', palette='viridis')
    #     plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
    #     plt.xlabel('Importance Score', fontsize=12)
    #     plt.ylabel('Features', fontsize=12)
    #     plt.tight_layout()
    #
    #     path = os.path.join(self.config.output_dir, 'feature_importance.png')
    #     plt.savefig(path, dpi=self.config.plot_dpi, bbox_inches='tight')
    #     plt.close()
    #     self.logger.info(f"✓ 特征重要性图已保存: {path}")
        
        # 更多可视化可以在这里添加...


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 解析命令行参数
    cfg = Config()
    parser = argparse.ArgumentParser(description='CatBoost Gold Price Prediction Training')
    parser.add_argument('--data_path', type=str, default=cfg.data_path,
                       help='数据文件夹路径')
    parser.add_argument('--output_dir', type=str, default=cfg.output_dir,
                       help='输出文件夹路径')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='CatBoost迭代次数')
    parser.add_argument('--learning_rate', type=float, default=0.03,
                       help='学习率')
    parser.add_argument('--depth', type=int, default=6,
                       help='树深度')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.iterations = args.iterations
    config.learning_rate = args.learning_rate
    config.depth = args.depth
    
    # 设置日志
    logger = setup_logging(config.output_dir)
    
    try:
        logger.info("="*60)
        logger.info("CatBoost Gold Price Prediction - 开始训练")
        logger.info("="*60)
        logger.info(f"\n配置信息:")
        logger.info(f"  数据路径: {config.data_path}")
        logger.info(f"  输出路径: {config.output_dir}")
        logger.info(f"  训练范围: {config.train_start} - {config.train_end}")
        logger.info(f"  OOT范围:  {config.oot_start} - {config.oot_end}")
        
        # 1. 加载数据
        data_loader = DataLoader(config, logger)
        train_df, oot_df = data_loader.load_data()
        
        # 合并所有数据用于预处理
        if len(oot_df) > 0:
            df = pd.concat([train_df, oot_df], ignore_index=True)
        else:
            df = train_df
        
        # 2. 数据预处理
        preprocessor = DataPreprocessor(config, logger)
        df = preprocessor.preprocess(df)
        
        if preprocessor.target_col is None:
            raise ValueError("未找到目标变量，无法继续训练！")
        
        # 3. 准备训练数据
        trainer = ModelTrainer(config, logger)
        X_train, X_test, y_train, y_test, X_oot, y_oot = trainer.prepare_data(
            df, preprocessor.target_col, preprocessor.cat_features_names
        )
        
        # 4. 训练模型
        model = trainer.train(X_train, y_train, X_test, y_test)
        
        # 5. 评估模型
        evaluator = ModelEvaluator(config, logger)
        results = evaluator.evaluate(model, X_test, y_test, X_oot, y_oot)
        
        # 6. 保存结果
        saver = ResultSaver(config, logger)
        saver.save_all(model, X_train, y_test, results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ 训练流程全部完成！")
        logger.info("="*60)
        logger.info(f"\n生成的文件位于: {config.output_dir}")
        logger.info("\n模型加载示例:")
        logger.info("  from catboost import CatBoostClassifier")
        logger.info("  model = CatBoostClassifier()")
        logger.info(f"  model.load_model('{os.path.join(config.output_dir, 'catboost_gold_model.cbm')}')")
        
    except Exception as e:
        logger.error(f"\n❌ 训练过程出错: {str(e)}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
