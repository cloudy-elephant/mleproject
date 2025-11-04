#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost Gold Price Prediction - Inference Script
=================================================
黄金价格预测模型 - 预测脚本

使用方法:
    python catboost_gold_predict.py --model_path ./output/catboost_gold_model.cbm --data_path ./new_data.csv

作者: Auto-generated
日期: 2024
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model(model_path: str, logger: logging.Logger) -> CatBoostClassifier:
    """加载模型"""
    logger.info(f"加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    logger.info("✓ 模型加载成功")
    return model


def load_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    """加载待预测数据"""
    logger.info(f"加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ 数据加载成功: {df.shape[0]} 条 × {df.shape[1]} 列")
    
    return df


def preprocess_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """预处理数据（需要与训练时保持一致）"""
    logger.info("预处理数据...")
    
    df = df.copy()
    
    # 1. 移除ID列
    id_cols = ['customerID', 'id', 'ID', 'customer_id', 'CustomerId']
    for col in id_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            logger.info(f"  移除ID列: {col}")
    
    # 2. 处理日期特征
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
                logger.info(f"  提取日期特征: {col}")
            except:
                pass
    
    # 3. 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
    
    # 4. 转换类别特征为字符串（关键！）
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    logger.info("✓ 数据预处理完成")
    
    return df


def make_predictions(model: CatBoostClassifier, df: pd.DataFrame, 
                    logger: logging.Logger, output_path: str = None):
    """进行预测"""
    logger.info("开始预测...")
    
    # 排除不需要的列
    exclude_cols = ['year', 'month', 'year_month']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # 预测
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    # 如果原始数据有ID列，添加回去
    if 'customerID' in df.columns:
        results.insert(0, 'customerID', df['customerID'])
    
    logger.info(f"✓ 预测完成: {len(results)} 条记录")
    logger.info(f"\n预测分布:")
    logger.info(f"  类别0: {(predictions == 0).sum()} 条 ({(predictions == 0).sum() / len(predictions) * 100:.2f}%)")
    logger.info(f"  类别1: {(predictions == 1).sum()} 条 ({(predictions == 1).sum() / len(predictions) * 100:.2f}%)")
    logger.info(f"\n平均预测概率: {probabilities.mean():.4f}")
    
    # 保存结果
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"✓ 预测结果已保存: {output_path}")
    
    return results

# read data gold something issues, can only predict one month.
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CatBoost Gold Price Prediction - Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径 (.cbm)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='待预测数据文件路径 (.csv)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='预测结果保存路径 (.csv)')
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f'predictions_{timestamp}.csv'
    
    # 设置日志
    logger = setup_logging()
    
    try:
        logger.info("="*60)
        logger.info("CatBoost Gold Price Prediction - 开始预测")
        logger.info("="*60)
        
        # 1. 加载模型
        model = load_model(args.model_path, logger)
        
        # 2. 加载数据
        df = load_data(args.data_path, logger)
        
        # 3. 预处理数据
        df = preprocess_data(df, logger)
        
        # 4. 进行预测
        results = make_predictions(model, df, logger, args.output_path)
        
        logger.info("\n" + "="*60)
        logger.info("✅ 预测流程完成！")
        logger.info("="*60)
        
        # 显示前几行结果
        logger.info("\n预测结果示例（前5行）:")
        logger.info(f"\n{results.head()}")
        
    except Exception as e:
        logger.error(f"\n❌ 预测过程出错: {str(e)}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
