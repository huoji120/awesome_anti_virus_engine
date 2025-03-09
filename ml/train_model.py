#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_data(malware_csv, whitelist_csv):
    """
    加载恶意软件和白名单CSV文件
    """
    print(f"加载恶意软件数据: {malware_csv}")
    
    # 预处理：先获取CSV的列数
    # 读取第一行以确定正确的列数
    try:
        header = pd.read_csv(malware_csv, nrows=1)
        expected_columns = len(header.columns)
        print(f"预期列数: {expected_columns}")
        
        # 使用自定义函数读取CSV，处理字段不足的行
        malware_df = pd.read_csv(
            malware_csv, 
            header=0,
            low_memory=False,
            on_bad_lines='skip',  # 跳过无法解析的行
            dtype=float,          # 将所有数据列转为浮点型
            converters={0: str}   # 第一列为文件路径，保持为字符串类型
        )
        
        # 检查列数是否不足，如果不足则填充0
        actual_columns = len(malware_df.columns)
        if actual_columns < expected_columns:
            for i in range(actual_columns, expected_columns):
                col_name = f"col_{i}"
                malware_df[col_name] = 0.0
                
        print(f"成功读取恶意软件数据，形状: {malware_df.shape}")
    except Exception as e:
        print(f"读取恶意软件数据时出错: {e}")
        return None, None
    
    malware_df['label'] = 1  # 恶意软件标签为1
    
    print(f"加载白名单数据: {whitelist_csv}")
    try:
        # 同样处理白名单数据
        whitelist_df = pd.read_csv(
            whitelist_csv, 
            header=0,
            low_memory=False,
            on_bad_lines='skip',
            dtype=float,
            converters={0: str}
        )
        
        # 确保列数与恶意软件数据一致
        whitelist_cols = len(whitelist_df.columns)
        malware_cols = len(malware_df.columns) - 1  # 减去标签列
        
        if whitelist_cols < malware_cols:
            for i in range(whitelist_cols, malware_cols):
                col_name = f"col_{i}"
                whitelist_df[col_name] = 0.0
                
        print(f"成功读取白名单数据，形状: {whitelist_df.shape}")
    except Exception as e:
        print(f"读取白名单数据时出错: {e}")
        return None, None
        
    whitelist_df['label'] = 0  # 白名单软件标签为0
    
    # 确保两个DataFrame的列完全一致（除了可能的文件路径差异）
    malware_features = set(malware_df.columns)
    whitelist_features = set(whitelist_df.columns)
    
    # 找出不同的列
    malware_only = malware_features - whitelist_features
    whitelist_only = whitelist_features - malware_features
    
    # 为缺少的列添加0值
    for col in malware_only:
        if col != 'label':
            whitelist_df[col] = 0.0
            
    for col in whitelist_only:
        if col != 'label':
            malware_df[col] = 0.0
    
    # 合并数据
    combined_df = pd.concat([malware_df, whitelist_df], ignore_index=True, sort=False)
    
    # 第一列通常是文件路径，需要将其移除
    # 先保存文件路径以便后续参考
    file_paths = combined_df.iloc[:, 0].tolist()
    
    features = combined_df.iloc[:, 1:-1]  # 除去第一列(文件路径)和最后一列(标签)
    labels = combined_df['label']
    
    print(f"数据加载完成: {len(malware_df)} 个恶意样本, {len(whitelist_df)} 个白名单样本")
    print(f"特征维度: {features.shape}")
    
    return features, labels

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost模型
    """
    print("开始训练XGBoost模型...")
    
    # 处理数据中可能存在的NaN值
    print("检查并填充缺失值...")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # 检查是否还有无限值，并将其替换为0
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print(f"处理后的训练数据形状: {X_train.shape}")
    print(f"处理后的测试数据形状: {X_test.shape}")
    
    # 设置XGBoost参数
    params = {
        'max_depth': 6,               # 树的最大深度
        'learning_rate': 0.1,         # 学习率
        'n_estimators': 100,          # 树的数量
        'objective': 'binary:logistic', # 二分类问题
        'eval_metric': 'logloss',     # 评估指标
        'subsample': 0.8,             # 样本采样率
        'colsample_bytree': 0.8,      # 特征采样率
        'random_state': 42            # 随机种子
    }
    
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(**params)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    print("模型训练完成！")
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    print("评估模型性能...")
    
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['白名单', '恶意软件']))
    
    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['白名单', '恶意软件'], 
                yticklabels=['白名单', '恶意软件'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 显示特征重要性
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('特征重要性')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return accuracy

def save_model(model, output_path='xgboost_malware_detector.model'):
    """
    保存模型到文件
    """
    print(f"保存模型到 {output_path}")
    joblib.dump(model, output_path)
    print("模型保存完成！")

def main():
    """
    主函数：加载数据，训练模型，评估结果，保存模型
    """
    try:
        print("开始恶意软件检测模型训练...")
        
        # 设置文件路径
        malware_csv = 'data/malware_features.csv'
        whitelist_csv = 'data/whitelist_features.csv'
        
        # 检查文件是否存在
        if not os.path.exists(malware_csv):
            print(f"错误: 找不到恶意软件特征文件 {malware_csv}")
            return
            
        if not os.path.exists(whitelist_csv):
            print(f"错误: 找不到白名单特征文件 {whitelist_csv}")
            return
        
        # 加载数据
        X, y = load_data(malware_csv, whitelist_csv)
        
        if X is None or y is None:
            print("数据加载失败，终止训练")
            return
            
        print(f"数据集加载完成，共 {len(X)} 个样本")
        
        # 数据划分
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            print(f"训练集: {len(X_train)} 样本，测试集: {len(X_test)} 样本")
        except Exception as e:
            print(f"数据划分出错: {e}")
            return
        
        # 训练模型
        try:
            model = train_xgboost_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"模型训练出错: {e}")
            return
        
        # 评估模型
        try:
            evaluate_model(model, X_test, y_test)
        except Exception as e:
            print(f"模型评估出错: {e}")
        
        # 保存模型
        try:
            save_model(model)
            print("模型训练和评估完成！")
        except Exception as e:
            print(f"模型保存出错: {e}")
        
    except Exception as e:
        print(f"训练过程中发生未预期错误: {e}")

if __name__ == "__main__":
    main() 