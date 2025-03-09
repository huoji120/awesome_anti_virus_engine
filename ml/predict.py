#!/usr/bin/env python
# -*- coding: utf-8 -*-

import joblib
import pandas as pd
import numpy as np
import sys
import os

def load_model(model_path='xgboost_malware_detector.model'):
    """
    加载训练好的模型
    """
    print(f"正在加载模型: {model_path}")
    try:
        model = joblib.load(model_path)
        print("模型加载成功！")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def predict_file(model, csv_path):
    """
    对单个CSV文件进行预测
    """
    try:
        # 加载CSV文件
        df = pd.read_csv(csv_path)
        
        # 提取特征 (除去第一列文件路径)
        features = df.iloc[:, 1:]
        
        # 使用模型预测
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # 添加预测结果到数据框
        df['预测标签'] = predictions
        df['恶意软件概率'] = probabilities[:, 1]
        
        # 创建结果数据框
        results = pd.DataFrame({
            '文件路径': df.iloc[:, 0],
            '预测标签': predictions,
            '恶意软件概率': probabilities[:, 1]
        })
        
        # 保存结果到CSV
        output_path = os.path.splitext(csv_path)[0] + '_predictions.csv'
        results.to_csv(output_path, index=False)
        print(f"预测结果已保存到: {output_path}")
        
        # 打印概要
        malware_count = len(results[results['预测标签'] == 1])
        total_count = len(results)
        print(f"总样本数: {total_count}")
        print(f"检测为恶意软件: {malware_count} ({malware_count/total_count*100:.2f}%)")
        print(f"检测为白名单软件: {total_count - malware_count} ({(total_count-malware_count)/total_count*100:.2f}%)")
        
        return results
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None

def batch_predict(model, csv_paths):
    """
    批量预测多个CSV文件
    """
    results = {}
    for csv_path in csv_paths:
        print(f"\n分析文件: {csv_path}")
        result = predict_file(model, csv_path)
        if result is not None:
            results[csv_path] = result
    
    return results

def main():
    """
    主函数
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <csv文件路径1> [csv文件路径2] ...")
        return
    
    # 加载模型
    model = load_model()
    if model is None:
        return
    
    # 批量预测
    csv_paths = sys.argv[1:]
    batch_predict(model, csv_paths)

if __name__ == "__main__":
    main() 