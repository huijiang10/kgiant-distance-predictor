import pandas as pd
import numpy as np

def load_data(file_path):
    """
    加载数据并统一格式
    """
    # 读取由 CSV.py 生成的 data.csv
    df = pd.read_csv(file_path)
    
    # 无论 CSV 里是 rmag 还是 Rmag，统一用 df['rmag'] 调用
    df.columns = [col.strip().lower() for col in df.columns]
    
    return df

def clean_data(df):
    """
    物理筛选: 剔除异常值和空值，基于 Xue et al. (2014) 的物理逻辑
    """
    # 1. 剔除带有空值（NaN）的行
    df = df.dropna()

    # 2. 视星等 rmag 范围 
    # 剔除过亮（饱和）或过暗（信噪比低）的异常恒星
    df = df[(df['rmag'] > 14) & (df['rmag'] < 22)]

    # 3. 颜色 g_r 范围 
    # K巨星的典型颜色范围
    df = df[(df['g_r'] > 0.5) & (df['g_r'] < 1.4)]

    # 4. 金属丰度 feh 范围 
    df = df[(df['feh'] > -2.5) & (df['feh'] < 0.5)]

    # 5. 表面重力 logg 范围 
    # 剔除矮星 (logg > 3.5) 和极端亮星 (logg < 2.0)
    df = df[(df['logg'] > 1.0) & (df['logg'] < 3.5)]

    # 6. 筛选 e_feh 极小的数据，减少噪声并降低计算量
    df = df[df['e_feh'] < 0.08]

    # 重置索引
    df.reset_index(drop=True, inplace=True)
    
    return df