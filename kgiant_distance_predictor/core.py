from .data import load_data, clean_data
from .model import build_model, train_model
from .utils import evaluate, plot_result,plot_error_distribution,plot_learning_curve,plot_hrd,plot_distance_distribution, plot_residual_analysis 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_pipeline(csv_path, features, m_col="rmag", target_col="dm50"):
    """
    完整流程：
    1. 加载真实数据并清洗
    2. 划分训练集和测试集（模拟未知数据）
    3. 训练 SVR 模型预测距离模数 (DM)
    4. 评估模型在测试集上的表现
    5. 计算测试集恒星的真实距离 (pc)
    """
    # 1. 加载与清洗数据
    df = load_data(csv_path)
    df = clean_data(df)
    plot_hrd(df)
    plot_distance_distribution(df[target_col]) 

    # 2. 准备特征(X)、目标(y)和视星等(m)
    X = df[features].values
    y = df[target_col].values
    m = df[m_col].values

    # 3. 划分训练集和测试集 (关键：80%训练，20%测试)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X, y, m, 
        test_size=0.2, 
        random_state=42
    )

    # 4. 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. 训练模型
    model = build_model()
    train_model(model, X_train, y_train)
    plot_learning_curve(model.best_estimator_, X_train, y_train)

    # 6. 预测测试集的距离模数 (DM_pred)
    dm_pred = model.predict(X_test)

    # 7. 模型评估 (对比 预测的DM 和 真实的DM)
    r2, rmse = evaluate(y_test, dm_pred)
    plot_result(y_test, dm_pred)
    plot_error_distribution(y_test, dm_pred) 

    # 8.残差分析图
    plot_residual_analysis(y_test, dm_pred) 

    # 9. 计算距离 (d = 10^((m - M + 5) / 5))
    # 使用测试集的视星等(m_test)和预测的距离模数(dm_pred)计算
    distance_pc = 10 ** ((m_test - dm_pred + 5) / 5)

    return {
        "r2": r2,
        "rmse": rmse,
        "model": model,
        "scaler": scaler,
        "distance_pc": distance_pc  # 返回计算好的秒差距距离
    }