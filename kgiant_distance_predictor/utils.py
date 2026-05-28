import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
import numpy as np

def evaluate(y_true, y_pred):
    """模型评估:R²、RMSE"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def plot_result(y_true, y_pred, save_path="result.png"):
    """绘制真实值 vs 预测值"""
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_distribution(y_true, y_pred, save_path="error_dist.png"):
    """
    功能：绘制误差（残差）的分布直方图
    """
    # 1. 计算误差：真实值 - 预测值
    errors = y_true - y_pred
    
    # 2. 开始绘图
    plt.figure(figsize=(8, 6))
    
    # 3. 画直方图
    plt.hist(errors, bins=50, color='skyblue', alpha=0.8, edgecolor='black')
    
    # 4. 添加均值线
    mean_err = np.mean(errors)
    plt.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_err:.2f}')
    
    # 5. 添加标签和标题
    plt.xlabel("Error (True - Pred)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    # 6. 保存并关闭
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"误差分布图已保存: {save_path}")

def plot_learning_curve(model, X, y, save_path="learning_curve.png"):
    """
    功能：绘制模型的学习曲线
    """
    # 生成学习曲线数据
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10), # 选取10个不同比例的训练集大小
        cv=5,
        scoring='r2', # 因为你的评估指标是 R²
        n_jobs=-1
    )

    # 计算均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # 开始绘图
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve (SVR)")
    plt.xlabel("Training Examples")
    plt.ylabel("Score (R²)")
    
    # 绘制训练集得分
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    
    # 绘制交叉验证得分
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"学习曲线图已保存: {save_path}")

def plot_hrd(df, teff_col="teff", logg_col="logg", save_path="hr_diagram.png"):
    """
    绘制赫罗图 (HR Diagram)
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制散点图
    sc = plt.scatter(df[teff_col], df[logg_col], c=df['rmag'], 
                     cmap='plasma', s=5, alpha=0.6, edgecolors='none')
    
    # 反向坐标轴
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    plt.title("Hertzsprung-Russell Diagram (K-Giants)", fontsize=15)
    plt.xlabel("Effective Temperature (Teff) [K]", fontsize=12)
    plt.ylabel("Surface Gravity (log g)", fontsize=12)
    
    # 加上颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('r mag')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"赫罗图已保存: {save_path}")

def plot_distance_distribution(distance_modulus_array, save_path="distance_distribution.png"):
    """
    功能：绘制 2万+ 样本的距离模数（距离）分布直方图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，bins设为50个柱子以观察分布细节
    plt.hist(distance_modulus_array, bins=50, color='#1f77b4', alpha=0.75, edgecolor='black')
    
    plt.xlabel('Distance Modulus (mag)', fontsize=12)
    plt.ylabel('Number of Stars', fontsize=12)
    plt.title(f'Distribution of Distance Modulus for {len(distance_modulus_array)} K-Giants', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加均值参考线
    mean_dm = np.mean(distance_modulus_array)
    plt.axvline(mean_dm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dm:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"距离分布图已保存: {save_path}")

def plot_residual_analysis(y_true, y_pred, save_path="residual_analysis.png"):
    """
    功能：绘制残差分析图 (Residual Analysis)
    """
    # 1. 计算残差
    residuals = y_true - y_pred
    
    # 2. 计算指标 
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 3. 开始绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(y_pred, residuals, alpha=0.6, color='#1f77b4', edgecolors='w', s=50)
    
    # 绘制 y=0 的红色基准线
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='dashed', linewidth=2)
    
    # 标签和标题
    plt.xlabel("Predicted Distance Modulus", fontsize=12)
    plt.ylabel("Residuals (True - Predicted)", fontsize=12)
    plt.title("Residual Analysis: Predicted vs. Residuals", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在图上标注 R² 和 RMSE
    stats_text = f'$R^2 = {r2:.4f}$\nRMSE = {rmse:.4f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存并关闭
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"残差分析图已保存: {save_path}")    