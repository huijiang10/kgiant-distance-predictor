from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def build_model():
    """带有自动调参功能的 SVR 模型"""
    # 定义机器自动搜索的参数范围
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'epsilon': [0.05, 0.1, 0.2] ,
    }
    # 使用网格搜索自动寻找最佳的 C 和 gamma,epsilon
    # cv=5 代表5折交叉验证，防止过拟合；n_jobs=-1 调用所有CPU核心加速
    model = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=-1,verbose=1)
    return model

def train_model(model, X_train, y_train):
    """训练模型(GridSearchCV 会在训练时自动跑完所有参数组合）"""
    model.fit(X_train, y_train)
    # 训练结束后，打印出机器所能选出的最佳参数
    print("选出最佳参数：", model.best_params_)