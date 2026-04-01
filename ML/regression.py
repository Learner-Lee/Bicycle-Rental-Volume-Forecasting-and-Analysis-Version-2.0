"""
回归预测模块
模型：线性回归、岭回归、随机森林、梯度提升、XGBoost
目标：预测 cnt_log，评估后还原为真实租车量
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import load, get_X_y, inverse_transform

# ── 路径 ───────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)
CSV_PATH  = os.path.join(MODEL_DIR, "..", "hour.csv")
WASH_PATH = os.path.join(MODEL_DIR, "..", "hour_wash.csv")


# ── 模型定义 ───────────────────────────────────────────────────
MODELS = {
    "线性回归":   LinearRegression(),
    "岭回归":     Ridge(alpha=1.0),
    "随机森林":   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "梯度提升":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":    XGBRegressor(n_estimators=100, random_state=42,
                               verbosity=0, n_jobs=-1),
}


# ── 评估 ───────────────────────────────────────────────────────
def evaluate(name: str, model, X_test: pd.DataFrame,
             y_test: pd.Series) -> dict:
    """计算 MAE / RMSE / R²，预测值还原为真实租车量后再计算"""
    y_pred_log = model.predict(X_test)
    y_pred     = inverse_transform(y_pred_log)
    y_true     = inverse_transform(y_test.values)

    return {
        "模型":  name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R²":    round(r2_score(y_true, y_pred), 4),
    }


# ── 训练与保存 ─────────────────────────────────────────────────
def train(test_size: float = 0.2) -> pd.DataFrame:
    """
    训练所有回归模型，保存到 ML/ 目录，返回评估结果汇总表。

    参数
    ----
    test_size : 测试集比例，默认 0.2

    返回
    ----
    pd.DataFrame  各模型的 MAE / RMSE / R²
    """
    df = load(csv_path=CSV_PATH, output_path=WASH_PATH)
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = []
    for name, model in MODELS.items():
        print(f"[regression] 训练 {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        results.append(evaluate(name, model, X_test, y_test))

    result_df = pd.DataFrame(results).set_index("模型")
    result_df.to_csv(os.path.join(MODEL_DIR, "regression_results.csv"))
    print("\n── 回归评估结果 ──────────────────────────────")
    print(result_df.to_string())
    return result_df


def load_model(name: str):
    """读取已保存的回归模型"""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在：{path}，请先运行 train()")
    return joblib.load(path)


if __name__ == "__main__":
    train()
