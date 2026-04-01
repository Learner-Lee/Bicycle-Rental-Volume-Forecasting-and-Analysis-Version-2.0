"""
特征重要性分析模块
方法：基于树模型（随机森林 / XGBoost）的分裂增益
依赖：需先运行 regression.py 完成训练
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import FEATURE_COLS

MODEL_DIR = os.path.dirname(__file__)


# ── 核心函数 ───────────────────────────────────────────────────
def get_importance(model_name: str = "随机森林") -> pd.DataFrame:
    """
    提取指定树模型的特征重要性，按得分降序排列。

    参数
    ----
    model_name : "随机森林" 或 "XGBoost"（梯度提升同样支持）

    返回
    ----
    pd.DataFrame  columns = ["特征", "重要性得分"]
    """
    # 动态导入避免循环依赖
    from regression import load_model

    model = load_model(model_name)

    importance = pd.DataFrame({
        "特征":     FEATURE_COLS,
        "重要性得分": model.feature_importances_,
    }).sort_values("重要性得分", ascending=False).reset_index(drop=True)

    return importance


def analyze(model_name: str = "随机森林") -> pd.DataFrame:
    """
    打印特征重要性排名并保存结果。

    参数
    ----
    model_name : 使用哪个树模型，默认随机森林

    返回
    ----
    pd.DataFrame  特征重要性排名表
    """
    importance = get_importance(model_name)
    importance["排名"] = range(1, len(importance) + 1)
    importance = importance[["排名", "特征", "重要性得分"]]

    save_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    importance.to_csv(save_path, index=False)

    print(f"\n── 特征重要性排名（{model_name}）────────────────────")
    print(importance.to_string(index=False))
    print(f"\n[feature_importance] 已保存 → {save_path}")
    return importance


if __name__ == "__main__":
    analyze("随机森林")
    print()
    analyze("XGBoost")
