"""
分类任务模块
方法：决策树
目标：将 cnt 划分为低 / 中 / 高需求三档，预测需求级别
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import load, FEATURE_COLS

MODEL_DIR = os.path.dirname(__file__)
CSV_PATH  = os.path.join(MODEL_DIR, "..", "hour.csv")
WASH_PATH = os.path.join(MODEL_DIR, "..", "hour_wash.csv")

# 需求档位定义（对应文档第五章）
BINS   = [0, 100, 300, float("inf")]
LABELS = [0, 1, 2]                    # 0=低需求  1=中需求  2=高需求
LABEL_NAMES = {0: "低需求", 1: "中需求", 2: "高需求"}


# ── 构造分类目标 ───────────────────────────────────────────────
def make_target(df: pd.DataFrame) -> pd.Series:
    """将连续的 cnt 按阈值（100 / 300）划分为 0 / 1 / 2 三档"""
    return pd.cut(df["cnt"], bins=BINS, labels=LABELS).astype(int)


# ── 训练 ───────────────────────────────────────────────────────
def train(max_depth: int = 5, test_size: float = 0.2) -> dict:
    """
    训练决策树分类模型，保存模型与评估结果。

    参数
    ----
    max_depth : 决策树最大深度，默认 5（限制深度防止过拟合，同时保持可解释性）
    test_size : 测试集比例，默认 0.2

    返回
    ----
    dict  包含 accuracy / f1 / confusion_matrix / report
    """
    df = load(csv_path=CSV_PATH, output_path=WASH_PATH)
    X  = df[FEATURE_COLS]
    y  = make_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"[classification] 训练决策树（max_depth={max_depth}）...")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, os.path.join(MODEL_DIR, "decision_tree.pkl"))

    # 评估
    y_pred   = model.predict(X_test)
    acc      = round(accuracy_score(y_test, y_pred), 4)
    f1       = round(f1_score(y_test, y_pred, average="macro"), 4)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in LABELS]
    )

    # 类别分布
    dist = y.value_counts().sort_index()
    dist.index = [LABEL_NAMES[i] for i in dist.index]

    print("\n── 类别分布 ──────────────────────────────────────")
    for label, count in dist.items():
        print(f"  {label}：{count} 条（{count/len(y):.1%}）")

    print(f"\n── 评估结果 ──────────────────────────────────────")
    print(f"  准确率 Accuracy : {acc}")
    print(f"  F1 Score (macro): {f1}")
    print(f"\n── 混淆矩阵 ──────────────────────────────────────")
    cm_df = pd.DataFrame(
        cm,
        index   = [f"实际-{LABEL_NAMES[i]}" for i in LABELS],
        columns = [f"预测-{LABEL_NAMES[i]}" for i in LABELS],
    )
    print(cm_df.to_string())
    print(f"\n── 分类报告 ──────────────────────────────────────")
    print(report)

    # 决策树规则（前 5 层）
    rules = export_text(model, feature_names=FEATURE_COLS, max_depth=3)
    print("── 决策树规则（前 3 层）──────────────────────────")
    print(rules)

    # 保存评估结果
    result = {"accuracy": acc, "f1": f1,
              "confusion_matrix": cm, "report": report}
    cm_df.to_csv(os.path.join(MODEL_DIR, "classification_confusion_matrix.csv"))

    return result


def load_model():
    """读取已保存的决策树模型"""
    path = os.path.join(MODEL_DIR, "decision_tree.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在：{path}，请先运行 train()")
    return joblib.load(path)


if __name__ == "__main__":
    train()
