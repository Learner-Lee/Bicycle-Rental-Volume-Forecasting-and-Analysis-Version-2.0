"""
聚类分析模块
方法：K-Means
特征：hr, workingday, season, weathersit, atemp
附带：肘部法则选 K，PCA 降维用于可视化
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import load

MODEL_DIR  = os.path.dirname(__file__)
CSV_PATH   = os.path.join(MODEL_DIR, "..", "hour.csv")
WASH_PATH  = os.path.join(MODEL_DIR, "..", "hour_wash.csv")

# 聚类使用的特征（不含目标变量）
CLUSTER_FEATURES = ["hr", "workingday", "season", "weathersit", "atemp"]

# 簇标签含义映射（根据聚类中心特征人工命名，训练后可按实际调整）
CLUSTER_LABELS = {
    0: "通勤型",
    1: "休闲型",
    2: "低活跃型",
}


# ── 肘部法则 ───────────────────────────────────────────────────
def elbow(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
    """
    计算不同 K 值下的 Inertia（簇内误差平方和），用于肘部法则选 K。

    返回
    ----
    pd.DataFrame  columns = ["K", "Inertia"]
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        records.append({"K": k, "Inertia": round(km.inertia_, 2)})

    elbow_df = pd.DataFrame(records)
    print("\n── 肘部法则（选取曲线拐点作为 K）────────────────────")
    print(elbow_df.to_string(index=False))
    return elbow_df


# ── 训练 ───────────────────────────────────────────────────────
def train(k: int = 3) -> pd.DataFrame:
    """
    训练 K-Means 聚类模型，保存模型与结果。

    参数
    ----
    k : 聚类数量，默认 3（通勤型 / 休闲型 / 低活跃型）

    返回
    ----
    pd.DataFrame  原始数据附加 cluster（簇编号）和 cluster_label（簇名称）列
    """
    df = load(csv_path=CSV_PATH, output_path=WASH_PATH)
    X  = df[CLUSTER_FEATURES].values

    # 标准化（K-Means 对量纲敏感）
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练
    print(f"[clustering] 训练 K-Means（K={k}）...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS).fillna("其他")

    # 保存模型与 scaler
    joblib.dump(km,     os.path.join(MODEL_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "kmeans_scaler.pkl"))

    # 保存带标签的数据
    result_path = os.path.join(MODEL_DIR, "cluster_result.csv")
    df.to_csv(result_path, index=False)

    # 打印各簇统计
    summary = df.groupby("cluster_label").agg(
        样本数=("cnt", "count"),
        平均租车量=("cnt", lambda x: round(x.mean(), 1)),
        平均小时=("hr", lambda x: round(x.mean(), 1)),
        工作日占比=("workingday", lambda x: f"{x.mean():.1%}"),
    )
    print("\n── 聚类结果统计 ──────────────────────────────────")
    print(summary.to_string())
    print(f"\n[clustering] 已保存 → {result_path}")
    return df


# ── PCA 降维（供可视化使用）──────────────────────────────────────
def pca_transform(k: int = 3) -> pd.DataFrame:
    """
    对聚类特征做 PCA 降至 2 维，返回含坐标和簇标签的 DataFrame，供散点图使用。

    返回
    ----
    pd.DataFrame  columns = ["PC1", "PC2", "cluster", "cluster_label"]
    """
    result_path = os.path.join(MODEL_DIR, "cluster_result.csv")
    if not os.path.exists(result_path):
        train(k)

    df     = pd.read_csv(result_path)
    X      = df[CLUSTER_FEATURES].values
    scaler = joblib.load(os.path.join(MODEL_DIR, "kmeans_scaler.pkl"))
    X_scaled = scaler.transform(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1":          coords[:, 0],
        "PC2":          coords[:, 1],
        "cluster":      df["cluster"],
        "cluster_label": df["cluster_label"],
    })

    pca_path = os.path.join(MODEL_DIR, "pca_result.csv")
    pca_df.to_csv(pca_path, index=False)
    print(f"[clustering] PCA 结果已保存 → {pca_path}")
    return pca_df


if __name__ == "__main__":
    df = load(csv_path=CSV_PATH, output_path=WASH_PATH)
    X  = df[CLUSTER_FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow(X_scaled)
    train(k=3)
    pca_transform(k=3)
