"""
共享单车数据预处理模块
对应文档：数据预处理.md
输入：hour.csv
输出：处理后的 DataFrame，包含 23 个特征 + cnt_log + cnt

持久化：
  save(df, path)   → 保存为 Parquet
  load(path)       → 读取 Parquet，若不存在则重新处理并保存
"""

import os
import numpy as np
import pandas as pd


# ── 1. 数据清洗 ────────────────────────────────────────────────

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除无效列：行号 instant、日期 dteday（已被时间字段覆盖）"""
    return df.drop(columns=["instant", "dteday"])


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除数据泄漏列：casual 和 registered 直接构成目标变量 cnt"""
    return df.drop(columns=["casual", "registered"])


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除冗余列：temp 与 atemp 相关系数 0.99，保留体感温度 atemp"""
    return df.drop(columns=["temp"])


def merge_rare_weathersit(df: pd.DataFrame) -> pd.DataFrame:
    """将极稀少的 weathersit=4（暴雨，仅 3 条）并入 weathersit=3（恶劣天气）"""
    df = df.copy()
    df["weathersit"] = df["weathersit"].replace(4, 3)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗总入口：依次执行所有清洗步骤"""
    df = drop_useless_columns(df)
    df = drop_leakage_columns(df)
    df = drop_redundant_columns(df)
    df = merge_rare_weathersit(df)
    return df


# ── 2. 特征工程 ────────────────────────────────────────────────

def add_time_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增 time_period：将 24 小时划分为 5 个骑行时段
      1 深夜    0–6 时
      2 早高峰  7–9 时
      3 日间   10–16 时
      4 晚高峰 17–19 时
      5 夜间   20–23 时
    """
    def _label(hr):
        if hr <= 6:
            return 1
        elif hr <= 9:
            return 2
        elif hr <= 16:
            return 3
        elif hr <= 19:
            return 4
        else:
            return 5

    df = df.copy()
    df["time_period"] = df["hr"].map(_label)
    return df


def add_cyclic_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增正余弦循环编码，处理时间字段的周期性：
      hr (周期 24)      → hr_sin,   hr_cos
      mnth (周期 12)    → mnth_sin, mnth_cos
      weekday (周期 7)  → wek_sin,  wek_cos
      season (周期 4)   → seas_sin, seas_cos
    """
    df = df.copy()
    df["hr_sin"]   = np.sin(2 * np.pi * df["hr"]      / 24)
    df["hr_cos"]   = np.cos(2 * np.pi * df["hr"]      / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"]    / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"]    / 12)
    df["wek_sin"]  = np.sin(2 * np.pi * df["weekday"] / 7)
    df["wek_cos"]  = np.cos(2 * np.pi * df["weekday"] / 7)
    df["seas_sin"] = np.sin(2 * np.pi * df["season"]  / 4)
    df["seas_cos"] = np.cos(2 * np.pi * df["season"]  / 4)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增交互特征：
      hr_workingday  = hr × workingday      捕捉工作日通勤双峰 vs 周末单峰
      temp_hum       = atemp × (1 - hum)   体感舒适度（温暖且干燥时最高）
      season_hr      = season × hr          不同季节的活跃时段偏好
    """
    df = df.copy()
    df["hr_workingday"] = df["hr"]    * df["workingday"]
    df["temp_hum"]      = df["atemp"] * (1 - df["hum"])
    df["season_hr"]     = df["season"] * df["hr"]
    return df


def add_target_log(df: pd.DataFrame) -> pd.DataFrame:
    """新增 cnt_log = log(1 + cnt)，压缩右偏分布，用于模型训练"""
    df = df.copy()
    df["cnt_log"] = np.log1p(df["cnt"])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程总入口：依次执行所有特征构造步骤"""
    df = add_time_period(df)
    df = add_cyclic_encoding(df)
    df = add_interaction_features(df)
    df = add_target_log(df)
    return df


# ── 3. 完整流水线 ──────────────────────────────────────────────

# 23 个输入特征（与文档第五节一致）
FEATURE_COLS = [
    # 时间原始
    "yr", "mnth", "hr",
    # 时间循环编码
    "hr_sin", "hr_cos", "mnth_sin", "mnth_cos",
    "wek_sin", "wek_cos", "seas_sin", "seas_cos",
    # 时间分段
    "time_period",
    # 日历
    "season", "holiday", "weekday", "workingday",
    # 天气
    "weathersit", "atemp", "hum", "windspeed",
    # 交互
    "hr_workingday", "temp_hum", "season_hr",
]

TARGET_LOG = "cnt_log"   # 建模用
TARGET_RAW = "cnt"       # 还原用


def preprocess(path: str) -> pd.DataFrame:
    """
    完整预处理流水线。

    参数
    ----
    path : str
        hour.csv 的文件路径

    返回
    ----
    pd.DataFrame
        包含 FEATURE_COLS + cnt_log + cnt 的处理后数据集
    """
    df = pd.read_csv(path)
    df = clean(df)
    df = engineer(df)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """从处理后的 DataFrame 中分离特征矩阵 X 和目标向量 y"""
    X = df[FEATURE_COLS]
    y = df[TARGET_LOG]
    return X, y


def inverse_transform(y_pred: np.ndarray) -> np.ndarray:
    """将模型预测的 cnt_log 还原为真实租车量：e^y - 1"""
    return np.expm1(y_pred)


# ── 4. 持久化 ──────────────────────────────────────────────────

PROCESSED_FILE = os.path.join(os.path.dirname(__file__), "hour_wash.csv")


def save(df: pd.DataFrame, path: str = PROCESSED_FILE) -> None:
    """将处理后的 DataFrame 保存为 CSV 文件"""
    df.to_csv(path, index=False)
    print(f"[save] 已保存 → {path}  ({df.shape[0]} 行 × {df.shape[1]} 列)")


def load(csv_path: str = None, output_path: str = PROCESSED_FILE) -> pd.DataFrame:
    """
    读取持久化文件。若文件不存在，则从 csv_path 重新预处理并保存。

    参数
    ----
    csv_path    : 原始 CSV 路径，仅在 hour_wash.csv 不存在时使用
    output_path : 持久化文件路径，默认 hour_wash.csv

    返回
    ----
    pd.DataFrame  处理后的完整数据集
    """
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"[load] 读取缓存 ← {output_path}  ({df.shape[0]} 行 × {df.shape[1]} 列)")
        return df

    if csv_path is None:
        raise FileNotFoundError(
            f"缓存文件 {output_path} 不存在，请提供 csv_path 以重新生成。"
        )

    print(f"[load] 缓存不存在，从 {csv_path} 重新预处理...")
    df = preprocess(csv_path)
    save(df, output_path)
    return df


# ── 5. 快速验证 ────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "hour.csv")

    # 首次运行：预处理并保存
    df = load(csv_path=csv_path)
    # 再次运行：直接读取缓存（注释掉上一行，取消注释下一行即可验证）
    # df = load()

    X, y = get_X_y(df)
    print(f"输入特征数：{X.shape[1]}")
    print(f"目标变量  ：{TARGET_LOG}（范围 {y.min():.2f} ~ {y.max():.2f}）")
