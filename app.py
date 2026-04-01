# -*- coding: utf-8 -*-
"""
自行车租赁数据挖掘平台 / Bike Sharing Data Mining Platform
Port: 8400  |  UCI Bike Sharing Dataset (hour.csv)
"""
import time, math, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib, os, sys

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import FEATURE_COLS, inverse_transform

# ── Paths ─────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
WASH  = os.path.join(BASE, "hour_wash.csv")
RAW   = os.path.join(BASE, "hour.csv")
MLDIR = os.path.join(BASE, "ML")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🚲 Bike Sharing · Data Mining",
    page_icon="🚲", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 全局样式 / Global Styles
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #283593 60%, #1565c0 100%);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #e8edf5 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
[data-testid="stHeader"] { display: none; }
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.main { background: #f5f7fa; }

/* ── KPI Card ── */
.kpi {
    background: white; border-radius: 14px; padding: 22px 20px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07); text-align: center;
    border-top: 5px solid #2196F3; margin-bottom: 8px;
}
.kpi-v { font-size: 2.1rem; font-weight: 800; color: #1a237e; line-height: 1.1; }
.kpi-l { font-size: 0.85rem; color: #6b7280; margin-top: 6px; }

/* ── Insight Box ── */
.insight {
    background: #e8f4fd; border-left: 5px solid #1976d2;
    border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 8px 0 16px;
    font-size: 0.9rem; line-height: 1.7; color: #1a237e;
}
/* ── Warning Box ── */
.warn-box {
    background: #fff8e1; border-left: 5px solid #f9a825;
    border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 8px 0 12px;
    font-size: .9rem; line-height: 1.75; color: #5d4037;
}
/* ── Algo Description ── */
.algo-desc {
    background: #fafafa; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 18px 22px; line-height: 1.8; font-size: 0.92rem; margin-bottom: 18px;
}
/* ── Page Title ── */
.page-title {
    font-size: 1.75rem; font-weight: 800; color: #1a237e;
    margin-bottom: 1.4rem; padding-bottom: 0.5rem; border-bottom: 3px solid #1976d2;
}
/* ── Section Header ── */
.sec { font-size: 1.1rem; font-weight: 700; color: #1a237e; margin: 1.4rem 0 .6rem; }
/* ── Summary Banner ── */
.banner {
    background: linear-gradient(135deg,#1a237e,#1565c0);
    border-radius: 12px; padding: 14px 20px; color: white !important;
    font-size: .95rem; margin-bottom: 16px;
}
.banner * { color: white !important; }

/* ── Note TextArea (student notes) ── */
[data-testid="stTextArea"] { margin: 4px 0 12px !important; }
[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] textarea {
    background: #e8f4fd !important;
    border-left: 5px solid #1976d2 !important;
    border-top: none !important; border-right: none !important; border-bottom: none !important;
    border-radius: 0 10px 10px 0 !important;
    color: #1a237e !important; font-size: 0.9rem !important; line-height: 1.7 !important;
    padding: 12px 16px !important; resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    box-shadow: 0 0 0 2px rgba(25,118,210,0.25) !important;
    outline: none !important;
}

/* ── Apple-style Segmented Control (语言切换) ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type {
    background: rgba(0,0,0,0.25);
    border-radius: 12px; padding: 3px !important; gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
    margin-bottom: 4px;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type button {
    border-radius: 9px !important; border: none !important;
    background: transparent !important; color: rgba(200,220,255,0.55) !important;
    font-size: 0.80rem !important; font-weight: 500 !important;
    letter-spacing: 0.02em !important; height: 32px !important;
    padding: 0 6px !important;
    transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="baseButton-primary"] {
    background: rgba(255,255,255,0.93) !important; color: #1a237e !important;
    font-weight: 650 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.28), 0 0.5px 1px rgba(0,0,0,0.12) !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="baseButton-secondary"]:hover {
    background: rgba(255,255,255,0.12) !important; color: rgba(220,235,255,0.85) !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="baseButton-primary"]:hover {
    background: white !important; color: #1a237e !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 语言初始化 / Language Init
# ══════════════════════════════════════════════════════════════
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# ── 双语文本表 / Bilingual Text Table ────────────────────────
TEXTS = {
    "app_title":       {"zh": "🚲 数据挖掘平台",         "en": "🚲 Data Mining Platform"},
    "app_subtitle":    {"zh": "自行车租赁量预测分析",     "en": "Bike Sharing Prediction"},
    "choose_algo":     {"zh": "选择算法：",                "en": "Choose Algorithm:"},
    "dataset_info": {
        "zh": "📦 数据集：UCI Bike Sharing<br>📊 记录数：17,379 条<br>📅 时间跨度：2011–2012",
        "en": "📦 Dataset: UCI Bike Sharing<br>📊 Records: 17,379<br>📅 Period: 2011–2012",
    },
    "done_label":      {"zh": "✅ 已完成模型：",           "en": "✅ Completed Models:"},
    "nav_dashboard":   {"zh": "📊 Dashboard",             "en": "📊 Dashboard"},
    "nav_preprocess":  {"zh": "🔧 数据预处理",             "en": "🔧 Preprocessing"},
    "nav_ml":          {"zh": "🤖 Deep Learning",         "en": "🤖 Deep Learning"},
    "nav_summary":     {"zh": "📋 总结报告",               "en": "📋 Summary"},
    "nav_test":        {"zh": "🔮 测试模型",               "en": "🔮 Test Model"},

    # Dashboard
    "dash_title":      {"zh": "📊 数据总览 — Dashboard",  "en": "📊 Data Overview — Dashboard"},
    "kpi_rows":        {"zh": "数据总条数（条）",           "en": "Total Records"},
    "kpi_total":       {"zh": "总租赁次数（次）",           "en": "Total Rentals"},
    "kpi_avg":         {"zh": "小时均租赁量（次/时）",      "en": "Avg Hourly Rentals"},
    "kpi_max":         {"zh": "单时最高租赁量（次）",       "en": "Peak Hourly Rentals"},
    "note_placeholder":{"zh": "在此记录你的观察与洞察...", "en": "Write your observations here..."},
    "raw_data_expander":{"zh": "📋 原始数据预览（前100行）","en": "📋 Raw Data Preview (first 100 rows)"},

    # Model page
    "complexity":      {"zh": "复杂度：",     "en": "Complexity:"},
    "algo_intro":      {"zh": "📚 算法介绍（点击展开/收起）", "en": "📚 Algorithm Introduction"},
    "run_btn":         {"zh": "▶ 运行 {name}",              "en": "▶ Run {name}"},
    "spinner":         {"zh": "⏳ 正在训练 {name}，请稍候...", "en": "⏳ Training {name}, please wait..."},
    "success":         {"zh": "✅ {name} 训练完成！",        "en": "✅ {name} training complete!"},
    "run_hint":        {"zh": "👆 点击上方「运行」按钮，开始训练模型并查看结果。",
                        "en": "👆 Click the Run button above to train the model and view results."},
    "metrics_title":   {"zh": "📊 模型评估指标",             "en": "📊 Model Evaluation Metrics"},
    "metric_rmse":     {"zh": "RMSE（均方根误差）",          "en": "RMSE"},
    "metric_rmse_help":{"zh": "越小越好，单位与租赁量相同",  "en": "Lower is better; same unit as rentals"},
    "metric_mae":      {"zh": "MAE（平均绝对误差）",         "en": "MAE"},
    "metric_mae_help": {"zh": "越小越好，直观反映平均误差",  "en": "Lower is better; average absolute error"},
    "metric_r2":       {"zh": "R²（决定系数）",              "en": "R² Score"},
    "metric_r2_help":  {"zh": "越接近1越好；>0.9为优秀",     "en": "Closer to 1 is better; >0.9 is excellent"},
    "metric_time":     {"zh": "训练时长",                    "en": "Train Time"},
    "metric_time_help":{"zh": "模型拟合所用时间",            "en": "Time to fit the model"},

    # Summary page
    "sum_title":       {"zh": "📋 模型总结报告 — Summary",  "en": "📋 Model Summary Report"},
    "sum_no_model": {
        "zh": "⚠️ 尚未运行任何模型，请先前往「Deep Learning」页面运行模型。",
        "en": "⚠️ No models have been run yet. Please go to the Deep Learning page first.",
    },
    "sum_hint": {
        "zh": "建议运行顺序：线性回归 → 岭回归 → 随机森林 → 梯度提升 → XGBoost",
        "en": "Suggested order: Linear → Ridge → Random Forest → Gradient Boosting → XGBoost",
    },
    "col_model":       {"zh": "模型",         "en": "Model"},
    "col_train_time":  {"zh": "训练时长(s)",  "en": "Train Time(s)"},
    "excellent_threshold": {"zh": "优秀阈值 0.90", "en": "Excellent 0.90"},
    "sum_analysis_title":  {"zh": "💬 综合分析与学习建议",  "en": "💬 Analysis & Learning Suggestions"},
    "radar_r2":    {"zh": "R²精度",    "en": "R² Score"},
    "radar_rmse":  {"zh": "RMSE(低好)", "en": "RMSE(↓)"},
    "radar_mae":   {"zh": "MAE(低好)", "en": "MAE(↓)"},
    "radar_spd":   {"zh": "训练速度",  "en": "Train Speed"},
}


def t(key_or_zh: str, en: str = None, **kwargs) -> str:
    """Dual-mode translation: t('key') for TEXTS dict, t(zh, en) for inline."""
    lang = st.session_state.lang
    if en is None:
        val = TEXTS.get(key_or_zh, {}).get(lang, key_or_zh)
    else:
        val = en if lang == "en" else key_or_zh
    return val.format(**kwargs) if kwargs else val


# ══════════════════════════════════════════════════════════════
# 常量 / Constants
# ══════════════════════════════════════════════════════════════
SEASON_MAP_ZH  = {1: "春季", 2: "夏季", 3: "秋季", 4: "冬季"}
SEASON_MAP_EN  = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
WEATHER_MAP_ZH = {1: "晴天/少云", 2: "雾天/多云", 3: "小雨/小雪", 4: "大雨/大雪"}
WEATHER_MAP_EN = {1: "Clear/Partly Cloudy", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}

REGRESSION_KEYS = ["线性回归", "岭回归", "决策树回归", "随机森林", "梯度提升", "XGBoost"]
ALL_ALGO_KEYS   = REGRESSION_KEYS
MODEL_NAMES_EN  = {
    "线性回归": "Linear Regression",
    "岭回归":   "Ridge Regression",
    "随机森林": "Random Forest",
    "梯度提升": "Gradient Boosting",
    "XGBoost":  "XGBoost",
    "决策树回归":  "Decision Tree Regressor",
}
MODEL_COLORS = {
    "线性回归": "#2196F3", "岭回归": "#4CAF50", "随机森林": "#FF9800",
    "梯度提升": "#9C27B0", "XGBoost": "#F44336", "决策树回归": "#00BCD4",
    "Linear Regression": "#2196F3", "Ridge Regression": "#4CAF50",
    "Random Forest": "#FF9800", "Gradient Boosting": "#9C27B0",
    "Decision Tree Regressor": "#00BCD4",
}
PC = dict(plot_bgcolor="white", paper_bgcolor="white")


def mname(key: str) -> str:
    if st.session_state.lang == "en":
        return MODEL_NAMES_EN.get(key, key)
    return key


# ══════════════════════════════════════════════════════════════
# 模型算法介绍（双语）/ Model Info
# ══════════════════════════════════════════════════════════════
MODEL_INFO = {
    "线性回归": {
        "icon": "📈", "complexity": "⭐",
        "cls": LinearRegression, "kwargs": {},
        "code": "LinearRegression()\nmodel.fit(X_train, y_train)",
        "zh": """**线性回归（Linear Regression）** 是机器学习中最基础的监督学习算法，也是理解其他复杂模型的基石。

**核心思想：** 寻找一组权重 *w*，使得 $\\hat{y} = w_0 + w_1x_1 + \\cdots + w_nx_n$ 与真实值 *y* 的误差最小（最小二乘法）。

**✅ 优点：**
- 概念简单，结果可直接解释（系数 = 特征对租赁量的影响大小）
- 训练速度极快（毫秒级）
- 适合作为基准模型（Baseline），衡量其他算法的提升幅度

**❌ 缺点：**
- 假设特征与目标之间是线性关系，无法捕捉非线性模式
- 对异常值敏感""",
        "en": """**Linear Regression** is the most fundamental supervised learning algorithm — the foundation for understanding more complex models.

**Core Idea:** Find weights *w* that minimize the difference between $\\hat{y} = w_0 + w_1x_1 + \\cdots + w_nx_n$ and the true value *y* (Ordinary Least Squares).

**✅ Pros:**
- Simple and interpretable (coefficient = feature's influence on rentals)
- Extremely fast to train (milliseconds)
- Great baseline model to measure improvement from other algorithms

**❌ Cons:**
- Assumes linear relationship — cannot capture nonlinear patterns
- Sensitive to outliers""",
    },
    "岭回归": {
        "icon": "📉", "complexity": "⭐⭐",
        "cls": Ridge, "kwargs": {"alpha": 1.0},
        "code": "Ridge(alpha=1.0)\nmodel.fit(X_train, y_train)",
        "zh": """**岭回归（Ridge Regression）** 是线性回归加上 **L2 正则化** 的改进版本，通过在损失函数中加入权重平方和惩罚项来解决过拟合。

**数学表达：** 最小化 $||y - Xw||^2 + \\alpha ||w||^2$，其中 $\\alpha$ 控制正则化强度。

**✅ 优点：**
- 有效解决**多重共线性**问题（本数据集中 temp 和 atemp 高度相关，r=0.99）
- 防止过拟合，泛化能力优于普通线性回归
- 仍然保持线性模型的可解释性

**❌ 缺点：**
- 仍然是线性模型，无法建模非线性关系
- 需要调整超参数 α""",
        "en": """**Ridge Regression** improves on Linear Regression by adding **L2 regularization**, adding a penalty to the loss function to prevent overfitting.

**Math:** Minimize $||y - Xw||^2 + \\alpha ||w||^2$, where $\\alpha$ controls regularization strength.

**✅ Pros:**
- Handles **multicollinearity** (temp and atemp correlation = 0.99 in this dataset)
- Better generalization than plain linear regression
- Still interpretable as a linear model

**❌ Cons:**
- Still a linear model — cannot model nonlinear relationships
- Requires tuning hyperparameter α""",
    },
    "决策树回归": {
        "icon": "🌳", "complexity": "⭐⭐",
        "cls": DecisionTreeRegressor, "kwargs": {"max_depth": 8, "random_state": 42},
        "code": "DecisionTreeRegressor(max_depth=8, random_state=42)",
        "zh": """**决策树回归（Decision Tree Regressor）** 通过对特征空间进行递归二分，构建一棵"如果…那么…"规则树，直接预测连续的租赁量。

**核心思想：** 每次选择能最大程度降低**均方误差（MSE）**的特征和阈值进行分裂，直到达到最大深度或样本数量不足为止。

**✅ 优点：**
- 决策路径可视化，**最易解释**的非线性模型
- 无需特征缩放，天然处理混合类型特征
- 可直接翻译为业务规则（如："工作日且 8 ≤ 小时 ≤ 9 → 高需求"）

**❌ 缺点：**
- 单棵树容易**过拟合**（对训练数据过度记忆）
- 对数据微小变化敏感，稳定性不如集成方法
- 精度通常低于随机森林/XGBoost（集成了多棵树）""",
        "en": """**Decision Tree Regressor** recursively splits the feature space into regions using "if-then" rules, directly predicting the continuous rental count in each region.

**Core Idea:** At each node, choose the feature and threshold that most reduces **Mean Squared Error (MSE)**, repeating until max depth is reached or samples are too few.

**✅ Pros:**
- Decision paths are fully visualizable — **most interpretable** nonlinear model
- No feature scaling required; handles mixed feature types naturally
- Rules translate directly into business insights (e.g., "Workday & 8≤hr≤9 → High Demand")

**❌ Cons:**
- A single tree is prone to **overfitting** — memorizes training data
- Sensitive to small data changes; less stable than ensemble methods
- Lower accuracy than Random Forest/XGBoost (which combine many trees)""",
    },
    "随机森林": {
        "icon": "🌲", "complexity": "⭐⭐⭐",
        "cls": RandomForestRegressor, "kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "code": "RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)",
        "zh": """**随机森林（Random Forest）** 是一种**集成学习**算法，通过构建多棵决策树并对预测结果取平均来提升精度（Bagging 思想）。

**核心思想：** "三个臭皮匠，顶个诸葛亮" —— 许多弱学习器（浅决策树）的集成，比单个强学习器更稳健。

**✅ 优点：**
- 天然处理**非线性关系**，无需特征缩放
- 对异常值和噪声鲁棒
- 内置**特征重要性**评估
- 可并行训练（`n_jobs=-1`），速度较快

**❌ 缺点：**
- 模型体积较大（100棵树），可解释性不如线性模型""",
        "en": """**Random Forest** is an **ensemble learning** algorithm that builds multiple decision trees and averages their predictions (Bagging).

**Core Idea:** "Wisdom of the crowd" — many weak learners (shallow trees) combined are more robust than a single strong learner.

**✅ Pros:**
- Naturally handles **nonlinear relationships**, no feature scaling needed
- Robust to outliers and noise
- Built-in **feature importance** ranking
- Parallel training (`n_jobs=-1`) — fast

**❌ Cons:**
- Large model size (100 trees); less interpretable than linear models""",
    },
    "梯度提升": {
        "icon": "🚀", "complexity": "⭐⭐⭐⭐",
        "cls": GradientBoostingRegressor, "kwargs": {"n_estimators": 100, "random_state": 42},
        "code": "GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)",
        "zh": """**梯度提升（Gradient Boosting）** 是另一种强大的集成方法，以**串行**方式逐步构建树：每棵新树专注于修正前一棵树的**残差**（预测误差）。

**核心思想：** 沿梯度下降方向不断减小残差，每一步让模型"精准补刀"弱点。

**✅ 优点：**
- 通常比随机森林精度更高（Boosting > Bagging）
- 对混合类型特征处理能力强
- 内置特征重要性

**❌ 缺点：**
- **串行训练**较慢，无法并行化
- 超参数较多，更容易过拟合""",
        "en": """**Gradient Boosting** builds trees **sequentially**: each new tree focuses on correcting the **residuals** (errors) of the previous tree.

**Core Idea:** Gradually reduce residuals along the gradient descent direction — each step precisely targets the model's weaknesses.

**✅ Pros:**
- Generally higher accuracy than Random Forest (Boosting > Bagging)
- Handles mixed feature types well
- Built-in feature importance

**❌ Cons:**
- **Sequential training** — cannot be parallelized, slower
- More hyperparameters — prone to overfitting without careful tuning""",
    },
    "XGBoost": {
        "icon": "⚡", "complexity": "⭐⭐⭐⭐⭐",
        "cls": XGBRegressor, "kwargs": {"n_estimators": 100, "random_state": 42, "verbosity": 0, "n_jobs": -1},
        "code": "XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)",
        "zh": """**XGBoost（Extreme Gradient Boosting）** 是梯度提升的**工程优化版**，由陈天奇于2014年提出，曾横扫众多机器学习竞赛冠军（Kaggle首选算法）。

**核心改进：**
- 🔬 二阶泰勒展开——更精确的梯度近似
- 🛡 内置正则化（L1/L2）——防止过拟合
- ⚡ 并行化树节点分裂——速度大幅提升
- 🏗 直方图算法——内存高效

**✅ 优点：** 速度快、精度高，综合性能是表格数据的天花板，超参数丰富可精细调优

**❌ 缺点：** 超参数众多，调参复杂；可解释性不如线性模型""",
        "en": """**XGBoost (Extreme Gradient Boosting)** is an engineered optimization of Gradient Boosting by Tianqi Chen (2014) — a dominant algorithm in Kaggle competitions.

**Key Improvements:**
- 🔬 Second-order Taylor expansion — more accurate gradient approximation
- 🛡 Built-in L1/L2 regularization — prevents overfitting
- ⚡ Parallelized node splitting — much faster training
- 🏗 Histogram algorithm — memory efficient

**✅ Pros:** Fast and highly accurate; state-of-the-art on tabular data; rich hyperparameters for fine-tuning

**❌ Cons:** Many hyperparameters — complex to tune; less interpretable""",
    },
}

# ── Summary Commentary ─────────────────────────────────────────
SUMMARY_COMMENT = {
    "zh": """#### 1. 线性模型（线性回归 & 岭回归）
训练速度极快，是学习机器学习的最佳起点。但自行车租赁量受**时间、天气、季节**的非线性影响，线性假设本身存在局限。岭回归通过 L2 正则化解决了 `atemp` 与 `temp` 高度相关（多重共线性）的问题，表现略优于普通线性回归。

#### 2. 集成树模型（随机森林、梯度提升、XGBoost）
三种基于决策树的集成算法均能捕捉特征间的**非线性关系**，R² 显著更高。
- **随机森林** — 并行 Bagging，降低方差，训练快且稳定
- **梯度提升** — 串行 Boosting，偏差更低，训练较慢
- **XGBoost** — 精度与速度兼顾，综合性能最佳

#### 3. 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 初学入门 / 概念教学 | **线性回归** | 公式直观，系数可解释 |
| 生产部署 | **XGBoost** | 精度最高，支持大规模数据 |
| 重视可解释性 | **随机森林** | 特征重要性清晰 |

#### 4. 最重要的特征
1. **`hr`（小时）** — 通勤规律决定了一天内的租赁峰谷
2. **`atemp`（体感温度）** — 天气舒适度是核心驱动
3. **`yr`（年份）** — 反映平台用户增长趋势
4. **`workingday`（工作日）** — 决定出行模式类型

> 💡 **No Free Lunch 定理：** 没有任何模型在所有问题上都最优。实际应用需综合考虑精度、速度与可解释性。""",
    "en": """#### 1. Linear Models (Linear Regression & Ridge)
Fast to train and great starting points for learning ML. However, bike rentals are influenced by **nonlinear** time-weather-season interactions. Ridge Regression handles `atemp`/`temp` multicollinearity (corr=0.99) better than plain linear regression.

#### 2. Ensemble Tree Models (Random Forest, Gradient Boosting, XGBoost)
All three capture **nonlinear feature interactions**, achieving much higher R²:
- **Random Forest** — parallel Bagging, reduces variance, fast and stable
- **Gradient Boosting** — sequential Boosting, lower bias, slower
- **XGBoost** — best balance of accuracy and speed; top performer on tabular data

#### 3. Model Selection Guide

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Learning / teaching | **Linear Regression** | Intuitive formula, interpretable coefficients |
| Production deployment | **XGBoost** | Highest accuracy, scalable |
| Interpretability | **Random Forest** | Clear feature importance |

#### 4. Key Drivers
1. **`hr` (Hour)** — commuter patterns drive intra-day peaks and valleys
2. **`atemp` (Temperature)** — weather comfort is the core driver
3. **`yr` (Year)** — reflects platform user growth trend
4. **`workingday`** — determines commuting vs. leisure travel mode

> 💡 **No Free Lunch Theorem:** No single model is best for all problems. Balance accuracy, speed, and interpretability.""",
}


# ══════════════════════════════════════════════════════════════
# 数据加载 / Data Loading
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df

@st.cache_data
def load_wash() -> pd.DataFrame:
    return pd.read_csv(WASH)

@st.cache_data
def get_split():
    df = load_wash()
    X  = df[FEATURE_COLS]
    y  = df["cnt_log"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ══════════════════════════════════════════════════════════════
# 训练辅助函数 / Train Helpers
# ══════════════════════════════════════════════════════════════
def train_reg(name: str, cls, kwargs: dict) -> dict:
    X_tr, X_te, y_tr, y_te = get_split()
    t0 = time.time()
    m = cls(**kwargs)
    m.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)
    yp = inverse_transform(m.predict(X_te))
    yt = inverse_transform(y_te.values)
    fi = None
    if hasattr(m, "feature_importances_"):
        fi = pd.Series(m.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(10)
    elif hasattr(m, "coef_"):
        fi = pd.Series(np.abs(m.coef_), index=FEATURE_COLS).sort_values(ascending=False).head(10)
    return {
        "yt": yt, "yp": yp, "fi": fi, "elapsed": elapsed,
        "MAE":  round(mean_absolute_error(yt, yp), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 2),
        "R²":   round(r2_score(yt, yp), 4),
    }

@st.cache_data
def rf_curve():
    X_tr, X_te, y_tr, y_te = get_split()
    ns, rm, r2 = [], [], []
    for n in range(5, 26, 2):
        m = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        yp = inverse_transform(m.predict(X_te))
        yt = inverse_transform(y_te.values)
        ns.append(n)
        rm.append(round(np.sqrt(mean_squared_error(yt, yp)), 2))
        r2.append(round(r2_score(yt, yp), 4))
    return ns, rm, r2

@st.cache_data
def xgb_learning_curve():
    X_tr, X_te, y_tr, y_te = get_split()
    ns, tr_e, te_e = [], [], []
    for n in list(range(1, 11)) + list(range(15, 101, 5)):
        m = XGBRegressor(n_estimators=n, random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_tr, y_tr)
        tr_p = inverse_transform(m.predict(X_tr))
        te_p = inverse_transform(m.predict(X_te))
        ns.append(n)
        tr_e.append(round(np.sqrt(mean_squared_error(inverse_transform(y_tr.values), tr_p)), 2))
        te_e.append(round(np.sqrt(mean_squared_error(inverse_transform(y_te.values), te_p)), 2))
    return ns, tr_e, te_e

@st.cache_data
def xgb_boost_rounds():
    X_tr, X_te, y_tr, y_te = get_split()
    res = {}
    for n in [1, 5, 20, 100]:
        m = XGBRegressor(n_estimators=n, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        res[str(n)] = inverse_transform(m.predict(X_te))
    res["true"] = inverse_transform(y_te.values)
    return res




# ══════════════════════════════════════════════════════════════
# 渲染辅助函数 / Render Helpers
# ══════════════════════════════════════════════════════════════
def kpi(col, val, label, color="#2196F3", val_color="#1a237e"):
    col.markdown(
        f'<div class="kpi" style="border-top-color:{color}">'
        f'<div class="kpi-v" style="color:{val_color}">{val}</div>'
        f'<div class="kpi-l">{label}</div></div>',
        unsafe_allow_html=True,
    )

def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)

def metrics_row(res):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("metric_rmse"), f"{res['RMSE']:.2f}",  help=t("metric_rmse_help"))
    c2.metric(t("metric_mae"),  f"{res['MAE']:.2f}",   help=t("metric_mae_help"))
    c3.metric(t("metric_r2"),   f"{res['R²']:.4f}",    help=t("metric_r2_help"))
    c4.metric(t("metric_time"), f"{res['elapsed']}s",  help=t("metric_time_help"))

def reg_charts(res):
    yt, yp = res["yt"], res["yp"]
    n = min(500, len(yt))

    c1, c2 = st.columns([3, 2])
    with c1:
        sec(t("📉 实际值 vs 预测值（前500点）", "📉 Actual vs Predicted (first 500 pts)"))
        fig = go.Figure()
        fig.add_scatter(y=yt[:n], name=t("实际值", "Actual"),
                        line=dict(color="#1E88E5", width=1.5))
        fig.add_scatter(y=yp[:n], name=t("预测值", "Predicted"),
                        line=dict(color="#E53935", width=1.5, dash="dot"))
        fig.update_layout(**PC, height=340, hovermode="x unified",
                          xaxis_title=t("样本序号", "Sample Index"),
                          yaxis_title=t("租赁量", "Rentals"),
                          yaxis_gridcolor="#f0f0f0", legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sec(t("🎯 预测散点图", "🎯 Scatter Plot"))
        fig = px.scatter(x=yt, y=yp, opacity=0.3,
                         color_discrete_sequence=["#7B1FA2"],
                         labels={"x": t("实际值", "Actual"), "y": t("预测值", "Predicted")})
        mv = float(max(yt.max(), yp.max())) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=mv, y1=mv,
                      line=dict(color="#E53935", dash="dash", width=1.5))
        fig.add_annotation(x=mv * .85, y=mv * .92, text=t("完美预测线", "Perfect line"),
                           showarrow=False, font=dict(color="#E53935", size=11))
        fig.update_layout(**PC, height=340)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if res.get("fi") is not None:
            sec(t("🔑 特征重要性（Top 10）", "🔑 Feature Importance (Top 10)"))
            fi = res["fi"].reset_index()
            fi.columns = [t("特征", "Feature"), t("重要性", "Importance")]
            fi = fi.sort_values(t("重要性", "Importance"))
            fig = px.bar(fi, x=t("重要性", "Importance"), y=t("特征", "Feature"),
                         orientation="h", color=t("重要性", "Importance"),
                         color_continuous_scale="Blues")
            fig.update_layout(**PC, height=340, coloraxis_showscale=False,
                              xaxis_gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        sec(t("📊 残差分布", "📊 Residual Distribution"))
        res_vals = yt - yp
        fig = px.histogram(res_vals, nbins=60, color_discrete_sequence=["#43A047"],
                           labels={"value": t("残差", "Residual"), "count": t("频次", "Count")})
        fig.add_vline(x=0, line_dash="dash", line_color="#E53935",
                      annotation_text=t("零误差", "Zero Error"), annotation_position="top")
        fig.update_layout(**PC, height=340, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    insight(t("📌 <b>残差解读：</b>理想情况下残差应近似<b>正态分布</b>且以0为中心，说明模型无系统性偏差。"
              "若残差明显偏斜，说明模型对高峰时段存在系统性低估或高估。",
              "📌 <b>Residual:</b> Ideally residuals follow a <b>normal distribution</b> centered at 0. "
              "Skewed residuals indicate systematic under/over-estimation on peak hours."))

    with st.expander(t("📋 预测明细（前100行）", "📋 Prediction Details (first 100 rows)")):
        tbl = pd.DataFrame({
            t("实际租赁量", "Actual"):   yt[:100].astype(int),
            t("预测租赁量", "Predicted"): np.round(yp[:100]).astype(int),
            t("误差", "Error"):          (yt[:100] - yp[:100]).round(1),
            t("绝对误差率(%)", "Err%"):   (np.abs(yt[:100] - yp[:100]) / (yt[:100] + 1) * 100).round(1),
        })
        st.dataframe(tbl, use_container_width=True)




# ══════════════════════════════════════════════════════════════
# 页面：Dashboard
# ══════════════════════════════════════════════════════════════
def page_dashboard():
    df = load_raw().copy()
    lang = st.session_state.lang
    df["season_l"]  = df["season"].map(SEASON_MAP_EN if lang == "en" else SEASON_MAP_ZH)
    df["weather_l"] = df["weathersit"].map(WEATHER_MAP_EN if lang == "en" else WEATHER_MAP_ZH)
    season_order  = list((SEASON_MAP_EN if lang == "en" else SEASON_MAP_ZH).values())
    weather_order = list((WEATHER_MAP_EN if lang == "en" else WEATHER_MAP_ZH).values())

    st.markdown(f'<div class="page-title">{t("dash_title")}</div>', unsafe_allow_html=True)

    # KPI 卡片
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(df):,}",            t("kpi_rows"),  "#2196F3", "#0d47a1")
    kpi(c2, f"{int(df.cnt.sum()):,}",  t("kpi_total"), "#4CAF50", "#1b5e20")
    kpi(c3, f"{int(df.cnt.mean())}",   t("kpi_avg"),   "#FF9800", "#e65100")
    kpi(c4, f"{int(df.cnt.max())}",    t("kpi_max"),   "#9C27B0", "#4a148c")
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. 每日趋势
    st.subheader(t("📈 每日总租赁量趋势（2011–2012）", "📈 Daily Total Rentals Trend (2011–2012)"))
    daily = df.groupby("dteday")["cnt"].sum().reset_index()
    fig = px.area(daily, x="dteday", y="cnt",
                  labels={"dteday": t("日期", "Date"), "cnt": t("日租赁量", "Daily Rentals")},
                  color_discrete_sequence=["#2196F3"])
    fig.update_traces(line_width=1.5, fillcolor="rgba(33,150,243,.15)")
    fig.update_layout(**PC, height=300, hovermode="x unified",
                      xaxis_title="", yaxis_gridcolor="#f0f0f0", xaxis_showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    insight(t("📌 <b>趋势洞察：</b>租赁量整体呈上升趋势，2012年明显高于2011年，说明共享单车平台处于高速增长期。"
              "每年呈现明显的<b>季节性波动</b>：夏秋（6–9月）租赁量达到峰值，冬季（12–2月）显著下降。",
              "📌 <b>Trend Insight:</b> Rentals show a clear upward trend — 2012 significantly higher than 2011. "
              "Strong <b>seasonal cycles</b>: peak Jun–Sep, sharp decline Dec–Feb, closely tracking temperature."))
    st.divider()

    # 2. 温度折线图 & 季节箱型图
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t("🌡️ 温度与平均租赁量的关系", "🌡️ Temperature vs Avg Rentals"))
        tmp = df.copy()
        tmp["tc"] = (tmp["temp"] * 41).round(0).astype(int)
        ta = tmp.groupby("tc")["cnt"].agg(["mean", "count"]).reset_index()
        ta = ta[ta["count"] >= 30].rename(columns={"mean": "cnt"})
        fig = go.Figure()
        fig.add_scatter(x=ta["tc"], y=ta["cnt"], mode="lines+markers",
                        fill="tozeroy", fillcolor="rgba(255,112,67,.1)",
                        line=dict(color="#FF7043", width=2.5), marker=dict(size=5))
        fig.update_layout(**PC, height=300, hovermode="x unified",
                          xaxis_title=t("温度（°C）", "Temperature (°C)"),
                          yaxis_title=t("平均租赁量", "Avg Rentals"), yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("租赁量随气温升高持续增加，在 20–30°C 区间达峰（约250–300次/时）；"
                  "超过38°C后因高温不适开始回落。温度是影响需求最直接的连续型特征，在预测模型中权重最高。",
                  "Rentals rise steadily with temperature, peaking at 20–30°C (~250–300 rides/hr), "
                  "then declining above 38°C. Temperature is the strongest continuous predictor in all models."))

    with col_b:
        st.subheader(t("🌡️ 各季节租赁量分布", "🌡️ Rental Distribution by Season"))
        fig = px.box(df, x="season_l", y="cnt", color="season_l",
                     category_orders={"season_l": season_order},
                     color_discrete_sequence=["#66BB6A", "#FFA726", "#EF5350", "#42A5F5"],
                     labels={"season_l": t("季节", "Season"), "cnt": t("租赁量", "Rentals")})
        fig.update_layout(**PC, height=300, showlegend=False, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("📌 <b>季节效应：</b>秋季租赁中位数最高，夏季其次。"
                  "春季偏低（天气不稳定），冬季最低（严寒限制出行）。",
                  "📌 <b>Seasonal Effect:</b> Autumn has the highest median, followed by summer. "
                  "Spring is lower (unstable weather), winter is lowest (cold restricts cycling)."))

    st.divider()

    # 3. 天气箱型图 & 工作日折线图
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader(t("☁️ 天气状况对租赁量分布的影响", "☁️ Rental Distribution by Weather"))
        fig = px.box(df, x="weather_l", y="cnt", color="weather_l",
                     category_orders={"weather_l": weather_order},
                     color_discrete_sequence=["#42A5F5", "#AB47BC", "#EF5350", "#78909C"],
                     labels={"weather_l": t("天气", "Weather"), "cnt": t("租赁量", "Rentals")})
        fig.update_layout(**PC, height=300, showlegend=False, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("晴天租赁量分布最高且最分散（中位数约180，上四分位超400）；"
                  "雾天约120；小雨小雪时显著下降；大雨大雪时需求几乎归零。"
                  "天气类型是预测模型中最重要的分类型特征。",
                  "Clear weather shows highest and most spread distribution (median ~180, Q3 >400). "
                  "Mist drops to ~120; light rain/snow causes a sharp decline; heavy rain/snow nearly eliminates demand. "
                  "Weather category is the most important categorical predictor."))

    with col_d:
        st.subheader(t("📅 工作日 vs 休息日（按小时）", "📅 Workday vs Weekend (by Hour)"))
        hwd = df.groupby(["hr", "workingday"])["cnt"].mean().reset_index()
        hwd["type"] = hwd["workingday"].map({0: t("🏖 休息日", "🏖 Weekend"),
                                              1: t("💼 工作日", "💼 Workday")})
        fig = px.line(hwd, x="hr", y="cnt", color="type",
                      labels={"hr": t("小时", "Hour"), "cnt": t("平均租赁量", "Avg Rentals"), "type": ""},
                      color_discrete_sequence=["#FF7043", "#1E88E5"])
        fig.update_layout(**PC, height=300, hovermode="x unified", yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("📌 <b>出行模式差异：</b>工作日呈 <b>双峰曲线</b>（通勤主导），"
                  "休息日为 <b>单峰平缓曲线</b>（10–15时，休闲主导）。两类用户行为模式截然不同。",
                  "📌 <b>Pattern Difference:</b> Workdays show a <b>bimodal curve</b> (commuter-driven); "
                  "weekends show a <b>single broad peak</b> (10am–3pm, leisure-driven)."))

    with st.expander(t("raw_data_expander")):
        st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 页面：数据预处理展示
# ══════════════════════════════════════════════════════════════
def page_preprocess():
    st.markdown(f'<div class="page-title">{t("🔧 数据预处理流程", "🔧 Data Preprocessing Pipeline")}</div>',
                unsafe_allow_html=True)

    df_raw = load_raw().copy()
    df_w   = load_wash()

    # ── 一、数据概览 KPI ──────────────────────────────────────
    sec(t("📋 一、数据概览", "📋 Ⅰ. Dataset Overview"))
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "17,379", t("原始行数",    "Raw Rows"),       "#2196F3", "#0d47a1")
    kpi(c2, "17",     t("原始列数",    "Raw Columns"),    "#4CAF50", "#1b5e20")
    kpi(c3, "0",      t("缺失值",      "Missing Values"), "#FF9800", "#e65100")
    kpi(c4, "23",     t("最终特征数",  "Final Features"), "#9C27B0", "#4a148c")
    st.markdown("<br>", unsafe_allow_html=True)

    # 原始字段说明表
    col_df = pd.DataFrame({
        t("字段", "Field"): ["instant","dteday","yr","mnth","hr","season","holiday",
                              "weekday","workingday","weathersit","temp","atemp",
                              "hum","windspeed","casual","registered","cnt"],
        t("说明", "Description"): [
            t("行编号（无意义）","Row index (meaningless)"),
            t("日期","Date"),
            t("年份编码 0=2011, 1=2012","Year 0=2011, 1=2012"),
            t("月份 1–12","Month 1–12"),
            t("小时 0–23","Hour 0–23"),
            t("季节 1=春 2=夏 3=秋 4=冬","Season 1=Spr 2=Sum 3=Aut 4=Win"),
            t("是否节假日","Public holiday flag"),
            t("星期几 0=周日","Weekday 0=Sun"),
            t("是否工作日","Working day flag"),
            t("天气等级 1=晴 2=雾 3=小雨 4=暴雨","Weather 1=Clear 2=Mist 3=Rain 4=Storm"),
            t("气温（归一化 ×41=°C）","Temp (norm ×41=°C)"),
            t("体感温度（归一化 ×50=°C）","Feels-like temp (norm ×50=°C)"),
            t("湿度（归一化 ×100=%）","Humidity (norm ×100=%)"),
            t("风速（归一化 ×67=km/h）","Windspeed (norm ×67=km/h)"),
            t("临时用户量（泄漏特征）","Casual users (leakage)"),
            t("注册用户量（泄漏特征）","Registered users (leakage)"),
            t("总租车量 = casual + registered（预测目标）","Total rentals = casual+registered (target)"),
        ],
        t("处理", "Action"): [
            t("❌ 删除","❌ Drop"), t("❌ 删除","❌ Drop"),
            t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"),
            t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"),
            t("✅ 保留","✅ Keep"), t("✅ 合并4→3","✅ Merge 4→3"),
            t("❌ 删除（与atemp相关0.99）","❌ Drop (corr=0.99 w/ atemp)"),
            t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"), t("✅ 保留","✅ Keep"),
            t("❌ 删除（数据泄漏）","❌ Drop (data leakage)"),
            t("❌ 删除（数据泄漏）","❌ Drop (data leakage)"),
            t("🎯 目标变量（取log1p）","🎯 Target (log1p transform)"),
        ],
    })
    st.dataframe(col_df, use_container_width=True, hide_index=True)
    st.divider()

    # ── 二、数据清洗 ──────────────────────────────────────────
    sec(t("🧹 二、数据清洗", "🧹 Ⅱ. Data Cleaning"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("2.1 稀有类别合并（weathersit 4 → 3）",
                       "2.1 Rare Category Merge (weathersit 4 → 3)"))
        ws_counts = df_raw["weathersit"].value_counts().sort_index()
        ws_labels = {1: t("晴/少云","Clear"), 2: t("薄雾","Mist"),
                     3: t("小雨雪","Light Rain"), 4: t("暴雨（→并入3）","Storm (→3)")}
        ws_colors = ["#42A5F5", "#AB47BC", "#EF5350", "#FF7043"]
        fig = go.Figure(go.Bar(
            x=[ws_labels[i] for i in ws_counts.index],
            y=ws_counts.values,
            marker_color=ws_colors[:len(ws_counts)],
            text=ws_counts.values, textposition="outside",
        ))
        fig.update_layout(**PC, height=300, yaxis_gridcolor="#f0f0f0",
                          yaxis_title=t("记录数","Count"))
        st.plotly_chart(fig, use_container_width=True)
        insight(t("weathersit=4（暴雨/冰雹）仅 <b>3 条</b>，占比 0.02%，"
                  "样本量极少导致模型无法有效学习，合并入 weathersit=3（恶劣天气）。",
                  "weathersit=4 (storm) has only <b>3 records</b> (0.02%). "
                  "Too few samples for the model to learn — merged into weathersit=3 (bad weather)."))

    with c2:
        st.subheader(t("2.2 共线特征：temp vs atemp（相关系数 0.99）",
                       "2.2 Multicollinearity: temp vs atemp (corr=0.99)"))
        sample = df_raw.sample(2000, random_state=42)
        fig = px.scatter(sample, x="temp", y="atemp", opacity=0.3,
                         color_discrete_sequence=["#1E88E5"],
                         labels={"temp": t("气温 temp（归一化）","temp (norm)"),
                                 "atemp": t("体感温度 atemp（归一化）","atemp (norm)")})
        corr_val = df_raw[["temp","atemp"]].corr().iloc[0,1]
        fig.add_annotation(x=0.7, y=0.15,
                           text=f"r = {corr_val:.4f}",
                           showarrow=False,
                           font=dict(size=16, color="#E53935", family="monospace"),
                           bgcolor="white", bordercolor="#E53935", borderwidth=1)
        fig.update_layout(**PC, height=300)
        st.plotly_chart(fig, use_container_width=True)
        insight(t("两者相关系数 <b>0.99</b>，几乎完全线性相关，同时保留会造成多重共线性。"
                  "删除 temp，保留 atemp（体感温度对骑行决策更直接）。",
                  "Correlation <b>0.99</b> — near-perfect linear relationship. Keeping both causes multicollinearity. "
                  "Drop temp, keep atemp (feels-like temp is a more direct driver of cycling decisions)."))
    st.divider()

    # ── 三、特征工程 ──────────────────────────────────────────
    sec(t("⚙️ 三、特征工程", "⚙️ Ⅲ. Feature Engineering"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("3.1 时间段分类（time_period）",
                       "3.1 Time Period Classification"))
        tp_map  = {1: t("深夜 0–6时","Late Night 0–6"),
                   2: t("早高峰 7–9时","AM Rush 7–9"),
                   3: t("日间 10–16时","Daytime 10–16"),
                   4: t("晚高峰 17–19时","PM Rush 17–19"),
                   5: t("夜间 20–23时","Evening 20–23")}
        tp_colors = ["#78909C","#FF9800","#42A5F5","#E53935","#7B1FA2"]
        avg_by_hr = df_raw.groupby("hr")["cnt"].mean().reset_index()
        avg_by_hr["period"] = avg_by_hr["hr"].apply(
            lambda h: (1 if h<=6 else 2 if h<=9 else 3 if h<=16 else 4 if h<=19 else 5))
        avg_by_hr["period_label"] = avg_by_hr["period"].map(tp_map)
        fig = px.bar(avg_by_hr, x="hr", y="cnt", color="period_label",
                     color_discrete_sequence=tp_colors,
                     labels={"hr": t("小时","Hour"), "cnt": t("平均租赁量","Avg Rentals"),
                             "period_label": t("时间段","Period")})
        fig.update_layout(**PC, height=300, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("将24小时划分为5段，赋予语义标签，比原始小时整数更能反映骑行行为模式。",
                  "24 hours split into 5 semantic segments — richer behavioral signal than raw hour integers."))

    with c2:
        st.subheader(t("3.2 循环编码（Sin-Cos Encoding）",
                       "3.2 Cyclic Encoding (Sin-Cos)"))
        hrs = np.arange(24)
        hr_sin = np.sin(2 * np.pi * hrs / 24)
        hr_cos = np.cos(2 * np.pi * hrs / 24)
        fig = go.Figure()
        fig.add_scatter(x=hrs, y=hr_sin, mode="lines+markers",
                        line=dict(color="#1E88E5", width=2),
                        marker=dict(size=5), name="hr_sin")
        fig.add_scatter(x=hrs, y=hr_cos, mode="lines+markers",
                        line=dict(color="#FF9800", width=2),
                        marker=dict(size=5), name="hr_cos")
        fig.add_annotation(x=0, y=hr_sin[0], text="0时≈23时",
                           showarrow=True, arrowhead=2,
                           font=dict(size=11, color="#E53935"),
                           arrowcolor="#E53935", ax=40, ay=-30)
        fig.update_layout(**PC, height=300, hovermode="x unified",
                          xaxis_title=t("小时 hr","Hour"),
                          yaxis_title=t("编码值","Encoded Value"),
                          yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
        insight(t("原始小时数值中 0 与 23 相差 23，但实际只差 1 小时。"
                  "正余弦编码将时间映射到单位圆，使首尾在数学上自然连接。",
                  "Raw hours: 0 and 23 differ by 23 but are only 1 hour apart. "
                  "Sin-Cos maps time onto a unit circle so endpoints connect naturally."))

    st.subheader(t("3.3 交互特征", "3.3 Interaction Features"))
    interact_df = pd.DataFrame({
        t("交互特征","Feature"):     ["hr_workingday", "temp_hum", "season_hr"],
        t("构成","Formula"):         ["hr × workingday", "atemp × (1 − hum)", "season × hr"],
        t("业务含义","Business Meaning"): [
            t("工作日通勤双峰 vs 周末休闲单峰，两种出行模式差异显著",
              "Workday bimodal commute vs weekend leisure peak — distinct travel patterns"),
            t("高温高湿时体感极差，骑行意愿下降；单独看温度或湿度不足以反映真实舒适度",
              "High temp + high humidity = extreme discomfort; individual features miss the combined effect"),
            t("夏季日间骑行时段长，冬季集中在正午前后，不同季节活跃时段分布不同",
              "Summer: long active window; Winter: concentrated midday — season shifts the optimal riding hours"),
        ],
    })
    st.dataframe(interact_df, use_container_width=True, hide_index=True)
    st.divider()

    # ── 四、目标变量处理 ──────────────────────────────────────
    sec(t("🎯 四、目标变量处理（Log1p 变换）", "🎯 Ⅳ. Target Variable Transform (Log1p)"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("原始 cnt 分布（右偏）", "Raw cnt Distribution (Right-skewed)"))
        fig = px.histogram(df_raw, x="cnt", nbins=60,
                           color_discrete_sequence=["#EF5350"],
                           labels={"cnt": t("总租车量 cnt","Total Rentals cnt")})
        fig.add_vline(x=df_raw["cnt"].mean(), line_dash="dash", line_color="#1E88E5",
                      annotation_text=f'mean={df_raw["cnt"].mean():.0f}',
                      annotation_position="top right")
        fig.add_vline(x=df_raw["cnt"].median(), line_dash="dot", line_color="#43A047",
                      annotation_text=f'median={df_raw["cnt"].median():.0f}',
                      annotation_position="top left")
        fig.update_layout(**PC, height=300, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(t("log1p(cnt) 分布（更接近正态）", "log1p(cnt) Distribution (More Normal)"))
        cnt_log = np.log1p(df_raw["cnt"])
        fig = px.histogram(x=cnt_log, nbins=60,
                           color_discrete_sequence=["#42A5F5"],
                           labels={"x": "cnt_log = log1p(cnt)"})
        fig.add_vline(x=cnt_log.mean(), line_dash="dash", line_color="#1E88E5",
                      annotation_text=f'mean={cnt_log.mean():.2f}',
                      annotation_position="top right")
        fig.add_vline(x=cnt_log.median(), line_dash="dot", line_color="#43A047",
                      annotation_text=f'median={cnt_log.median():.2f}',
                      annotation_position="top left")
        fig.update_layout(**PC, height=300, yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    insight(t('原始 cnt 均值（189.5）> 中位数（142），典型右偏分布。'
              '取 <b>log(1+cnt)</b> 后分布更接近正态，高峰极值被"拉近"，'
              '低值区间被"展开"，模型训练更稳定，低租车量时段的预测误差不再被系统性忽视。',
              "Raw cnt: mean (189.5) > median (142) — classic right skew. "
              "After <b>log(1+cnt)</b> the distribution is more symmetric, extreme peaks are compressed, "
              "low values are spread out, training is more stable, and low-demand hours get fair treatment."))
    st.divider()

    # ── 五、最终特征列表 ──────────────────────────────────────
    sec(t("📊 五、最终特征列表（共23个）", "📊 Ⅴ. Final Feature List (23 Total)"))
    feat_df = pd.DataFrame({
        t("类别","Category"): [
            t("时间原始","Time Raw")]*3 +
            [t("时间循环编码","Cyclic Encoding")]*8 +
            [t("时间分段","Time Segment")] +
            [t("日历","Calendar")]*4 +
            [t("天气","Weather")]*4 +
            [t("交互","Interaction")]*3,
        t("特征","Feature"): [
            "yr","mnth","hr",
            "hr_sin","hr_cos","mnth_sin","mnth_cos","wek_sin","wek_cos","seas_sin","seas_cos",
            "time_period",
            "season","holiday","weekday","workingday",
            "weathersit","atemp","hum","windspeed",
            "hr_workingday","temp_hum","season_hr",
        ],
        t("说明","Description"): [
            t("年份编码","Year 0/1"),
            t("月份 1–12","Month 1–12"),
            t("小时 0–23","Hour 0–23"),
            "sin(2π·hr/24)","cos(2π·hr/24)",
            "sin(2π·mnth/12)","cos(2π·mnth/12)",
            "sin(2π·weekday/7)","cos(2π·weekday/7)",
            "sin(2π·season/4)","cos(2π·season/4)",
            t("5段时间分类","5-segment time label"),
            t("季节 1–4","Season 1–4"),
            t("节假日 0/1","Holiday 0/1"),
            t("星期 0–6","Weekday 0–6"),
            t("工作日 0/1","Workday 0/1"),
            t("天气等级 1–3","Weather level 1–3"),
            t("体感温度（归一化）","Normalized feels-like temp"),
            t("湿度（归一化）","Normalized humidity"),
            t("风速（归一化）","Normalized windspeed"),
            "hr × workingday",
            "atemp × (1 − hum)",
            "season × hr",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 页面：通用回归模型
# ══════════════════════════════════════════════════════════════
def page_model(name: str):
    info    = MODEL_INFO[name]
    display = mname(name)
    desc    = info["en"] if st.session_state.lang == "en" else info["zh"]

    st.markdown(
        f'<div class="page-title">{info["icon"]} {display} '
        f'<span style="font-size:1rem;color:#64748b;">{t("complexity")}{info["complexity"]}</span></div>',
        unsafe_allow_html=True,
    )

    with st.expander(t("algo_intro"), expanded=True):
        st.markdown(f'<div class="algo-desc">{desc}</div>', unsafe_allow_html=True)
        st.code(info["code"], language="python")

    key = f"res_{name}"
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        run_clicked = st.button(t("run_btn", name=display), type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner(t("spinner", name=display)):
            st.session_state[key] = train_reg(name, info["cls"], info["kwargs"])
        st.success(t("success", name=display))

    if key not in st.session_state:
        st.info(t("run_hint"))
        return

    res = st.session_state[key]
    st.divider()
    sec(t("metrics_title"))
    metrics_row(res)
    st.divider()
    reg_charts(res)


# ══════════════════════════════════════════════════════════════
# 页面：随机森林（含5–25棵树曲线）
# ══════════════════════════════════════════════════════════════
def page_random_forest():
    info    = MODEL_INFO["随机森林"]
    display = mname("随机森林")
    desc    = info["en"] if st.session_state.lang == "en" else info["zh"]

    st.markdown(
        f'<div class="page-title">🌲 {display} '
        f'<span style="font-size:1rem;color:#64748b;">{t("complexity")}{info["complexity"]}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander(t("algo_intro"), expanded=True):
        st.markdown(f'<div class="algo-desc">{desc}</div>', unsafe_allow_html=True)
        st.code(info["code"], language="python")

    key = "res_随机森林"
    btn, _ = st.columns([1, 4])
    if btn.button(t("▶ 运行随机森林", "▶ Run Random Forest"), type="primary", use_container_width=True):
        with st.spinner(t("⏳ 训练中…", "⏳ Training…")):
            st.session_state[key] = train_reg("随机森林", info["cls"], info["kwargs"])
        st.success(t("✅ 完成！", "✅ Done!"))

    if key not in st.session_state:
        st.info(t("run_hint"))
        return

    res = st.session_state[key]
    st.divider()
    sec(t("📊 评估指标（100棵树）", "📊 Metrics (100 trees)"))
    metrics_row(res)
    st.divider()

    # 5–25 trees curve
    sec(t("🌳 树数量 vs 模型性能（5–25棵）", "🌳 n_trees vs Performance (5–25)"))
    with st.spinner(t("计算中…", "Computing…")):
        ns, rm, r2 = rf_curve()
    mi = int(np.argmin(rm))
    mr = int(np.argmax(r2))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=ns, y=rm, mode="lines+markers",
                        line=dict(color="#E53935", width=2), marker=dict(size=7), name="RMSE")
        fig.add_scatter(x=[ns[mi]], y=[rm[mi]], mode="markers",
                        marker=dict(size=18, color="#43A047", symbol="star"),
                        name=f"{t('最优','Best')} n={ns[mi]}")
        fig.update_layout(**PC, height=320, xaxis_title=t("决策树数量", "n_trees"),
                          yaxis_title="RMSE", yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=ns, y=r2, mode="lines+markers",
                        line=dict(color="#1E88E5", width=2), marker=dict(size=7), name="R²")
        fig.add_scatter(x=[ns[mr]], y=[r2[mr]], mode="markers",
                        marker=dict(size=18, color="#FF9800", symbol="star"),
                        name=f"{t('最优','Best')} n={ns[mr]}")
        fig.update_layout(**PC, height=320, xaxis_title=t("决策树数量", "n_trees"),
                          yaxis_title="R²", yaxis_gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)
    insight(t(f"📌 {ns[mi]} 棵树时 RMSE 最低；继续增加树数量后性能提升边际递减，超过约15棵后曲线趋于平稳。",
              f"📌 RMSE bottoms at n={ns[mi]} trees; marginal gain diminishes beyond ~15 trees — curve plateaus."))
    st.divider()
    reg_charts(res)


# ══════════════════════════════════════════════════════════════
# 页面：XGBoost（含学习曲线 + Boosting轮数可视化）
# ══════════════════════════════════════════════════════════════
def page_xgboost():
    info    = MODEL_INFO["XGBoost"]
    display = "XGBoost"
    desc    = info["en"] if st.session_state.lang == "en" else info["zh"]

    st.markdown(
        f'<div class="page-title">⚡ XGBoost '
        f'<span style="font-size:1rem;color:#64748b;">{t("complexity")}{info["complexity"]}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander(t("algo_intro"), expanded=True):
        st.markdown(f'<div class="algo-desc">{desc}</div>', unsafe_allow_html=True)
        st.code(info["code"], language="python")

    key = "res_XGBoost"
    btn, _ = st.columns([1, 4])
    if btn.button(t("▶ 运行 XGBoost", "▶ Run XGBoost"), type="primary", use_container_width=True):
        with st.spinner(t("⏳ 训练中…", "⏳ Training…")):
            st.session_state[key] = train_reg("XGBoost", info["cls"], info["kwargs"])
        st.success(t("✅ 完成！", "✅ Done!"))

    if key not in st.session_state:
        st.info(t("run_hint"))
        return

    res = st.session_state[key]
    st.divider()
    sec(t("📊 评估指标", "📊 Metrics"))
    metrics_row(res)
    st.divider()

    # 学习曲线
    sec(t("📚 原理可视化①：学习曲线（Boosting轮数 vs RMSE）",
          "📚 Principle ①: Learning Curve (n_estimators vs RMSE)"))
    with st.spinner(t("计算中…", "Computing…")):
        ns, tr, te = xgb_learning_curve()
    mi    = int(np.argmin(te))
    d2    = np.diff(np.diff(te))
    elbow = int(np.argmax(np.abs(d2))) + 1
    fig = go.Figure()
    fig.add_scatter(x=ns, y=tr, mode="lines", line=dict(color="#FF9800", width=2),
                    name=t("训练集", "Train"))
    fig.add_scatter(x=ns, y=te, mode="lines", line=dict(color="#1E88E5", width=2),
                    name=t("测试集", "Test"))
    fig.add_scatter(x=[ns[mi]], y=[te[mi]], mode="markers",
                    marker=dict(size=16, color="#43A047", symbol="star"),
                    name=f"{t('最优','Best')} n={ns[mi]}")
    fig.add_scatter(x=[ns[elbow]], y=[te[elbow]], mode="markers",
                    marker=dict(size=14, color="#E53935", symbol="diamond"),
                    name=f"{t('拐点','Elbow')} n={ns[elbow]}")
    fig.add_vline(x=ns[elbow], line_dash="dash", line_color="#E53935", opacity=0.4)
    fig.update_layout(**PC, height=340, hovermode="x unified",
                      xaxis_title=t("Boosting轮数", "n_estimators"),
                      yaxis_title="RMSE", yaxis_gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
    insight(t(f"📌 训练误差持续下降；测试误差在约 {ns[elbow]} 轮后收益递减（拐点），"
              f"{ns[mi]} 轮时达到最优泛化性能。XGBoost正则化有效抑制过拟合，训练/测试曲线差距较小。",
              f"📌 Train error keeps dropping; test error plateaus around n={ns[elbow]} (elbow), "
              f"achieving best generalization at n={ns[mi]}. Regularization keeps train/test gap small."))
    st.divider()

    # Boost轮数逐步拟合
    sec(t("📚 原理可视化②：逐步残差拟合", "📚 Principle ②: Progressive Residual Fitting"))
    with st.spinner(t("计算中…", "Computing…")):
        bd = xgb_boost_rounds()
    sidx = np.argsort(bd["true"])[:100]
    fig = go.Figure()
    for n, c in [("1", "#bdbdbd"), ("5", "#FF9800"), ("20", "#1E88E5"), ("100", "#43A047")]:
        fig.add_scatter(x=list(range(100)), y=np.array(bd[n])[sidx],
                        mode="lines", line=dict(color=c, width=1.5),
                        name=f"{n} {t('轮', 'rounds')}")
    fig.add_scatter(x=list(range(100)), y=np.array(bd["true"])[sidx],
                    mode="lines", line=dict(color="#E53935", width=2, dash="dash"),
                    name=t("真实值", "Actual"))
    fig.update_layout(**PC, height=340, hovermode="x unified",
                      xaxis_title=t("样本（按真实值排序）", "Samples (sorted by actual)"),
                      yaxis_title=t("租赁量", "Rentals"), yaxis_gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
    insight(t("📌 随着Boosting轮数增加，预测曲线越来越逼近真实值。"
              "每一轮新树都在修正上一轮的残差，体现了 Boosting 的核心思想。",
              "📌 As boosting rounds increase, predictions progressively converge to actual values. "
              "Each new tree corrects the residuals of the previous — the core idea of Boosting."))
    st.divider()
    reg_charts(res)


# ══════════════════════════════════════════════════════════════
# 页面：决策树回归
# ══════════════════════════════════════════════════════════════
def page_decision_tree():
    info    = MODEL_INFO["决策树回归"]
    display = mname("决策树回归")
    desc    = info["en"] if st.session_state.lang == "en" else info["zh"]

    st.markdown(
        f'<div class="page-title">{info["icon"]} {display} '
        f'<span style="font-size:1rem;color:#64748b;">{t("complexity")}{info["complexity"]}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander(t("algo_intro"), expanded=True):
        st.markdown(f'<div class="algo-desc">{desc}</div>', unsafe_allow_html=True)
        st.code(info["code"], language="python")

    key = "res_决策树回归"
    btn, _ = st.columns([1, 4])
    if btn.button(t("▶ 运行决策树回归", "▶ Run Decision Tree"), type="primary", use_container_width=True):
        with st.spinner(t("训练中…", "Training…")):
            Xtr, Xte, ytr, yte = get_split()
            t0 = time.time()
            m  = DecisionTreeRegressor(max_depth=8, random_state=42)
            m.fit(Xtr, ytr)
            elapsed = round(time.time() - t0, 2)
            yp   = m.predict(Xte)
            yp_r = inverse_transform(yp)
            yt_r = inverse_transform(yte.values)
            rmse = round(float(np.sqrt(mean_squared_error(yt_r, yp_r))), 2)
            mae  = round(float(mean_absolute_error(yt_r, yp_r)), 2)
            r2   = round(float(r2_score(yte.values, yp)), 4)
            fi   = pd.Series(m.feature_importances_, index=FEATURE_COLS).nlargest(10)
            st.session_state[key] = {
                "yt": yt_r, "yp": yp_r, "elapsed": elapsed,
                "RMSE": rmse, "MAE": mae, "R²": r2, "fi": fi,
            }
        st.success(t("✅ 完成", "✅ Done"))

    if key not in st.session_state:
        st.info(t("run_hint")); return

    d = st.session_state[key]
    st.divider()
    metrics_row(d)
    st.divider()
    reg_charts(d)


# ══════════════════════════════════════════════════════════════
# 页面：总结报告
# ══════════════════════════════════════════════════════════════
def page_summary():
    st.markdown(f'<div class="page-title">{t("sum_title")}</div>', unsafe_allow_html=True)

    rows = []; missing = []
    for name in REGRESSION_KEYS:
        key = f"res_{name}"
        if key in st.session_state:
            r = st.session_state[key]
            rows.append({
                t("col_model"):      mname(name),
                "RMSE":              r["RMSE"],
                "MAE":               r["MAE"],
                "R²":                r["R²"],
                t("col_train_time"): r["elapsed"],
            })
        else:
            missing.append(mname(name))

    if not rows:
        st.warning(t("sum_no_model"))
        st.info(t("sum_hint"))
        return

    if missing:
        st.markdown(
            f'<div class="warn-box">⚠️ {t("以下模型尚未运行：","Models not yet run:")} '
            f'{", ".join(missing)}</div>',
            unsafe_allow_html=True,
        )

    df_s      = pd.DataFrame(rows)
    mc        = t("col_model")
    tc        = t("col_train_time")
    best_r2   = df_s.loc[df_s["R²"].idxmax(),   mc]
    best_rmse = df_s.loc[df_s["RMSE"].idxmin(), mc]

    st.markdown(
        f'<div class="banner">🏆 {t("已完成","Completed")} <b>{len(rows)}/{len(REGRESSION_KEYS)}</b> '
        f'{t("个模型","models")} &nbsp;|&nbsp; '
        f'{t("最高R²：","Best R²:")} <b>{best_r2}</b> &nbsp;|&nbsp; '
        f'{t("最低RMSE：","Lowest RMSE:")} <b>{best_rmse}</b></div>',
        unsafe_allow_html=True,
    )

    # 对比表
    sec(t("📊 模型性能对比表", "📊 Model Performance Comparison"))
    st.dataframe(
        df_s.style
        .highlight_max(subset=["R²"], color="#c8e6c9")
        .highlight_min(subset=["RMSE", "MAE", tc], color="#c8e6c9")
        .format({"R²": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        use_container_width=True, height=220,
    )
    st.divider()

    models = df_s[mc].tolist()
    colors = [MODEL_COLORS.get(m, "#9E9E9E") for m in models]

    c1, c2 = st.columns(2)
    with c1:
        sec(t("📈 R² 对比（越高越好）", "📈 R² Comparison (higher is better)"))
        fig = go.Figure(go.Bar(x=models, y=df_s["R²"], marker_color=colors,
                               text=[f"{v:.4f}" for v in df_s["R²"]], textposition="outside"))
        fig.add_hline(y=0.9, line_dash="dot", line_color="#4CAF50",
                      annotation_text=t("excellent_threshold"), annotation_position="top right")
        fig.update_layout(**PC, height=340, yaxis_range=[0, 1.08],
                          yaxis_gridcolor="#f0f0f0", yaxis_title="R²")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec(t("📉 RMSE 对比（越低越好）", "📉 RMSE Comparison (lower is better)"))
        fig = go.Figure(go.Bar(x=models, y=df_s["RMSE"], marker_color=colors,
                               text=[f"{v:.1f}" for v in df_s["RMSE"]], textposition="outside"))
        fig.update_layout(**PC, height=340, yaxis_gridcolor="#f0f0f0", yaxis_title="RMSE")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec(t("⏱️ 训练时长对比", "⏱️ Training Time Comparison"))
        fig = go.Figure(go.Bar(x=models, y=df_s[tc], marker_color=colors,
                               text=[f"{v}s" for v in df_s[tc]], textposition="outside"))
        fig.update_layout(**PC, height=320, yaxis_gridcolor="#f0f0f0",
                          yaxis_title=t("训练时长（秒）", "Seconds"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec(t("🕸️ 综合雷达对比图", "🕸️ Comprehensive Radar Chart"))
        if len(rows) >= 2:
            eps  = 1e-9
            r2n  = (df_s["R²"] - df_s["R²"].min()) / (df_s["R²"].max() - df_s["R²"].min() + eps)
            rmn  = 1 - (df_s["RMSE"] - df_s["RMSE"].min()) / (df_s["RMSE"].max() - df_s["RMSE"].min() + eps)
            man  = 1 - (df_s["MAE"]  - df_s["MAE"].min())  / (df_s["MAE"].max()  - df_s["MAE"].min()  + eps)
            spd  = 1 - (df_s[tc] - df_s[tc].min()) / (df_s[tc].max() - df_s[tc].min() + eps)
            cats = [t("radar_r2"), t("radar_rmse"), t("radar_mae"), t("radar_spd")]
            fig  = go.Figure()
            for i, (idx, row) in enumerate(df_s.iterrows()):
                vals = [r2n[idx], rmn[idx], man[idx], spd[idx], r2n[idx]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]], fill="toself",
                    name=row[mc], line=dict(color=colors[i]), opacity=0.65,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                paper_bgcolor="white", height=320, showlegend=True,
                legend=dict(x=1.05, y=0.5),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(t("运行 2 个以上模型后显示雷达图。", "Run 2+ models to display the radar chart."))

    st.divider()
    sec(t("sum_analysis_title"))
    lang = st.session_state.lang
    st.markdown(f'<div class="algo-desc">{SUMMARY_COMMENT["en" if lang=="en" else "zh"]}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 页面：测试模型
# ══════════════════════════════════════════════════════════════
def page_test():
    st.markdown(
        f'<div class="page-title">🔮 {t("测试模型 · 实时预测", "Test Model · Live Prediction")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="algo-desc">'
        f'{t("输入骑行条件，选择已训练的模型，即时获取预计租车量。","Enter riding conditions, select a trained model, and get instant rental predictions.")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    TRAINED = [k.replace("res_", "") for k in st.session_state
               if k.startswith("res_") and
               k.replace("res_", "") in REGRESSION_KEYS]
    if not TRAINED:
        st.warning(t("⚠️ 请先在 Deep Learning 中训练至少一个模型。",
                     "⚠️ Please train at least one model in Deep Learning first."))
        return

    # 月份 → 季节映射（与数据集一致：1=春 3-5，2=夏 6-8，3=秋 9-11，4=冬 12/1/2）
    def month_to_season(m):
        if m in (3, 4, 5):  return 1
        if m in (6, 7, 8):  return 2
        if m in (9, 10, 11): return 3
        return 4  # 12, 1, 2

    SEASON_LABELS = {
        1: t("🌸 春季 (3–5月)", "🌸 Spring (Mar–May)"),
        2: t("☀️ 夏季 (6–8月)", "☀️ Summer (Jun–Aug)"),
        3: t("🍂 秋季 (9–11月)", "🍂 Autumn (Sep–Nov)"),
        4: t("❄️ 冬季 (12–2月)", "❄️ Winter (Dec–Feb)"),
    }

    c_in, c_out = st.columns([2, 1])
    with c_in:
        sec(t("⚙️ 输入参数", "⚙️ Input Parameters"))
        r1c1, r1c2, r1c3 = st.columns(3)
        hr  = r1c1.slider(t("小时 hr", "Hour"),  0, 23, 8)
        mth = r1c2.slider(t("月份 mnth", "Month"), 1, 12, 6)
        yr  = r1c3.selectbox(t("年份", "Year"), [0, 1],
                              format_func=lambda x: ["2011", "2012"][x])

        # 季节由月份自动推导，只读展示
        season = month_to_season(mth)
        st.markdown(
            f'<div style="background:#f0f4ff;border-left:3px solid #1E88E5;'
            f'padding:.45rem .8rem;border-radius:4px;margin-bottom:.6rem;font-size:.9rem;">'
            f'🗓 <b>{t("季节（由月份自动确定）","Season (auto from month)")}</b>：'
            f'{SEASON_LABELS[season]}</div>',
            unsafe_allow_html=True,
        )

        r2c1, r2c2 = st.columns(2)
        workday = r2c1.selectbox(t("工作日", "Workday"), [0, 1],
                                  format_func=lambda x: [t("否","No"),t("是","Yes")][x])
        weather = r2c2.selectbox(t("天气", "Weather"), [1, 2, 3],
                                  format_func=lambda x: [t("晴","Clear"),t("薄雾","Mist"),t("雨雪","Rain")][x-1])
        r3c1, r3c2, r3c3 = st.columns(3)
        atemp_c  = r3c1.slider(t("体感温度（°C）", "Feels Like (°C)"),  0, 50,  25, 1)
        hum_pct  = r3c2.slider(t("湿度（%）",    "Humidity (%)"),       0, 100, 60, 1)
        wspd_kmh = r3c3.slider(t("风速（km/h）", "Wind Speed (km/h)"),  0, 57,  13, 1)
        # 转回归一化值供模型使用 / convert to normalized for model
        atemp = atemp_c  / 50.0
        hum   = hum_pct  / 100.0
        wspd  = wspd_kmh / 67.0
        r4c1, r4c2 = st.columns(2)
        wday    = r4c1.slider(t("星期几（0=周日）", "Weekday (0=Sun)"), 0, 6, 1)
        holiday = r4c2.selectbox(t("节假日", "Holiday"), [0, 1],
                                  format_func=lambda x: [t("否","No"),t("是","Yes")][x])

        model_c = st.selectbox(t("选择模型", "Select Model"), TRAINED)
        run_btn = st.button(t("🚀 开始预测", "🚀 Predict"), type="primary", use_container_width=True)

    with c_out:
        sec(t("📊 预测结果", "📊 Prediction"))
        if run_btn:
            # Build feature vector matching FEATURE_COLS
            feat = pd.DataFrame([[
                yr, mth, hr,
                math.sin(2*math.pi*hr/24),   math.cos(2*math.pi*hr/24),
                math.sin(2*math.pi*mth/12),  math.cos(2*math.pi*mth/12),
                math.sin(2*math.pi*wday/7),  math.cos(2*math.pi*wday/7),
                math.sin(2*math.pi*season/4),math.cos(2*math.pi*season/4),
                (1 if hr<=6 else 2 if hr<=9 else 3 if hr<=16 else 4 if hr<=19 else 5),
                season, holiday, wday, workday, weather, atemp, hum, wspd,
                hr * workday, atemp * (1 - hum), season * hr,
            ]], columns=FEATURE_COLS)

            CLS_MAP = {
                "线性回归": LinearRegression, "岭回归": Ridge,
                "决策树回归": DecisionTreeRegressor,
                "随机森林": RandomForestRegressor, "梯度提升": GradientBoostingRegressor,
                "XGBoost":  XGBRegressor,
            }
            info2 = MODEL_INFO[model_c]
            pkl   = os.path.join(MLDIR, f"{model_c}.pkl")
            if os.path.exists(pkl):
                m2 = joblib.load(pkl)
            else:
                X_tr, X_te, y_tr, y_te = get_split()
                m2 = CLS_MAP[model_c](**info2["kwargs"])
                m2.fit(X_tr, y_tr)
                joblib.dump(m2, pkl)

            pred_log = float(m2.predict(feat)[0])
            pred_cnt = max(0, int(round(inverse_transform(np.array([pred_log]))[0])))
            level = ("🔴 " + t("高需求", "High") if pred_cnt >= 300
                     else "🟡 " + t("中需求", "Mid") if pred_cnt >= 100
                     else "🟢 " + t("低需求", "Low"))

            st.markdown(
                f'<div class="kpi" style="border-top-color:#43A047;margin-top:.5rem">'
                f'<div class="kpi-v" style="color:#2e7d32">{pred_cnt}</div>'
                f'<div class="kpi-l">{t("预计租车量（次/小时）","Predicted Rentals/hr")}</div>'
                f'</div>', unsafe_allow_html=True,
            )
            st.markdown(f"**{t('需求级别','Demand Level')}：** {level}")
            st.markdown(f"**{t('使用模型','Model')}：** {model_c}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pred_cnt,
                title={"text": t("预测租车量", "Predicted Rentals"), "font": {"color": "#333"}},
                number={"font": {"color": "#2e7d32", "size": 46}},
                gauge={
                    "axis": {"range": [0, 977]},
                    "bar":  {"color": "#43A047"},
                    "steps": [{"range": [0,   100], "color": "#f5f5f5"},
                               {"range": [100, 300], "color": "#e8f5e9"},
                               {"range": [300, 977], "color": "#ffebee"}],
                    "threshold": {"line": {"color": "#E53935", "width": 3},
                                  "thickness": .75, "value": pred_cnt},
                },
            ))
            fig.update_layout(paper_bgcolor="white", font_color="#333",
                              height=250, margin=dict(l=15, r=15, t=35, b=5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(t("设置参数后点击预测", "Set parameters and click Predict"))


# ══════════════════════════════════════════════════════════════
# 主入口 / Main
# ══════════════════════════════════════════════════════════════
def main():
    if "nav_idx" not in st.session_state:
        st.session_state.nav_idx = 0
    if "algo_idx" not in st.session_state:
        st.session_state.algo_idx = 0

    with st.sidebar:
        # ── 语言切换（Apple Segmented Control）──
        lang_c1, lang_c2 = st.columns(2)
        with lang_c1:
            if st.button("🇨🇳 中文", use_container_width=True, key="btn_zh",
                         type="primary" if st.session_state.lang == "zh" else "secondary"):
                st.session_state.lang = "zh"; st.rerun()
        with lang_c2:
            if st.button("🇬🇧 English", use_container_width=True, key="btn_en",
                         type="primary" if st.session_state.lang == "en" else "secondary"):
                st.session_state.lang = "en"; st.rerun()

        st.markdown("---")
        st.markdown(f"## {t('app_title')}")
        st.markdown(f"<span style='font-size:.8rem;color:#90caf9;'>{t('app_subtitle')}</span>",
                    unsafe_allow_html=True)
        st.divider()

        nav_opts = [t("nav_dashboard"), t("nav_preprocess"), t("nav_ml"), t("nav_summary"), t("nav_test")]
        nav = st.radio("nav", nav_opts, index=st.session_state.nav_idx,
                       label_visibility="collapsed")
        st.session_state.nav_idx = nav_opts.index(nav)

        algo_choice = None
        if nav == t("nav_ml"):
            st.markdown(f"<span style='font-size:.85rem;color:#bbdefb;'>{t('choose_algo')}</span>",
                        unsafe_allow_html=True)
            algo_display = [mname(k) for k in ALL_ALGO_KEYS]
            algo_sel = st.radio("algo", algo_display, index=st.session_state.algo_idx,
                                label_visibility="collapsed")
            st.session_state.algo_idx = algo_display.index(algo_sel)
            algo_choice = ALL_ALGO_KEYS[st.session_state.algo_idx]

        # 已完成模型 badge
        done = [n for n in REGRESSION_KEYS if f"res_{n}" in st.session_state]
        if done:
            st.divider()
            st.markdown(f"<span style='font-size:.82rem;color:#a5d6a7;'>{t('done_label')}</span>",
                        unsafe_allow_html=True)
            for n in done:
                r2 = st.session_state[f"res_{n}"]["R²"]
                st.markdown(f"<span style='font-size:.8rem;color:#c8e6c9;'>&nbsp;&nbsp;{mname(n)} — R²={r2:.3f}</span>",
                            unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<span style='font-size:.75rem;color:rgba(200,215,255,.6);'>{t('dataset_info')}</span>",
                    unsafe_allow_html=True)

    # ── 路由 ───────────────────────────────────────────────────
    idx = st.session_state.nav_idx
    if idx == 0:
        page_dashboard()
    elif idx == 1:
        page_preprocess()
    elif idx == 2 and algo_choice:
        if algo_choice == "线性回归":    page_model("线性回归")
        elif algo_choice == "岭回归":    page_model("岭回归")
        elif algo_choice == "随机森林":  page_random_forest()
        elif algo_choice == "梯度提升":  page_model("梯度提升")
        elif algo_choice == "XGBoost":   page_xgboost()
        elif algo_choice == "决策树回归":   page_decision_tree()
    elif idx == 3:
        page_summary()
    elif idx == 4:
        page_test()


if __name__ == "__main__":
    main()
