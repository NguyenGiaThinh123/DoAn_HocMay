import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ĐỒ ÁN HỌC MÁY: DỰ ĐOÁN NGUY CƠ BỆNH TIỂU ĐƯỜNG\n",
            "### So sánh hiệu năng 8 mô hình học máy trên 2 kịch bản dữ liệu (Lâm sàng vs Khảo sát cộng đồng)\n",
            "\n",
            "**Các mô hình thực nghiệm:**\n",
            "1. Logistic Regression\n",
            "2. Decision Tree\n",
            "3. K-Nearest Neighbors (KNN)\n",
            "4. Random Forest\n",
            "5. Support Vector Machine (Linear SVM)\n",
            "6. XGBoost\n",
            "7. LightGBM\n",
            "8. Stacking Ensemble Classifier\n",
            "\n",
            "**Dữ liệu thực nghiệm:**\n",
            "- **Kịch bản 1:** Dữ liệu xét nghiệm máu & chỉ số sinh hóa (Clinical Dataset - 100k records)\n",
            "- **Kịch bản 2:** Dữ liệu khảo sát thói quen sinh hoạt CDC BRFSS (Survey Dataset - 70k records)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Khai báo thư viện cần thiết\n",
            "import os\n",
            "import time\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    roc_auc_score, confusion_matrix, roc_curve\n",
            ")\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.ensemble import RandomForestClassifier, StackingClassifier\n",
            "from sklearn.svm import LinearSVC\n",
            "from sklearn.calibration import CalibratedClassifierCV\n",
            "from xgboost import XGBClassifier\n",
            "from lightgbm import LGBMClassifier\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "print(\"Đã import thành công các thư viện!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Kịch bản 1: Phân tích & Đánh giá trên Dữ liệu Lâm sàng (Dataset 1)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df1 = pd.read_csv('diabetes_prediction_dataset.csv')\n",
            "df1 = df1[df1['gender'] != 'Other']\n",
            "print(\"Kích thước ban đầu:\", df1.shape)\n",
            "display(df1.head())\n",
            "\n",
            "# Phân tầng lấy mẫu 30.000 dòng để tối ưu tốc độ huấn luyện\n",
            "if len(df1) > 30000:\n",
            "    df1, _ = train_test_split(df1, train_size=30000, stratify=df1['diabetes'], random_state=42)\n",
            "\n",
            "y1 = df1['diabetes']\n",
            "X1 = df1.drop(columns=['diabetes'])\n",
            "\n",
            "cat_cols1 = ['gender', 'smoking_history']\n",
            "num_cols1 = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']\n",
            "\n",
            "preprocessor1 = ColumnTransformer([\n",
            "    ('num', StandardScaler(), num_cols1),\n",
            "    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols1)\n",
            "])\n",
            "\n",
            "X1_processed = preprocessor1.fit_transform(X1)\n",
            "X1_train, X1_test, y1_train, y1_test = train_test_split(X1_processed, y1, test_size=0.2, stratify=y1, random_state=42)\n",
            "print(f\"X_train: {X1_train.shape}, X_test: {X1_test.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Kịch bản 2: Phân tích & Đánh giá trên Dữ liệu Khảo sát Thói quen (Dataset 2)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df2 = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')\n",
            "print(\"Kích thước ban đầu:\", df2.shape)\n",
            "display(df2.head())\n",
            "\n",
            "if len(df2) > 30000:\n",
            "    df2, _ = train_test_split(df2, train_size=30000, stratify=df2['Diabetes_binary'], random_state=42)\n",
            "\n",
            "y2 = df2['Diabetes_binary'].astype(int)\n",
            "X2 = df2.drop(columns=['Diabetes_binary'])\n",
            "\n",
            "scaler2 = StandardScaler()\n",
            "X2_processed = scaler2.fit_transform(X2)\n",
            "X2_train, X2_test, y2_train, y2_test = train_test_split(X2_processed, y2, test_size=0.2, stratify=y2, random_state=42)\n",
            "print(f\"X_train: {X2_train.shape}, X_test: {X2_test.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Xem bảng so sánh kết quả đã thực nghiệm và biểu đồ trực quan"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== BẢNG SO SÁNH DATASET 1 (LÂM SÀNG) ===\")\n",
            "res1 = pd.read_csv('results/Dataset1_Clinical_comparison.csv')\n",
            "display(res1)\n",
            "\n",
            "print(\"\\n=== BẢNG SO SÁNH DATASET 2 (KHẢO SÁT THÓI QUEN) ===\")\n",
            "res2 = pd.read_csv('results/Dataset2_Survey_comparison.csv')\n",
            "display(res2)"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Diabetes_Prediction_Benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Đã tạo file notebook: Diabetes_Prediction_Benchmark.ipynb")
