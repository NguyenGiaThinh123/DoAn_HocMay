import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent Tkinter thread conflicts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Create output directories for charts, tables, and saved models
os.makedirs("results", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 70)
print(" MACHINE LEARNING DIABETES PREDICTION BENCHMARK (LEAK-FREE PIPELINE)")
print("=" * 70)


def build_model_pipelines(preprocessor):
    """
    Khởi tạo 8 Pipeline mô hình độc lập.
    - Với 6 mô hình đơn lẻ (ngoài SVM), mỗi mô hình được đóng gói trong
      Pipeline([('preprocessor', clone(preprocessor)), ('classifier', clf)])
      để bộ tiền xử lý chỉ được fit trên X_train và transform trên X_test.
    - Riêng SVM (Linear), Pipeline([('preprocessor', clone(preprocessor)), ('classifier', LinearSVC(...))])
      được đặt BÊN TRONG CalibratedClassifierCV(estimator=svm_pipeline, cv=3) để trong mỗi fold hiệu chỉnh
      xác suất, bộ tiền xử lý chỉ được fit trên phần dữ liệu huấn luyện của chính fold đó (16.000 mẫu),
      không có preprocessor nào fit trước toàn bộ quá trình cross-validation của SVM.
    - Với Stacking Ensemble, mỗi mô hình thành phần (base estimator) là một Pipeline riêng có chứa
      bước clone(preprocessor), giúp quá trình tạo dự đoán out-of-fold bằng cross-validation (cv=3)
      chỉ fit bộ tiền xử lý trên từng fold huấn luyện, tuyệt đối không rò rỉ dữ liệu giữa các fold hay từ tập test.
    """
    single_classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        "SVM (Linear)": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    }

    pipelines = {}
    for name, clf in single_classifiers.items():
        if name == "SVM (Linear)":
            svm_pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("classifier", clf)
            ])
            calibrated_svm = CalibratedClassifierCV(
                estimator=svm_pipeline,
                cv=3
            )
            pipelines[name] = calibrated_svm
        else:
            pipelines[name] = Pipeline(steps=[
                ('preprocessor', clone(preprocessor)),
                ('classifier', clf)
            ])

    # 8. Stacking Ensemble: mỗi base estimator có Pipeline tiền xử lý riêng để tránh data leakage trong CV
    stacking_base_estimators = [
        ('lr', Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])),
        ('rf', Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1))
        ])),
        ('lgbm', Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))
        ])),
    ]

    stacking_clf = StackingClassifier(
        estimators=stacking_base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=3,
        n_jobs=1
    )
    pipelines["Stacking Ensemble"] = Pipeline(steps=[
        ('classifier', stacking_clf)
    ])

    return pipelines


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor, dataset_name, model_save_path, single_model_save_path=None):
    print(f"\n[+] Huấn luyện 8 mô hình cho: {dataset_name}")
    print(f"    Số lượng mẫu Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Số cột đầu vào: {X_train.shape[1]}")

    pipelines = build_model_pipelines(preprocessor)

    results = []
    roc_curves = {}
    trained_pipelines = {}

    for name, pipe in pipelines.items():
        t0 = time.time()
        print(f"  --> Đang train {name:20s} ... ", end="", flush=True)
        # Fit toàn bộ pipeline CHỈ trên tập huấn luyện (X_train, y_train)
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # Đánh giá trên cùng tập kiểm thử độc lập X_test
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            classes = list(pipe.classes_)
            pos_idx = classes.index(1)
            y_prob = pipe.predict_proba(X_test)[:, pos_idx]
        else:
            y_prob = pipe.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, roc)
        trained_pipelines[name] = pipe

        results.append({
            "Mô hình": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Độ nhạy)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4),
            "Thời gian train (s)": round(train_time, 2)
        })
        print(f"Xong! (Acc: {acc*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%, AUC: {roc*100:.2f}%, Time: {train_time:.2f}s)")

    df_res = pd.DataFrame(results)

    # Lưu bảng kết quả thực nghiệm vào CSV
    csv_path = f"results/{dataset_name}_comparison.csv"
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[V] Đã lưu bảng kết quả vào: {csv_path}")

    # Lưu toàn bộ 8 pipeline mô hình đã huấn luyện trên X_train
    joblib.dump(trained_pipelines, model_save_path)
    print(f"[V] Đã lưu 8 pipeline mô hình (không rò rỉ dữ liệu) vào: {model_save_path}")

    if single_model_save_path is not None and "LightGBM" in trained_pipelines:
        joblib.dump(trained_pipelines["LightGBM"], single_model_save_path)
        print(f"[V] Đã cập nhật mô hình đơn lẻ LightGBM vào: {single_model_save_path}")

    # Vẽ biểu đồ so sánh ROC Curve
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, roc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall / TPR)', fontsize=12)
    plt.title(f'Đường cong ROC - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_chart_path = f"results/charts/{dataset_name}_roc_curves.png"
    plt.tight_layout()
    plt.savefig(roc_chart_path, dpi=300)
    plt.close()

    # Vẽ biểu đồ cột so sánh F1 và Recall
    plt.figure(figsize=(12, 6))
    df_melted = df_res.melt(
        id_vars=["Mô hình"],
        value_vars=["Recall (Độ nhạy)", "F1-Score", "ROC-AUC"],
        var_name="Chỉ số",
        value_name="Điểm"
    )
    sns.barplot(data=df_melted, x="Mô hình", y="Điểm", hue="Chỉ số", palette="Blues_r")
    plt.title(f'So sánh Recall, F1-Score và ROC-AUC - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    bar_chart_path = f"results/charts/{dataset_name}_metrics_bar.png"
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()

    return df_res, trained_pipelines


if __name__ == "__main__":
    # ==============================================================================
    # 1. XỬ LÝ DATASET 1: Clinical Data (diabetes_prediction_dataset.csv)
    # ==============================================================================
    print("\n" + "=" * 50)
    print("1. XỬ LÝ DATASET 1 (Lâm sàng & Xét nghiệm máu)")
    print("=" * 50)
    df1 = pd.read_csv("diabetes_prediction_dataset.csv")

    # Lọc bỏ giá trị không rõ ở gender
    df1 = df1[df1['gender'] != 'Other']

    # Subsampling 30,000 dòng phân tầng
    if len(df1) > 30000:
        df1, _ = train_test_split(df1, train_size=30000, stratify=df1['diabetes'], random_state=42)

    # Tách biến đầu vào X1 và biến mục tiêu y1
    y1 = df1['diabetes'].astype(int)
    X1 = df1.drop(columns=['diabetes'])

    # Chia train/test TRƯỚC khi fit bất kỳ bộ tiền xử lý nào
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, stratify=y1, random_state=42
    )

    cat_cols1 = ['gender', 'smoking_history']
    num_cols1 = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    preprocessor1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols1),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols1)
        ]
    )

    df_res1, _ = evaluate_models(
        X1_train, X1_test, y1_train, y1_test,
        preprocessor=preprocessor1,
        dataset_name="Dataset1_Clinical",
        model_save_path="models/clinical_all_models.joblib",
        single_model_save_path="models/clinical_model.joblib"
    )

    # ==============================================================================
    # 2. XỬ LÝ DATASET 2: Survey / Lifestyle (diabetes_binary_5050split_health_indicators_BRFSS2015.csv)
    # ==============================================================================
    print("\n" + "=" * 50)
    print("2. XỬ LÝ DATASET 2 (Khảo sát thói quen / Sinh hoạt CDC)")
    print("=" * 50)
    df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    if len(df2) > 30000:
        df2, _ = train_test_split(df2, train_size=30000, stratify=df2['Diabetes_binary'], random_state=42)

    # Tách biến đầu vào X2 và biến mục tiêu y2
    y2 = df2['Diabetes_binary'].astype(int)
    X2 = df2.drop(columns=['Diabetes_binary'])

    # Chia train/test TRƯỚC khi fit bất kỳ bộ tiền xử lý nào
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, stratify=y2, random_state=42
    )

    num_cols2 = list(X2.columns)
    preprocessor2 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols2)
        ]
    )

    df_res2, _ = evaluate_models(
        X2_train, X2_test, y2_train, y2_test,
        preprocessor=preprocessor2,
        dataset_name="Dataset2_Survey",
        model_save_path="models/survey_all_models.joblib",
        single_model_save_path="models/survey_model.joblib"
    )

    print("\n" + "=" * 70)
    print(" TẤT CẢ QUÁ TRÌNH HUẤN LUYỆN, ĐÁNH GIÁ VÀ LƯU MÔ HÌNH ĐÃ HOÀN TẤT!")
    print(" - Thư mục kết quả: results/ (chứa CSV và charts)")
    print(" - Thư mục mô hình: models/ (chứa pipeline đã fit trên X_train)")
    print("=" * 70)
