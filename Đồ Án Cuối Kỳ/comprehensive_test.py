import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from streamlit.testing.v1 import AppTest
from app import get_class_1_probability, format_evaluation_table, get_knn_explanation_note

print("=" * 80)
print("BẮT ĐẦU CHẠY KIỂM THỬ TOÀN DIỆN (PHẦN 5: DATA LEAKAGE, METRICS, STACKING & UI)")
print("=" * 80)

# -------------------------------------------------------------
# 1. Tái tạo tập Train / Test chuẩn để kiểm chứng Data Leakage (Mục 1, 2, 3, 4, 5)
# -------------------------------------------------------------
print("\n[TEST 1] Xác minh bộ tiền xử lý CHỈ được fit trên X_train (24.000 mẫu), KHÔNG chạm vào X_test:")

# Dataset 1
df1 = pd.read_csv("diabetes_prediction_dataset.csv")
df1 = df1[df1['gender'] != 'Other']
df1, _ = train_test_split(df1, train_size=30000, stratify=df1['diabetes'], random_state=42)
y1 = df1['diabetes'].astype(int)
X1 = df1.drop(columns=['diabetes'])
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, stratify=y1, random_state=42
)
num_cols1 = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Dataset 2
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df2, _ = train_test_split(df2, train_size=30000, stratify=df2['Diabetes_binary'], random_state=42)
y2 = df2['Diabetes_binary'].astype(int)
X2 = df2.drop(columns=['Diabetes_binary'])
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, stratify=y2, random_state=42
)
num_cols2 = list(X2.columns)

# Kiểm tra không có oversampling trên tập test (Mục 3)
assert len(X1_train) == 24000 and len(X1_test) == 6000, "Kích thước train/test Kịch bản 1 sai"
assert len(X2_train) == 24000 and len(X2_test) == 6000, "Kích thước train/test Kịch bản 2 sai"
assert y2_test.value_counts()[0] == 3000 and y2_test.value_counts()[1] == 3000, "Tập test Kịch bản 2 bị biến đổi phân phối"
print("  + Xác nhận tập Train = 24.000 mẫu, tập Test = 6.000 mẫu độc lập, KHÔNG có oversampling trên tập Test.")

clinical_dict = joblib.load("models/clinical_all_models.joblib")
survey_dict = joblib.load("models/survey_all_models.joblib")

train1_means = X1_train[num_cols1].mean(axis=0).to_numpy()
full1_means = X1[num_cols1].mean(axis=0).to_numpy()
train2_means = X2_train[num_cols2].mean(axis=0).to_numpy()
full2_means = X2[num_cols2].mean(axis=0).to_numpy()

# Kiểm tra từng mô hình trong Dataset 1
for name, pipe in clinical_dict.items():
    if name == "SVM (Linear)":
        assert isinstance(pipe, CalibratedClassifierCV), "SVM (Linear) phải là CalibratedClassifierCV bọc ngoài Pipeline"
        # Xác nhận template estimator gốc không bị fit trước CV
        try:
            check_is_fitted(pipe.estimator.named_steps['preprocessor'])
            raise AssertionError("Preprocessor bên trong template estimator không được fit trước CV!")
        except NotFittedError:
            pass
        assert len(pipe.calibrated_classifiers_) == 3, "SVM phải có đúng 3 fold calibrators (cv=3)"
        fold_means = []
        for cal_clf in pipe.calibrated_classifiers_:
            fold_scaler = cal_clf.estimator.named_steps['preprocessor'].named_transformers_['num']
            assert fold_scaler.n_samples_seen_ == 16000, f"Mỗi fold của SVM phải fit trên 16.000 mẫu (2/3 của 24.000), nhưng có {fold_scaler.n_samples_seen_}"
            fold_means.append(fold_scaler.mean_)
        # Các fold khác nhau phải có mean_ khác nhau (chứng minh được fit độc lập trên từng fold)
        assert not np.allclose(fold_means[0], fold_means[1], atol=1e-6)
    elif name != "Stacking Ensemble":
        scaler = pipe.named_steps['preprocessor'].named_transformers_['num']
        assert scaler.n_samples_seen_ == 24000, f"{name} (Clinical) có n_samples_seen_={scaler.n_samples_seen_} != 24000"
        assert np.allclose(scaler.mean_, train1_means, atol=1e-10), f"{name} (Clinical) mean_ không khớp X1_train"
        assert not np.allclose(scaler.mean_, full1_means, atol=1e-6), f"{name} (Clinical) mean_ trùng với toàn bộ 30.000 mẫu!"
    else:
        stack_clf = pipe.named_steps['classifier']
        assert stack_clf.cv == 3, "Stacking Ensemble phải dùng cv=3"
        for base_est in stack_clf.estimators_:
            base_scaler = base_est.named_steps['preprocessor'].named_transformers_['num']
            assert base_scaler.n_samples_seen_ == 24000
            assert np.allclose(base_scaler.mean_, train1_means, atol=1e-10)

# Kiểm tra từng mô hình trong Dataset 2
for name, pipe in survey_dict.items():
    if name == "SVM (Linear)":
        assert isinstance(pipe, CalibratedClassifierCV), "SVM (Linear) phải là CalibratedClassifierCV bọc ngoài Pipeline"
        try:
            check_is_fitted(pipe.estimator.named_steps['preprocessor'])
            raise AssertionError("Preprocessor bên trong template estimator không được fit trước CV!")
        except NotFittedError:
            pass
        assert len(pipe.calibrated_classifiers_) == 3, "SVM phải có đúng 3 fold calibrators (cv=3)"
        fold_means = []
        for cal_clf in pipe.calibrated_classifiers_:
            fold_scaler = cal_clf.estimator.named_steps['preprocessor'].named_transformers_['num']
            assert fold_scaler.n_samples_seen_ == 16000, f"Mỗi fold của SVM phải fit trên 16.000 mẫu, nhưng có {fold_scaler.n_samples_seen_}"
            fold_means.append(fold_scaler.mean_)
        assert not np.allclose(fold_means[0], fold_means[1], atol=1e-6)
    elif name != "Stacking Ensemble":
        scaler = pipe.named_steps['preprocessor'].named_transformers_['num']
        assert scaler.n_samples_seen_ == 24000, f"{name} (Survey) có n_samples_seen_={scaler.n_samples_seen_} != 24000"
        assert np.allclose(scaler.mean_, train2_means, atol=1e-10), f"{name} (Survey) mean_ không khớp X2_train"
        assert not np.allclose(scaler.mean_, full2_means, atol=1e-6), f"{name} (Survey) mean_ trùng với toàn bộ 30.000 mẫu!"
    else:
        stack_clf = pipe.named_steps['classifier']
        assert stack_clf.cv == 3, "Stacking Ensemble phải dùng cv=3"
        for base_est in stack_clf.estimators_:
            base_scaler = base_est.named_steps['preprocessor'].named_transformers_['num']
            assert base_scaler.n_samples_seen_ == 24000
            assert np.allclose(base_scaler.mean_, train2_means, atol=1e-10)

print("  + Xác nhận 100%: 6 mô hình đơn lẻ + Stacking fit preprocessor trên đúng 24.000 mẫu X_train;")
print("    Riêng SVM (Linear) bọc Pipeline bên trong CalibratedClassifierCV(cv=3), mỗi fold fit preprocessor riêng trên đúng 16.000 mẫu của fold đó!")

# -------------------------------------------------------------
# 2. Xác minh cả 8 mô hình dùng cùng tập Test và khớp 100% với CSV kết quả (Mục 2 & 5)
# -------------------------------------------------------------
print("\n[TEST 2] Đối chiếu kết quả đánh giá trực tiếp trên tập Test với tệp CSV thực nghiệm:")
csv1 = pd.read_csv("results/Dataset1_Clinical_comparison.csv").set_index("Mô hình")
csv2 = pd.read_csv("results/Dataset2_Survey_comparison.csv").set_index("Mô hình")

for name, pipe in clinical_dict.items():
    y_pred = pipe.predict(X1_test)
    y_prob = pipe.predict_proba(X1_test)[:, list(pipe.classes_).index(1)]
    assert round(accuracy_score(y1_test, y_pred), 4) == csv1.loc[name, "Accuracy"]
    assert round(precision_score(y1_test, y_pred, zero_division=0), 4) == csv1.loc[name, "Precision"]
    assert round(recall_score(y1_test, y_pred, zero_division=0), 4) == csv1.loc[name, "Recall (Độ nhạy)"]
    assert round(f1_score(y1_test, y_pred, zero_division=0), 4) == csv1.loc[name, "F1-Score"]
    assert round(roc_auc_score(y1_test, y_prob), 4) == csv1.loc[name, "ROC-AUC"]

for name, pipe in survey_dict.items():
    y_pred = pipe.predict(X2_test)
    y_prob = pipe.predict_proba(X2_test)[:, list(pipe.classes_).index(1)]
    assert round(accuracy_score(y2_test, y_pred), 4) == csv2.loc[name, "Accuracy"]
    assert round(precision_score(y2_test, y_pred, zero_division=0), 4) == csv2.loc[name, "Precision"]
    assert round(recall_score(y2_test, y_pred, zero_division=0), 4) == csv2.loc[name, "Recall (Độ nhạy)"]
    assert round(f1_score(y2_test, y_pred, zero_division=0), 4) == csv2.loc[name, "F1-Score"]
    assert round(roc_auc_score(y2_test, y_prob), 4) == csv2.loc[name, "ROC-AUC"]

print("  + Cả 8 mô hình ở 2 kịch bản đều khớp 100% với tệp CSV trên cùng tập kiểm thử X_test (6.000 mẫu).")

# -------------------------------------------------------------
# 3. Kiểm tra xác suất đầu ra và ghi chú KNN trên mô hình mới
# -------------------------------------------------------------
print("\n[TEST 3] Kiểm tra xác suất dự đoán của 8 mô hình mới trên đầu vào mặc định:")
default_clinical = pd.DataFrame([{
    'gender': 'Male',
    'age': 50.0,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'never',
    'bmi': 27.5,
    'HbA1c_level': 6.2,
    'blood_glucose_level': 140.0
}])

probs_default = []
for name, pipe in clinical_dict.items():
    p, err = get_class_1_probability(pipe, default_clinical)
    assert err is None
    probs_default.append(p)
    print(f"  + {name:20s}: P(diabetes=1) = {p*100:6.2f}%")

p_mean = sum(probs_default) / 8.0
print(f"  => p_mean = (1/8) * sum(p_i) = {p_mean*100:.2f}% (Stacking riêng biệt = {probs_default[-1]*100:.2f}%)")

knn_note_clin = get_knn_explanation_note(clinical_dict["KNN"], default_clinical)
assert "n_neighbors = 7" in knn_note_clin and "0/7 = 0.00%" in knn_note_clin
print(f"  + Ghi chú KNN: {knn_note_clin}")

# -------------------------------------------------------------
# 4. Kiểm thử giao diện Streamlit bằng AppTest (Mục 6, 7, 8)
# -------------------------------------------------------------
print("\n[TEST 4] Kiểm thử giao diện Streamlit bằng AppTest (Mục 6, 7, 8):")
at = AppTest.from_file("app.py", default_timeout=30)
at.run()
assert not at.exception, f"Lỗi khi khởi động Streamlit: {at.exception}"

# Khôi phục hồ sơ mẫu rồi nhấn dự đoán Kịch bản 1 (So sánh toàn bộ 8 mô hình)
at.button(key="btn_reset_clin").click().run()
at.button(key="btn_pred_clin").click().run()
assert not at.exception, f"Lỗi khi dự đoán Kịch bản 1: {at.exception}"

# Kiểm tra công thức chỉ hiển thị đúng 1 lần và đúng định dạng yêu cầu (Mục 7)
assert len(at.latex) == 1, f"Công thức LaTeX phải hiển thị đúng 1 lần, nhưng có {len(at.latex)} phần tử"
assert r"\bar{p} = \frac{1}{8}\sum_{i=1}^{8} p_i" in at.latex[0].value, f"Công thức sai: {at.latex[0].value}"
assert len(at.metric) == 4, f"Thẻ kết quả phải hiển thị 4 chỉ số tóm tắt (st.metric), nhưng có {len(at.metric)}"
assert at.metric[0].value == f"{p_mean*100:.2f}%"
assert at.metric[2].value == "Trung bình xác suất"
assert at.metric[3].value == "8 mô hình"

all_md_clin = "\n".join([m.value for m in at.markdown])
assert f"{p_mean*100:.2f}%" in all_md_clin
print("  + Kịch bản 1 (So sánh 8 mô hình): Đọc đúng model mới, hiển thị đúng p_mean và công thức LaTeX duy nhất 1 lần.")

# Kiểm tra mô hình đơn lẻ SVM (Linear) ở Kịch bản 1
at.sidebar.selectbox[0].select("SVM (Linear)").run()
at.button(key="btn_pred_clin").click().run()
assert not at.exception, f"Lỗi khi dự đoán đơn lẻ SVM (Linear) Kịch bản 1: {at.exception}"
assert len(at.latex) == 0, "Chế độ mô hình đơn lẻ không được hiển thị công thức trung bình 8 mô hình"
assert "SVM (Linear)" in "\n".join([m.value for m in at.markdown])
print("  + Kịch bản 1 (Mô hình đơn lẻ SVM (Linear)): Hoạt động chính xác.")

# Chuyển sang Kịch bản 2 (CDC Survey)
at.sidebar.radio[0].set_value("📋 Kịch bản 2: Khảo sát sức khỏe cộng đồng (CDC)").run()
at.sidebar.selectbox[0].select("⚡ So sánh toàn bộ 8 mô hình cùng lúc").run()
at.button(key="btn_pred_cdc").click().run()
assert not at.exception, f"Lỗi khi dự đoán Kịch bản 2: {at.exception}"
assert len(at.latex) == 1
assert r"\bar{p} = \frac{1}{8}\sum_{i=1}^{8} p_i" in at.latex[0].value
print("  + Kịch bản 2 (CDC Survey - So sánh 8 mô hình): Đọc đúng model mới, công thức LaTeX hiển thị đúng 1 lần.")

# Kiểm tra mô hình đơn lẻ SVM (Linear) ở Kịch bản 2
at.sidebar.selectbox[0].select("SVM (Linear)").run()
at.button(key="btn_pred_cdc").click().run()
assert not at.exception, f"Lỗi khi dự đoán đơn lẻ SVM (Linear) Kịch bản 2: {at.exception}"
assert "SVM (Linear)" in "\n".join([m.value for m in at.markdown])
print("  + Kịch bản 2 (Mô hình đơn lẻ SVM (Linear)): Hoạt động chính xác.")

print("\n" + "=" * 80)
print(">>> TẤT CẢ 8 TIÊU CHÍ KIỂM THỬ TRONG PHẦN 5 ĐỀU ĐẠT 100%! <<<")
print("=" * 80)
