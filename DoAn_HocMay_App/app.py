import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Bệnh Tiểu Đường - Machine Learning",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# GIAO DIỆN CSS TÙY CHỈNH (NAVY / TEAL CLINICAL THEME)
# -------------------------------------------------------------
st.markdown(
    """
    <meta name="google" content="notranslate">
    <style>
        html, body, [class*="css"], div, span, p, label {
            translate: no !important;
        }
        /* Ẩn triệt để lớp katex-mathml để tránh hiện tượng nhân đôi text toán học trên các trình duyệt */
        .katex-mathml, .katex-mathml * {
            display: none !important;
            visibility: hidden !important;
            font-size: 0 !important;
            line-height: 0 !important;
            width: 0 !important;
            height: 0 !important;
        }

        /* Bố cục trang tổng thể tinh gọn */
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 2.2rem !important;
            max-width: 1400px !important;
        }

        /* Khung thẻ container Streamlit đồng nhất trên nền sáng */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 0.5rem !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
        }

        /* Đảm bảo các ô st.number_input và st.selectbox luôn hiển thị rõ chữ đậm trên nền sáng */
        [data-testid="stNumberInput"] input,
        [data-baseweb="input"] input {
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            background-color: #F8FAFC !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            opacity: 1 !important;
        }
        [data-testid="stNumberInput"] [data-baseweb="input"] {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 6px !important;
        }

        /* Nút bấm chính (Primary Button) tông xanh Teal / Emerald y sinh */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0D9488 0%, #059669 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 3px 10px rgba(13, 148, 136, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0F766E 0%, #047857 100%) !important;
            box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35) !important;
        }

        /* Metric hiển thị nổi bật trong khu vực kết quả */
        [data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            color: #64748B !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.02em !important;
        }

        /* Bố cục thích ứng (Responsive) */
        @media (max-width: 900px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------
# CÁC HÀM TIỆN ÍCH XỬ LÝ DỮ LIỆU & HIỂN THỊ (EXPORTED FOR TESTS)
# -------------------------------------------------------------
def get_class_1_probability(pipeline, input_df):
    """
    Trích xuất xác suất P(target = 1) một cách động dựa trên classes_.
    Không mặc định cột thứ 2 luôn là lớp 1.
    Kiểm tra tính hợp lệ trong đoạn [0, 1].
    """
    if not hasattr(pipeline, "predict_proba"):
        return None, "Mô hình không hỗ trợ hàm predict_proba để tính xác suất."

    try:
        probs = pipeline.predict_proba(input_df)[0]
    except Exception as e:
        return None, f"Lỗi trong quá trình tính predict_proba: {str(e)}"

    if hasattr(pipeline, "classes_"):
        classes = list(pipeline.classes_)
    elif hasattr(pipeline, "named_steps") and hasattr(pipeline.named_steps.get("classifier", None), "classes_"):
        classes = list(pipeline.named_steps["classifier"].classes_)
    else:
        return None, "Không tìm thấy thuộc tính classes_ trong mô hình."

    if 1 not in classes:
        return None, f"Không tìm thấy nhãn dương tính (1) trong danh sách classes_ ({classes})."

    idx_1 = classes.index(1)
    prob_1 = float(probs[idx_1])

    if not (0.0 <= prob_1 <= 1.0):
        return None, f"Giá trị xác suất không hợp lệ ngoài đoạn [0, 1]: {prob_1}"

    return prob_1, None


def format_evaluation_table(csv_path):
    """
    Đọc và định dạng bảng kết quả đánh giá thực nghiệm từ tệp CSV.
    Định dạng các cột Accuracy, Precision, Recall, F1-Score và ROC-AUC thành phần trăm
    với 2 chữ số thập phân ở lớp hiển thị, thống nhất với các thẻ KPI phía trên.
    Không thay đổi tệp CSV gốc.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    styled_df = df.copy()
    float_cols = ['Accuracy', 'Precision', 'Recall (Độ nhạy)', 'F1-Score', 'ROC-AUC']
    for col in float_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{float(x) * 100.0:.2f}%" if pd.notnull(x) else "-"
            )
    if 'Thời gian train (s)' in styled_df.columns:
        styled_df['Thời gian train (s)'] = styled_df['Thời gian train (s)'].apply(
            lambda x: f"{float(x):.2f}s" if pd.notnull(x) else "-"
        )
    return styled_df


def get_knn_explanation_note(knn_pipeline, input_df):
    """
    Trích xuất và giải thích số lượng láng giềng gần nhất k-NN thuộc lớp dương tính.
    """
    try:
        classifier = knn_pipeline.named_steps.get("classifier", None)
        preprocessor = knn_pipeline.named_steps.get("preprocessor", None)

        if classifier is not None and preprocessor is not None and hasattr(classifier, "kneighbors"):
            x_trans = preprocessor.transform(input_df)
            indices = classifier.kneighbors(x_trans, return_distance=False)[0]
            y_train = classifier._y
            neighbor_labels = y_train[indices]
            positive_neighbors = int(np.sum(neighbor_labels == 1))
            total_neighbors = len(neighbor_labels)
            prob_knn, _ = get_class_1_probability(knn_pipeline, input_df)
            n_neighbors = getattr(classifier, "n_neighbors", total_neighbors)
            weights = getattr(classifier, "weights", "uniform")

            return (
                f"Ghi chú KNN: Đã xác minh trực tiếp từ mô hình KNN đã lưu (n_neighbors = {n_neighbors}, weights = '{weights}') — "
                f"với đầu vào hiện tại, có {positive_neighbors}/{total_neighbors} láng giềng gần nhất trong tập huấn luyện thuộc lớp dương tính "
                f"(xác suất = {positive_neighbors}/{total_neighbors} = {prob_knn*100:.2f}%)."
            )
    except Exception as e:
        return f"Ghi chú KNN: Không thể trích xuất chi tiết láng giềng ({str(e)})."

    return "Ghi chú KNN: Không thể trích xuất chi tiết láng giềng."


# -------------------------------------------------------------
# ĐÁNH GIÁ ĐỘNG TRÊN TẬP KIỂM THỬ ĐỘC LẬP (KHÔNG HARDCODE)
# -------------------------------------------------------------
CLINICAL_MODELS_PATH = "models/clinical_all_models.joblib"
SURVEY_MODELS_PATH = "models/survey_all_models.joblib"
CLINICAL_CSV_PATH = "results/Dataset1_Clinical_comparison.csv"
SURVEY_CSV_PATH = "results/Dataset2_Survey_comparison.csv"

# Bảng ánh xạ hiển thị thân thiện trên UI mà vẫn bảo toàn giá trị chuỗi nội bộ cho Pipeline
GENDER_MAP = {
    "Male": "Nam",
    "Female": "Nữ"
}

SMOKING_MAP = {
    "never": "Chưa từng hút thuốc (never)",
    "No Info": "Không có thông tin (No Info)",
    "current": "Đang hút thuốc (current)",
    "former": "Đã từng hút thuốc (former)",
    "ever": "Từng hút thuốc (ever)",
    "not current": "Hiện không hút (not current)"
}

SAMPLE_CLINICAL_DEFAULTS = {
    "gender": "Male",
    "age": 50,
    "smoking": "never",
    "hypertension": "Không",
    "heart_disease": "Không",
    "bmi": 27.5,
    "hba1c": 6.2,
    "glucose": 140.0
}


def init_clinical_sample_state():
    """Khởi tạo và duy trì trạng thái của biểu mẫu lâm sàng theo cơ chế session_state."""
    for field, default_val in [
        ("clin_gender", SAMPLE_CLINICAL_DEFAULTS["gender"]),
        ("clin_age", SAMPLE_CLINICAL_DEFAULTS["age"]),
        ("clin_smoking", SAMPLE_CLINICAL_DEFAULTS["smoking"]),
        ("clin_hypertension", SAMPLE_CLINICAL_DEFAULTS["hypertension"]),
        ("clin_heart_disease", SAMPLE_CLINICAL_DEFAULTS["heart_disease"]),
        ("clin_bmi", None),
        ("clin_hba1c", None),
        ("clin_glucose", None),
    ]:
        if field not in st.session_state:
            st.session_state[field] = st.session_state.get(f"_saved_{field}", default_val)


def reset_clinical_sample():
    """Khôi phục toàn bộ biểu mẫu về đúng giá trị hồ sơ mẫu (Nam, 50 tuổi, 27.5 / 6.2 / 140)."""
    for k, v in SAMPLE_CLINICAL_DEFAULTS.items():
        st.session_state[f"clin_{k}"] = v
        st.session_state[f"_saved_clin_{k}"] = v
    # Vô hiệu hóa kết quả dự đoán cũ cho đến khi người dùng bấm dự đoán lại
    st.session_state["clinical_prediction"] = None


def clear_clinical_numbers():
    """Xóa trắng các trường số BMI, HbA1c, Đường huyết để nhập ca bệnh mới."""
    st.session_state["clin_bmi"] = None
    st.session_state["clin_hba1c"] = None
    st.session_state["clin_glucose"] = None
    st.session_state["_saved_clin_bmi"] = None
    st.session_state["_saved_clin_hba1c"] = None
    st.session_state["_saved_clin_glucose"] = None
    # Xóa hoàn toàn kết quả dự đoán cũ
    st.session_state["clinical_prediction"] = None


def are_clinical_inputs_matching(saved_snapshot, current_snapshot):
    """
    Kiểm tra xem dữ liệu đầu vào hiện tại trên form có khớp 100% với
    snapshot dữ liệu đã dùng để tạo kết quả dự đoán trước đó hay không.
    """
    if saved_snapshot is None or current_snapshot is None:
        return False
    keys_to_compare = ["gender", "age", "smoking", "hypertension", "heart_disease", "bmi", "hba1c", "glucose"]
    for k in keys_to_compare:
        v_saved = saved_snapshot.get(k)
        v_curr = current_snapshot.get(k)
        if v_saved is None or v_curr is None:
            if v_saved != v_curr:
                return False
        elif isinstance(v_saved, (int, float)) and isinstance(v_curr, (int, float)):
            if abs(float(v_saved) - float(v_curr)) > 1e-4:
                return False
        elif v_saved != v_curr:
            return False
    return True


def build_clinical_snapshot(gender, age, smoking, hypertension, heart_disease, bmi, hba1c, glucose):
    """
    Tạo một snapshot dữ liệu đầu vào duy nhất từ trạng thái hiện tại của biểu mẫu, dùng chung cho:
    1) Hồ sơ bệnh nhân bên phải biểu mẫu
    2) Kết quả dự đoán của mô hình
    Nếu người dùng xóa hoặc nhập ngoài phạm vi bất kỳ trường số nào, trả về is_valid=False và input_df=None.
    """
    missing_fields = []
    bmi_valid = bmi is not None and (10.0 <= float(bmi) <= 60.0)
    hba1c_valid = hba1c is not None and (3.0 <= float(hba1c) <= 15.0)
    glucose_valid = glucose is not None and (50.0 <= float(glucose) <= 300.0)

    if not bmi_valid:
        missing_fields.append("BMI (cần trong khoảng 10.0 – 60.0 kg/m²)")
    if not hba1c_valid:
        missing_fields.append("HbA1c (cần trong khoảng 3.0 – 15.0%)")
    if not glucose_valid:
        missing_fields.append("Đường huyết lúc đói (cần trong khoảng 50 – 300 mg/dL)")

    bmi_val = round(float(bmi), 1) if bmi_valid else None
    hba1c_val = round(float(hba1c), 1) if hba1c_valid else None
    glucose_val = round(float(glucose), 0) if glucose_valid else None
    age_val = int(age)

    is_valid = len(missing_fields) == 0
    is_sample = (
        is_valid
        and gender == SAMPLE_CLINICAL_DEFAULTS["gender"]
        and age_val == SAMPLE_CLINICAL_DEFAULTS["age"]
        and smoking == SAMPLE_CLINICAL_DEFAULTS["smoking"]
        and hypertension == SAMPLE_CLINICAL_DEFAULTS["hypertension"]
        and heart_disease == SAMPLE_CLINICAL_DEFAULTS["heart_disease"]
        and abs(bmi_val - SAMPLE_CLINICAL_DEFAULTS["bmi"]) < 1e-4
        and abs(hba1c_val - SAMPLE_CLINICAL_DEFAULTS["hba1c"]) < 1e-4
        and abs(glucose_val - SAMPLE_CLINICAL_DEFAULTS["glucose"]) < 1e-4
    )

    snapshot = {
        "gender": gender,
        "gender_label": GENDER_MAP.get(gender, gender),
        "age": age_val,
        "smoking": smoking,
        "smoking_label": SMOKING_MAP.get(smoking, smoking),
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi_val,
        "hba1c": hba1c_val,
        "glucose": glucose_val,
        "is_sample": is_sample
    }

    if not is_valid:
        return False, missing_fields, snapshot, None

    input_df = pd.DataFrame([{
        'gender': gender,
        'age': float(age_val),
        'hypertension': 1 if hypertension == "Có" else 0,
        'heart_disease': 1 if heart_disease == "Có" else 0,
        'smoking_history': smoking,
        'bmi': float(bmi_val),
        'HbA1c_level': float(hba1c_val),
        'blood_glucose_level': float(glucose_val)
    }])

    return True, [], snapshot, input_df


@st.cache_data
def get_test_set_and_predictions(scenario_type, model_name):
    """
    Tải tập kiểm thử chuẩn (holdout test set 20% = 6.000 mẫu) được tạo độc lập
    với seed random_state=42 giống hệt run_pipeline.py và comprehensive_test.py.
    Tính toán thực tế:
    - Confusion Matrix (TN, FP, FN, TP)
    - Nhãn thực tế (y_test) và nhãn dự đoán (y_pred)
    - Xác suất dự đoán (y_prob)
    - Đường cong ROC (fpr, tpr, roc_auc)
    - Hệ số tương quan Matthews (MCC)
    Đảm bảo 100% số liệu lấy từ mô hình thực tế, không hardcode.
    """
    if "Kịch bản 1" in scenario_type:
        df = pd.read_csv("diabetes_prediction_dataset.csv")
        df = df[df['gender'] != 'Other']
        df, _ = train_test_split(df, train_size=30000, stratify=df['diabetes'], random_state=42)
        y = df['diabetes'].astype(int)
        X = df.drop(columns=['diabetes'])
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        models = joblib.load(CLINICAL_MODELS_PATH)
    else:
        df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
        df, _ = train_test_split(df, train_size=30000, stratify=df['Diabetes_binary'], random_state=42)
        y = df['Diabetes_binary'].astype(int)
        X = df.drop(columns=['Diabetes_binary'])
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        models = joblib.load(SURVEY_MODELS_PATH)

    pipe = models[model_name]
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        classes = list(pipe.classes_)
        pos_idx = classes.index(1)
        y_prob = pipe.predict_proba(X_test)[:, pos_idx]
    else:
        y_prob = pipe.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    mcc_val = matthews_corrcoef(y_test, y_pred)
    acc_val = accuracy_score(y_test, y_pred)
    prec_val = precision_score(y_test, y_pred, zero_division=0)
    rec_val = recall_score(y_test, y_pred, zero_division=0)
    f1_val = f1_score(y_test, y_pred, zero_division=0)

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "acc": float(acc_val),
        "prec": float(prec_val),
        "rec": float(rec_val),
        "f1": float(f1_val),
        "auc": float(auc_val),
        "mcc": float(mcc_val),
        "fpr": fpr,
        "tpr": tpr,
        "total": len(y_test)
    }


def get_model_feature_importance(pipeline, scenario_type):
    """
    Trích xuất độ quan trọng của đặc trưng từ mô hình thực tế.
    Nếu mô hình không có feature_importances_ (ví dụ SVM/Stacking),
    trả về đặc trưng từ mô hình cây đại diện của cùng kịch bản.
    """
    if "Kịch bản 1" in scenario_type:
        num_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        cat_cols = ['gender', 'smoking_history']
        try:
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feature_names = num_cols + cat_names
        except Exception:
            feature_names = num_cols + ['gender_Female', 'gender_Male', 'smoking_never']
    else:
        df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv", nrows=2)
        feature_names = list(df2.drop(columns=['Diabetes_binary']).columns)

    clf = pipeline.named_steps.get('classifier', None) if hasattr(pipeline, 'named_steps') else None
    if clf is not None and hasattr(clf, 'feature_importances_'):
        raw_fi = clf.feature_importances_
    elif clf is not None and hasattr(clf, 'coef_'):
        raw_fi = np.abs(clf.coef_[0])
    else:
        # Fallback lấy từ Random Forest hoặc LightGBM tương ứng
        if "Kịch bản 1" in scenario_type:
            models_ref = joblib.load(CLINICAL_MODELS_PATH)
            clf_ref = models_ref["Random Forest"].named_steps['classifier']
            raw_fi = clf_ref.feature_importances_
        else:
            models_ref = joblib.load(SURVEY_MODELS_PATH)
            clf_ref = models_ref["LightGBM"].named_steps['classifier']
            raw_fi = clf_ref.feature_importances_

    fi_pct = (raw_fi / np.sum(raw_fi)) * 100.0
    sorted_idx = np.argsort(fi_pct)[::-1][:6]
    top_features = [feature_names[i] for i in sorted_idx][::-1]
    top_values = [float(fi_pct[i]) for i in sorted_idx][::-1]
    return top_features, top_values


# -------------------------------------------------------------
# CÁC HÀM TẠO BIỂU ĐỒ BẰNG MATPLOTLIB (KHÔNG DÙNG RAW HTML/SVG)
# -------------------------------------------------------------
def create_risk_spectrum_chart(prob_percent, threshold_percent=50.0):
    """
    Biểu đồ phân tầng rủi ro & định vị ngưỡng quyết định (Hình 3).
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    ax.axvspan(0, threshold_percent, color='#ECFDF5', alpha=0.9, label='Vùng dưới ngưỡng (P < 0.50)')
    ax.axvspan(threshold_percent, 100, color='#FFF1F2', alpha=0.9, label='Vùng đạt/vượt ngưỡng (P >= 0.50)')
    ax.axvline(threshold_percent, color='#E11D48', linestyle='--', linewidth=2, label=f'Ngưỡng phân định ({threshold_percent:.0f}%)')

    marker_color = '#059669' if prob_percent < threshold_percent else '#E11D48'
    ax.plot([prob_percent], [0.5], marker='o', markersize=12, color=marker_color, markeredgecolor='#FFFFFF', markeredgewidth=2.5)

    ax.annotate(
        f'{prob_percent:.2f}%',
        xy=(prob_percent, 0.5),
        xytext=(0, 16),
        textcoords='offset points',
        ha='center',
        fontweight='bold',
        fontsize=11,
        color=marker_color,
        bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec=marker_color, lw=1.5)
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Thang đo xác suất ước lượng của mô hình (%)', fontsize=9, color='#475569', labelpad=6)
    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig


def create_comparison_chart(res_df, avg_prob=None):
    """
    Biểu đồ cột ngang so sánh xác suất giữa các mô hình.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    df_plot = res_df.copy().sort_values(by="Xác suất số", ascending=True)
    models = df_plot["Mô hình"]
    probs = df_plot["Xác suất số"]

    bar_colors = ['#0D9488' if p < 50.0 else '#F59E0B' for p in probs]
    bars = ax.barh(models, probs, color=bar_colors, height=0.55, edgecolor='#0F766E', lw=0.5)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}%",
            va='center',
            ha='left',
            fontsize=8.5,
            fontweight='bold',
            color='#1E293B'
        )

    ax.axvline(50.0, color='#E11D48', linestyle='--', linewidth=1.5, alpha=0.8, label='Ngưỡng chuẩn (50%)')

    if avg_prob is not None:
        ax.axvline(
            avg_prob * 100,
            color='#0284C7',
            linestyle='-',
            linewidth=2.0,
            alpha=0.9,
            label=f'Trung bình ({avg_prob*100:.2f}%)'
        )

    ax.set_xlim(0, max(100.0, max(probs) + 14 if len(probs) > 0 else 100.0))
    ax.set_xlabel("Xác suất ước lượng P(Y=1) (%)", fontsize=9, color='#475569', labelpad=6)
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.legend(loc='lower right', framealpha=0.9, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=8)

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig


def create_roc_curve_chart(fpr, tpr, auc_val, model_name=''):
    """
    Biểu đồ ROC Curve thực tế từ tập test (Hình 4 - Top-Left).
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    ax.plot(fpr, tpr, color='#0284C7', lw=2.5, label=f'{model_name} (AUC = {auc_val:.4f})')
    ax.fill_between(fpr, tpr, color='#E0F2FE', alpha=0.5)
    ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', lw=1.2, label='Ngẫu nhiên (AUC = 0.50)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=8.5, color='#475569')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=8.5, color='#475569')
    ax.set_title(f'Đường cong ROC — {model_name}', fontsize=10.5, fontweight='bold', color='#0F172A')
    ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, facecolor='#FFFFFF', edgecolor='#E2E8F0')

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig


def create_circular_gauges_chart(metrics_list):
    """
    Hiển thị 6 vòng tròn đo tiến trình (Circular Progress Gauges) bằng Matplotlib (Hình 4 - Top-Right).
    metrics_list: list of (label, percentage_for_ring, display_text, color)
    """
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.8), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')

    for ax, (name, val_num, display_str, color) in zip(axes.flatten(), metrics_list):
        ax.set_facecolor('#FFFFFF')
        # Vòng tròn nền
        ax.pie([100], radius=1.0, colors=['#E2E8F0'], wedgeprops=dict(width=0.22, edgecolor='none'))
        # Vòng tiến trình
        pct = min(max(val_num, 0.0), 100.0)
        ax.pie([pct, 100.0 - pct], radius=1.0, colors=[color, 'none'], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.22, edgecolor='none'))
        # Nhãn & giá trị
        ax.text(0, 0.14, name, ha='center', va='center', fontsize=9, fontweight='bold', color='#64748B')
        ax.text(0, -0.16, display_str, ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')

    plt.tight_layout()
    return fig


def create_confusion_matrix_chart(tn, fp, fn, tp, model_name=''):
    """
    Hiển thị ma trận nhầm lẫn bằng bảng Matplotlib chuẩn y tế (Hình 4 - Bottom-Left).
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')

    total_neg = tn + fp
    total_pos = fn + tp

    spec = (tn / total_neg * 100.0) if total_neg > 0 else 0.0
    fpr_rate = (fp / total_neg * 100.0) if total_neg > 0 else 0.0
    fnr_rate = (fn / total_pos * 100.0) if total_pos > 0 else 0.0
    rec_rate = (tp / total_pos * 100.0) if total_pos > 0 else 0.0

    cell_text = [
        [f'TN: {tn:,}\n(Đúng âm: {spec:.1f}%)', f'FP: {fp:,}\n(Dương giả: {fpr_rate:.1f}%)'],
        [f'FN: {fn:,}\n(Âm giả: {fnr_rate:.1f}%)', f'TP: {tp:,}\n(Đúng dương: {rec_rate:.1f}%)']
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=[f'Thực tế: Âm (0)\n({total_neg:,} mẫu)', f'Thực tế: Dương (1)\n({total_pos:,} mẫu)'],
        colLabels=['Dự đoán: Âm (0)', 'Dự đoán: Dương (1)'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#0284C7')
            cell.get_text().set_color('#FFFFFF')
            cell.get_text().set_weight('bold')
        elif c == -1:
            cell.set_facecolor('#0284C7')
            cell.get_text().set_color('#FFFFFF')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#F8FAFC')
            cell.get_text().set_color('#0F172A')

    title = f'Ma trận nhầm lẫn — {model_name}' if model_name else 'Ma trận nhầm lẫn'
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#0F172A', pad=12)
    plt.tight_layout()
    return fig


def create_feature_importance_chart(features, importances, model_name=''):
    """
    Biểu đồ tầm quan trọng đặc trưng (Hình 4 - Bottom-Right).
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    bars = ax.barh(features, importances, color='#0284C7', height=0.55, edgecolor='#0369A1', lw=0.5)

    for bar, val in zip(bars, importances):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center',
            ha='left',
            fontsize=8.5,
            fontweight='bold',
            color='#0369A1'
        )

    max_val = max(importances) if len(importances) > 0 else 50
    ax.set_xlim(0, max_val + 8)
    ax.set_xlabel('Mức độ ảnh hưởng (%)', fontsize=8.5, color='#475569')
    title = f'Đặc trưng quan trọng ({model_name})' if model_name else 'Tầm quan trọng của đặc trưng'
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#0F172A')
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='#CBD5E1')

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    return fig


# -------------------------------------------------------------
# HÀM HIỂN THỊ THẺ KẾT QUẢ DỰ ĐOÁN (FIGURE 3 — KHÔNG LẶP THÔNG TIN)
# -------------------------------------------------------------
def render_prediction_card(
    model_name,
    prob,
    target_var,
    class_desc,
    positive_label_text,
    negative_label_text,
    threshold=0.50,
    is_averaging=False,
    res_df=None,
    knn_pipe=None,
    input_df=None
):
    """
    Hiển thị giao diện kết quả dự đoán gọn gàng, không lặp lại thông tin:
    1) Một thẻ kết quả chính duy nhất (gồm snapshot đầu vào, 4 st.metric: xác suất, nhãn phân loại,
       phương pháp, số mô hình, kết luận ngưỡng 0.50 và biểu đồ định vị ngưỡng).
    2) Khu vực riêng cho bảng & biểu đồ xác suất của 8 mô hình (khi ở chế độ so sánh 8 mô hình).
    3) Khu vực riêng giải thích phương pháp (với đúng 1 công thức toán học duy nhất).
    """
    is_positive = prob >= threshold
    prob_pct = prob * 100.0
    pred_label_code = "1" if is_positive else "0"
    status_text = (
        "Xác suất ước lượng đạt hoặc vượt ngưỡng phân loại 0.50"
        if is_positive
        else "Xác suất ước lượng dưới ngưỡng phân loại 0.50"
    )
    conclusion_text = positive_label_text if is_positive else negative_label_text

    st.divider()

    # 1. THẺ KẾT QUẢ DỰ ĐOÁN CHÍNH DUY NHẤT
    with st.container(border=True):
        st.markdown(f"#### 🩺 Kết quả đánh giá từ mô hình: **{model_name}**")
        if input_df is not None:
            if "HbA1c_level" in input_df.columns:
                gender_raw = input_df['gender'].iloc[0]
                gender_txt = GENDER_MAP.get(gender_raw, gender_raw)
                st.caption(
                    f"Dữ liệu đầu vào thực tế: **Tuổi {int(input_df['age'].iloc[0])}** &bull; "
                    f"**Giới tính: {gender_txt}** &bull; "
                    f"**BMI: {input_df['bmi'].iloc[0]:.1f} kg/m²** &bull; "
                    f"**HbA1c: {input_df['HbA1c_level'].iloc[0]:.1f}%** &bull; "
                    f"**Đường huyết: {input_df['blood_glucose_level'].iloc[0]:.0f} mg/dL** | "
                    f"Ngưỡng phân loại chuẩn: **τ = {threshold:.2f}**"
                )
            else:
                st.caption(
                    f"Dữ liệu đầu vào khảo sát CDC: **Giới tính: {'Nam' if input_df['Sex'].iloc[0]==1 else 'Nữ'}** &bull; "
                    f"**BMI: {input_df['BMI'].iloc[0]:.1f}** &bull; "
                    f"**Tăng huyết áp: {'Có' if input_df['HighBP'].iloc[0]==1 else 'Không'}** &bull; "
                    f"**Mỡ máu cao: {'Có' if input_df['HighChol'].iloc[0]==1 else 'Không'}** | "
                    f"Ngưỡng phân loại chuẩn: **τ = {threshold:.2f}**"
                )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label=f"Xác suất ước lượng P({target_var}=1)", value=f"{prob_pct:.2f}%")
        with m_col2:
            st.metric(
                label="Phân loại theo ngưỡng",
                value=f"Nhãn {pred_label_code} (Ngưỡng {threshold:.2f})"
            )
        with m_col3:
            method_title = "Trung bình xác suất" if is_averaging else model_name
            st.metric(label="Phương pháp tổng hợp" if is_averaging else "Thuật toán sử dụng", value=method_title)
        with m_col4:
            count_title = "8 mô hình" if is_averaging else "1 mô hình"
            st.metric(label="Số mô hình tham gia", value=count_title)

        col_summary, col_chart = st.columns([1.05, 1.35])
        with col_summary:
            st.markdown(
                f"**{status_text}** (`{prob_pct:.2f}%` so với ngưỡng `τ = {threshold:.2f}`).\n\n"
                f"**Đầu ra phân loại của mô hình:** {conclusion_text}"
            )
            if input_df is not None and "HbA1c_level" in input_df.columns:
                hba1c_in = float(input_df["HbA1c_level"].iloc[0])
                glu_in = float(input_df["blood_glucose_level"].iloc[0])
                if hba1c_in >= 5.7 or glu_in >= 100.0:
                    st.warning(
                        f"⚠️ **Cảnh báo tách biệt đầu ra mô hình và ngưỡng xét nghiệm tham khảo:** "
                        f"Chỉ số hiện tại (**HbA1c = {hba1c_in:.1f}%**, **Đường huyết lúc đói = {glu_in:.0f} mg/dL**) "
                        f"đang nằm trong vùng cần được đánh giá lâm sàng theo ngưỡng tham khảo ADA (HbA1c ≥ 5.7% hoặc Đường huyết đói ≥ 100 mg/dL). "
                        f"Đầu ra **“Mô hình dự đoán nhãn {pred_label_code} theo định nghĩa của dataset”** không phủ định các chỉ số xét nghiệm đó "
                        f"và không thay thế đánh giá của nhân viên y tế."
                    )
        with col_chart:
            fig_spec = create_risk_spectrum_chart(prob_pct, threshold * 100)
            st.pyplot(fig_spec, width="stretch")

    # 2. KHU VỰC RIÊNG: BẢNG XÁC SUẤT & BIỂU ĐỒ SO SÁNH 8 MÔ HÌNH
    if is_averaging and res_df is not None:
        with st.container(border=True):
            st.markdown("#### 📊 Bảng xác suất của 8 mô hình (Model Breakdown)")
            c_tbl, c_bar = st.columns([1.1, 1])

            with c_tbl:
                target_prob_col = (
                    "Xác suất ước lượng P(diabetes=1)"
                    if "Xác suất ước lượng P(diabetes=1)" in res_df.columns
                    else "Xác suất ước lượng P(Diabetes_binary=1)"
                )
                display_df = res_df[["Mô hình", target_prob_col, "Nhãn dự đoán (τ=0.50)"]].copy()
                st.dataframe(display_df, hide_index=True, width="stretch")

            with c_bar:
                fig_bar = create_comparison_chart(res_df, avg_prob=prob)
                st.pyplot(fig_bar, width="stretch")

    # 3. KHU VỰC RIÊNG: GIẢI THÍCH PHƯƠNG PHÁP & CẢNH BÁO Y KHOA (CHỈ 1 CÔNG THỨC LATEX DUY NHẤT)
    with st.container(border=True):
        st.markdown("#### 📚 Cơ sở phương pháp tổng hợp & Lưu ý y sinh")
        if is_averaging:
            st.markdown("##### 1. Công thức Equal-weight Probability Averaging")
            st.markdown(
                "Xác suất đại diện được tính bằng trung bình cộng không trọng số của xác suất hậu nghiệm từ 8 mô hình:"
            )
            st.latex(r"\bar{p} = \frac{1}{8}\sum_{i=1}^{8} p_i")
            st.caption(
                f"Giá trị trung bình xác suất thực tế của 8 mô hình: **{prob_pct:.2f}%**. "
                "Khác với Stacking Ensemble (sử dụng Logistic Regression làm Meta-learner học trọng số từ out-of-fold predictions), "
                "phương pháp này lấy trung bình cộng trực tiếp với trọng số đều 1/8 cho mỗi mô hình."
            )

        if knn_pipe is not None and input_df is not None:
            knn_note = get_knn_explanation_note(knn_pipe, input_df)
            st.info(f"🔍 **Giải thích kết quả KNN:** {knn_note}")

        st.markdown("##### Hiệu chuẩn xác suất & Khuyến cáo y khoa")
        st.markdown(
            "- **Hiệu chuẩn SVM:** Chỉ mô hình SVM (Linear) sử dụng Platt Scaling (`CalibratedClassifierCV(cv=3)`) được bao bọc bên ngoài Pipeline tiền xử lý để tránh rò rỉ dữ liệu.\n"
            "- **Khuyến cáo y khoa:** Kết quả mang tính chất tham khảo sàng lọc trong nghiên cứu khoa học máy tính, không thay thế chẩn đoán y khoa chính thức của bác sĩ."
        )


# -------------------------------------------------------------
# TẢI DỮ LIỆU & TẬP MÔ HÌNH THỰC NGHIỆM
# -------------------------------------------------------------
clinical_models = joblib.load(CLINICAL_MODELS_PATH) if os.path.exists(CLINICAL_MODELS_PATH) else None
survey_models = joblib.load(SURVEY_MODELS_PATH) if os.path.exists(SURVEY_MODELS_PATH) else None


# -------------------------------------------------------------
# THANH ĐIỀU HƯỚNG BÊN (SIDEBAR)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 MediLife ML System")
    st.caption("Đồ án Machine Learning &mdash; HUIT")

    st.markdown("---")
    st.markdown("#### ⚙️ Cấu hình hệ thống")

    # Radio chọn kịch bản dữ liệu (Phải là sidebar.radio[0] cho AppTest)
    scenario = st.radio(
        "Chọn kịch bản dữ liệu:",
        [
            "🩺 Kịch bản 1: Dữ liệu xét nghiệm lâm sàng (30.000 mẫu)",
            "📋 Kịch bản 2: Khảo sát sức khỏe cộng đồng (CDC)"
        ],
        index=0
    )

    # Selectbox chọn mô hình (Phải là sidebar.selectbox[0] cho AppTest)
    model_choice = st.selectbox(
        "Lựa chọn mô hình dự đoán:",
        [
            "⚡ So sánh toàn bộ 8 mô hình cùng lúc",
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Random Forest",
            "SVM (Linear)",
            "XGBoost",
            "LightGBM",
            "Stacking Ensemble"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown(
        """
        **Thông số thực nghiệm:**
        - **Tập kiểm thử độc lập:** 6.000 mẫu (20% holdout tách trước khi fit bộ tiền xử lý)
        - **Số mô hình:** 8 mô hình ML
        - **Ngưỡng phân loại chuẩn:** τ = 0.50
        - **Hiệu chuẩn xác suất SVM:** Platt Scaling (`CalibratedClassifierCV(cv=3)`)
        - **Các mô hình khác:** Sử dụng `predict_proba` trực tiếp từ thuật toán
        """
    )


# -------------------------------------------------------------
# TOP HEADER BAR & 4 THẺ KPI CÓ NGUỒN GỐC RÕ RÀNG (MỤC 1 & 4)
# -------------------------------------------------------------
st.markdown("#### ĐỒ ÁN MACHINE LEARNING &mdash; TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM (HUIT)")
st.title("Hệ Thống Dự Đoán Bệnh Tiểu Đường")
st.caption("Bảng điều khiển phân tích học máy và dự báo nguy cơ đái tháo đường &bull; Dữ liệu cập nhật thực nghiệm 2025 - 2026")

is_clin_scenario = "Kịch bản 1" in scenario
csv_path = CLINICAL_CSV_PATH if is_clin_scenario else SURVEY_CSV_PATH
csv_df = pd.read_csv(csv_path, index_col="Mô hình") if os.path.exists(csv_path) else None

# 4 KPI Cards hiển thị số liệu thực nghiệm minh bạch (Mục 1)
if csv_df is not None:
    kpi_cols = st.columns(4)

    if model_choice in csv_df.index:
        # Trường hợp A: Người dùng chọn 1 mô hình cụ thể -> Hiển thị bộ metric đầy đủ của mô hình đó
        row_m = csv_df.loc[model_choice]
        with kpi_cols[0]:
            with st.container(border=True):
                st.caption(f"**ACCURACY** &bull; `{model_choice}`")
                st.subheader(f"{row_m['Accuracy']*100:.2f}%")
                st.caption("Độ chính xác toàn thể")
        with kpi_cols[1]:
            with st.container(border=True):
                st.caption(f"**RECALL** &bull; `{model_choice}`")
                st.subheader(f"{row_m['Recall (Độ nhạy)']*100:.2f}%")
                st.caption("Độ nhạy phát hiện bệnh")
        with kpi_cols[2]:
            with st.container(border=True):
                st.caption(f"**PRECISION** &bull; `{model_choice}`")
                st.subheader(f"{row_m['Precision']*100:.2f}%")
                st.caption("Độ chuẩn xác dương tính")
        with kpi_cols[3]:
            with st.container(border=True):
                st.caption(f"**F1-SCORE** &bull; `{model_choice}`")
                st.subheader(f"{row_m['F1-Score']*100:.2f}%")
                st.caption(f"ROC-AUC: {row_m['ROC-AUC']*100:.2f}%")
        st.caption(
            f"📌 **Nguồn số liệu:** File kết quả `{os.path.basename(csv_path)}` &mdash; Đánh giá độc lập trên 6.000 mẫu kiểm thử (20% holdout). "
            f"Thời gian huấn luyện đo được: **{row_m['Thời gian train (s)']}s**."
        )

    else:
        # Trường hợp B: So sánh toàn bộ 8 mô hình -> Hiển thị giá trị tốt nhất và GHI RÕ TÊN MÔ HÌNH TƯƠNG ỨNG
        best_acc_m = csv_df['Accuracy'].idxmax()
        best_rec_m = csv_df['Recall (Độ nhạy)'].idxmax()
        best_f1_m = csv_df['F1-Score'].idxmax()
        best_auc_m = csv_df['ROC-AUC'].idxmax()

        with kpi_cols[0]:
            with st.container(border=True):
                st.caption("**ACCURACY CAO NHẤT**")
                st.subheader(f"{csv_df.loc[best_acc_m, 'Accuracy']*100:.2f}%")
                st.caption(f"Mô hình: **`{best_acc_m}`**")
        with kpi_cols[1]:
            with st.container(border=True):
                st.caption("**RECALL CAO NHẤT**")
                st.subheader(f"{csv_df.loc[best_rec_m, 'Recall (Độ nhạy)']*100:.2f}%")
                st.caption(f"Mô hình: **`{best_rec_m}`**")
        with kpi_cols[2]:
            with st.container(border=True):
                st.caption("**F1-SCORE CAO NHẤT**")
                st.subheader(f"{csv_df.loc[best_f1_m, 'F1-Score']*100:.2f}%")
                st.caption(f"Mô hình: **`{best_f1_m}`**")
        with kpi_cols[3]:
            with st.container(border=True):
                st.caption("**ROC-AUC CAO NHẤT**")
                st.subheader(f"{csv_df.loc[best_auc_m, 'ROC-AUC']*100:.2f}%")
                st.caption(f"Mô hình: **`{best_auc_m}`**")
        st.caption(
            f"📌 **Nguồn số liệu:** Bảng so sánh 8 mô hình trong `{os.path.basename(csv_path)}`. "
            "Mỗi chỉ số ghi rõ mô hình tương ứng đạt kết quả cao nhất trên cùng tập kiểm thử độc lập 6.000 mẫu."
        )


# -------------------------------------------------------------
# BỐ CỤC CHÍNH BẰNG TABS
# -------------------------------------------------------------
tab_main, tab_compare, tab_docs = st.tabs([
    "🩺 Hồ sơ bệnh nhân & Dự đoán (Hình 1 & 3)",
    "🔬 Bảng so sánh chuyên sâu các mô hình (Hình 4)",
    "📚 Dữ liệu & Phương pháp thực nghiệm"
])


# =============================================================
# TAB 1: HỒ SƠ BỆNH NHÂN & DỰ ĐOÁN (HÌNH 1 & HÌNH 3)
# =============================================================
with tab_main:
    # ---------------------------------------------------------
    # KỊCH BẢN 1: DỮ LIỆU LÂM SÀNG
    # ---------------------------------------------------------
    if is_clin_scenario:
        # Stepper tiêu đề gọn gàng
        sc1, sc2 = st.columns(2)
        with sc1:
            with st.container(border=True):
                st.markdown("👤 **Bước 1** &mdash; Thông tin bệnh nhân & Sinh trắc học")
        with sc2:
            with st.container(border=True):
                st.markdown("🩸 **Bước 2** &mdash; Chỉ số đường huyết & Chuyển hóa lâm sàng")

        init_clinical_sample_state()

        col_form, col_side = st.columns([1.5, 1])

        with col_form:
            st.markdown("##### 📝 Phiếu nhập liệu lâm sàng (New Patient Intake)")
            st.info(
                "💡 **Nhập liệu lâm sàng:** "
                "Bạn có thể nhập trực tiếp các chỉ số cận lâm sàng của bệnh nhân, hoặc nhấn nút **'🔄 Khôi phục hồ sơ mẫu (27.5 / 6.2 / 140)'** "
                "để tự động điền nhanh dữ liệu mẫu (Nam, 50 tuổi, BMI 27.5, HbA1c 6.2%, Đường huyết đói 140 mg/dL) phục vụ kiểm thử mô hình. "
                "Nhấn **'🗑️ Xóa trắng ô số'** để xóa nhanh các chỉ số xét nghiệm."
            )
            f_col1, f_col2 = st.columns(2)

            with f_col1:
                gender = st.selectbox(
                    "Giới tính sinh học",
                    ["Male", "Female"],
                    format_func=lambda x: GENDER_MAP.get(x, x),
                    key="clin_gender",
                    help="Giới tính ghi nhận trên hồ sơ bệnh án"
                )
                age = st.slider("Tuổi (năm)", 1, 100, key="clin_age", help="Độ tuổi của bệnh nhân")
                smoking = st.selectbox(
                    "Tiền sử hút thuốc",
                    ["never", "No Info", "current", "former", "ever", "not current"],
                    format_func=lambda x: SMOKING_MAP.get(x, x),
                    key="clin_smoking",
                    help="Tình trạng phơi nhiễm khói thuốc"
                )
                bmi = st.number_input(
                    "Chỉ số khối cơ thể (BMI)",
                    min_value=0.0,
                    max_value=60.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    key="clin_bmi",
                    placeholder="Nhập BMI (10.0 - 60.0)",
                    help="Trọng lượng (kg) / chiều cao bình phương (m²). Hợp lệ: 10.0 - 60.0"
                )

            with f_col2:
                hypertension = st.selectbox(
                    "Tăng huyết áp",
                    ["Không", "Có"],
                    key="clin_hypertension",
                    help="Đã được bác sĩ chẩn đoán tăng huyết áp"
                )
                heart_disease = st.selectbox(
                    "Bệnh tim mạch",
                    ["Không", "Có"],
                    key="clin_heart_disease",
                    help="Tiền sử bệnh lý động mạch vành hoặc tim mạch"
                )
                hba1c = st.number_input(
                    "Chỉ số HbA1c (%)",
                    min_value=0.0,
                    max_value=15.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    key="clin_hba1c",
                    placeholder="Nhập HbA1c (3.0 - 15.0)",
                    help="Chỉ số hemoglobin gắn đường trong 3 tháng. Hợp lệ: 3.0 - 15.0"
                )
                glucose = st.number_input(
                    "Đường huyết lúc đói (mg/dL)",
                    min_value=0.0,
                    max_value=300.0,
                    value=None,
                    step=1.0,
                    format="%.0f",
                    key="clin_glucose",
                    placeholder="Nhập Đường huyết (50 - 300)",
                    help="Nồng độ glucose huyết tương lúc đói. Hợp lệ: 50 - 300"
                )

            # Lưu lại trạng thái người dùng vừa nhập để giữ nguyên khi chuyển đổi kịch bản
            st.session_state["_saved_clin_gender"] = gender
            st.session_state["_saved_clin_age"] = age
            st.session_state["_saved_clin_smoking"] = smoking
            st.session_state["_saved_clin_hypertension"] = hypertension
            st.session_state["_saved_clin_heart_disease"] = heart_disease
            st.session_state["_saved_clin_bmi"] = bmi
            st.session_state["_saved_clin_hba1c"] = hba1c
            st.session_state["_saved_clin_glucose"] = glucose

            st.write("")
            btn_col1, btn_col2, btn_col3 = st.columns([1.25, 1.1, 0.9])
            with btn_col1:
                btn_pred_clin = st.button("DỰ ĐOÁN TỪ MÔ HÌNH", type="primary", key="btn_pred_clin")
            with btn_col2:
                st.button(
                    "🔄 Khôi phục hồ sơ mẫu (27.5 / 6.2 / 140)",
                    key="btn_reset_clin",
                    on_click=reset_clinical_sample
                )
            with btn_col3:
                st.button(
                    "🗑️ Xóa trắng ô số",
                    key="btn_clear_clin",
                    on_click=clear_clinical_numbers
                )

        # Tạo snapshot dữ liệu đầu vào duy nhất từ trạng thái hiện tại dùng chung cho cả Hồ sơ bệnh nhân và Dự đoán
        is_clin_valid, missing_clin_fields, clin_snapshot, clin_input_df = build_clinical_snapshot(
            gender, age, smoking, hypertension, heart_disease, bmi, hba1c, glucose
        )

        with col_side:
            # Thẻ tóm tắt tiền sử bệnh án — đọc 100% từ clin_snapshot hiện tại
            with st.container(border=True):
                st.markdown("##### 📋 Hồ sơ bệnh lý & Yếu tố nguy cơ")
                if is_clin_valid and clin_snapshot["is_sample"]:
                    st.caption("🏷️ **Chế độ:** Dữ liệu hồ sơ bệnh nhân mẫu (Minh họa)")
                elif is_clin_valid:
                    st.caption("🏷️ **Chế độ:** Dữ liệu bệnh nhân vừa nhập trên biểu mẫu")
                else:
                    st.caption("⚠️ **Chế độ:** Biểu mẫu đang thiếu hoặc có chỉ số chưa hợp lệ")

                hyp_badge = "🔴 **Có bệnh nền**" if clin_snapshot["hypertension"] == "Có" else "🟢 **Không ghi nhận**"
                hd_badge = "🔴 **Có bệnh nền**" if clin_snapshot["heart_disease"] == "Có" else "🟢 **Không ghi nhận**"
                gender_txt = clin_snapshot["gender_label"]
                smoking_txt = clin_snapshot["smoking_label"]

                st.markdown(f"- **Giới tính:** **{gender_txt}** &bull; **Tuổi:** **{clin_snapshot['age']}**")
                st.markdown(f"- **Tăng huyết áp:** {hyp_badge}")
                st.markdown(f"- **Bệnh tim mạch:** {hd_badge}")
                st.markdown(f"- **Tiền sử hút thuốc:** `{smoking_txt}`")

                if clin_snapshot["bmi"] is not None:
                    bmi_v = clin_snapshot["bmi"]
                    bmi_status = "Bình thường" if bmi_v < 25.0 else ("Thừa cân" if bmi_v < 30.0 else "Béo phì")
                    bmi_icon = "🟢" if bmi_v < 25.0 else ("🟡" if bmi_v < 30.0 else "🔴")
                    st.markdown(f"- **Thể trạng (BMI):** {bmi_icon} **{bmi_status}** (`{bmi_v:.1f}` kg/m²)")
                else:
                    st.markdown("- **Thể trạng (BMI):** ⚠️ *Chưa có dữ liệu hợp lệ (Chưa nhập)*")

                if clin_snapshot["hba1c"] is not None:
                    hba1c_v = clin_snapshot["hba1c"]
                    hba1c_eval = "🟢 Bình thường (< 5.7%)" if hba1c_v < 5.7 else ("🟡 Tiền tiểu đường (5.7 - 6.4%)" if hba1c_v < 6.5 else "🔴 Ngưỡng tiểu đường (≥ 6.5%)")
                    st.markdown(f"- **Chỉ số HbA1c:** `{hba1c_v:.1f}%` &rarr; {hba1c_eval}")
                else:
                    st.markdown("- **Chỉ số HbA1c:** ⚠️ *Chưa có dữ liệu hợp lệ (Chưa nhập)*")

                if clin_snapshot["glucose"] is not None:
                    glu_v = clin_snapshot["glucose"]
                    glu_eval = "🟢 Bình thường (< 100)" if glu_v < 100 else ("🟡 Rối loạn (100 - 125)" if glu_v < 126 else "🔴 Ngưỡng tiểu đường (≥ 126)")
                    st.markdown(f"- **Đường huyết đói:** `{glu_v:.0f} mg/dL` &rarr; {glu_eval}")
                else:
                    st.markdown("- **Đường huyết đói:** ⚠️ *Chưa có dữ liệu hợp lệ (Chưa nhập)*")

                st.divider()

                st.markdown("##### 🩺 Diễn giải xét nghiệm theo ngưỡng tham khảo (ADA)")
                if is_clin_valid:
                    st.caption(
                        "📌 **Tách biệt chuyên môn:** Đây là diễn giải xét nghiệm độc lập theo ngưỡng tham khảo Hiệp hội Đái tháo đường Hoa Kỳ (ADA), "
                        "tách biệt với đầu ra phân loại của mô hình học máy. Nếu chỉ số nằm trong vùng cần đánh giá lâm sàng, kết quả mô hình không phủ định chỉ số đó."
                    )
                else:
                    st.warning(
                        "⚠️ *Phần diễn giải cận lâm sàng theo chuẩn ADA đang tạm ẩn do biểu mẫu chưa đủ hoặc chưa hợp lệ ở các chỉ số: "
                        + ", ".join(missing_clin_fields)
                        + ". Vui lòng nhập đầy đủ các trường trên.*"
                    )

        # Xử lý dự đoán Lâm sàng — lưu trữ và đồng bộ hóa trạng thái dự đoán
        if btn_pred_clin:
            if not is_clin_valid or clin_input_df is None:
                st.session_state["clinical_prediction"] = None
                st.warning(
                    "⚠️ Vui lòng nhập đầy đủ và hợp lệ các chỉ số bắt buộc ("
                    + ", ".join(missing_clin_fields)
                    + ") trước khi thực hiện dự đoán."
                )
            elif clinical_models is None:
                st.session_state["clinical_prediction"] = None
                st.error("Không tìm thấy tệp mô hình `models/clinical_all_models.joblib`.")
            else:
                input_df = clin_input_df

                if model_choice == "⚡ So sánh toàn bộ 8 mô hình cùng lúc":
                    results_list = []
                    valid_probs = []

                    for m_name, pipe in clinical_models.items():
                        prob_val, err = get_class_1_probability(pipe, input_df)
                        if err is not None:
                            results_list.append({
                                "Mô hình": m_name,
                                "Xác suất ước lượng P(diabetes=1)": "Lỗi",
                                "Xác suất số": 0.0,
                                "Nhãn dự đoán (τ=0.50)": f"Lỗi: {err}"
                            })
                        else:
                            valid_probs.append(prob_val)
                            pred_lbl = (
                                "Mô hình dự đoán nhãn 1 theo định nghĩa của dataset"
                                if prob_val >= 0.50
                                else "Mô hình dự đoán nhãn 0 theo định nghĩa của dataset"
                            )
                            results_list.append({
                                "Mô hình": m_name,
                                "Xác suất ước lượng P(diabetes=1)": f"{prob_val*100:.2f}%",
                                "Xác suất số": prob_val * 100,
                                "Nhãn dự đoán (τ=0.50)": pred_lbl
                            })

                    res_df = pd.DataFrame(results_list)

                    if len(valid_probs) == 8:
                        avg_prob = sum(valid_probs) / 8.0
                        st.session_state["clinical_prediction"] = {
                            "snapshot": dict(clin_snapshot),
                            "input_df": input_df.copy(),
                            "model_choice": model_choice,
                            "is_averaging": True,
                            "prob": avg_prob,
                            "res_df": res_df.copy(),
                            "model_name": "Trung bình xác suất không trọng số của 8 mô hình (Equal-weight Probability Averaging)",
                            "target_var": "diabetes",
                            "class_desc": "thuộc lớp dương tính theo định nghĩa của dataset",
                            "positive_label_text": "Mô hình dự đoán nhãn 1 theo định nghĩa của dataset.",
                            "negative_label_text": "Mô hình dự đoán nhãn 0 theo định nghĩa của dataset.",
                        }
                else:
                    if model_choice not in clinical_models:
                        st.session_state["clinical_prediction"] = None
                        st.error(f"Mô hình `{model_choice}` không có trong tập mô hình đã huấn luyện.")
                    else:
                        target_pipe = clinical_models[model_choice]
                        prob_val, err = get_class_1_probability(target_pipe, input_df)
                        if err is not None:
                            st.session_state["clinical_prediction"] = None
                            st.error(f"Không thể dự đoán với mô hình {model_choice}: {err}")
                        else:
                            st.session_state["clinical_prediction"] = {
                                "snapshot": dict(clin_snapshot),
                                "input_df": input_df.copy(),
                                "model_choice": model_choice,
                                "is_averaging": False,
                                "prob": prob_val,
                                "res_df": None,
                                "model_name": model_choice,
                                "target_var": "diabetes",
                                "class_desc": "thuộc lớp dương tính theo định nghĩa của dataset",
                                "positive_label_text": "Mô hình dự đoán nhãn 1 theo định nghĩa của dataset.",
                                "negative_label_text": "Mô hình dự đoán nhãn 0 theo định nghĩa của dataset.",
                            }

        # Hiển thị kết quả dự đoán và quản lý trạng thái đồng bộ
        saved_pred = st.session_state.get("clinical_prediction", None)

        if saved_pred is not None:
            inputs_match = are_clinical_inputs_matching(saved_pred.get("snapshot"), clin_snapshot)
            model_matches = (saved_pred.get("model_choice") == model_choice)

            if not inputs_match or not model_matches:
                if not is_clin_valid:
                    st.warning(
                        "⚠️ **Dữ liệu đầu vào đã thay đổi và hiện đang thiếu trường bắt buộc:** "
                        + ", ".join(missing_clin_fields)
                        + ". Kết quả cũ đã được ẩn. Vui lòng nhập đầy đủ trước khi bấm **'DỰ ĐOÁN TỪ MÔ HÌNH'**."
                    )
                else:
                    st.warning(
                        "⚠️ **Dữ liệu đầu vào hoặc mô hình đã thay đổi so với lượt dự đoán trước (Kết quả đã lỗi thời — cần dự đoán lại).** "
                        "Kết quả cũ đã được ẩn để tránh gây nhầm lẫn. "
                        "Vui lòng nhấn nút **'DỰ ĐOÁN TỪ MÔ HÌNH'** để cập nhật kết quả mới cho dữ liệu hiện tại."
                    )
            else:
                render_prediction_card(
                    model_name=saved_pred["model_name"],
                    prob=saved_pred["prob"],
                    target_var=saved_pred["target_var"],
                    class_desc=saved_pred["class_desc"],
                    positive_label_text=saved_pred["positive_label_text"],
                    negative_label_text=saved_pred["negative_label_text"],
                    threshold=0.50,
                    is_averaging=saved_pred["is_averaging"],
                    res_df=saved_pred.get("res_df"),
                    knn_pipe=clinical_models.get("KNN") if clinical_models else None,
                    input_df=saved_pred["input_df"]
                )
        elif not btn_pred_clin:
            if is_clin_valid:
                st.info("ℹ️ **Hướng dẫn:** Hồ sơ đầu vào hiện tại đã hợp lệ. Nhấn nút **'DỰ ĐOÁN TỪ MÔ HÌNH'** để nhận kết quả phân tích mới.")
            else:
                st.warning(
                    "⚠️ **Thiếu dữ liệu đầu vào bắt buộc:** "
                    + ", ".join(missing_clin_fields)
                    + ". Vui lòng nhập đầy đủ hoặc nhấn **'Khôi phục hồ sơ mẫu'** trước khi nhấn **'DỰ ĐOÁN TỪ MÔ HÌNH'**."
                )

    # ---------------------------------------------------------
    # KỊCH BẢN 2: KHẢO SÁT SỨC KHỎE CỘNG ĐỒNG (CDC BRFSS)
    # ---------------------------------------------------------
    else:
        st.info("📋 **Phiếu điều tra dịch tễ CDC BRFSS** &mdash; 21 chỉ số khảo sát sức khỏe cộng đồng")

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("##### 🩺 Tiền sử & Bệnh lý nền")
                highbp = st.selectbox("Tăng huyết áp (HighBP)", ["Không", "Có"], index=0)
                highchol = st.selectbox("Mỡ máu cao (HighChol)", ["Không", "Có"], index=0)
                cholcheck = st.selectbox("Đã kiểm tra mỡ máu trong 5 năm", ["Có", "Không"], index=0)
                stroke = st.selectbox("Tiền sử đột quỵ (Stroke)", ["Không", "Có"], index=0)
                heart_attack = st.selectbox("Bệnh tim / Nhồi máu cơ tim", ["Không", "Có"], index=0)
                diffwalk = st.selectbox("Khó khăn khi đi lại / leo cầu thang", ["Không", "Có"], index=0)
                genhlth = st.select_slider(
                    "Tự đánh giá sức khỏe chung (GenHlth)",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: {1: "1 - Xuất sắc", 2: "2 - Rất tốt", 3: "3 - Tốt", 4: "4 - Tạm được", 5: "5 - Kém"}[x]
                )

            with c2:
                st.markdown("##### 🏃 Lối sống & Thể chất")
                bmi_cdc = st.number_input("Chỉ số BMI", 0.0, 70.0, 28.0, step=0.1, key="cdc_bmi")
                smoker = st.selectbox("Hút trên 100 điếu thuốc trong đời", ["Không", "Có"], index=0)
                phys_act = st.selectbox("Hoạt động thể chất trong 30 ngày qua", ["Có", "Không"], index=0)
                fruits = st.selectbox("Ăn trái cây ít nhất 1 lần/ngày", ["Có", "Không"], index=0)
                veggies = st.selectbox("Ăn rau ít nhất 1 lần/ngày", ["Có", "Không"], index=0)
                alcohol = st.selectbox("Uống nhiều rượu bia (>14 ly/tuần nam, >7 ly nữ)", ["Không", "Có"], index=0)
                physhlth = st.slider("Số ngày sức khỏe thể chất kém (trong 30 ngày)", 0, 30, 2)
                menthlth = st.slider("Số ngày sức khỏe tinh thần kém (trong 30 ngày)", 0, 30, 2)

            with c3:
                st.markdown("##### 👥 Nhân khẩu học & Tiếp cận y tế")
                sex = st.selectbox("Giới tính (Sex)", ["Nam", "Nữ"], index=0)
                age_group = st.select_slider(
                    "Nhóm tuổi (Age 1-13)",
                    options=list(range(1, 14)),
                    value=8,
                    format_func=lambda x: f"Mức {x} ({18+(x-1)*5} - {18+x*5-1} tuổi)" if x < 13 else "Mức 13 (>= 80 tuổi)"
                )
                healthcare = st.selectbox("Có bảo hiểm y tế bất kỳ", ["Có", "Không"], index=0)
                nodoc = st.selectbox("Không đi khám vì lý do chi phí trong 12 tháng qua", ["Không", "Có"], index=0)
                education = st.select_slider(
                    "Trình độ học vấn (Education 1-6)",
                    options=list(range(1, 7)),
                    value=5,
                    format_func=lambda x: {1: "Chưa học tiểu học", 2: "Tiểu học", 3: "THCS", 4: "THPT", 5: "Cao đẳng/Đại học 1-3 năm", 6: "Tốt nghiệp ĐH trở lên"}[x]
                )
                income = st.select_slider(
                    "Mức thu nhập gia đình (Income 1-8)",
                    options=list(range(1, 9)),
                    value=6,
                    format_func=lambda x: f"Mức {x}"
                )

            st.write("")
            btn_pred_cdc = st.button("DỰ ĐOÁN TỪ MÔ HÌNH", type="primary", key="btn_pred_cdc")

        # Xử lý dự đoán CDC
        if btn_pred_cdc:
            if bmi_cdc is None or bmi_cdc <= 0:
                st.warning("⚠️ Vui lòng nhập chỉ số BMI hợp lệ trước khi thực hiện dự đoán.")
            elif survey_models is None:
                st.error("Không tìm thấy tệp mô hình `models/survey_all_models.joblib`.")
            else:
                input_df = pd.DataFrame([{
                    'HighBP': 1 if highbp == "Có" else 0,
                    'HighChol': 1 if highchol == "Có" else 0,
                    'CholCheck': 1 if cholcheck == "Có" else 0,
                    'BMI': float(bmi_cdc),
                    'Smoker': 1 if smoker == "Có" else 0,
                    'Stroke': 1 if stroke == "Có" else 0,
                    'HeartDiseaseorAttack': 1 if heart_attack == "Có" else 0,
                    'PhysActivity': 1 if phys_act == "Có" else 0,
                    'Fruits': 1 if fruits == "Có" else 0,
                    'Veggies': 1 if veggies == "Có" else 0,
                    'HvyAlcoholConsump': 1 if alcohol == "Có" else 0,
                    'AnyHealthcare': 1 if healthcare == "Có" else 0,
                    'NoDocbcCost': 1 if nodoc == "Có" else 0,
                    'GenHlth': float(genhlth),
                    'MentHlth': float(menthlth),
                    'PhysHlth': float(physhlth),
                    'DiffWalk': 1 if diffwalk == "Có" else 0,
                    'Sex': 1 if sex == "Nam" else 0,
                    'Age': float(age_group),
                    'Education': float(education),
                    'Income': float(income)
                }])

                if model_choice == "⚡ So sánh toàn bộ 8 mô hình cùng lúc":
                    results_list = []
                    valid_probs = []

                    for m_name, pipe in survey_models.items():
                        prob_val, err = get_class_1_probability(pipe, input_df)
                        if err is not None:
                            results_list.append({
                                "Mô hình": m_name,
                                "Xác suất ước lượng P(Diabetes_binary=1)": "Lỗi",
                                "Xác suất số": 0.0,
                                "Nhãn dự đoán (τ=0.50)": f"Lỗi: {err}"
                            })
                        else:
                            valid_probs.append(prob_val)
                            pred_lbl = (
                                "Mô hình dự đoán nhãn 1 theo định nghĩa của dataset"
                                if prob_val >= 0.50
                                else "Mô hình dự đoán nhãn 0 theo định nghĩa của dataset"
                            )
                            results_list.append({
                                "Mô hình": m_name,
                                "Xác suất ước lượng P(Diabetes_binary=1)": f"{prob_val*100:.2f}%",
                                "Xác suất số": prob_val * 100,
                                "Nhãn dự đoán (τ=0.50)": pred_lbl
                            })

                    res_df = pd.DataFrame(results_list)

                    if len(valid_probs) == 8:
                        avg_prob = sum(valid_probs) / 8.0
                        render_prediction_card(
                            model_name="Trung bình xác suất không trọng số của 8 mô hình (Equal-weight Probability Averaging)",
                            prob=avg_prob,
                            target_var="Diabetes_binary",
                            class_desc="thuộc lớp dương tính theo định nghĩa của dataset",
                            positive_label_text="Mô hình dự đoán nhãn 1 theo định nghĩa của dataset.",
                            negative_label_text="Mô hình dự đoán nhãn 0 theo định nghĩa của dataset.",
                            threshold=0.50,
                            is_averaging=True,
                            res_df=res_df,
                            knn_pipe=survey_models.get("KNN"),
                            input_df=input_df
                        )
                else:
                    if model_choice not in survey_models:
                        st.error(f"Mô hình `{model_choice}` không có trong tập mô hình đã huấn luyện.")
                    else:
                        target_pipe = survey_models[model_choice]
                        prob_val, err = get_class_1_probability(target_pipe, input_df)
                        if err is not None:
                            st.error(f"Không thể dự đoán với mô hình {model_choice}: {err}")
                        else:
                            render_prediction_card(
                                model_name=model_choice,
                                prob=prob_val,
                                target_var="Diabetes_binary",
                                class_desc="thuộc lớp dương tính theo định nghĩa của dataset",
                                positive_label_text="Mô hình dự đoán nhãn 1 theo định nghĩa của dataset.",
                                negative_label_text="Mô hình dự đoán nhãn 0 theo định nghĩa của dataset.",
                                threshold=0.50,
                                is_averaging=False,
                                input_df=input_df
                            )
        else:
            st.info("ℹ️ **Hướng dẫn:** Vui lòng hoàn thành các mục khảo sát CDC ở biểu mẫu trên và nhấn nút **'DỰ ĐOÁN TỪ MÔ HÌNH'** để nhận kết quả phân tích.")


# =============================================================
# TAB 2: BẢNG SO SÁNH CHUYÊN SÂU CÁC MÔ HÌNH (HÌNH 4)
# =============================================================
with tab_compare:
    st.markdown("### 🔬 Bảng So Sánh Hiệu Năng Mô Hình Học Máy (Model Comparison Dashboard)")
    st.caption("Bố cục 4 góc phần tư trực quan hóa hoàn toàn bằng Matplotlib: ROC Curve, Circular Progress Gauges, Confusion Matrix và Feature Importance.")

    target_models_dict = clinical_models if is_clin_scenario else survey_models

    if target_models_dict is not None and len(target_models_dict) > 0:
        model_names_list = list(target_models_dict.keys())

        # Chọn mô hình trọng tâm để phân tích động (Mục 1 & 2)
        default_selected = model_choice if model_choice in model_names_list else ("Stacking Ensemble" if "Stacking Ensemble" in model_names_list else model_names_list[-1])
        default_idx = model_names_list.index(default_selected)

        c_select, c_note = st.columns([1.2, 2])
        with c_select:
            focus_model = st.selectbox(
                "Chọn mô hình để phân tích chuyên sâu (ROC, Gauges, Confusion Matrix):",
                model_names_list,
                index=default_idx
            )
        with c_note:
            st.info(f"Đang hiển thị dữ liệu kiểm định động của mô hình: **`{focus_model}`** trên cùng tập kiểm thử độc lập 6.000 mẫu.")

        # Lấy dữ liệu đánh giá thực tế từ tập test (KHÔNG HARDCODE)
        eval_data = get_test_set_and_predictions(scenario, focus_model)

        # Hàng 1 (Top-Left: ROC Curve & Top-Right: 6 Circular Progress Gauges)
        col_roc, col_gauge = st.columns([1, 1.1])

        with col_roc:
            st.markdown("##### 1. Đường cong ROC (ROC Curve)")
            fig_roc = create_roc_curve_chart(
                eval_data["fpr"],
                eval_data["tpr"],
                eval_data["auc"],
                model_name=focus_model
            )
            st.pyplot(fig_roc, width="stretch")

        with col_gauge:
            st.markdown("##### 2. Các chỉ số cốt lõi (Core Performance Metrics)")
            # MCC có thang đo [-1.0, +1.0], chuẩn hóa sang 0-100% cho vòng tròn trực quan
            mcc_normalized = (eval_data["mcc"] + 1.0) / 2.0 * 100.0
            gauges_list = [
                ("F1-Score", eval_data["f1"] * 100.0, f"{eval_data['f1']*100:.2f}%", "#0284C7"),
                ("MCC", mcc_normalized, f"{eval_data['mcc']:+.3f}", "#0284C7"),
                ("Accuracy", eval_data["acc"] * 100.0, f"{eval_data['acc']*100:.2f}%", "#0284C7"),
                ("Precision", eval_data["prec"] * 100.0, f"{eval_data['prec']*100:.2f}%", "#0284C7"),
                ("Recall", eval_data["rec"] * 100.0, f"{eval_data['rec']*100:.2f}%", "#0284C7"),
                ("ROC-AUC", eval_data["auc"] * 100.0, f"{eval_data['auc']*100:.2f}%", "#0284C7")
            ]
            fig_gauges = create_circular_gauges_chart(gauges_list)
            st.pyplot(fig_gauges, width="stretch")
            st.caption("📌 **Ghi chú MCC:** Hệ số tương quan Matthews có thang đo chuẩn [-1.0, +1.0]. Được tính trực tiếp từ nhãn thực tế và dự đoán.")

        st.write("")

        # Hàng 2 (Bottom-Left: Confusion Matrix & Bottom-Right: Feature Importance)
        col_cm, col_fi = st.columns([1, 1.1])

        with col_cm:
            st.markdown("##### 3. Ma trận nhầm lẫn y tế (Confusion Matrix)")
            fig_cm = create_confusion_matrix_chart(
                eval_data["tn"],
                eval_data["fp"],
                eval_data["fn"],
                eval_data["tp"],
                model_name=focus_model
            )
            st.pyplot(fig_cm, width="stretch")
            if focus_model == "Stacking Ensemble" and is_clin_scenario:
                st.caption(
                    f"Tính toán thực tế từ 6.000 mẫu kiểm thử độc lập (5.490 mẫu âm tính, 510 mẫu dương tính): "
                    f"**TN = {eval_data['tn']:,}** (Đúng âm), **FP = {eval_data['fp']:,}** (Dương giả), "
                    f"**FN = {eval_data['fn']:,}** (Âm giả), **TP = {eval_data['tp']:,}** (Đúng dương). "
                    f"Số liệu tính động từ nhãn thực tế và nhãn dự đoán của mô hình, khớp 100% với Accuracy {eval_data['acc']*100:.2f}% và Recall {eval_data['rec']*100:.2f}%."
                )
            else:
                st.caption(
                    f"Tính toán thực tế từ 6.000 mẫu kiểm thử: Tổng âm tính thực tế = **{eval_data['tn']+eval_data['fp']:,}**, "
                    f"Tổng dương tính thực tế = **{eval_data['fn']+eval_data['tp']:,}**. "
                    f"TN = **{eval_data['tn']:,}**, FP = **{eval_data['fp']:,}**, FN = **{eval_data['fn']:,}**, TP = **{eval_data['tp']:,}**."
                )

        with col_fi:
            st.markdown("##### 4. Tầm quan trọng của đặc trưng (Feature Importance)")
            top_feats, top_vals = get_model_feature_importance(target_models_dict[focus_model], scenario)
            fig_fi = create_feature_importance_chart(top_feats, top_vals, model_name=focus_model)
            st.pyplot(fig_fi, width="stretch")

        st.write("")
        st.markdown("##### 5. Bảng tổng hợp số liệu thực nghiệm toàn bộ 8 mô hình (Test Set: 6.000 mẫu)")
        formatted_eval = format_evaluation_table(csv_path)
        if formatted_eval is not None:
            st.dataframe(formatted_eval, hide_index=True, width="stretch")
        else:
            st.warning("Chưa tìm thấy tệp CSV kết quả thực nghiệm.")
    else:
        st.error("Chưa tải được tập mô hình đã huấn luyện.")


# =============================================================
# TAB 3: DỮ LIỆU & PHƯƠNG PHÁP THỰC NGHIỆM
# =============================================================
with tab_docs:
    st.markdown("### 📚 Chi Tiết Bộ Dữ Liệu & Quy Trình Ngăn Chặn Data Leakage")

    d_col1, d_col2 = st.columns(2)

    with d_col1:
        with st.container(border=True):
            st.markdown("##### 🧪 Kịch bản 1: Xét nghiệm lâm sàng (Clinical)")
            st.markdown(
                """
                - **Nguồn dữ liệu:** `diabetes_prediction_dataset.csv` (100.000 bản ghi gốc).
                - **Quy mô lấy mẫu:** 30.000 mẫu đại diện theo tỷ lệ tầng (stratified sampling).
                - **Phân bố nhãn:**
                  - Lớp 0 (Không đái tháo đường): ~91.5%
                  - Lớp 1 (Đái tháo đường): ~8.5%
                - **Phân chia dữ liệu:** Train (24.000 mẫu - 80%) / Test (6.000 mẫu - 20%).
                - **Tiền xử lý độc lập:** `StandardScaler` và `OneHotEncoder` chỉ được `fit` trên `X_train` (24.000 mẫu), sau đó mới `transform` trên `X_test` (6.000 mẫu).
                """
            )

    with d_col2:
        with st.container(border=True):
            st.markdown("##### 📋 Kịch bản 2: Khảo sát cộng đồng CDC BRFSS")
            st.markdown(
                """
                - **Nguồn dữ liệu:** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (70.692 bản ghi gốc).
                - **Quy mô lấy mẫu:** 30.000 mẫu cân bằng nhãn 50:50.
                - **Phân bố nhãn:**
                  - Lớp 0 (Không có tiền đái tháo đường/đái tháo đường): 15.000 mẫu (50.0%)
                  - Lớp 1 (Có tiền đái tháo đường hoặc đái tháo đường): 15.000 mẫu (50.0%)
                - **Phân chia dữ liệu:** Train (24.000 mẫu - 80%) / Test (6.000 mẫu - 20%).
                - **Đặc trưng:** 21 chỉ số khảo sát hành vi, lối sống, sức khỏe thể chất, tinh thần và nhân khẩu học.
                """
            )

    st.write("")
    with st.container(border=True):
        st.markdown("#### 🔬 Quy trình đánh giá thực nghiệm & Vai trò của Cross-Validation")
        st.markdown(
            """
            1. **Đánh giá trên tập kiểm định độc lập (Holdout Test 80/20):**
               - Toàn bộ 16 pipeline mô hình (8 mô hình &times; 2 kịch bản) được huấn luyện trên tập Train cố định (24.000 mẫu) và đánh giá độc lập trên tập Test cố định (6.000 mẫu).
               - Bảng số liệu thực nghiệm hiển thị các độ đo (Accuracy, Precision, Recall, F1-Score, ROC-AUC) được tính toán trực tiếp trên tập Test 6.000 mẫu này; **không phải kết quả trung bình của k-fold cross-validation toàn cục**.
            2. **Vai trò cục bộ của 3-fold Cross-Validation (`cv=3`):**
               - **SVM (Linear):** Sử dụng cấu trúc `CalibratedClassifierCV(estimator=svm_pipeline, cv=3)` bọc ngoài Pipeline tiền xử lý. Cơ chế này chia nội bộ tập Train (24.000 mẫu) thành 3 fold (mỗi fold 16.000 mẫu fit và 8.000 mẫu calibration) để khớp hàm sigmoid Platt Scaling mà không làm rò rỉ phân phối xác suất.
               - **Stacking Ensemble:** Sử dụng `StackingClassifier(cv=3)` bên trong tập Train (24.000 mẫu) nhằm sinh dự đoán out-of-fold từ 3 base estimator (Logistic Regression, Random Forest, LightGBM) để làm ma trận đặc trưng meta huấn luyện bộ ước lượng cuối cùng (Logistic Regression).
            """
        )

    with st.container(border=True):
        st.markdown("#### ⚖️ Phân tích nguy cơ rò rỉ dữ liệu (Data Leakage Analysis)")
        st.markdown(
            """
            - **Rò rỉ tiền xử lý (Preprocessing Leakage &mdash; Đã loại trừ hoàn toàn):**
              - Các bước chuẩn hóa số học (`StandardScaler`) và mã hóa biến phân loại (`OneHotEncoder`) được đóng gói hoàn toàn bên trong đối tượng `Pipeline` của Scikit-Learn.
              - Tham số thống kê (giá trị trung bình, độ lệch chuẩn) chỉ được học (`fit`) trên tập huấn luyện (24.000 mẫu hoặc 16.000 mẫu ở từng fold CV), tuyệt đối không tiếp cận tập kiểm thử (6.000 mẫu) trước khi suy luận (`transform`).
            - **Rủi ro rò rỉ mục tiêu (Target Leakage qua Proxy Features lâm sàng &mdash; Cần nhìn nhận thận trọng):**
              - Ở **Kịch bản 1 (Clinical)**, các mô hình đạt hiệu năng rất cao (Accuracy ~96–97%, ROC-AUC ~98%). Về mặt bản chất y sinh, hai chỉ số sinh hóa `HbA1c_level` (&ge; 6.5%) và `blood_glucose_level` (&ge; 126 mg/dL) chính là các tiêu chuẩn vàng dùng để chẩn đoán xác định bệnh đái tháo đường theo hướng dẫn của Hiệp hội Đái tháo đường Hoa Kỳ (ADA).
              - Việc đưa các biến chẩn đoán tương đương định nghĩa bệnh vào mô hình phân loại hoạt động tương tự như biến đại diện trực tiếp của nhãn (diagnostic proxy), dẫn đến nguy cơ **target leakage** trong bài toán sàng lọc sớm. Nhóm nghiên cứu ghi nhận đây là kết quả của việc mô hình học theo tiêu chuẩn chẩn đoán, không khẳng định tuyệt đối là dataset hoàn toàn miễn nhiễm rò rỉ mục tiêu.
              - Ngược lại, ở **Kịch bản 2 (CDC BRFSS)**, 21 chỉ số hoàn toàn là khảo sát dịch tễ, hành vi và nhân khẩu học (không chứa xét nghiệm sinh hóa máu), giúp phản ánh khách quan hơn năng lực sàng lọc cộng đồng thực tế (Accuracy ~75%, ROC-AUC ~83%).
            """
        )
