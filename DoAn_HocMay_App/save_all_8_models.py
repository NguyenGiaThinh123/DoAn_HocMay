import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from run_pipeline import build_model_pipelines

os.makedirs("models", exist_ok=True)
print("Đang huấn luyện và đóng gói TOÀN BỘ 8 MÔ HÌNH cho cả 2 kịch bản (Chỉ fit trên tập Train)...")

# -------------------------------------------------------------
# 1. DATASET 1 (LÂM SÀNG)
# -------------------------------------------------------------
print("[1/2] Huấn luyện 8 mô hình cho Dataset 1 (Lâm sàng)...")
df1 = pd.read_csv("diabetes_prediction_dataset.csv")
df1 = df1[df1['gender'] != 'Other']

if len(df1) > 30000:
    df1, _ = train_test_split(df1, train_size=30000, stratify=df1['diabetes'], random_state=42)

y1 = df1['diabetes'].astype(int)
X1 = df1.drop(columns=['diabetes'])

# Chia train/test trước khi fit bộ tiền xử lý
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

clinical_trained_pipelines = {}
for name, pipe in build_model_pipelines(preprocessor1).items():
    print(f"  + Train {name} trên X1_train ({len(X1_train)} mẫu)...")
    pipe.fit(X1_train, y1_train)
    clinical_trained_pipelines[name] = pipe

joblib.dump(clinical_trained_pipelines, "models/clinical_all_models.joblib")
print("[V] Đã lưu 8 mô hình lâm sàng vào: models/clinical_all_models.joblib")

# -------------------------------------------------------------
# 2. DATASET 2 (KHẢO SÁT CDC)
# -------------------------------------------------------------
print("[2/2] Huấn luyện 8 mô hình cho Dataset 2 (Khảo sát CDC)...")
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

if len(df2) > 30000:
    df2, _ = train_test_split(df2, train_size=30000, stratify=df2['Diabetes_binary'], random_state=42)

y2 = df2['Diabetes_binary'].astype(int)
X2 = df2.drop(columns=['Diabetes_binary'])

# Chia train/test trước khi fit bộ tiền xử lý
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, stratify=y2, random_state=42
)

num_cols2 = list(X2.columns)
preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols2)
    ]
)

survey_trained_pipelines = {}
for name, pipe in build_model_pipelines(preprocessor2).items():
    print(f"  + Train {name} trên X2_train ({len(X2_train)} mẫu)...")
    pipe.fit(X2_train, y2_train)
    survey_trained_pipelines[name] = pipe

joblib.dump(survey_trained_pipelines, "models/survey_all_models.joblib")
print("[V] Đã lưu 8 mô hình khảo sát vào: models/survey_all_models.joblib")
print("\n>>> TẤT CẢ 16 MÔ HÌNH ĐÃ ĐƯỢC HUẤN LUYỆN TRÊN TẬP TRAIN VÀ LƯU THÀNH CÔNG! <<<")
