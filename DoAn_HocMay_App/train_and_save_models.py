import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

os.makedirs("models", exist_ok=True)
print("Đang huấn luyện và đóng gói 2 mô hình tối ưu cho Web Demo...")

# -------------------------------------------------------------
# 1. Model Lâm sàng (Clinical Dataset 1)
# -------------------------------------------------------------
df1 = pd.read_csv("diabetes_prediction_dataset.csv")
df1 = df1[df1['gender'] != 'Other']

if len(df1) > 40000:
    df1, _ = train_test_split(df1, train_size=40000, stratify=df1['diabetes'], random_state=42)

y1 = df1['diabetes']
X1 = df1.drop(columns=['diabetes'])

cat_cols1 = ['gender', 'smoking_history']
num_cols1 = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols1),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols1)
    ]
)

# Sử dụng LightGBM - mô hình đạt Recall cao (88.6%) và ROC-AUC 0.9736
pipeline1 = Pipeline(steps=[
    ('preprocessor', preprocessor1),
    ('classifier', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, verbose=-1))
])

pipeline1.fit(X1, y1)
joblib.dump(pipeline1, "models/clinical_model.joblib")
print("[V] Đã lưu model lâm sàng: models/clinical_model.joblib")

# -------------------------------------------------------------
# 2. Model Khảo sát Thói quen (Survey Dataset 2)
# -------------------------------------------------------------
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

if len(df2) > 40000:
    df2, _ = train_test_split(df2, train_size=40000, stratify=df2['Diabetes_binary'], random_state=42)

y2 = df2['Diabetes_binary'].astype(int)
X2 = df2.drop(columns=['Diabetes_binary'])

pipeline2 = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42, verbose=-1))
])

pipeline2.fit(X2, y2)
joblib.dump(pipeline2, "models/survey_model.joblib")
print("[V] Đã lưu model khảo sát: models/survey_model.joblib")
print("Hoàn tất đóng gói mô hình!")
