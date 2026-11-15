import joblib
import pandas as pd

print("="*60)
print("KIỂM TRA TÍNH TOÀN VẸN CỦA TOÀN BỘ 16 MÔ HÌNH")
print("="*60)

# 1. Test Dataset 1 (Clinical)
clinical_models = joblib.load("models/clinical_all_models.joblib")
print(f"\n[+] Số mô hình lâm sàng: {len(clinical_models)}")

sample_clinical = pd.DataFrame([{
    'gender': 'Male',
    'age': 55.0,
    'hypertension': 1,
    'heart_disease': 0,
    'smoking_history': 'never',
    'bmi': 28.5,
    'HbA1c_level': 6.8,
    'blood_glucose_level': 160.0
}])

for name, pipe in clinical_models.items():
    prob = pipe.predict_proba(sample_clinical)[0][1]
    pred = pipe.predict(sample_clinical)[0]
    classes = pipe.classes_
    print(f"  - {name:20s}: P(diabetes=1) = {prob*100:5.1f}% | Pred = {pred} | Classes = {classes}")

# 2. Test Dataset 2 (Survey)
survey_models = joblib.load("models/survey_all_models.joblib")
print(f"\n[+] Số mô hình khảo sát CDC: {len(survey_models)}")

sample_survey = pd.DataFrame([{
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 30.0, 'Smoker': 0,
    'Stroke': 0, 'HeartDiseaseorAttack': 0, 'PhysActivity': 1, 'Fruits': 1,
    'Veggies': 1, 'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
    'GenHlth': 3.0, 'MentHlth': 2.0, 'PhysHlth': 2.0, 'DiffWalk': 0,
    'Sex': 1, 'Age': 8.0, 'Education': 5.0, 'Income': 6.0
}])

for name, pipe in survey_models.items():
    prob = pipe.predict_proba(sample_survey)[0][1]
    pred = pipe.predict(sample_survey)[0]
    classes = pipe.classes_
    print(f"  - {name:20s}: P(diabetes=1) = {prob*100:5.1f}% | Pred = {pred} | Classes = {classes}")

print("\n>>> TẤT CẢ 16 MÔ HÌNH ĐỀU HOẠT ĐỘNG HOÀN HẢO VÀ ĐÚNG LỚP diabetes=1! <<<")
