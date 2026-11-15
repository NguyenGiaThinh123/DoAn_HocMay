import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Dataset 1 EDA
df1 = pd.read_csv("diabetes_prediction_dataset.csv")
df1 = df1[df1['gender'] != 'Other']

# Biểu đồ phân phối nhãn Dataset 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tỷ lệ nhãn tiểu đường
diabetes_counts = df1['diabetes'].value_counts()
labels = ['Không tiểu đường (0)', 'Tiểu đường (1)']
axes[0].pie(diabetes_counts, labels=labels, autopct='%1.1f%%', colors=['#60A5FA', '#F87171'], explode=(0, 0.1), startangle=140)
axes[0].set_title("Phân phối nhãn bệnh - Dataset 1 (Lâm sàng)", fontsize=13, fontweight='bold')

# Phân phối chỉ số đường huyết theo nhãn bệnh
sns.boxplot(ax=axes[1], data=df1, x='diabetes', y='blood_glucose_level', palette=['#60A5FA', '#F87171'])
axes[1].set_xticklabels(['Không tiểu đường', 'Tiểu đường'])
axes[1].set_xlabel("Tình trạng bệnh", fontsize=11)
axes[1].set_ylabel("Mức đường huyết (blood_glucose_level)", fontsize=11)
axes[1].set_title("So sánh mức đường huyết giữa 2 nhóm", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("results/charts/eda_dataset1_distribution.png", dpi=300)
plt.close()

# Ma trận tương quan Dataset 1
plt.figure(figsize=(10, 8))
numeric_cols1 = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
corr1 = df1[numeric_cols1].corr()
sns.heatmap(corr1, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Ma trận tương quan (Correlation Matrix) - Dataset 1", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/charts/eda_dataset1_correlation.png", dpi=300)
plt.close()

# 2. Dataset 2 EDA
df2 = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Tương quan giữa Huyết áp cao & Tiểu đường
sns.countplot(ax=axes[0], data=df2, x='HighBP', hue='Diabetes_binary', palette=['#34D399', '#F87171'])
axes[0].set_title("Tỷ lệ Huyết áp cao vs Bệnh tiểu đường", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Huyết áp cao (0: Không, 1: Có)")
axes[0].set_ylabel("Số lượng mẫu")
axes[0].legend(['Không bệnh', 'Có bệnh'])

# Tương quan giữa Sức khỏe tổng quát (GenHlth) & Tiểu đường
sns.countplot(ax=axes[1], data=df2, x='GenHlth', hue='Diabetes_binary', palette=['#34D399', '#F87171'])
axes[1].set_title("Sức khỏe tổng quát (1: Rất tốt -> 5: Kém) vs Tiểu đường", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Đánh giá sức khỏe tổng quát (GenHlth)")
axes[1].set_ylabel("Số lượng mẫu")
axes[1].legend(['Không bệnh', 'Có bệnh'])

plt.tight_layout()
plt.savefig("results/charts/eda_dataset2_lifestyle.png", dpi=300)
plt.close()

print("Đã tạo xong tất cả biểu đồ trực quan hóa dữ liệu (EDA)!")
