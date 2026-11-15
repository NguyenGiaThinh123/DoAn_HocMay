# BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY (MACHINE LEARNING)
### BỘ MÔN KHOA HỌC DỮ LIỆU - TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM (HUIT)

---
**TÊN ĐỀ TÀI:**
## NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG DỰA TRÊN CÁC MÔ HÌNH HỌC MÁY VÀ ENSEMBLE LEARNING

* **Giảng viên hướng dẫn:** [Tên Giảng Viên]
* **Sinh viên / Nhóm thực hiện:** [Họ và Tên Sinh Viên]
* **Mã số sinh viên (MSSV):** [Nhập MSSV]
* **Lớp học phần:** [Nhập Mã Lớp]
* **Năm học:** 2026 - 2027

---

## MỤC LỤC
1. [KẾ HOẠCH LÀM VIỆC CỦA NHÓM (CLO3)](#1-kế-hoạch-làm-việc-của-nhóm-clo3)
2. [TÌM HIỂU LÝ THUYẾT VỀ BÀI TOÁN VÀ THUẬT TOÁN (CLO1.1, CLO4)](#2-tìm-hiểu-lý-thuyết-về-bài-toán-và-thuật-toán)
3. [VÍ DỤ MINH HỌA TÍNH TOÁN BẰNG TAY (CLO1, CLO2)](#3-ví-dụ-minh-họa-tính-toán-bằng-tay)
4. [BỘ DỮ LIỆU, TRỰC QUAN HÓA VÀ TIỀN XỬ LÝ (CLO1.1, CLO1.2)](#4-bộ-dữ-liệu-trực-quan-hóa-và-tiền-xử-lý)
5. [CÀI ĐẶT THỰC NGHIỆM, BASELINE VÀ ĐÁNH GIÁ MÔ HÌNH](#5-cài-đặt-thực-nghiệm-baseline-và-đánh-giá-mô-hình)
6. [XÂY DỰNG ỨNG DỤNG MINH HỌA (DEMO WEB APP)](#6-xây-dựng-ứng-dụng-minh-họa)
7. [BỘ CÂU HỎI VẤN ĐÁP VÀ TRẢ LỜI BẢO VỆ ĐỒ ÁN](#7-bộ-câu-hỏi-vấn-đáp-và-trả-lời)

---

## 1. KẾ HOẠCH LÀM VIỆC CỦA NHÓM (CLO3)

Nhằm đảm bảo tiến độ thực hiện đề tài trong suốt thời gian học tập, nhóm đã xây dựng bảng phân công nhiệm vụ và lịch trình làm việc chi tiết:

| Tuần | Nội dung công việc | Chuẩn đầu ra (CLO) | Người phụ trách | Sản phẩm bàn giao | Tiến độ |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **Tuần 1** | Tìm kiếm đề tài, lựa chọn 2 bộ dữ liệu trên Kaggle (Lâm sàng & Khảo sát CDC). | CLO1.1 | Cả nhóm | 2 file dữ liệu thô (.csv) | 100% |
| **Tuần 2** | Tìm hiểu cơ sở lý thuyết 8 thuật toán học máy và chuẩn bị ví dụ tính toán minh họa bằng tay. | CLO1.1, CLO2 | Thành viên A | Bản soạn lý thuyết toán học & ví dụ | 100% |
| **Tuần 3** | Khám phá dữ liệu (EDA), vẽ biểu đồ phân phối, tiền xử lý dữ liệu (Encoding, Scaling, Imbalance). | CLO1.1, CLO1.2 | Thành viên B | Script EDA & biểu đồ trực quan hóa | 100% |
| **Tuần 4** | Cài đặt thực nghiệm mô hình Baseline (Logistic Regression) và 7 mô hình nâng cao, đánh giá metrics. | CLO1.1, CLO1.2 | Cả nhóm | Bảng so sánh kết quả 5 metrics thực nghiệm | 100% |
| **Tuần 5** | Tinh chỉnh siêu tham số, lưu trữ mô hình tối ưu (.joblib) và xây dựng Web Demo (Streamlit). | CLO1.1, CLO1.2 | Thành viên A | File `app.py`, models `.joblib`, giao diện web | 100% |
| **Tuần 6** | Viết báo cáo đồ án, thiết kế slide thuyết trình và luyện tập trả lời câu hỏi vấn đáp. | CLO1, CLO3 | Cả nhóm | Báo cáo hoàn chỉnh, Slide trình chiếu | 100% |

---

## 2. TÌM HIỂU LÝ THUYẾT VỀ BÀI TOÁN VÀ THUẬT TOÁN

### 2.1. Đặt vấn đề bài toán
Bệnh đái tháo đường (tiểu đường) là một trong những bệnh mãn tính phổ biến và nguy hiểm nhất thế giới, dẫn đến biến chứng tim mạch, mù lòa, suy thận. Việc phát hiện sớm bệnh có ý nghĩa sống còn. Trong đề tài này, bài toán được mô hình hóa dưới dạng **Phân lớp nhị phân (Binary Classification)**:
$$y \in \{0, 1\}$$
Trong đó $y = 1$ đại diện cho người mắc bệnh tiểu đường, $y = 0$ là người không mắc bệnh.

### 2.2. Lý thuyết chi tiết 8 thuật toán học máy thực nghiệm

#### 1. Logistic Regression (Hồi quy Logistic - Baseline Model)
* **Nguyên lý:** Sử dụng một hàm tuyến tính kết hợp với hàm phi tuyến Sigmoid để ánh xạ đầu ra về khoảng xác suất $[0, 1]$:
  $$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{với } z = w^T X + b$$
* **Hàm mất mát (Binary Cross-Entropy Loss):**
  $$L(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
* **Ưu/Nhược điểm:** Tốc độ tính toán tức thì, dễ diễn giải, nhưng khó học được các ranh giới phi tuyến phức tạp.

#### 2. Decision Tree (Cây quyết định)
* **Nguyên lý:** Phân chia không gian dữ liệu bằng các câu hỏi điều kiện `If-Else`. Tiêu chí chọn biến phân nhánh dựa trên độ thuần khiết (Purity) thông qua chỉ số **Gini Impurity** hoặc **Entropy**:
  $$\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$$
* **Ưu/Nhược điểm:** Dễ giải thích cho bác sĩ; tuy nhiên rất dễ bị quá khớp (overfitting) nếu không giới hạn độ sâu (`max_depth`).

#### 3. K-Nearest Neighbors (KNN)
* **Nguyên lý:** Là thuật toán học lười (Lazy learning), dự đoán nhãn của một điểm dữ liệu mới dựa trên đa số phiếu bầu của $K$ điểm láng giềng gần nhất trong không gian đặc trưng:
  $$d_{\text{Euclidean}}(x_a, x_b) = \sqrt{\sum_{j=1}^m (x_{aj} - x_{bj})^2}$$
* **Ưu/Nhược điểm:** Không cần giả định phân phối dữ liệu; nhưng tốc độ dự đoán chậm trên tập dữ liệu lớn và cực kỳ nhạy cảm với thang đo của biến (cần chuẩn hóa dữ liệu).

#### 4. Random Forest (Rừng ngẫu nhiên)
* **Nguyên lý:** Thuộc họ **Bagging (Bootstrap Aggregating)**. Xây dựng một tập hợp (Ensemble) gồm nhiều cây quyết định độc lập. Mỗi cây được huấn luyện trên một tập mẫu con ngẫu nhiên (có hoàn lại) và chỉ chọn ngẫu nhiên một tập con các đặc trưng khi phân nhánh. Kết quả cuối cùng là bầu chọn theo số đông:
  $$\hat{y} = \text{mode}\left(\{T_1(x), T_2(x), ..., T_B(x)\}\right)$$
* **Ưu điểm:** Giảm phương sai (variance) đáng kể so với cây đơn lẻ, hạn chế tối đa overfit.

#### 5. Support Vector Machine (SVM)
* **Nguyên lý:** Tìm kiếm một siêu phẳng (Hyperplane) phân tách hai lớp dữ liệu sao cho khoảng cách biên (Margin) giữa hai lớp là lớn nhất:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.a.} \quad y_i (w^T x_i + b) \ge 1 - \xi_i$$
* **Ưu điểm:** Tối ưu hóa toàn cục, hiệu quả trên không gian nhiều chiều. Trong đề tài sử dụng `LinearSVC` với `CalibratedClassifierCV` để tối ưu thời gian huấn luyện và tính xác suất.

#### 6. XGBoost (eXtreme Gradient Boosting)
* **Nguyên lý:** Thuộc họ **Boosting**. Xây dựng tuần tự các cây quyết định, trong đó mỗi cây mới được sinh ra để khắc phục sai số (residual) của các cây trước đó. XGBoost bổ sung thêm số hạng hiệu chỉnh (Regularization) vào hàm mục tiêu:
  $$\mathcal{L} = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \left( \gamma T_k + \frac{1}{2}\lambda \|w_k\|^2 \right)$$
* **Ưu điểm:** Khả năng xử lý dữ liệu dạng bảng (Tabular) vượt trội, hỗ trợ xử lý song song và tránh overfit rất tốt.

#### 7. LightGBM (Light Gradient Boosting Machine)
* **Nguyên lý:** Phát triển bởi Microsoft, sử dụng kỹ thuật phân nhánh theo lá (**Leaf-wise tree growth**) thay vì theo tầng (Level-wise) như XGBoost, kết hợp với các thuật toán GOSS (Gradient-based One-Side Sampling) và EFB (Exclusive Feature Bundling).
* **Ưu điểm:** Tốc độ huấn luyện nhanh gấp nhiều lần, tiêu tốn ít bộ nhớ RAM, rất phù hợp với dữ liệu hàng trăm ngàn dòng.

#### 8. Stacking Ensemble Classifier
* **Nguyên lý:** Kỹ thuật học siêu cấp (Meta-learning). Kết hợp dự đoán của nhiều mô hình cơ sở (Base Models) có kiến trúc khác nhau (ví dụ: Random Forest, LightGBM, Logistic Regression). Sau đó, một mô hình cấp cao (Meta-Classifier - thường là Logistic Regression) sẽ học cách kết hợp các xác suất đầu ra này để đưa ra quyết định cuối cùng.
* **Ưu điểm:** Tận dụng được thế mạnh riêng của từng thuật toán thành phần.

---

## 3. VÍ DỤ MINH HỌA TÍNH TOÁN BẰNG TAY (CLO1, CLO2)

> **Mục tiêu:** Chứng minh năng lực hiểu sâu bản chất toán học của mô hình thông qua tính toán số học từng bước.

### 3.1. Ví dụ 1: Tính toán dự đoán bằng Logistic Regression
Giả sử ta đã huấn luyện xong mô hình Logistic Regression với 2 đặc trưng chuẩn hóa:
* $x_1$: Chỉ số khối cơ thể (BMI đã chuẩn hóa)
* $x_2$: Đường huyết lúc đói (Glucose đã chuẩn hóa)

Hệ số trọng số học được là:
$$w_1 = 0.85, \quad w_2 = 1.45, \quad \text{hệ số chặn (bias) } b = -0.50$$

Xét một bệnh nhân mới có giá trị đặc trưng sau chuẩn hóa là $x = [x_1 = 1.2, \quad x_2 = 1.8]$.
1. **Bước 1: Tính tổng tuyến tính $z$:**
   $$z = w_1 x_1 + w_2 x_2 + b = (0.85 \times 1.2) + (1.45 \times 1.8) - 0.50$$
   $$z = 1.02 + 2.61 - 0.50 = 3.13$$
2. **Bước 2: Áp dụng hàm kích hoạt Sigmoid để tính xác suất:**
   $$P(y=1|X) = \sigma(3.13) = \frac{1}{1 + e^{-3.13}} = \frac{1}{1 + 0.0437} = \frac{1}{1.0437} \approx 0.9581 \quad (95.81\%)$$
3. **Bước 3: Đưa ra quyết định phân lớp:**
   Với ngưỡng quyết định chuẩn $\tau = 0.5$, vì $P = 0.9581 > 0.5 \implies$ **Kết luận: Bệnh nhân có nguy cơ cao mắc bệnh tiểu đường (Nhãn 1).**

---

### 3.2. Ví dụ 2: Tính toán phân nhánh trong Cây quyết định (Decision Tree bằng Gini)
Xét một tập dữ liệu mẫu nhỏ gồm 10 bệnh nhân:
* Tổng số mẫu $N = 10$, gồm: 4 người Mắc bệnh ($y=1$) và 6 người Không mắc bệnh ($y=0$).

1. **Tính độ thuần khiết ban đầu (Gini của nút gốc):**
   $$p_1 = \frac{4}{10} = 0.4, \quad p_0 = \frac{6}{10} = 0.6$$
   $$\text{Gini}(D_{\text{root}}) = 1 - (p_1^2 + p_0^2) = 1 - (0.4^2 + 0.6^2) = 1 - (0.16 + 0.36) = 0.48$$

2. **Giả sử ta phân nhánh theo đặc trưng "Đường huyết cao > 126 mg/dL":**
   * **Nhánh Trái (Đường huyết > 126):** Có 4 người, trong đó cả 4 người đều bị bệnh ($y=1$).
     $$\text{Gini}(D_{\text{left}}) = 1 - \left[\left(\frac{4}{4}\right)^2 + \left(\frac{0}{4}\right)^2\right] = 1 - (1 + 0) = 0.0 \quad (\text{Nút thuần khiết tuyệt đối})$$
   * **Nhánh Phải (Đường huyết $\le$ 126):** Có 6 người, trong đó 0 người bị bệnh, 6 người không bị bệnh ($y=0$).
     $$\text{Gini}(D_{\text{right}}) = 1 - \left[\left(\frac{0}{6}\right)^2 + \left(\frac{6}{6}\right)^2\right] = 0.0$$

3. **Tính Gini sau khi phân chia (Weighted Gini):**
   $$\text{Gini}_{\text{split}} = \frac{4}{10} \times 0.0 + \frac{6}{10} \times 0.0 = 0.0$$
4. **Mức giảm độ bất thuần khiết (Gini Gain):**
   $$\Delta \text{Gini} = \text{Gini}(D_{\text{root}}) - \text{Gini}_{\text{split}} = 0.48 - 0.0 = 0.48$$
   $\implies$ Đây là phân nhánh hoàn hảo giúp phân tách 100% hai nhóm bệnh nhân.

---

## 4. BỘ DỮ LIỆU, TRỰC QUAN HÓA VÀ TIỀN XỬ LÝ (CLO1.1, CLO1.2)

### 4.1. Khám phá 2 Bộ dữ liệu
Đề tài so sánh 2 kịch bản dữ liệu hoàn toàn khác biệt:
1. **Bộ dữ liệu 1 (Lâm sàng - Clinical Dataset):** 100.000 bản ghi gồm các chỉ số cận lâm sàng sinh hóa: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `diabetes` (nhãn 0/1).
2. **Bộ dữ liệu 2 (Khảo sát cộng đồng - CDC BRFSS Survey):** Bản chuẩn hóa 70.692 bản ghi (tỷ lệ 50/50) gồm 21 câu hỏi về hành vi thói quen: `HighBP`, `HighChol`, `Smoker`, `Stroke`, `PhysActivity`, `Fruits`, `Veggies`, `GenHlth`, `Income`, `Education`,...

### 4.2. Trực quan hóa dữ liệu (EDA)
Các biểu đồ trực quan hóa được lưu trong thư mục `results/charts/`:
* Biểu đồ phân phối nhãn và phân phối đường huyết: `results/charts/eda_dataset1_distribution.png`
* Ma trận tương quan giữa các biến lâm sàng: `results/charts/eda_dataset1_correlation.png`
* Biểu đồ ảnh hưởng của Lối sống (Huyết áp cao, Sức khỏe tổng quát): `results/charts/eda_dataset2_lifestyle.png`

**Nhận xét từ EDA:**
* Ở dữ liệu lâm sàng, 2 đặc trưng có độ tương quan mạnh nhất với bệnh tiểu đường là `HbA1c_level` ($r \approx 0.40$) và `blood_glucose_level` ($r \approx 0.42$).
* Ở dữ liệu khảo sát thói quen, những người có huyết áp cao (`HighBP=1`) và tự đánh giá sức khỏe kém (`GenHlth \ge 3`) có tỷ lệ mắc tiểu đường cao gấp gần 3 lần so với nhóm còn lại.

### 4.3. Quy trình tiền xử lý dữ liệu
1. **Xử lý giá trị bất thường:** Lọc bỏ các bản ghi không rõ ràng (như `gender == 'Other'`).
2. **Mã hóa đặc trưng (Encoding):**
   * Sử dụng `OneHotEncoder(drop='first')` cho biến định danh `gender` và `smoking_history`.
3. **Chuẩn hóa dữ liệu (Feature Scaling):**
   * Sử dụng `StandardScaler` đưa các biến định lượng về phân phối chuẩn có $\mu = 0, \sigma = 1$. Việc này bắt buộc đối với Logistic Regression, SVM và KNN.
4. **Xử lý mất cân bằng nhãn (Class Imbalance):**
   * Áp dụng tham số `class_weight='balanced'` trong Logistic Regression, Random Forest, Decision Tree và LightGBM để tự động phạt nặng hơn vào lỗi dự đoán sai ca dương tính.

---

## 5. CÀI ĐẶT THỰC NGHIỆM, BASELINE VÀ ĐÁNH GIÁ MÔ HÌNH

### 5.1. Thiết kế thực nghiệm
* Dữ liệu được chia theo tỷ lệ **80% tập Train** và **20% tập Test**, sử dụng kỹ thuật phân tầng `StratifiedShuffleSplit` để giữ nguyên tỷ lệ nhãn.
* **Mô hình cơ sở (Baseline):** Sử dụng **Logistic Regression**.
* **Các thang đo đánh giá:** Accuracy, Precision, Recall (Độ nhạy), F1-Score, ROC-AUC.
  > *Lưu ý trong Y tế:* **Recall** là chỉ số quan trọng nhất vì cần tránh tối đa việc bỏ sót bệnh nhân có bệnh thực sự ($FN$).

### 5.2. Kết quả thực nghiệm Kịch bản 1 (Lâm sàng - Clinical Dataset)

| Mô hình | Accuracy | Precision | Recall (Độ nhạy) | F1-Score | ROC-AUC | Thời gian train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 89.43% | 0.4376 | 85.29% | 0.5785 | 0.9598 | 0.02s |
| **Decision Tree** | 86.08% | 0.3701 | **90.78%** | 0.5258 | 0.9604 | 0.02s |
| **KNN** | 95.90% | 0.9151 | 57.06% | 0.7029 | 0.9026 | 0.03s |
| **Random Forest** | 90.68% | 0.4736 | 86.27% | 0.6115 | 0.9697 | 0.34s |
| **SVM (Linear)** | 95.83% | 0.8736 | 59.61% | 0.7086 | 0.9592 | 0.11s |
| **XGBoost** | **96.92%** | **0.9683** | 65.88% | **0.7841** | **0.9736** | 0.14s |
| **LightGBM** | 91.03% | 0.4850 | 88.63% | 0.6269 | **0.9736** | 0.07s |
| **Stacking Ensemble** | 96.43% | 0.8348 | 72.35% | 0.7752 | 0.9728 | 1.74s |

### 5.3. Kết quả thực nghiệm Kịch bản 2 (Khảo sát thói quen - CDC Survey)

| Mô hình | Accuracy | Precision | Recall (Độ nhạy) | F1-Score | ROC-AUC | Thời gian train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 74.62% | 0.7378 | 76.37% | 0.7505 | 0.8234 | 0.03s |
| **Decision Tree** | 73.80% | 0.7356 | 74.30% | 0.7393 | 0.8111 | 0.04s |
| **KNN** | 71.77% | 0.7078 | 74.13% | 0.7242 | 0.7758 | 0.00s |
| **Random Forest** | 74.88% | 0.7299 | 79.00% | 0.7588 | 0.8288 | 0.31s |
| **SVM (Linear)** | 74.77% | 0.7384 | 76.70% | 0.7525 | 0.8236 | 0.13s |
| **XGBoost** | 74.97% | 0.7299 | 79.27% | 0.7600 | 0.8302 | 0.09s |
| **LightGBM** | 75.03% | 0.7304 | **79.37%** | **0.7607** | 0.8306 | 0.09s |
| **Stacking Ensemble** | **75.17%** | **0.7382** | 78.00% | 0.7585 | **0.8311** | 1.82s |

### 5.4. Đóng gói và lưu trữ mô hình
Các mô hình đạt kết quả tối ưu nhất kèm pipeline tiền xử lý đã được đóng gói bằng thư viện `joblib` để tích hợp trực tiếp vào ứng dụng web:
* `models/clinical_model.joblib`: Mô hình LightGBM cho kịch bản lâm sàng.
* `models/survey_model.joblib`: Mô hình LightGBM cho kịch bản khảo sát.

---

## 6. XÂY DỰNG ỨNG DỤNG MINH HỌA (DEMO WEB APP)

* **Công nghệ phát triển:** Python & Framework **Streamlit**.
* **File khởi động ứng dụng:** `app.py` và tệp thực thi nhanh `run_demo.bat`.
* **Địa chỉ truy cập cục bộ:** `http://localhost:8501`.
* **Tính năng chính:**
  1. Cho phép người dùng chuyển đổi linh hoạt giữa 2 kịch bản ứng dụng (Phòng khám vs Khảo sát cộng đồng).
  2. Giao diện trực quan với thanh trượt và hộp chọn để nhập các chỉ số sức khỏe.
  3. Dự đoán nguy cơ theo thời gian thực (tính xác suất %).
  4. Trả về cảnh báo 3 cấp độ màu sắc kèm lời khuyên y tế chuẩn:
     * 🟢 **Xanh lá (<35%):** Nguy cơ thấp.
     * 🟡 **Vàng (35% - 65%):** Cảnh báo tiền tiểu đường.
     * 🔴 **Đỏ (>65%):** Nguy cơ cao, khuyến nghị xét nghiệm chẩn đoán tại bệnh viện.

---

## 7. BỘ CÂU HỎI VẤN ĐÁP VÀ TRẢ LỜI BẢO VỆ ĐỒ ÁN (GIẢNG VIÊN HAY HỎI)

**Câu hỏi 1: Tại sao nhóm không gộp 2 bộ dữ liệu lại với nhau mà lại phân tách thành 2 kịch bản?**
* *Trả lời:* Vì 2 bộ dữ liệu có schema hoàn toàn khác nhau và đại diện cho hai bối cảnh y tế khác nhau. Dataset 1 là dữ liệu định lượng phòng thí nghiệm (có HbA1c, Glucose máu), phù hợp hỗ trợ bác sĩ chẩn đoán tại phòng khám. Dataset 2 là khảo sát hành vi lối sống của CDC, phù hợp để người dân tự sàng lọc tại nhà không cần lấy máu. Việc so sánh song song giúp đánh giá giá trị chẩn đoán của xét nghiệm sinh hóa so với thói quen sinh hoạt.

**Câu hỏi 2: Tại sao trong bài toán y tế nhóm không chỉ nhìn vào độ chính xác (Accuracy)?**
* *Trả lời:* Vì dữ liệu y tế thường mất cân bằng (số người không bệnh chiếm áp đảo). Một mô hình đoán tất cả là "Không bệnh" vẫn đạt Accuracy 90% nhưng bỏ sót toàn bộ người bệnh, gây nguy hiểm đến tính mạng bệnh nhân. Do đó, chỉ số **Recall (Độ nhạy)** và **ROC-AUC** là thước đo quan trọng hàng đầu để đảm bảo giảm tối đa ca bỏ sót (False Negative).

**Câu hỏi 3: Sự khác biệt cơ bản giữa Random Forest (Bagging) và XGBoost/LightGBM (Boosting) là gì?**
* *Trả lời:* 
  * Random Forest (Bagging) huấn luyện các cây quyết định độc lập, song song và lấy trung bình/bầu cử để giảm phương sai (giảm variance).
  * XGBoost và LightGBM (Boosting) huấn luyện các cây tuần tự, cây sau tập trung học để sửa lỗi của cây trước đó nhằm giảm độ chệch (giảm bias).

**Câu hỏi 4: Nhóm đã làm thế nào để tránh hiện tượng rò rỉ dữ liệu (Data Leakage)?**
* *Trả lời:* Nhóm đã đóng gói toàn bộ bước tiền xử lý (StandardScaler, OneHotEncoder) và mô hình vào trong `sklearn.pipeline.Pipeline`. Bộ chuẩn hóa chỉ học các tham số ($\mu, \sigma$) trên tập Train và chỉ dùng các tham số đó để biến đổi sang tập Test, tuyệt đối không fit trên toàn bộ dữ liệu trước khi chia fold.

---
**HẾT BÁO CÁO**
