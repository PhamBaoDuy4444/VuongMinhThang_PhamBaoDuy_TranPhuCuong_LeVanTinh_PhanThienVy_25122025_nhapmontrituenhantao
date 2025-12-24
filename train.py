# File: email-spam-project/train.py (Phiên bản Hoàn chỉnh)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# --- Thư viện để vẽ biểu đồ ---
import seaborn as sns
import matplotlib.pyplot as plt

# --- Import các mô hình "bằng tay" từ thư mục src ---
from src.custom_models import CustomMultinomialNB, SimpleGradientBoostingClassifier

# --- Tải tài nguyên NLTK ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Định nghĩa các đường dẫn ---
DATA_PATH = 'data/enron_spam_data.csv'
MODELS_PATH = 'models/'
OUTPUTS_PATH = 'outputs/' # Thư mục mới để lưu các kết quả trực quan

# Đảm bảo các thư mục đầu ra tồn tại
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.exists(OUTPUTS_PATH):
    os.makedirs(OUTPUTS_PATH)

# --- Hàm tiền xử lý văn bản ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# --- BẮT ĐẦU QUY TRÌNH CHÍNH ---
print("Bắt đầu quy trình huấn luyện...")

# 1. Tải dữ liệu
print(f"Đang tải dữ liệu từ {DATA_PATH}...")
columns_to_use = ['Subject', 'Message', 'Spam/Ham']
df = pd.read_csv(DATA_PATH, encoding='latin1', usecols=columns_to_use)
df.rename(columns={'Spam/Ham': 'Category', 'Message': 'Body'}, inplace=True)

# 2. Tiền xử lý dữ liệu
print("Đang tiền xử lý dữ liệu...")
df['Subject'] = df['Subject'].fillna('')
df['Body'] = df['Body'].fillna('')
df['Full_Message'] = df['Subject'] + ' ' + df['Body']
df['Cleaned_Message'] = df['Full_Message'].apply(preprocess_text)
#  in ra 5 dòng đầu sau khi tiền xử lý
print("\n--- 5 dòng đầu của dữ liệu sau khi tiền xử lý ---")
# Chọn các cột quan trọng để hiển thị cho gọn gàng
columns_to_show = ['Category', 'Full_Message', 'Cleaned_Message']
print(df[columns_to_show].head())
# 3. Trích xuất đặc trưng TF-IDF
print("Đang trích xuất đặc trưng TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Cleaned_Message'])
y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_dense = X.toarray()
y_array = y.to_numpy()

# 4. Chia dữ liệu Train/Test
print("Đang chia dữ liệu thành tập Train và Test (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_dense, y_array, test_size=0.2, random_state=42, stratify=y_array
)

# 5. Huấn luyện, Đánh giá và Trực quan hóa Custom Naive Bayes
print("\n--- Huấn luyện Custom Naive Bayes ---")
custom_nb = CustomMultinomialNB(alpha=1.0)
custom_nb.fit(X_train, y_train)
y_pred_nb = custom_nb.predict(X_test)
print("--- Kết quả đánh giá Custom Naive Bayes ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(classification_report(y_test, y_pred_nb, target_names=['Ham', 'Spam']))

# Tạo và lưu Ma trận nhầm lẫn cho Naive Bayes
print("Đang tạo Ma trận nhầm lẫn cho Naive Bayes...")
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Ma trận nhầm lẫn - Custom Naive Bayes')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
nb_chart_path = os.path.join(OUTPUTS_PATH, 'confusion_matrix_nb.png')
plt.savefig(nb_chart_path)
print(f"Đã lưu ma trận nhầm lẫn tại: {nb_chart_path}")
plt.close() # Đóng biểu đồ để giải phóng bộ nhớ

# 6. Huấn luyện, Đánh giá và Trực quan hóa Simple Gradient Boosting
print("\n--- Huấn luyện Simple Gradient Boosting ---")
simple_gbm = SimpleGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
simple_gbm.fit(X_train, y_train)
y_pred_gbm = simple_gbm.predict(X_test)
print("--- Kết quả đánh giá Simple Gradient Boosting ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gbm):.4f}")
print(classification_report(y_test, y_pred_gbm, target_names=['Ham', 'Spam']))

# Tạo và lưu Ma trận nhầm lẫn cho Simple GBM
print("Đang tạo Ma trận nhầm lẫn cho Simple GBM...")
cm_gbm = confusion_matrix(y_test, y_pred_gbm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gbm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Ma trận nhầm lẫn - Simple GBM')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
gbm_chart_path = os.path.join(OUTPUTS_PATH, 'confusion_matrix_gbm.png')
plt.savefig(gbm_chart_path)
print(f"Đã lưu ma trận nhầm lẫn tại: {gbm_chart_path}")
plt.close() # Đóng biểu đồ để giải phóng bộ nhớ

# 7. Lưu các mô hình và vectorizer
print("\nĐang lưu các mô hình và TfidfVectorizer...")
joblib.dump(tfidf_vectorizer, os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'))
joblib.dump(custom_nb, os.path.join(MODELS_PATH, 'custom_naive_bayes.pkl'))
joblib.dump(simple_gbm, os.path.join(MODELS_PATH, 'simple_gbm.pkl'))

print(f"\nQuy trình huấn luyện hoàn tất. Các mô hình đã được lưu tại '{MODELS_PATH}'.")
print(f"Các biểu đồ đã được lưu tại '{OUTPUTS_PATH}'.")