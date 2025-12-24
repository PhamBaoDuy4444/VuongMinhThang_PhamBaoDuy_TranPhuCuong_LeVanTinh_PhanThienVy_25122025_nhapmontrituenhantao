
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# --- Tải các mô hình và vectorizer đã được lưu ---
# Sử dụng đường dẫn tương đối để đảm bảo nó hoạt động
MODELS_PATH = 'models/'
tfidf_vectorizer = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.pkl'))
nb_model = joblib.load(os.path.join(MODELS_PATH, 'custom_naive_bayes.pkl'))
gbm_model = joblib.load(os.path.join(MODELS_PATH, 'simple_gbm.pkl'))



# Tải stop words nếu chưa có
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


# --- Thiết lập giao diện Streamlit ---
st.title("Hệ thống Phát hiện Email Spam")
st.write("Huấn luyện trên bộ dữ liệu Enron sử dụng các mô hình triển khai từ đầu.")

# Chọn mô hình
model_choice = st.selectbox(
    "Chọn thuật toán để dự đoán:",
    ("Simple Gradient Boosting", "Custom Naive Bayes")
)

# Ô nhập liệu cho người dùng
user_input = st.text_area("Nhập nội dung email (bao gồm cả tiêu đề):", height=200)

# Nút để thực hiện dự đoán
if st.button("Kiểm tra Email"):
    if user_input:
        # 1. Tiền xử lý dữ liệu đầu vào
        cleaned_input = preprocess_text(user_input)
        
        # 2. Vector hóa văn bản bằng TfidfVectorizer đã tải
        input_vector = tfidf_vectorizer.transform([cleaned_input]).toarray()
        
        # 3. Chọn mô hình và dự đoán
        if model_choice == "Custom Naive Bayes":
            prediction = nb_model.predict(input_vector)
            st.write("Đang sử dụng mô hình Naive Bayes...")
        else:
            prediction = gbm_model.predict(input_vector)
            st.write("Đang sử dụng mô hình Gradient Boosting...")
            
        # 4. Hiển thị kết quả
        st.subheader("Kết quả:")
        if prediction[0] == 1:
            st.error("Email này được phân loại là SPAM.")
        else:
            st.success("Email này được phân loại là HAM (Không phải spam).")
    else:
        st.warning("Vui lòng nhập nội dung email để kiểm tra.")