# File: email-spam-project/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến file dữ liệu
DATA_FILE_PATH = 'data/enron_spam_data.csv'

print(f"Đang đọc dữ liệu từ: {DATA_FILE_PATH}")

try:
    # Đọc dữ liệu
    df = pd.read_csv(DATA_FILE_PATH, encoding='latin1')

    # --- Bắt đầu Khám phá Dữ liệu (EDA) ---
    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head())

    print("\nThông tin tổng quan về dữ liệu:")
    df.info()

    # Kiểm tra các giá trị thiếu
    print("\nSố lượng giá trị thiếu trong mỗi cột:")
    print(df.isnull().sum())

    # Đổi tên cột để dễ làm việc hơn
    df.rename(columns={'Spam/Ham': 'Category', 'Message': 'Body'}, inplace=True)

    # Thống kê số lượng email spam và ham
    print("\nPhân phối email Spam và Ham:")
    print(df['Category'].value_counts())

    # Trực quan hóa tỷ lệ spam/ham
    print("\nĐang vẽ biểu đồ phân phối...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Category', data=df)
    plt.title('Phân phối số lượng Email Spam và Ham trong bộ dữ liệu Enron')
    plt.xlabel('Loại Email')
    plt.ylabel('Số lượng')
    plt.show()

    print("\nKhám phá dữ liệu hoàn tất!")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file dữ liệu tại đường dẫn '{DATA_FILE_PATH}'.")
    print("Hãy đảm bảo rằng file 'enron_spam_data.csv' nằm trong thư mục 'data'.")