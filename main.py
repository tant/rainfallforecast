# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình LSTM - Phiên bản đọc file CSV và vẽ biểu đồ.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# ==============================================================================
# BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==============================================================================
print("Bắt đầu Bước 1: Tải và chuẩn bị dữ liệu...")

# --- PHẦN BẠN CẦN THAY THẾ ---
# Hãy thay thế 'your_data_file.csv' bằng tên file hoặc đường dẫn đầy đủ đến file của bạn.
file_path = 'weather.csv'
# --- KẾT THÚC PHẦN THAY THẾ ---

try:
    df = pd.read_csv(file_path)
    print(f"Đọc thành công file: '{file_path}'")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng kiểm tra lại tên file.")
    # Thoát chương trình nếu không tìm thấy file
    exit()

# Xử lý cột date và time
try:
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %H:%M', errors='coerce')
    df.set_index('datetime', inplace=True)
    df.drop(['date', 'time'], axis=1, inplace=True)
    df.rename(columns={'temp': 'Temperature', 'humidity': 'Humidity', 'pressure': 'Pressure', 'rainfall': 'Rainfall'}, inplace=True)
except Exception as e:
    print(f"Lỗi khi xử lý các cột ngày giờ và tên cột. Vui lòng kiểm tra format file. Lỗi: {e}")
    exit()

print("5 dòng dữ liệu đầu tiên sau khi xử lý:")
print(df.head())


# ==============================================================================
# CÁC BƯỚC 2, 3, 4, 5, 6, 7 (Giữ nguyên như trước)
# ==============================================================================

# Bước 2: Phân chia dữ liệu
print("\nBắt đầu Bước 2: Phân chia dữ liệu...")
train_df = df[df.index.year <= 2023]
val_df = df[df.index.year == 2024]
test_df = df[df.index.year == 2025]

# Bước 3: Chuẩn hóa dữ liệu
print("\nBắt đầu Bước 3: Chuẩn hóa dữ liệu...")
features = ['Temperature', 'Humidity', 'Pressure']
target_col = 'Rainfall'
scaler = MinMaxScaler()
scaler.fit(train_df[features])
train_scaled = scaler.transform(train_df[features])
val_scaled = scaler.transform(val_df[features])
test_scaled = scaler.transform(test_df[features])

# Bước 4: Tạo dữ liệu chuỗi (Sequences)
print("\nBắt đầu Bước 4: Tạo dữ liệu chuỗi (sequences)...")
LOOKBACK = 8
TARGET_COL_INDEX = features.index(target_col)
def create_sequences(data, lookback, target_col_index):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback, target_col_index])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_scaled, LOOKBACK, TARGET_COL_INDEX)
X_val, y_val = create_sequences(val_scaled, LOOKBACK, TARGET_COL_INDEX)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, TARGET_COL_INDEX)

# Bước 5: Xây dựng mô hình
print("\nBắt đầu Bước 5: Xây dựng mô hình LSTM...")
n_features = X_train.shape[2]
model = Sequential([
    LSTM(units=64, input_shape=(LOOKBACK, n_features)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Bước 6: Huấn luyện mô hình
print("\nBắt đầu Bước 6: Huấn luyện mô hình...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32, validation_data=(X_val, y_val),
    callbacks=[early_stopping], verbose=1
)

# Bước 7: Đánh giá mô hình
print("\nBắt đầu Bước 7: Đánh giá mô hình trên tập Test...")
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Squared Error trên tập Test (dữ liệu đã scale): {test_loss:.6f}")


# ==============================================================================
# BƯỚC 8: TRỰC QUAN HÓA KẾT QUẢ (REPORT)
# ==============================================================================
print("\nBắt đầu Bước 8: Trực quan hóa kết quả...")

# Report 1: Biểu đồ Training Loss và Validation Loss
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Báo cáo 1: Diễn biến Training & Validation Loss', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (Mean Squared Error)', fontsize=12)
ax1.legend()
plt.show()

# Report 2: So sánh giá trị Thực tế và Dự đoán trên tập Test
# Lấy dự đoán từ mô hình (dữ liệu vẫn đang ở dạng scale 0-1)
test_predictions_scaled = model.predict(X_test)

# Tạo một array rỗng có cùng số cột như lúc scale để thực hiện inverse_transform
dummy_array = np.zeros(shape=(len(test_predictions_scaled), n_features))
# Điền giá trị dự đoán vào đúng cột 'Rainfall'
dummy_array[:, TARGET_COL_INDEX] = test_predictions_scaled.flatten()
# Thực hiện inverse transform để đưa về giá trị gốc
test_predictions = scaler.inverse_transform(dummy_array)[:, TARGET_COL_INDEX]

# Lấy giá trị thực tế (y_test) và inverse_transform tương tự
dummy_array_actual = np.zeros(shape=(len(y_test), n_features))
dummy_array_actual[:, TARGET_COL_INDEX] = y_test.flatten()
test_actual = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COL_INDEX]

# Lấy ra đúng các ngày tháng của tập test để vẽ biểu đồ
test_dates = test_df.index[LOOKBACK:]

# Vẽ biểu đồ
fig2, ax2 = plt.subplots(figsize=(15, 7))
ax2.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
ax2.plot(test_dates, test_predictions, label='Lượng mưa dự đoán', color='red', alpha=0.7, linestyle='--')
ax2.set_title('Báo cáo 2: So sánh kết quả Thực tế và Dự đoán trên tập Test (2025)', fontsize=16)
ax2.set_xlabel('Ngày', fontsize=12)
ax2.set_ylabel('Lượng mưa', fontsize=12)
ax2.legend()
ax2.grid(True)
plt.show()

print("\n--- QUÁ TRÌNH HOÀN TẤT ---")