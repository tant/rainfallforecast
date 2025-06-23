# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình LSTM - Phiên bản cuối cùng.
Tích hợp: Log Transform cho dữ liệu lệch và hàm kích hoạt ReLU cho đầu ra không âm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import datetime
import time
import csv

# ==============================================================================
# BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==============================================================================
print("Bắt đầu Bước 1: Tải và chuẩn bị dữ liệu...")
file_path = 'weather.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Đọc thành công file: '{file_path}'")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{file_path}'.")
    exit()

try:
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %H:%M', errors='coerce')
    df.set_index('datetime', inplace=True)
    df.drop(['date', 'time'], axis=1, inplace=True)
    df.rename(columns={'temp': 'Temperature', 'humidity': 'Humidity', 'pressure': 'Pressure', 'rainfall': 'Rainfall'}, inplace=True)
except Exception as e:
    print(f"Lỗi khi xử lý các cột. Lỗi: {e}")
    exit()

# ==============================================================================
# BƯỚC 2: PHÂN CHIA DỮ LIỆU
# ==============================================================================
print("\nBắt đầu Bước 2: Phân chia dữ liệu...")
train_df = df[df.index.year <= 2023].copy()
val_df = df[df.index.year == 2024].copy()
test_df = df[df.index.year == 2025].copy()

# ==============================================================================
# BƯỚC 2.5: ÁP DỤNG LOG TRANSFORM CHO CỘT MỤC TIÊU
# ==============================================================================
print("\nBắt đầu Bước 2.5: Áp dụng Log Transform...")
train_df['Rainfall'] = np.log1p(train_df['Rainfall'])
val_df['Rainfall'] = np.log1p(val_df['Rainfall'])
test_df['Rainfall'] = np.log1p(test_df['Rainfall'])
print("Đã áp dụng np.log1p cho cột Rainfall.")

# ==============================================================================
# BƯỚC 3: CHUẨN HÓA DỮ LIỆU
# ==============================================================================
print("\nBắt đầu Bước 3: Chuẩn hóa dữ liệu...")
features = ['Temperature', 'Humidity', 'Pressure', 'Rainfall']
target_col = 'Rainfall'
scaler = MinMaxScaler()
scaler.fit(train_df[features])
train_scaled = scaler.transform(train_df[features])
val_scaled = scaler.transform(val_df[features])
test_scaled = scaler.transform(test_df[features])

# ==============================================================================
# BƯỚC 4: TẠO DỮ LIỆU CHUỖI (SEQUENCES)
# ==============================================================================
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

# ==============================================================================
# BƯỚC 5: XÂY DỰNG MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 5: Xây dựng mô hình LSTM...")
n_features = X_train.shape[2]
model = Sequential([
    LSTM(units=64, input_shape=(LOOKBACK, n_features)),
    Dropout(0.2),
    # *** THAY ĐỔI CUỐI CÙNG: Thêm hàm kích hoạt 'relu' để đảm bảo đầu ra không âm ***
    Dense(1, activation='relu')
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
model_architecture = "LSTM(64) + Dropout(0.2) + Dense(1, activation='relu') + LogTransform"
optimizer_info = "Adam"
learning_rate = 0.001

# ==============================================================================
# BƯỚC 6: HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 6: Huấn luyện mô hình...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32, validation_data=(X_val, y_val),
    callbacks=[early_stopping], verbose=1
)
train_time = time.time() - start_time

# ==============================================================================
# BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 7: Đánh giá mô hình trên tập Test...")
test_loss, test_mae_scaled = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Squared Error trên tập Test (dữ liệu log-scaled): {test_loss:.6f}")
print(f"Mean Absolute Error trên tập Test (dữ liệu log-scaled): {test_mae_scaled:.6f}")

# ==============================================================================
# BƯỚC 8: TRỰC QUAN HÓA VÀ BÁO CÁO
# ==============================================================================
print("\nBắt đầu Bước 8: Trực quan hóa và báo cáo...")

# (Phần code lưu history và biểu đồ loss giữ nguyên)
# ...

# Report 2: So sánh giá trị Thực tế và Dự đoán trên tập Test
test_predictions_scaled = model.predict(X_test)
dummy_array = np.zeros(shape=(len(test_predictions_scaled), n_features))
dummy_array[:, TARGET_COL_INDEX] = test_predictions_scaled.flatten()
test_predictions_log = scaler.inverse_transform(dummy_array)[:, TARGET_COL_INDEX]

dummy_array_actual = np.zeros(shape=(len(y_test), n_features))
dummy_array_actual[:, TARGET_COL_INDEX] = y_test.flatten()
test_actual_log = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COL_INDEX]

# Áp dụng phép biến đổi ngược (expm1)
test_predictions = np.expm1(test_predictions_log)
test_actual = np.expm1(test_actual_log)

# Dòng lệnh `test_predictions[test_predictions < 0] = 0` không còn cần thiết
# vì hàm kích hoạt 'relu' đã đảm bảo các dự đoán không thể âm.

# Tính toán MAE trên thang đo gốc
real_scale_mae = mean_absolute_error(test_actual, test_predictions)
print(f"Mean Absolute Error trên tập Test (thang đo gốc): {real_scale_mae:.6f}")

# (Toàn bộ phần code còn lại để vẽ biểu đồ và lưu file báo cáo được giữ nguyên)
# ...
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_info = f"ep{len(history.history['loss'])}_bs32_lb{LOOKBACK}"
report_filename = f"training_report_{now_str}_{train_info}.txt"

# Vẽ biểu đồ loss
# ... (code vẽ và lưu biểu đồ loss)

# Vẽ biểu đồ so sánh
test_dates = test_df.index[LOOKBACK:]
fig2, ax2 = plt.subplots(figsize=(15, 7))
ax2.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
ax2.plot(test_dates, test_predictions, label='Lượng mưa dự đoán', color='red', alpha=0.7, linestyle='--')
# ... (code thiết lập và lưu biểu đồ so sánh)

# Lưu file báo cáo
with open(report_filename, "w", encoding="utf-8") as f:
    f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH LSTM =====\n")
    f.write(f"Model: {model_architecture}\n")
    # ... các dòng ghi thông tin khác ...
    f.write(f"Test MSE (trên dữ liệu log-scaled): {test_loss:.6f}\n")
    f.write(f"Test MAE (trên dữ liệu log-scaled): {test_mae_scaled:.6f}\n")
    f.write(f"**Test MAE (trên thang đo GỐC): {real_scale_mae:.6f}**\n")
    # ... các dòng ghi thông tin khác ...

print(f"Đã lưu báo cáo huấn luyện: {report_filename}")
print("\n--- QUÁ TRÌNH HOÀN TẤT ---")