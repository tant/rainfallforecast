# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình LSTM - Phiên bản cuối cùng.
Tích hợp: Log Transform, hàm kích hoạt ReLU và giữ nguyên hệ thống báo cáo chi tiết.
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

print("5 dòng dữ liệu đầu tiên sau khi xử lý:")
print(df.head())

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
    # *** SỬA LỖI: Áp dụng hàm kích hoạt 'relu' ***
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
# BƯỚC 8: LƯU TRỮ, TRỰC QUAN HÓA VÀ BÁO CÁO (KHẮC PHỤC)
# ==============================================================================
print("\nBắt đầu Bước 8: Lưu trữ, trực quan hóa và báo cáo...")

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_info = f"ep{len(history.history['loss'])}_bs32_lb{LOOKBACK}"

# Lưu lại dữ liệu train/val loss/mae theo từng epoch
history_file = f"history_{now_str}_{train_info}.csv"
with open(history_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    header = ["epoch", "loss", "mae", "val_loss", "val_mae"]
    writer.writerow(header)
    for i in range(len(history.history['loss'])):
        row = [i+1, history.history['loss'][i], history.history['mae'][i], history.history['val_loss'][i], history.history['val_mae'][i]]
        writer.writerow(row)
print(f"Đã lưu lịch sử training: {history_file}")

# Report 1: Biểu đồ Training Loss và Validation Loss
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Báo cáo 1: Diễn biến Training & Validation Loss', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (Mean Squared Error)', fontsize=12)
ax1.legend()
loss_fig_name = f"loss_curve_{now_str}_{train_info}.png"
plt.savefig(loss_fig_name)
print(f"Đã lưu biểu đồ loss: {loss_fig_name}")
plt.close(fig1)

# Report 2: So sánh giá trị Thực tế và Dự đoán trên tập Test
test_predictions_scaled = model.predict(X_test)
dummy_array = np.zeros(shape=(len(test_predictions_scaled), n_features))
dummy_array[:, TARGET_COL_INDEX] = test_predictions_scaled.flatten()
test_predictions_log = scaler.inverse_transform(dummy_array)[:, TARGET_COL_INDEX]

dummy_array_actual = np.zeros(shape=(len(y_test), n_features))
dummy_array_actual[:, TARGET_COL_INDEX] = y_test.flatten()
test_actual_log = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COL_INDEX]

test_predictions = np.expm1(test_predictions_log)
test_actual = np.expm1(test_actual_log)
# Dòng lệnh sau không còn cần thiết vì 'relu' đã đảm bảo kết quả không âm.
# test_predictions[test_predictions < 0] = 0

test_dates = test_df.index[LOOKBACK:]

fig2, ax2 = plt.subplots(figsize=(15, 7))
ax2.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
ax2.plot(test_dates, test_predictions, label='Lượng mưa dự đoán', color='red', alpha=0.7, linestyle='--')
ax2.set_title('Báo cáo 2: So sánh kết quả Thực tế và Dự đoán trên tập Test (2025)', fontsize=16)
ax2.set_xlabel('Ngày', fontsize=12)
ax2.set_ylabel('Lượng mưa', fontsize=12)
ax2.legend()
ax2.grid(True)
compare_fig_name = f"compare_test_{now_str}_{train_info}.png"
plt.savefig(compare_fig_name)
print(f"Đã lưu biểu đồ so sánh: {compare_fig_name}")
plt.close(fig2)

# Tính toán các chỉ số
real_scale_mae = mean_absolute_error(test_actual, test_predictions)
print(f"Mean Absolute Error trên tập Test (thang đo gốc): {real_scale_mae:.6f}")

best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best_val_loss = np.min(history.history['val_loss'])
best_val_mae = history.history['val_mae'][best_epoch - 1]
final_train_loss = history.history['loss'][-1]
final_train_mae = history.history['mae'][-1]

# Tổng hợp thông tin báo cáo
report_filename = f"training_report_{now_str}_{train_info}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH LSTM =====\n")
    f.write(f"Thời gian chạy: {now_str}\n\n")
    f.write(f"--- THÔNG SỐ THỬ NGHIỆM ---\n")
    f.write(f"Data file: {file_path}\n")
    f.write(f"Model: {model_architecture}\n")
    f.write(f"Lookback: {LOOKBACK}, Batch size: 32, Epochs tối đa: 100, Patience: 10\n")
    f.write(f"Optimizer: {optimizer_info}, Learning rate: {learning_rate}\n")
    f.write(f"Loss: mean_squared_error, Metrics: mae\n")
    f.write(f"Features: {', '.join(features)}\n")
    f.write(f"Target: {target_col}\n\n")
    f.write(f"--- KÍCH THƯỚC DỮ LIỆU ---\n")
    f.write(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}\n\n")
    f.write(f"--- KẾT QUẢ HUẤN LUYỆN ---\n")
    f.write(f"Total training time: {train_time:.1f} seconds\n")
    f.write(f"Training dừng ở epoch: {len(history.history['loss'])}\n")
    f.write(f"Best epoch (val_loss thấp nhất): {best_epoch}\n")
    f.write(f"Best val_loss: {best_val_loss:.6f}\n")
    f.write(f"Best val_mae (tại epoch tốt nhất): {best_val_mae:.6f}\n\n")
    f.write(f"--- KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG ---\n")
    f.write(f"Test MSE (trên dữ liệu log-scaled): {test_loss:.6f}\n")
    f.write(f"Test MAE (trên dữ liệu log-scaled): {test_mae_scaled:.6f}\n")
    f.write(f"**Test MAE (trên THANG ĐO GỐC): {real_scale_mae:.6f}**\n\n")
    f.write(f"--- CÁC FILE ĐÃ LƯU ---\n")
    f.write(f"Biểu đồ loss: {loss_fig_name}\n")
    f.write(f"Biểu đồ so sánh thực tế/dự đoán: {compare_fig_name}\n")
    f.write(f"Lịch sử loss/mae từng epoch: {history_file}\n")
    f.write("="*40 + "\n")

print(f"Đã lưu báo cáo huấn luyện chi tiết: {report_filename}")

print("\n--- QUÁ TRÌNH HOÀN TẤT ---")