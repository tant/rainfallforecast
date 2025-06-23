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
import datetime

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
features = ['Temperature', 'Humidity', 'Pressure', 'Rainfall']
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

# print("Các thiết bị GPU khả dụng:", tf.config.list_physical_devices('GPU'))

# Bước 5: Xây dựng mô hình
print("\nBắt đầu Bước 5: Xây dựng mô hình LSTM...")
n_features = X_train.shape[2]
model = Sequential([
    LSTM(units=64, input_shape=(LOOKBACK, n_features)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# Bước 6: Huấn luyện mô hình
print("\nBắt đầu Bước 6: Huấn luyện mô hình...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
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

# Tạo chuỗi thông tin cho tên file: ngày giờ, số epoch, batch size, lookback, v.v.
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_info = f"ep{len(history.history['loss'])}_bs32_lb{LOOKBACK}"

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
test_predictions = scaler.inverse_transform(dummy_array)[:, TARGET_COL_INDEX]
dummy_array_actual = np.zeros(shape=(len(y_test), n_features))
dummy_array_actual[:, TARGET_COL_INDEX] = y_test.flatten()
test_actual = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COL_INDEX]
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

# Tổng hợp thông tin báo cáo
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
best_val_loss = np.min(history.history['val_loss'])
best_val_mae = np.min(history.history['val_mae']) if 'val_mae' in history.history else None
final_train_loss = history.history['loss'][-1]
final_train_mae = history.history['mae'][-1] if 'mae' in history.history else None

test_metrics = model.evaluate(X_test, y_test, verbose=0)
if isinstance(test_metrics, (list, tuple)):
    test_mse, test_mae = test_metrics
else:
    test_mse, test_mae = test_metrics, None

with open(f"training_report_{now_str}_{train_info}.txt", "w", encoding="utf-8") as f:
    f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH LSTM =====\n")
    f.write(f"Thời gian chạy: {now_str}\n")
    f.write(f"Lookback: {LOOKBACK}\nBatch size: 32\nEpochs: {len(history.history['loss'])}\nPatience: 10\n")
    f.write(f"Best epoch (val_loss thấp nhất): {best_epoch}\n")
    f.write(f"Best val_loss: {best_val_loss:.6f}\n")
    if best_val_mae is not None:
        f.write(f"Best val_mae: {best_val_mae:.6f}\n")
    f.write(f"Final train loss: {final_train_loss:.6f}\n")
    if final_train_mae is not None:
        f.write(f"Final train mae: {final_train_mae:.6f}\n")
    f.write(f"Test MSE: {test_mse:.6f}\n")
    if test_mae is not None:
        f.write(f"Test MAE: {test_mae:.6f}\n")
    f.write(f"Biểu đồ loss: {loss_fig_name}\n")
    f.write(f"Biểu đồ so sánh thực tế/dự đoán: {compare_fig_name}\n")
    f.write("============================================\n")

print(f"Đã lưu báo cáo huấn luyện: training_report_{now_str}_{train_info}.txt")

print("\n--- QUÁ TRÌNH HOÀN TẤT ---")