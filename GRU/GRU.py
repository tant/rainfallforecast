# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình GRU - Phiên bản dự đoán lượng mưa.
Tích hợp: Log Transform cho dữ liệu lệch và hàm kích hoạt ReLU cho đầu ra không âm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
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
LOOKBACK = 24
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
# BƯỚC 5: XÂY DỰNG MÔ HÌNH (Kiến trúc 2 lớp GRU)
# ==============================================================================
print("\nBắt đầu Bước 5: Xây dựng mô hình GRU...")
n_features = X_train.shape[2]

model = Sequential([
    # GRU thứ nhất: trả về chuỗi cho lớp sau, có recurrent_dropout
    GRU(units=64, recurrent_dropout=0.25, return_sequences=True, input_shape=(LOOKBACK, n_features)),
    Dropout(0.3),
    # GRU thứ hai: chỉ trả về kết quả cuối cùng
    GRU(units=32, recurrent_dropout=0.25),
    Dropout(0.3),
    Dense(16, activation='relu'),  # Thêm lớp Dense trung gian
    Dense(1, activation='softplus')  # Đổi activation từ 'relu' sang
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
model.summary()

# Cập nhật lại thông tin kiến trúc để lưu vào báo cáo
model_architecture = "GRU(64,recurrent_dropout=0.25,return_seq=True) + Dropout(0.3) + GRU(32,recurrent_dropout=0.25) + Dropout(0.3) + Dense(16,relu) + Dense(1,softplus) + LogTransform"
optimizer_info = "Adam"
learning_rate = 0.001

# ==============================================================================
# BƯỚC 6: HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 6: Huấn luyện mô hình...")
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=32, validation_data=(X_val, y_val),
    callbacks=[early_stopping], verbose=1
)
train_time = time.time() - start_time

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_info = f"ep{len(history.history['loss'])}_bs32_lb{LOOKBACK}"

# Lưu lại mô hình đã train với tên chứa thông tin thời gian và cấu hình train
model_filename = f"gru_model_{now_str}_{train_info}.h5"
model.save(model_filename)
print(f"Đã lưu mô hình vào file: {model_filename}")

# ==============================================================================
# BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 7: Đánh giá mô hình trên tập Test...")
test_loss, test_mae_scaled = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Squared Error trên tập Test (dữ liệu log-scaled): {test_loss:.6f}")
print(f"Mean Absolute Error trên tập Test (dữ liệu log-scaled): {test_mae_scaled:.6f}")

# Tính toán dự đoán và giá trị thực tế trên tập test
test_predictions_scaled = model.predict(X_test)
test_predictions_scaled = np.maximum(0, test_predictions_scaled) # Áp dụng max(0, prediction) để xử lý giá trị âm
dummy_array = np.zeros(shape=(len(test_predictions_scaled), n_features))
dummy_array[:, TARGET_COL_INDEX] = test_predictions_scaled.flatten()
test_predictions_log = scaler.inverse_transform(dummy_array)[:, TARGET_COL_INDEX]

dummy_array_actual = np.zeros(shape=(len(y_test), n_features))
dummy_array_actual[:, TARGET_COL_INDEX] = y_test.flatten()
test_actual_log = scaler.inverse_transform(dummy_array_actual)[:, TARGET_COL_INDEX]

test_predictions = np.expm1(test_predictions_log)
test_actual = np.expm1(test_actual_log)
test_dates = test_df.index[LOOKBACK:]

# ==============================================================================
# BƯỚC 8: TRỰC QUAN HÓA VÀ BÁO CÁO
# ==============================================================================
print("\nBắt đầu Bước 8: Trực quan hóa và báo cáo...")

# Vẽ và lưu biểu đồ loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
loss_fig_name = f"loss_curve_{now_str}_{train_info}.png"
plt.savefig(loss_fig_name)
plt.close()

# Vẽ và lưu biểu đồ so sánh thực tế/dự đoán
plt.figure(figsize=(15, 7))
plt.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
plt.plot(test_dates, test_predictions, label='Lượng mưa dự đoán', color='red', alpha=0.7, linestyle='--')
plt.xlabel('Ngày')
plt.ylabel('Lượng mưa')
plt.title('So sánh thực tế và dự đoán trên tập Test')
plt.legend()
compare_fig_name = f"compare_test_{now_str}_{train_info}.png"
plt.savefig(compare_fig_name)
plt.close()

# Lưu lịch sử loss/mae từng epoch
history_file = f"history_{now_str}_{train_info}.csv"
with open(history_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    header = ["epoch", "loss", "mae", "val_loss", "val_mae"]
    writer.writerow(header)
    for i in range(len(history.history['loss'])):
        row = [
            i+1,
            history.history['loss'][i],
            history.history['mae'][i] if 'mae' in history.history else '',
            history.history['val_loss'][i],
            history.history['val_mae'][i] if 'val_mae' in history.history else ''
        ]
        writer.writerow(row)

# Tính toán MAE trên thang đo gốc
real_scale_mae = mean_absolute_error(test_actual, test_predictions)
print(f"Mean Absolute Error trên tập Test (thang đo gốc): {real_scale_mae:.6f}")

# Tính RMSE
rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
print(f"Root Mean Squared Error trên tập Test (thang đo gốc): {rmse:.6f}")

# Tính RRSE
mean_actual = np.mean(test_actual)
rrse_numerator = np.sum((test_actual - test_predictions) ** 2)
rrse_denominator = np.sum((test_actual - mean_actual) ** 2)
rrse = np.sqrt(rrse_numerator / rrse_denominator) if rrse_denominator != 0 else np.nan
print(f"Root Relative Squared Error trên tập Test (thang đo gốc): {rrse:.6f}")

# Lưu file báo cáo
report_filename = f"training_report_{now_str}_{train_info}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    # 1. Thông tin tổng quan
    f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH GRU =====\n")
    f.write(f"Thời gian chạy: {now_str}\n")
    f.write(f"Lookback: {LOOKBACK}\n")
    f.write(f"Batch size: 32\n")
    f.write(f"Epochs: {len(history.history['loss'])}\n")
    f.write(f"Patience: 20\n")
    f.write(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}\n")
    f.write(f"Features: {', '.join(features)}\n")
    f.write(f"Target: {target_col}\n")
    f.write(f"Data file: {file_path}\n")
    f.write(f"Model: {model_architecture}\n")
    f.write(f"Optimizer: {optimizer_info}, Learning rate: {learning_rate}\n")
    f.write(f"Loss: mean_squared_error, Metrics: mae\n")
    f.write(f"Total training time: {train_time:.1f} seconds\n")
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_val_loss = np.min(history.history['val_loss'])
    best_val_mae = np.min(history.history['val_mae']) if 'val_mae' in history.history else None
    final_train_loss = history.history['loss'][-1]
    final_train_mae = history.history['mae'][-1] if 'mae' in history.history else None
    f.write(f"Best epoch (val_loss thấp nhất): {best_epoch}\n")
    f.write(f"Best val_loss: {best_val_loss:.6f}\n")
    if best_val_mae is not None:
        f.write(f"Best val_mae: {best_val_mae:.6f}\n")
    f.write(f"Final train loss: {final_train_loss:.6f}\n")
    if final_train_mae is not None:
        f.write(f"Final train mae: {final_train_mae:.6f}\n")
    f.write(f"Test MSE: {test_loss:.6f}\n")
    f.write(f"Test MAE: {test_mae_scaled:.6f}\n")
    f.write(f"Test MAE (thang gốc): {real_scale_mae:.6f}\n")
    f.write(f"Test RMSE (thang gốc): {rmse:.6f}\n")
    f.write(f"Test RRSE (thang gốc): {rrse:.6f}\n")
    f.write(f"Biểu đồ loss: {loss_fig_name}\n")
    f.write(f"Biểu đồ so sánh thực tế/dự đoán: {compare_fig_name}\n")
    f.write(f"Lịch sử loss/mae từng epoch: {history_file}\n")
    f.write("============================================\n\n")

    # 2. Log training từng epoch
    f.write("===== LOG TRAINING TỪNG EPOCH =====\n")
    for i in range(len(history.history['loss'])):
        epoch_num = i + 1
        loss = history.history['loss'][i]
        mae = history.history['mae'][i] if 'mae' in history.history else 0
        val_loss = history.history['val_loss'][i]
        val_mae = history.history['val_mae'][i] if 'val_mae' in history.history else 0
        f.write(
            f"Epoch {epoch_num}/{len(history.history['loss'])} - "
            f"loss: {loss:.4f} - mae: {mae:.4f} - "
            f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}\n"
        )
    f.write("============================================\n\n")

    # 3. Dữ liệu so sánh thực tế và dự đoán trên tập test
    f.write("===== DỮ LIỆU SO SÁNH THỰC TẾ VÀ DỰ ĐOÁN TRÊN TẬP TEST =====\n")
    f.write("Ngày,Thực tế,Dự đoán\n")
    for date, actual, pred in zip(test_dates, test_actual, test_predictions):
        f.write(f"{date.strftime('%Y-%m-%d %H:%M')},{actual:.4f},{pred:.4f}\n")
    f.write("============================================\n")

print(f"Đã lưu báo cáo huấn luyện: {report_filename}")
print("\n--- QUÁ TRÌNH HOÀN TẤT ---")