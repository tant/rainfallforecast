# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình XGBoost để dự báo lượng mưa.
Bao gồm:
1. Đọc và xử lý dữ liệu.
2. Tạo đặc trưng dạng bảng (lag features).
3. Huấn luyện XGBoost với Early Stopping.
4. Đánh giá, lưu trữ mô hình và tạo báo cáo trực quan.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import time

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
# BƯỚC 2: PHÂN CHIA & BIẾN ĐỔI DỮ LIỆU
# ==============================================================================
print("\nBắt đầu Bước 2: Phân chia & biến đổi dữ liệu...")
train_df = df[df.index.year <= 2023].copy()
val_df = df[df.index.year == 2024].copy()
test_df = df[df.index.year == 2025].copy()

# Áp dụng Log Transform cho cột mục tiêu
print("Áp dụng Log Transform cho cột Rainfall...")
train_df['Rainfall'] = np.log1p(train_df['Rainfall'])
val_df['Rainfall'] = np.log1p(val_df['Rainfall'])
test_df['Rainfall'] = np.log1p(test_df['Rainfall'])

# ==============================================================================
# BƯỚC 3: TẠO ĐẶC TRƯNG DẠNG BẢNG (LAG FEATURES)
# ==============================================================================
print("\nBắt đầu Bước 3: Tạo đặc trưng dạng bảng (Lag Features)...")
LOOKBACK = 6
features_to_lag = ['Temperature', 'Humidity', 'Pressure', 'Rainfall']
target_col = 'Rainfall'

def create_tabular_features(df, lookback):
    """
    Sử dụng phương pháp shift để tạo các đặc trưng trễ hiệu quả.
    """
    df_lagged = df.copy()
    for i in range(1, lookback + 1):
        for col in features_to_lag:
            df_lagged[f'{col}_t-{i}'] = df_lagged[col].shift(i)
    
    df_lagged.dropna(inplace=True)
    y = df_lagged[target_col]
    X = df_lagged.drop(columns=features_to_lag)
    return X, y

X_train, y_train = create_tabular_features(train_df, LOOKBACK)
X_val, y_val = create_tabular_features(val_df, LOOKBACK)
X_test, y_test = create_tabular_features(test_df, LOOKBACK)
test_dates = X_test.index

print(f"Kích thước tập huấn luyện X: {X_train.shape}, y: {y_train.shape}")
print("Một vài đặc trưng được tạo ra:", list(X_train.columns[:5]))

# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH XGBoost
# ==============================================================================
print("\nBắt đầu Bước 4: Huấn luyện mô hình XGBoost...")

# Cấu hình XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'eta': 0.02,         
    'max_depth': 8,         
    'subsample': 0.9,     
    'colsample_bytree': 0.8, 
    'lambda': 1.2,           
    'alpha': 1.2,            
    'n_estimators': 1000,
    'seed': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 100,
    'validation_indicator_col': 'validation_0'
}

# Khởi tạo mô hình
model = xgb.XGBRegressor(**xgb_params)

# Huấn luyện với Early Stopping
print("Đang huấn luyện...")
start_time = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)
train_time = time.time() - start_time
print(f"Huấn luyện hoàn tất trong {train_time:.1f} giây.")

# Lấy lịch sử đánh giá
evals_result = model.evals_result()

# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 5: Đánh giá mô hình...")
test_predictions_log = model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)
test_actual = np.expm1(y_test)

# Xử lý giá trị âm
test_predictions[test_predictions < 0] = 0

# Tính toán sai số
mse_real = mean_squared_error(test_actual, test_predictions)
mae_real = mean_absolute_error(test_actual, test_predictions)
rmse_real = np.sqrt(mse_real)

# Tính RRSE
mean_actual = np.mean(test_actual)
rrse_real = np.sqrt(np.sum((test_actual - test_predictions) ** 2) / np.sum((test_actual - mean_actual) ** 2))

# In kết quả
print(f"Mean Squared Error trên tập Test (thang đo gốc): {mse_real:.6f}")
print(f"Mean Absolute Error trên tập Test (thang đo gốc): {mae_real:.6f}")
print(f"Root Mean Squared Error trên tập Test: {rmse_real:.6f}")
print(f"Root Relative Squared Error trên tập Test: {rrse_real:.6f}")

# ==============================================================================
# BƯỚC 6: LƯU TRỮ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 6: Lưu trữ mô hình...")
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgb_model_{now_str}.json"
model.save_model(model_filename)
print(f"Đã lưu mô hình tại: {model_filename}")

# ==============================================================================
# BƯỚC 7: TRỰC QUAN HÓA
# ==============================================================================
print("\nBắt đầu Bước 7: Trực quan hóa...")
plt.style.use('seaborn-v0_8-whitegrid')

train_info = f"xgb_rounds{model.best_iteration}_lr{xgb_params['eta']}_lb{LOOKBACK}"

# Report 1: Biểu đồ Lịch sử Huấn luyện (Evaluation Curve)
fig1, ax1 = plt.subplots(figsize=(12, 6))
val_mae = evals_result['validation_0']['mae']
epochs = range(1, len(val_mae) + 1)
ax1.plot(epochs, val_mae, label='Validation MAE', color='blue', linewidth=2)
ax1.set_title('Báo cáo 1: Diễn biến Validation MAE', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MAE', fontsize=12)
ax1.legend()
ax1.grid(True)
eval_fig_name = f"eval_curve_{now_str}_{train_info}.png"
try:
    plt.savefig(eval_fig_name, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ evaluation curve: {eval_fig_name}")
except Exception as e:
    print(f"Lỗi khi lưu biểu đồ evaluation curve: {e}")
plt.close(fig1)

# Report 2: Biểu đồ Mức độ quan trọng của Đặc trưng (Feature Importance)
fig2, ax2 = plt.subplots(figsize=(10, 8))
xgb.plot_importance(model, ax=ax2, max_num_features=20, height=0.8)
ax2.set_title('Báo cáo 2: 20 Đặc trưng quan trọng nhất', fontsize=16)
plt.tight_layout()
feat_imp_fig_name = f"feature_importance_{now_str}_{train_info}.png"
try:
    plt.savefig(feat_imp_fig_name, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ feature importance: {feat_imp_fig_name}")
except Exception as e:
    print(f"Lỗi khi lưu biểu đồ feature importance: {e}")
plt.close(fig2)

# Report 3: So sánh giá trị Thực tế và Dự đoán trên tập Test
fig3, ax3 = plt.subplots(figsize=(15, 7))
ax3.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
ax3.plot(test_dates, test_predictions, label='Lượng mưa dự đoán (XGBoost)', color='red', alpha=0.7, linestyle='--')
ax3.set_title('Báo cáo 3: So sánh kết quả Thực tế và Dự đoán trên tập Test', fontsize=16)
ax3.set_xlabel('Ngày', fontsize=12)
ax3.set_ylabel('Lượng mưa', fontsize=12)
ax3.legend()
ax3.grid(True)
compare_fig_name = f"compare_test_{now_str}_{train_info}.png"
try:
    plt.savefig(compare_fig_name, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ so sánh: {compare_fig_name}")
except Exception as e:
    print(f"Lỗi khi lưu biểu đồ so sánh: {e}")
plt.close(fig3)

# Lưu báo cáo
# Tổng hợp thông tin báo cáo
report_filename = f"training_report_xgb_{now_str}_{train_info}.txt"
try:
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH XGBoost =====\n")
        f.write(f"Thời gian chạy: {now_str}\n\n")
        f.write(f"--- THÔNG SỐ THỬ NGHIỆM ---\n")
        f.write(f"Model: XGBoost\n")
        for key, value in xgb_params.items():
            f.write(f"- {key}: {value}\n")
        f.write(f"Lookback: {LOOKBACK}\n")
        f.write(f"Data file: {file_path}\n\n")
        f.write(f"--- KẾT QUẢ HUẤN LUYỆN ---\n")
        f.write(f"Total training time: {train_time:.1f} seconds\n")
        f.write(f"Số cây (boosting rounds) đã chạy: {model.best_iteration}\n")
        f.write(f"Best score (val_mae): {min(val_mae):.6f}\n\n")
        f.write(f"--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (THANG ĐO GỐC) ---\n")
        f.write(f"Test MSE: {mse_real:.6f}\n")
        f.write(f"Test MAE: {mae_real:.6f}\n")
        f.write(f"Test RMSE: {rmse_real:.6f}\n")
        f.write(f"Test RRSE: {rrse_real:.6f}\n\n")
        f.write(f"--- CÁC FILE ĐÃ LƯU ---\n")
        f.write(f"Model đã lưu: {model_filename}\n")
        f.write(f"Biểu đồ evaluation curve: {eval_fig_name}\n")
        f.write(f"Biểu đồ feature importance: {feat_imp_fig_name}\n")
        f.write(f"Biểu đồ so sánh thực tế/dự đoán: {compare_fig_name}\n")
        f.write("="*50 + "\n")
        f.write("\n--- LỊCH SỬ MAE TRÊN VALIDATION QUA CÁC EPOCH ---\n")
        for i, mae in enumerate(val_mae, 1):
            f.write(f"Epoch {i}: val_mae = {mae:.6f}\n")
        f.write("="*50 + "\n")
    print(f"Đã lưu báo cáo huấn luyện chi tiết: {report_filename}")
except Exception as e:
    print(f"Lỗi khi lưu báo cáo: {e}")

print("\n--- QUÁ TRÌNH HOÀN TẤT ---")