# -*- coding: utf-8 -*-
"""
Mã nguồn huấn luyện mô hình LightGBM để dự báo lượng mưa.
Bao gồm:
1. Đọc và xử lý dữ liệu.
2. Tạo đặc trưng dạng bảng (lag features).
3. Huấn luyện LightGBM với Early Stopping.
4. Đánh giá, lưu trữ mô hình và tạo báo cáo trực quan.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import time

# ==============================================================================
# BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU (Tương tự code trước)
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
LOOKBACK = 8
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
    
    # Xóa các dòng có giá trị NaN được tạo ra do phép shift
    df_lagged.dropna(inplace=True)
    
    # Tách X (đặc trưng) và y (mục tiêu)
    y = df_lagged[target_col]
    X = df_lagged.drop(columns=features_to_lag)
    return X, y

# Tạo dữ liệu cho từng tập
X_train, y_train = create_tabular_features(train_df, LOOKBACK)
X_val, y_val = create_tabular_features(val_df, LOOKBACK)
X_test, y_test = create_tabular_features(test_df, LOOKBACK)
test_dates = X_test.index # Lưu lại ngày tháng của tập test để vẽ biểu đồ

print(f"Kích thước tập huấn luyện X: {X_train.shape}, y: {y_train.shape}")
print("Một vài đặc trưng được tạo ra:", list(X_train.columns[:5]))

# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH LIGHTGBM (VỚI THAM SỐ MỚI)
# ==============================================================================
print("\nBắt đầu Bước 4: Huấn luyện mô hình LightGBM với tham số mới...")

# *** THAY ĐỔI: Tinh chỉnh lại bộ tham số để mô hình "mạnh dạn" hơn ***
params = {
    'objective': 'tweedie',            # THAY ĐỔI QUAN TRỌNG: Phù hợp cho dữ liệu có nhiều số 0
    'tweedie_variance_power': 1.1,     # Tham số cho objective tweedie (1.0-2.0)
    'metric': 'mae',
    'n_estimators': 2000,              # Tăng số cây tối đa, early stopping sẽ tìm điểm dừng tốt nhất
    'learning_rate': 0.02,             # Giảm learning rate để mô hình học cẩn thận hơn
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,                  # Thêm một chút regularization
    'lambda_l2': 0.1,
    'num_leaves': 50,                  # Tăng độ phức tạp của cây
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# Khởi tạo mô hình
model = lgb.LGBMRegressor(**params)

# Huấn luyện mô hình với Early Stopping
print("Đang huấn luyện...")
start_time = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(10, verbose=True)]
)
train_time = time.time() - start_time
print(f"Huấn luyện hoàn tất trong {train_time:.1f} giây.")

# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 5: Đánh giá mô hình...")
# Lấy dự đoán trên tập test (dự đoán vẫn ở thang đo log)
test_predictions_log = model.predict(X_test)

# Chuyển về thang đo gốc
test_predictions = np.expm1(test_predictions_log)
test_actual = np.expm1(y_test)

# Xử lý các giá trị âm (nếu có)
test_predictions[test_predictions < 0] = 0

# Tính toán sai số trên thang đo gốc
mse_real = mean_squared_error(test_actual, test_predictions)
mae_real = mean_absolute_error(test_actual, test_predictions)

print(f"Mean Squared Error trên tập Test (thang đo gốc): {mse_real:.6f}")
print(f"Mean Absolute Error trên tập Test (thang đo gốc): {mae_real:.6f}")

# ==============================================================================
# BƯỚC 6: LƯU TRỮ MÔ HÌNH
# ==============================================================================
print("\nBắt đầu Bước 6: Lưu trữ mô hình...")
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"lgbm_model_{now_str}.txt"
model.booster_.save_model(model_filename)
print(f"Đã lưu mô hình tại: {model_filename}")

# ==============================================================================
# BƯỚC 7: TRỰC QUAN HÓA VÀ BÁO CÁO
# ==============================================================================
print("\nBắt đầu Bước 7: Trực quan hóa và báo cáo...")
plt.style.use('seaborn-v0_8-whitegrid')
train_info = f"lgbm_rounds{model.best_iteration_}_lr{params['learning_rate']}_lb{LOOKBACK}"

# Report 1: Biểu đồ Lịch sử Huấn luyện (Evaluation Curve)
fig1, ax1 = plt.subplots(figsize=(12, 6))
lgb.plot_metric(model, ax=ax1, title='Báo cáo 1: Diễn biến Validation MAE')
eval_fig_name = f"eval_curve_{now_str}_{train_info}.png"
plt.savefig(eval_fig_name)
print(f"Đã lưu biểu đồ evaluation curve: {eval_fig_name}")
plt.close(fig1)

# Report 2: Biểu đồ Mức độ quan trọng của Đặc trưng (Feature Importance)
fig2, ax2 = plt.subplots(figsize=(10, 8))
lgb.plot_importance(model, ax=ax2, max_num_features=20, height=0.8)
ax2.set_title('Báo cáo 2: 20 Đặc trưng quan trọng nhất', fontsize=16)
plt.tight_layout()
feat_imp_fig_name = f"feature_importance_{now_str}_{train_info}.png"
plt.savefig(feat_imp_fig_name)
print(f"Đã lưu biểu đồ feature importance: {feat_imp_fig_name}")
plt.close(fig2)

# Report 3: So sánh giá trị Thực tế và Dự đoán trên tập Test
fig3, ax3 = plt.subplots(figsize=(15, 7))
ax3.plot(test_dates, test_actual, label='Lượng mưa thực tế', color='blue', linewidth=2)
ax3.plot(test_dates, test_predictions, label='Lượng mưa dự đoán (LGBM)', color='red', alpha=0.7, linestyle='--')
ax3.set_title('Báo cáo 3: So sánh kết quả Thực tế và Dự đoán trên tập Test', fontsize=16)
ax3.set_xlabel('Ngày', fontsize=12)
ax3.set_ylabel('Lượng mưa', fontsize=12)
ax3.legend()
ax3.grid(True)
compare_fig_name = f"compare_test_{now_str}_{train_info}.png"
plt.savefig(compare_fig_name)
print(f"Đã lưu biểu đồ so sánh: {compare_fig_name}")
plt.close(fig3)

# Tổng hợp thông tin báo cáo
report_filename = f"training_report_{now_str}_{train_info}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write("===== BÁO CÁO HUẤN LUYỆN MÔ HÌNH LIGHTGBM =====\n")
    f.write(f"Thời gian chạy: {now_str}\n\n")
    f.write(f"--- THÔNG SỐ THỬ NGHIỆM ---\n")
    f.write(f"Model: LightGBM\n")
    for key, value in params.items():
        f.write(f"- {key}: {value}\n")
    f.write(f"Lookback: {LOOKBACK}\n")
    f.write(f"Data file: {file_path}\n\n")
    f.write(f"--- KẾT QUẢ HUẤN LUYỆN ---\n")
    f.write(f"Total training time: {train_time:.1f} seconds\n")
    f.write(f"Số cây (boosting rounds) đã chạy: {model.best_iteration_}\n")
    
    best_score_dict = model.best_score_['valid_0']
    if 'mae' in best_score_dict:
        best_score = best_score_dict['mae']
    elif 'l1' in best_score_dict:
        best_score = best_score_dict['l1']
    else:
        best_score = None

    if best_score is not None:
        f.write(f"Best score (val_mae): {best_score:.6f}\n\n")
    else:
        f.write("Best score (val_mae): Không tìm thấy metric phù hợp!\n\n")
    
    f.write(f"--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (THANG ĐO GỐC) ---\n")
    f.write(f"Test MSE: {mse_real:.6f}\n")
    f.write(f"Test MAE: {mae_real:.6f}\n\n")
    f.write(f"--- CÁC FILE ĐÃ LƯU ---\n")
    f.write(f"Model đã lưu: {model_filename}\n")
    f.write(f"Biểu đồ evaluation curve: {eval_fig_name}\n")
    f.write(f"Biểu đồ feature importance: {feat_imp_fig_name}\n")
    f.write(f"Biểu đồ so sánh thực tế/dự đoán: {compare_fig_name}\n")
    f.write("="*50 + "\n")

    # Lưu lịch sử train từng epoch
    f.write("\n--- LỊCH SỬ MAE TRÊN VALIDATION QUA CÁC EPOCH ---\n")
    evals_result = model.evals_result_
    val_mae_list = evals_result['valid_0']['l1'] if 'l1' in evals_result['valid_0'] else evals_result['valid_0']['mae']
    for i, val_mae in enumerate(val_mae_list, 1):
        f.write(f"Epoch {i}: val_mae = {val_mae:.6f}\n")
    f.write("="*50 + "\n")

print(f"Đã lưu báo cáo huấn luyện chi tiết: {report_filename}")

print("\n--- QUÁ TRÌNH HOÀN TẤT ---")