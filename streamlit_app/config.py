# ============================================================
# CẤU HÌNH HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN GIÁ CỔ PHIẾU
# ============================================================

# ----------------------------------------------------------
# CẤU HÌNH CƠ SỞ DỮ LIỆU (PostgreSQL/TimescaleDB)
# ----------------------------------------------------------
# Sử dụng cloudflared tunnel:
# cloudflared access tcp --hostname timescaledb.tranvienduyhung.id.vn --url localhost:5432

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "dev"
DB_USER = "db_user"
DB_PASSWORD = "db_password"
DB_SCHEMA = "dev"

# ----------------------------------------------------------
# CẤU HÌNH ỨNG DỤNG
# ----------------------------------------------------------

# Số ngày dữ liệu mặc định hiển thị
DEFAULT_DATA_DAYS = 365

# Số ngày dự đoán mặc định
DEFAULT_PREDICTION_DAYS = 14

# ----------------------------------------------------------
# CẤU HÌNH MÔ HÌNH DỰ ĐOÁN
# ----------------------------------------------------------

# ARIMA
ARIMA_MAX_P = 5  # Tham số AR tối đa
ARIMA_MAX_Q = 5  # Tham số MA tối đa

# Linear Regression
LR_TRAIN_SPLIT = 0.8  # Tỷ lệ train/test

# ----------------------------------------------------------
# CẤU HÌNH HIỂN THỊ
# ----------------------------------------------------------

# Định dạng tiền tệ
CURRENCY_SYMBOL = "$"
CURRENCY_FORMAT = "{:,.2f}"

# Màu sắc biểu đồ
COLOR_UP = "#26a69a"      # Màu tăng (xanh)
COLOR_DOWN = "#ef5350"    # Màu giảm (đỏ)
COLOR_MA5 = "#FF6B6B"
COLOR_MA10 = "#4ECDC4"
COLOR_MA20 = "#45B7D1"
COLOR_MA50 = "#96CEB4"

