# Hệ thống Phân tích và Dự đoán Giá Cổ phiếu

## Giới thiệu

Hệ thống phân tích và dự đoán giá cổ phiếu sử dụng các kỹ thuật Machine Learning và phân tích kỹ thuật. Dự án được xây dựng với Streamlit, PostgreSQL/TimescaleDB và các mô hình dự đoán như LSTM, Linear Regression, Auto ARIMA và Random Forest.

## Cấu trúc dự án

```
finance-elt-warehouse/
├── streamlit_app/              # Ứng dụng web Streamlit
│   ├── app.py                  # File chính của ứng dụng
│   ├── database.py             # Kết nối và truy vấn database
│   ├── config.py               # File cấu hình
│   └── __init__.py
├── models/                     # Các mô hình dự đoán
│   ├── __init__.py
│   ├── linear_regression.py    # Mô hình Linear Regression
│   ├── arima.py                # Mô hình Auto ARIMA
│   └── random_forest.py        # Mô hình Random Forest
├── dags/                       # Apache Airflow DAGs (ETL)
│   ├── api_request.py          # Lấy dữ liệu từ API
│   ├── etl_tasks.py            # Các task ETL
│   ├── transform_data.py       # Xử lý và biến đổi dữ liệu
│   ├── model.py                # Mô hình LSTM
│   ├── pipeline.py             # Pipeline xử lý
│   └── stock_quarterly_training_dag.py  # DAG training định kỳ
├── datas/                      # Dữ liệu mẫu CSV
├── requirements.txt            # Thư viện cần thiết
├── docker-compose.yaml         # Docker compose
└── README.md                   # Hướng dẫn này
```

## Yêu cầu hệ thống

- Python 3.10+
- PostgreSQL/TimescaleDB

## Cài đặt

### 1. Clone dự án

```bash
git clone <repository-url>
cd finance-elt-warehouse
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Với venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Hoặc với conda
conda create -n finance python=3.10
conda activate finance
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình database

Chỉnh sửa file `streamlit_app/config.py`:

```python
# Cấu hình cơ sở dữ liệu
DB_HOST = "localhost"       # Địa chỉ host
DB_PORT = 5432              # Cổng kết nối
DB_NAME = "dev"             # Tên database
DB_USER = "db_user"         # Tên đăng nhập
DB_PASSWORD = "db_password" # Mật khẩu
DB_SCHEMA = "dev"           # Schema sử dụng
```

## Chạy ứng dụng

### Bước 1: Di chuyển vào thư mục dự án

```bash
cd <đường-dẫn-tới-thư-mục-dự-án>/finance-elt-warehouse
```

### Bước 2: Chạy ứng dụng Streamlit

```bash
streamlit run streamlit_app/app.py
```

### Bước 3: Truy cập ứng dụng

Mở trình duyệt và truy cập: `http://localhost:8501`

## Tính năng chính

### 1. Dashboard
- Tổng quan thị trường
- Biểu đồ giá và trung bình động (MA5, MA10, MA20)
- Thông tin cổ phiếu và chỉ số cơ bản
- Biểu đồ khối lượng giao dịch

### 2. Phân tích kỹ thuật
- **Trung bình động (Moving Averages)**: MA5, MA10, MA20, MA50, EMA12, EMA26
- **RSI (Relative Strength Index)**: Chỉ báo quá mua/quá bán
- **MACD**: Xu hướng và động lượng
- **Dải Bollinger**: Biến động giá

### 3. Biểu đồ nâng cao
- Biểu đồ nến Nhật Bản (Candlestick)
- So sánh hiệu suất các cổ phiếu
- Phân tích biến động và phân phối lợi nhuận

### 4. Dự đoán AI
- **Hybrid (LSTM)**: Dự đoán từ mô hình đã train sẵn trong database
- **Linear Regression**: Hồi quy tuyến tính thời gian thực
- **Auto ARIMA**: Tự động tìm tham số ARIMA tối ưu
- **Random Forest**: Mô hình rừng ngẫu nhiên với nhiều features kỹ thuật

### 5. Cài đặt
- Cấu hình kết nối database
- Tùy chỉnh hiển thị
- Thông tin hệ thống

## Các mô hình dự đoán

### Hybrid LSTM (từ Database)
- Mô hình đã được train sẵn với Apache Airflow
- Sử dụng các features: Returns, RSI, MACD, Volume_MA7, Volatility
- Kết hợp chỉ số thị trường: VIX, TNX, Oil, USD
- Dự đoán 14 ngày

### Linear Regression
- Sử dụng sklearn LinearRegression
- Dự đoán dựa trên xu hướng tuyến tính
- Metrics: MAE, RMSE, MAPE

### Auto ARIMA
- Sử dụng pmdarima auto_arima
- Tự động tìm tham số (p, d, q) tối ưu
- Phù hợp với dữ liệu chuỗi thời gian

### Random Forest
- Sử dụng sklearn RandomForestRegressor
- Các features tương tự mô hình Hybrid:
  - Chỉ số kỹ thuật: Returns, RSI, MACD, Volatility
  - Volume features: Volume_MA7, Volume_Ratio
  - Moving Averages: MA5, MA10, MA20, EMA12, EMA26
  - Momentum: Momentum_5, Momentum_10, ROC_5, ROC_10
  - Bollinger Bands: BB_Width, BB_Position
  - Lag features: 14 ngày giá và returns
- Metrics: MAE, RMSE, MAPE, Feature Importance

## Cơ sở dữ liệu

Các bảng chính trong schema `dev`:

| Bảng | Mô tả |
|------|-------|
| `dimstock` | Thông tin cổ phiếu (mã, tên công ty, ngành) |
| `dimdate` | Bảng ngày tháng |
| `factstockprice` | Giá cổ phiếu hàng ngày (OHLCV) |
| `factfundamentals` | Chỉ số cơ bản (P/E, P/B, Market Cap) |
| `factstockprediction` | Dự đoán giá từ mô hình LSTM |
| `factmodelperformance` | Hiệu suất mô hình (MAE, RMSE, MAPE) |
| `factmarketindicators` | Chỉ số thị trường (VIX, TNX, Oil, USD) |

## Xử lý lỗi thường gặp

### 1. Không kết nối được database
```
Lỗi: Không thể kết nối database
```
**Giải pháp**: 
- Kiểm tra database đang chạy
- Kiểm tra thông tin cấu hình trong `config.py`

### 2. Module not found
```
ModuleNotFoundError: No module named 'pmdarima'
```
**Giải pháp**:
```bash
pip install pmdarima
```

### 3. Streamlit không chạy được
**Giải pháp**:
```bash
pip install streamlit --upgrade
```

### 4. Lỗi khi chạy Random Forest
```
IndexError: single positional indexer is out-of-bounds
```
**Giải pháp**: Đảm bảo dữ liệu có đủ số ngày (tối thiểu 60 ngày) để tính các chỉ số kỹ thuật.

