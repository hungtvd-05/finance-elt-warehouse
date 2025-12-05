"""
Mô hình Random Forest cho dự đoán giá cổ phiếu
Sử dụng các đặc trưng tương tự mô hình Hybrid (LSTM)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_technical_indicators(df):
    """
    Thêm các chỉ số kỹ thuật tương tự như mô hình Hybrid
    """
    data = df.copy()

    # Returns - Lợi nhuận theo ngày
    data['returns'] = data['close'].pct_change()

    # RSI - Relative Strength Index (14 ngày)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # MACD - Moving Average Convergence Divergence
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2

    # Volume MA7 - Trung bình khối lượng 7 ngày
    if 'volume' in data.columns:
        data['volume_ma7'] = data['volume'].rolling(window=7).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma7']
    else:
        data['volume_ma7'] = 0
        data['volume_ratio'] = 0

    # Volatility - Độ biến động (21 ngày)
    data['volatility'] = data['returns'].rolling(window=21).std()

    # Moving Averages
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['ma50'] = data['close'].rolling(window=50).mean()

    # EMA - Exponential Moving Average
    data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()

    # Momentum
    data['momentum_5'] = data['close'] - data['close'].shift(5)
    data['momentum_10'] = data['close'] - data['close'].shift(10)

    # Price Rate of Change (ROC)
    data['roc_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5) * 100
    data['roc_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10) * 100

    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    data['bb_std'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

    # Price relative to MAs
    data['price_ma5_ratio'] = data['close'] / data['ma5']
    data['price_ma20_ratio'] = data['close'] / data['ma20']

    # MA crossovers
    data['ma5_ma20_diff'] = data['ma5'] - data['ma20']
    data['ema12_ema26_diff'] = data['ema12'] - data['ema26']

    return data


def predict_random_forest(df, days_to_predict=14, df_market=None):
    """
    Dự đoán giá cổ phiếu sử dụng Random Forest
    Với các features tương tự mô hình Hybrid (LSTM)

    Args:
        df: DataFrame chứa dữ liệu giá cổ phiếu (phải có cột 'close', 'volume')
        days_to_predict: Số ngày cần dự đoán
        df_market: DataFrame chứa dữ liệu thị trường (VIX, TNX, Oil, USD) - tùy chọn

    Returns:
        predicted_prices: List giá dự đoán
        metrics: Dict chứa MAE, RMSE, MAPE
    """
    # Chuẩn bị dữ liệu
    df_copy = df.copy().reset_index(drop=True)

    # Thêm các chỉ số kỹ thuật
    df_copy = add_technical_indicators(df_copy)

    # Tạo lag features
    lookback = 14
    for i in range(1, lookback + 1):
        df_copy[f'close_lag_{i}'] = df_copy['close'].shift(i)
        df_copy[f'returns_lag_{i}'] = df_copy['returns'].shift(i)

    # Xóa các dòng có NaN
    df_copy = df_copy.dropna().reset_index(drop=True)

    # Định nghĩa features
    feature_cols = [
        # Chỉ số kỹ thuật chính (tương tự Hybrid)
        'returns', 'rsi', 'macd', 'volatility',
        # Volume features
        'volume_ma7', 'volume_ratio',
        # Moving Averages
        'ma5', 'ma10', 'ma20',
        # EMA
        'ema12', 'ema26',
        # Momentum
        'momentum_5', 'momentum_10',
        # ROC
        'roc_5', 'roc_10',
        # Bollinger Bands
        'bb_width', 'bb_position',
        # Price ratios
        'price_ma5_ratio', 'price_ma20_ratio',
        # MA crossovers
        'ma5_ma20_diff', 'ema12_ema26_diff',
    ]

    # Thêm lag features
    feature_cols += [f'close_lag_{i}' for i in range(1, lookback + 1)]
    feature_cols += [f'returns_lag_{i}' for i in range(1, lookback + 1)]

    # Lọc các cột tồn tại
    feature_cols = [col for col in feature_cols if col in df_copy.columns]

    X = df_copy[feature_cols].values
    y = df_copy['close'].values

    # Chia train/test (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model để đánh giá
    model_eval = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model_eval.fit(X_train, y_train)

    # Dự đoán trên test set
    y_pred_test = model_eval.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    # Train model trên toàn bộ dữ liệu
    model_full = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model_full.fit(X, y)

    # Dự đoán tương lai (recursive prediction)
    predicted_prices = []

    # Lấy features của dòng cuối cùng
    last_features = X[-1].copy()
    last_close = y[-1]

    # Lưu các giá trị lag gần nhất
    recent_closes = list(y[-(lookback+1):])  # Lấy lookback+1 giá gần nhất
    recent_returns = list(df_copy['returns'].iloc[-(lookback+1):])

    for day in range(days_to_predict):
        # Dự đoán giá tiếp theo
        next_price = model_full.predict(last_features.reshape(1, -1))[0]
        predicted_prices.append(next_price)

        # Cập nhật returns
        new_return = (next_price - last_close) / last_close if last_close != 0 else 0

        # Cập nhật recent lists
        recent_closes.append(next_price)
        recent_closes.pop(0)
        recent_returns.append(new_return)
        recent_returns.pop(0)

        # Cập nhật features cho dự đoán tiếp theo
        # Tìm index của các features và cập nhật
        for i, col in enumerate(feature_cols):
            if col == 'returns':
                last_features[i] = new_return
            elif col.startswith('close_lag_'):
                lag_num = int(col.split('_')[-1])
                if lag_num <= len(recent_closes) - 1:
                    last_features[i] = recent_closes[-(lag_num + 1)]
            elif col.startswith('returns_lag_'):
                lag_num = int(col.split('_')[-1])
                if lag_num <= len(recent_returns) - 1:
                    last_features[i] = recent_returns[-(lag_num + 1)]
            # Các features khác giữ nguyên giá trị gần nhất (approximation)

        last_close = next_price

    # Feature importance (top 10)
    feature_importance = dict(zip(feature_cols, model_full.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'model_name': 'Random Forest',
        'n_features': len(feature_cols),
        'top_features': top_features
    }

    return predicted_prices, metrics

