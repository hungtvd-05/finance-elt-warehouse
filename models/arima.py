"""
Mô hình ARIMA cho dự đoán giá cổ phiếu
Sử dụng Auto ARIMA để tự động tìm tham số tối ưu
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima


def predict_arima(df, days_to_predict=14):
    """
    Dự đoán giá cổ phiếu sử dụng Auto ARIMA

    Args:
        df: DataFrame chứa dữ liệu giá cổ phiếu (phải có cột 'close')
        days_to_predict: Số ngày cần dự đoán

    Returns:
        predicted_prices: List giá dự đoán
        metrics: Dict chứa MAE, RMSE, MAPE
    """
    # Chuẩn bị dữ liệu
    df_copy = df.copy().reset_index(drop=True)
    prices = df_copy['close'].values

    # Tính metrics với cross-validation đơn giản
    split_idx = int(len(prices) * 0.9)
    train_data = prices[:split_idx]
    test_data = prices[split_idx:]

    # Sử dụng Auto ARIMA để tìm tham số tối ưu
    model_eval = auto_arima(
        train_data,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,  # Tự động xác định d
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Dự đoán trên test set
    y_pred_test = model_eval.predict(n_periods=len(test_data))

    mae = mean_absolute_error(test_data, y_pred_test)
    rmse = np.sqrt(mean_squared_error(test_data, y_pred_test))
    mape = np.mean(np.abs((test_data - y_pred_test) / test_data)) * 100

    # Fit model trên toàn bộ dữ liệu để dự đoán tương lai
    model_full = auto_arima(
        prices,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Lấy order tối ưu
    order = model_full.order

    # Dự đoán tương lai
    predicted_prices = model_full.predict(n_periods=days_to_predict).tolist()

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'model_name': f'Auto ARIMA{order}'
    }

    return predicted_prices, metrics


