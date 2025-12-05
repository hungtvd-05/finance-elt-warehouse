"""
Mô hình Linear Regression cho dự đoán giá cổ phiếu
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def predict_linear_regression(df, days_to_predict=14):
    """
    Dự đoán giá cổ phiếu sử dụng Linear Regression

    Args:
        df: DataFrame chứa dữ liệu giá cổ phiếu (phải có cột 'close')
        days_to_predict: Số ngày cần dự đoán

    Returns:
        predicted_prices: List giá dự đoán
        metrics: Dict chứa MAE, RMSE, MAPE
    """
    # Chuẩn bị dữ liệu
    df_copy = df.copy().reset_index(drop=True)
    df_copy['day_index'] = np.arange(len(df_copy))

    # Features: day_index
    X = df_copy[['day_index']].values
    y = df_copy['close'].values

    # Train trên toàn bộ dữ liệu
    model = LinearRegression()
    model.fit(X, y)

    split_idx = int(len(df_copy) * 0.9)
    y_test = y[split_idx:]
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]

    model_eval = LinearRegression()
    model_eval.fit(X_train, y_train)
    y_pred_test = model_eval.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    # Dự đoán tương lai
    last_idx = len(df_copy) - 1
    future_indices = np.arange(last_idx + 1, last_idx + 1 + days_to_predict).reshape(-1, 1)
    predicted_prices = model.predict(future_indices).tolist()

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'model_name': 'Linear Regression'
    }

    return predicted_prices, metrics

