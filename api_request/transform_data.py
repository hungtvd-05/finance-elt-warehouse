import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df_stock, sector, df_market, date_map):
    data = df_stock.copy()

    data['DateKey'] = data['date'].apply(lambda x: date_map.get(x))
    data = data.dropna(subset=['DateKey'])
    data['DateKey'] = data['DateKey'].astype(int)

    data = pd.merge(data, df_market, on='DateKey', how='left')

    data['Returns'] = data['close'].pct_change()

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2

    data['Volume_MA7'] = data['volume'].rolling(window=7).mean()

    if sector in ['Energy', 'Industrials', 'Basic Materials']:
        data['Volatility'] = data['Returns'].rolling(window=21).std()

        data['Stock_Oil_Corr'] = data['Returns'].rolling(window=21).corr(data['Oil_Change'])
        data['Vol_x_Oil'] = data['volume'] * data['Oil_Change']

    elif sector in ['Technology', 'Communication Services', 'Consumer Cyclical']:
        data['Volatility'] = data['Returns'].rolling(window=21).std()
        data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()

    elif sector in ['Financial Services', 'Real Estate']:
        data['Stock_Rate_Corr'] = data['Returns'].rolling(window=21).corr(data['TNX_Change'])

    elif sector in ['Healthcare', 'Utilities', 'Consumer Defensive']:
        ma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        data['BB_Width'] = ((ma20 + (std20 * 2)) - (ma20 - (std20 * 2))) / ma20
        data['SMA_50'] = data['close'].rolling(window=50).mean()

    cols_to_drop = [
        'VIX_Close', 'TNX_Close', 'Oil_Close', 'USD_Close',
        'VIX_Change', 'TNX_Change', 'Oil_Change', 'USD_Change', 'date', 'name'
    ]

    data = data.drop(columns=cols_to_drop, errors='ignore')
    data = data.dropna()
    return data

def prepare_sequences(data, config, train_split=0.85):

    active_features, input_timesteps, output_horizon = config

    feature_columns = ['Close'] + active_features

    feature_columns = list(map(str.lower, feature_columns))

    data_train = data[feature_columns].copy()
    data_train.dropna(inplace=True)

    data_train.to_csv('debug_data_train.csv', index=False)

    features = data_train[feature_columns].values
    target = data_train['close'].values

    split_idx = int(len(features) * train_split)

    features_train = features[:split_idx]
    features_test = features[split_idx:]
    target_train = target[:split_idx]
    target_test = target[split_idx:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_train_scaled = scaler_X.fit_transform(features_train)
    features_test_scaled = scaler_X.transform(features_test)

    target_train_scaled = scaler_y.fit_transform(target_train.reshape(-1, 1))
    target_test_scaled = scaler_y.transform(target_test.reshape(-1, 1))

    def create_sequences(features_scaled, target_scaled, input_timesteps, output_horizon):
        X_encoder, X_decoder, y = [], [], []

        for i in range(input_timesteps, len(features_scaled) - output_horizon):
            X_encoder.append(features_scaled[i - input_timesteps:i])

            decoder_input = np.zeros((output_horizon, 1))
            decoder_input[0] = target_scaled[i - 1]
            for j in range(1, output_horizon):
                decoder_input[j] = target_scaled[i + j - 1]
            X_decoder.append(decoder_input)

            y.append(target_scaled[i:i + output_horizon])

        return np.array(X_encoder), np.array(X_decoder), np.array(y)

    X_encoder_train, X_decoder_train, y_train = create_sequences(
        features_train_scaled, target_train_scaled, input_timesteps, output_horizon
    )

    X_encoder_test, X_decoder_test, y_test = create_sequences(
        features_test_scaled, target_test_scaled, input_timesteps, output_horizon
    )

    return (X_encoder_train, X_decoder_train, y_train,
            X_encoder_test, X_decoder_test, y_test,
            scaler_X, scaler_y)