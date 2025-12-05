import json
import os

import joblib
# import matplotlib.pyplot as plt

import pandas as pd

import api_request as ar
import transform_data as td
from psycopg2 import pool, extras
import datetime
import numpy as np

def connect_to_db():
    print("Connecting to the PostgreSQL database...")
    try:
        connection_pool = pool.SimpleConnectionPool(
            minconn=5,
            maxconn=10,
            host="localhost",
            port="5432",
            database="dev",
            user="db_user",
            password="db_password"
        )
        print("Connection pool created successfully.")
        return connection_pool
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def safe_cast_to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_cast_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_or_create_date_key(connection_pool, target_date):

    if isinstance(target_date, str):
        target_date = datetime.date.fromisoformat(target_date)

    date_key = int(target_date.strftime("%Y%m%d"))

    insert_query = """
                   INSERT INTO dev.DimDate (DateKey, FullDate, Year, Month, Day, Weekday)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (DateKey) DO NOTHING; \
                   """

    conn = None

    try:
        conn = connection_pool.getconn()

        with conn.cursor() as cursor:
            cursor.execute(insert_query, (
                date_key,
                target_date,
                target_date.year,
                target_date.month,
                target_date.day,
                target_date.weekday()
            ))
            conn.commit()

        return date_key

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error in get_or_create_date_key for {date_key}: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def fetch_one_stock_info(ticker):
    try:
        df = ar.get_full_stock_dimensions(ticker)

        if df.empty:
            print("No data found for ticker:", ticker)
            return None

        row = df.iloc[0]

        if row['Country'] is None or row['Country'] != 'United States':
            return None

        return row
    except Exception as e:
        return None

def process_one_stock_fundamentals(connection_pool, stock_key, ticker, date_key):
    conn = None
    try:
        conn = connection_pool.getconn()

        df = ar.get_full_stock_dimensions(ticker)

        if df.empty:
            print(f"No fundamentals data for ticker {ticker}")
            return

        row = df.iloc[0]

        with conn.cursor() as cursor:
            update_dim_query = """
                                INSERT INTO dev.DimStock (TickerSymbol, CompanyName, Industry, Sector, Country)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (TickerSymbol) DO UPDATE SET
                                   CompanyName = EXCLUDED.CompanyName,
                                   Industry = EXCLUDED.Industry,
                                   Sector = EXCLUDED.Sector,
                                   Country = EXCLUDED.Country;
                               """

            cursor.execute(update_dim_query, (
                row['TickerSymbol'],
                row['CompanyName'],
                row['Industry'],
                row['Sector'],
                row['Country']
            ))

            update_fact_query = """
                            INSERT INTO dev.FactFundamentals (DateKey, StockKey, MarketCap, forwardPE, trailingPE, priceToBook, dividendYield, beta)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (DateKey, StockKey) DO UPDATE SET
                                MarketCap = EXCLUDED.MarketCap,
                                forwardPE = EXCLUDED.forwardPE,
                                trailingPE = EXCLUDED.trailingPE,
                                priceToBook = EXCLUDED.priceToBook,
                                dividendYield = EXCLUDED.dividendYield,
                                beta = EXCLUDED.beta;
                        """
            cursor.execute(update_fact_query, (
                date_key,
                stock_key,
                safe_cast_to_int(row['MarketCap']),
                safe_cast_to_float(row['forwardPE']),
                safe_cast_to_float(row['trailingPE']),
                safe_cast_to_float(row['priceToBook']),
                safe_cast_to_float(row['dividendYield']),
                safe_cast_to_float(row['beta'])
            ))

            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def process_one_stock_transformation(connection_pool, stock_key, ticker, sector, df_market, date_map, update=True):
    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            if not update:
                cursor.execute("""SELECT RawDate, Ticker, Open, High, Low, Close, Volume
                                  FROM staging.RawStockPrice
                                  WHERE Ticker = %s""", (ticker,))
            else:
                query_limit_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

                cursor.execute("""SELECT RawDate, Ticker, Open, High, Low, Close, Volume
                                  FROM staging.RawStockPrice
                                  WHERE Ticker = %s
                                    AND RawDate >= %s
                                  ORDER BY RawDate ASC""", (ticker, query_limit_date))
            raw_stock_data = cursor.fetchall()

            if not raw_stock_data:
                return

            df_raw = pd.DataFrame(raw_stock_data, columns=['date', 'name', 'open', 'high', 'low', 'close', 'volume'])
            df_raw = df_raw.sort_values(['name', 'date']).reset_index(drop=True)

            stock_trans = td.add_technical_indicators(df_raw, sector, df_market, date_map)

            if stock_trans.empty:
                return

            core_db_columns = {'DateKey', 'open', 'high', 'low', 'close', 'volume', 'Returns', 'Volatility', 'Volume_MA7', 'RSI', 'MACD'}

            records_to_insert = []

            for row in stock_trans.itertuples(index=False):
                features_dict = {}

                for col in stock_trans.columns:
                    if col not in core_db_columns:
                        val = getattr(row, col)

                        if pd.isna(val) or np.isnan(val):
                            val = None

                        features_dict[col] = val

                technical_features_json = json.dumps(features_dict)

                records_to_insert.append((
                    getattr(row, 'DateKey'),
                    stock_key,
                    getattr(row, 'open'),
                    getattr(row, 'high'),
                    getattr(row, 'low'),
                    getattr(row, 'close'),
                    getattr(row, 'volume'),
                    getattr(row, 'Returns'),
                    getattr(row, 'Volatility'),
                    getattr(row, 'Volume_MA7'),
                    getattr(row, 'RSI'),
                    getattr(row, 'MACD'),
                    technical_features_json
                ))

            if not records_to_insert:
                print(f"No transformed data to load for {ticker}.")
                return

            insert_query = """
                           INSERT INTO dev.FactStockPrice (DateKey, StockKey, Open, High, Low, Close, Volume, Returns, Volatility, Volume_MA7, RSI, MACD, 
                                                           TechnicalFeatures)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (DateKey, StockKey) DO \
                           UPDATE SET \
                               Open = EXCLUDED.Open, \
                               High = EXCLUDED.High, \
                               Low = EXCLUDED.Low, \
                               Close = EXCLUDED.Close, \
                               Volume = EXCLUDED.Volume, \
                                Returns = EXCLUDED.Returns, \
                                Volatility = EXCLUDED.Volatility, \
                                Volume_MA7 = EXCLUDED.Volume_MA7, \
                                RSI = EXCLUDED.RSI, \
                                MACD = EXCLUDED.MACD, \
                               TechnicalFeatures = EXCLUDED.TechnicalFeatures; \
                           """
            cursor.executemany(insert_query, records_to_insert)
            conn.commit()
            print(f"Successfully loaded transformed data for {ticker}.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"FAILED for {ticker}. Error: {e}")
        raise e
    finally:
        if conn:
            connection_pool.putconn(conn)

def train_one_stock_model(connection_pool, stock_key, ticker, df_market, training_date_key):
    from model import build_hybrid_model, train_model, evaluate_model

    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT ActiveFeatures, LookBackWindow, ForecastHorizon
                              FROM dev.DimStockModelConfig
                              WHERE StockKey = %s""", (stock_key,))
            config = cursor.fetchone()
            print(config)

        query = """
                SELECT DateKey, Open, High, Low, Close, Volume, Returns, Volatility, Volume_MA7, RSI, MACD, TechnicalFeatures
                FROM dev.FactStockPrice
                WHERE StockKey = %s
                ORDER BY DateKey ASC
                """

        df = pd.read_sql(query, conn, params=(stock_key,))

        if df.empty:
            print(f"No stock price data found for ticker {ticker}")
            return

        features_df = pd.json_normalize(df['technicalfeatures'])

        df_final = pd.concat([df.drop(columns=['technicalfeatures']), features_df], axis=1)

        df_final.columns = df_final.columns.str.lower()

        df_merged = pd.merge(df_final, df_market, on='datekey', how='left')

        (X_encoder_train, X_decoder_train, y_train,
         X_encoder_test, X_decoder_test, y_test,
         scaler_X, scaler_y) = td.prepare_sequences(df_merged, config)

        model = build_hybrid_model(len(X_encoder_train[0]), X_encoder_train.shape[2], config[2])

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        scaler_x_path = os.path.join(MODEL_DIR, f"{ticker}_scaler_X.pkl")
        joblib.dump(scaler_X, scaler_x_path)

        scaler_y_path = os.path.join(MODEL_DIR, f"{ticker}_scaler_y.pkl")
        joblib.dump(scaler_y, scaler_y_path)

        model_filename = f"{MODEL_DIR}/{ticker}_model.keras"

        history = train_model(model, X_encoder_train, X_decoder_train, y_train,
                                 X_encoder_test, X_decoder_test, y_test, model_save_path=model_filename,
                                 epochs=100, batch_size=32)

        predictions, mae, rmse, mape = evaluate_model(model, X_encoder_test, X_decoder_test, y_test, scaler_y)

        with conn.cursor() as cursor:
            query = """
                    INSERT INTO dev.FactModelPerformance 
                    (StockKey, TrainingDateKey, MAE, RMSE, MAPE, LookBackWindow, ForecastHorizon)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
            cursor.execute(query, (
                stock_key,
                training_date_key,
                float(mae),
                float(rmse),
                float(mape),
                int(config[1]),
                int(config[2])
            ))
            conn.commit()

    except Exception as e:
        print(f"Error fetching model config for {ticker}: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)


def predict_one_stock(connection_pool, stock_key, ticker, df_market, prediction_date_key):
    from tensorflow.keras.models import load_model
    from model import directional_loss

    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT ActiveFeatures, LookBackWindow, ForecastHorizon
                              FROM dev.DimStockModelConfig
                              WHERE StockKey = %s""", (stock_key,))
            config = cursor.fetchone()

        query = """
                SELECT * FROM (
                                  SELECT DateKey, Open, High, Low, Close, Volume, Returns, Volatility, Volume_MA7, RSI, MACD, TechnicalFeatures
                                  FROM dev.FactStockPrice
                                  WHERE StockKey = %s
                                  ORDER BY DateKey DESC
                                      LIMIT 355
                              ) AS sub
                ORDER BY DateKey ASC
                """

        df = pd.read_sql(query, conn, params=(stock_key,))

        if df.empty:
            print(f"No stock price data found for ticker {ticker}")
            return

        features_df = pd.json_normalize(df['technicalfeatures'])

        df_final = pd.concat([df.drop(columns=['technicalfeatures']), features_df], axis=1)

        df_final.columns = df_final.columns.str.lower()

        df_merged = pd.merge(df_final, df_market, on='datekey', how='left')

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

        model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
        scaler_x_path = os.path.join(MODEL_DIR, f"{ticker}_scaler_X.pkl")
        scaler_y_path = os.path.join(MODEL_DIR, f"{ticker}_scaler_y.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_x_path):
            print(f"Model or Scalers not found for {ticker}")
            return None

        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        encoder_input, decoder_input = td.prepare_sequences_predict(df_merged, config, scaler_X, scaler_y)

        if encoder_input is None:
            print(f"Skipping {ticker}: Failed to prepare sequences.")
            return None

        model = load_model(model_path, custom_objects={'directional_loss': directional_loss})
        predictions_scaled = model.predict([encoder_input, decoder_input])
        predictions_array = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1, config[2])

        print(f"Predictions for {ticker}: {predictions_array}")

        if hasattr(predictions_array, 'tolist'):
            pred_list = predictions_array.tolist()
        else:
            pred_list = predictions_array

        with conn.cursor() as cursor:
            query = """
                    INSERT INTO dev.FactStockPrediction
                        (PredictionDateKey, StockKey, ForecastPrices)
                    VALUES (%s, %s, %s) ON CONFLICT (PredictionDateKey, StockKey) 
                        DO \
                    UPDATE SET
                        ForecastPrices = EXCLUDED.ForecastPrices, \
                        CreatedAt = CURRENT_TIMESTAMP;
                    """
            cursor.execute(query, (prediction_date_key, stock_key, pred_list))
            conn.commit()

    except Exception as e:
        print(f"Error during prediction for {ticker}: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)


