import etl_tasks as etl

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import api_request as ar
from psycopg2 import extras
import datetime

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


def initialize_database_schema(connection_pool):
    print("Creating table...")
    conn = None
    create_table_query = """
                CREATE SCHEMA IF NOT EXISTS dev;  
                CREATE SCHEMA IF NOT EXISTS staging;
                CREATE EXTENSION IF NOT EXISTS timescaledb;
                CREATE TABLE IF NOT EXISTS staging.RawStockPrice (
                    Rawdate DATE NOT NULL, 
                    Ticker VARCHAR(10) NOT NULL,                 
                    Open FLOAT,
                    High FLOAT,
                    Low FLOAT,
                    Close FLOAT,
                    Volume BIGINT,

                    PRIMARY KEY (RawDate, Ticker)
                );
                CREATE TABLE IF NOT EXISTS dev.DimDate (
                    DateKey INT PRIMARY KEY,         
                    FullDate DATE NOT NULL UNIQUE,  
                    Year INT,
                    Month INT,
                    Day INT,
                    Weekday INT
                );
                CREATE TABLE IF NOT EXISTS dev.DimStock (
                    StockKey SERIAL PRIMARY KEY, 
                    TickerSymbol VARCHAR(10) NOT NULL UNIQUE, 
                    CompanyName VARCHAR(255),
                    Industry VARCHAR(255),
                    Sector VARCHAR(255),
                    Country VARCHAR(100)
                ); 
                CREATE TABLE IF NOT EXISTS dev.FactFundamentals (
                    StockKey INT NOT NULL,         
                    MarketCap BIGINT,
                    forwardPE FLOAT,
                    trailingPE FLOAT,
                    priceToBook FLOAT,
                    dividendYield FLOAT,
                    beta FLOAT,

                    FOREIGN KEY (StockKey) REFERENCES dev.DimStock(StockKey),

                    PRIMARY KEY (StockKey)
                );
                CREATE TABLE IF NOT EXISTS dev.FactStockPrice (
                    DateKey INT NOT NULL,
                    StockKey INT NOT NULL,

                    Open FLOAT,
                    High FLOAT,
                    Low FLOAT,
                    Close FLOAT,
                    Volume BIGINT,
                    
                    Returns FLOAT,
                    Volatility FLOAT,
                    Volume_MA7 FLOAT,
                    RSI FLOAT,
                    MACD FLOAT,

                    TechnicalFeatures JSONB,

                    FOREIGN KEY (DateKey) REFERENCES dev.DimDate(DateKey),
                    FOREIGN KEY (StockKey) REFERENCES dev.DimStock(StockKey),

                    PRIMARY KEY (DateKey, StockKey)
                );
                CREATE TABLE IF NOT EXISTS dev.FactMarketIndicators (
                    DateKey INT NOT NULL,
                    
                    VIX_Close FLOAT,
                    TNX_Close FLOAT,
                    Oil_Close FLOAT,
                    USD_Close FLOAT,
                    
                    Oil_Change FLOAT,
                    USD_Change FLOAT,
                    VIX_Change FLOAT,
                    TNX_Change FLOAT,

                    FOREIGN KEY (DateKey) REFERENCES dev.DimDate(DateKey),
                    PRIMARY KEY (DateKey)
                ); 
                CREATE TABLE IF NOT EXISTS staging.RawMarketIndicators (
                    RawDate DATE NOT NULL,
                    Ticker VARCHAR(10) NOT NULL, 
                    Open FLOAT,
                    High FLOAT,
                    Low FLOAT,
                    Close FLOAT,
                    Volume BIGINT,
                    PRIMARY KEY (RawDate, Ticker)
                );
                
                CREATE TABLE IF NOT EXISTS dev.DimStockModelConfig (
                    StockKey INT NOT NULL,
                    
                    ActiveFeatures TEXT[] NOT NULL, 
                    
                    LookBackWindow INT DEFAULT 60,
                    
                    ForecastHorizon INT DEFAULT 14,
                
                    CONSTRAINT pk_dim_stock_model_config PRIMARY KEY (StockKey),
                
                    CONSTRAINT fk_dim_stock_model_config_stock 
                        FOREIGN KEY (StockKey) 
                        REFERENCES dev.DimStock(StockKey) 
                        ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS dev.FactStockPrediction (                    
                    PredictionDateKey INT NOT NULL, 
                    StockKey INT NOT NULL,
                    
                    ForecastPrices DECIMAL[] NOT NULL, 
                    
                    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                    CONSTRAINT fk_prediction_stock 
                        FOREIGN KEY (StockKey) 
                        REFERENCES dev.DimStock(StockKey),
                        
                    CONSTRAINT fk_prediction_date 
                        FOREIGN KEY (PredictionDateKey) 
                        REFERENCES dev.DimDate(DateKey),
                
                    CONSTRAINT pk_fact_prediction PRIMARY KEY (PredictionDateKey, StockKey)
                );
                
                CREATE TABLE IF NOT EXISTS dev.FactModelPerformance (
                    PerformanceID SERIAL PRIMARY KEY,
                    StockKey INT NOT NULL,
                    
                    TrainingDateKey INT NOT NULL, 
                    
                    MAE FLOAT,
                    RMSE FLOAT,
                    MAPE FLOAT,
                    
                    LookBackWindow INT,
                    ForecastHorizon INT,
                    
                    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                    FOREIGN KEY (StockKey) REFERENCES dev.DimStock(StockKey),
                    FOREIGN KEY (TrainingDateKey) REFERENCES dev.DimDate(DateKey)
                );

                SELECT create_hypertable('staging.RawMarketIndicators', 'rawdate', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 year');
                SELECT create_hypertable('staging.RawStockPrice', 'rawdate', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 year');

                SELECT create_hypertable('dev.FactMarketIndicators', 'datekey', if_not_exists => TRUE, chunk_time_interval => 10000);
                SELECT create_hypertable('dev.FactStockPrice', 'datekey', if_not_exists => TRUE, chunk_time_interval => 10000);
                SELECT create_hypertable('dev.FactStockPrediction', 'predictiondatekey', if_not_exists => TRUE, chunk_time_interval => 10000);
                CREATE INDEX idx_perf_stock_date ON dev.FactModelPerformance(StockKey, TrainingDateKey);
                """
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
            conn.commit()
        print("Table created successfully.")
    except Exception as e:
        print(f"✗ Error processing batch: {e}")
    finally:
        if conn:
            connection_pool.putconn(conn)

def populate_dim_date(connection_pool, start_date_str='1990-01-01'):
    start_date = datetime.date.fromisoformat(start_date_str)
    today = datetime.date.today()
    end_date = datetime.date(today.year + 1, 12, 31)

    current_date = start_date
    date_records = []

    while current_date <= end_date:
        date_key = int(current_date.strftime("%Y%m%d"))
        date_records.append((
            date_key,
            current_date,
            current_date.year,
            current_date.month,
            current_date.day,
            current_date.weekday()
        ))
        current_date += datetime.timedelta(days=1)

    print("Populating dimension dates...")

    insert_query = """
                   INSERT INTO dev.DimDate (DateKey, FullDate, Year, Month, Day, Weekday)
                   VALUES %s ON CONFLICT (DateKey) DO NOTHING; \
                   """

    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            extras.execute_values(cursor, insert_query, date_records)
            conn.commit()
        print("Dimension dates populated successfully.")
    except Exception as e:
        print(f"Error populating dimension dates: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def populate_dim_stock(connection_pool, tickers):
    print("Inserting data into DimStock table...")

    all_fetched_data = []

    for ticker in tickers:
        result = etl.fetch_one_stock_info(ticker)
        if result is not None:
            all_fetched_data.append(result)

    if not all_fetched_data:
        print("No valid stock data fetched.")
        return

    dim_stock_records = []
    fact_fundamentals_records = {}
    dim_stock_model_config = {}

    for row in all_fetched_data:
        dim_stock_records.append((
            row['TickerSymbol'],
            row['CompanyName'],
            row['Industry'],
            row['Sector'],
            row['Country']
        ))

        fact_fundamentals_records[row['TickerSymbol']] = (
            safe_cast_to_int(row['MarketCap']),
            safe_cast_to_float(row['forwardPE']),
            safe_cast_to_float(row['trailingPE']),
            safe_cast_to_float(row['priceToBook']),
            safe_cast_to_float(row['dividendYield']),
            safe_cast_to_float(row['beta'])
        )

        if row['Sector'] in ['Energy', 'Industrials', 'Basic Materials', 'Consumer Cyclical', 'Consumer Defensive']:
            dim_stock_model_config[row['TickerSymbol']] = (
                ['Returns', 'Volatility', 'Volume_MA7', 'RSI', 'MACD', 'Oil_Change', 'Stock_Oil_Corr', 'Vol_x_Oil', 'USD_Change', 'VIX_Change'], 60, 14
            )
        elif row['Sector'] in ['Financial Services', 'Communication Services', 'Utilities', 'Healthcare']:
            dim_stock_model_config[row['TickerSymbol']] = (
                ['Returns', 'Volatility', 'Volume_MA7', 'RSI', 'MACD', 'Stock_Rate_Corr', 'USD_Change', 'VIX_Change',  'TNX_Change'], 60, 14
            )

    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            insert_query = """
                            INSERT INTO dev.DimStock (TickerSymbol, CompanyName, Industry, Sector, Country)
                            VALUES %s 
                            ON CONFLICT (TickerSymbol) DO NOTHING; 
                           """
            extras.execute_values(cursor, insert_query, dim_stock_records)

            cursor.execute("SELECT TickerSymbol, StockKey FROM dev.DimStock;")
            stock_key_map = dict(cursor.fetchall())

            fact_records_to_insert = []
            for ticker, data_tuple in fact_fundamentals_records.items():
                stock_key = stock_key_map.get(ticker)
                if stock_key:
                    record = (stock_key,) + data_tuple
                    fact_records_to_insert.append(record)

            dim_records_to_insert = []
            for ticker, config_tuple in dim_stock_model_config.items():
                stock_key = stock_key_map.get(ticker)
                if stock_key:
                    record = (stock_key,) + config_tuple
                    dim_records_to_insert.append(record)

            if fact_records_to_insert:
                insert_fact_query = """
                                    INSERT INTO dev.FactFundamentals (StockKey, MarketCap, forwardPE, trailingPE, priceToBook, dividendYield, beta)
                                    VALUES %s 
                                    ON CONFLICT (StockKey) DO UPDATE SET
                                        MarketCap = EXCLUDED.MarketCap,
                                        forwardPE = EXCLUDED.forwardPE,
                                        trailingPE = EXCLUDED.trailingPE,
                                        priceToBook = EXCLUDED.priceToBook,
                                        dividendYield = EXCLUDED.dividendYield,
                                        beta = EXCLUDED.beta;
                                   """
                extras.execute_values(cursor, insert_fact_query, fact_records_to_insert)

            if dim_records_to_insert:
                insert_dim_model_query = """
                                        INSERT INTO dev.DimStockModelConfig (StockKey, ActiveFeatures, LookBackWindow, ForecastHorizon)
                                        VALUES %s 
                                        ON CONFLICT (StockKey) DO UPDATE SET
                                            ActiveFeatures = EXCLUDED.ActiveFeatures,
                                            LookBackWindow = EXCLUDED.LookBackWindow,
                                            ForecastHorizon = EXCLUDED.ForecastHorizon;
                                       """
                extras.execute_values(cursor, insert_dim_model_query, dim_records_to_insert)
            conn.commit()
        print("DimStock, FactFundamentals, DimStockModelConfig tables populated successfully.")

    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def load_raw_historical_stock_prices(connection_pool, start_date='1990-01-01', retries=3):
    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT StockKey, TickerSymbol FROM dev.DimStock;""")
            stock_keys = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

    insert_query = """
                   INSERT INTO staging.RawStockPrice (Rawdate, Ticker, Open, High, Low, Close, Volume)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (RawDate, Ticker) DO NOTHING; \
                   """

    for stock_key, ticker in stock_keys:
        conn_worker = None
        try:
            conn_worker = connection_pool.getconn()
            stock_data = ar.download_and_format_stock_data(ticker, start_date, retries)

            if stock_data.empty:
                print(f"No data for {ticker}, skipping...")
                continue

            records_to_insert = []

            for _, row in stock_data.iterrows():
                records_to_insert.append((
                    row['date'].date(),
                    row['name'],
                    safe_cast_to_float(row['open']),
                    safe_cast_to_float(row['high']),
                    safe_cast_to_float(row['low']),
                    safe_cast_to_float(row['close']),
                    safe_cast_to_int(row['volume']),
                ))


            with conn_worker.cursor() as cursor:
                cursor.executemany(insert_query, records_to_insert)
                conn_worker.commit()

            print(f"Successfully loaded raw data for {ticker}.")
        except Exception as e:
            if conn_worker:
                conn_worker.rollback()
            print(f"Error loading raw data for {ticker}: {e}")
        finally:
            if conn_worker:
                connection_pool.putconn(conn_worker)

def load_raw_historical_market_indicators(connection_pool, start_date='1990-01-01', retries=3):
    stock_keys = ['^VIX', '^TNX', "CL=F", "DX-Y.NYB"]

    insert_query = """
                   INSERT INTO staging.RawMarketIndicators (Rawdate, Ticker, Open, High, Low, Close, Volume)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (RawDate, Ticker) DO NOTHING; \
                   """

    for ticker in stock_keys:
        conn_worker = None
        try:
            conn_worker = connection_pool.getconn()
            stock_data = ar.download_and_format_stock_data(ticker, start_date, retries)

            if stock_data.empty:
                print(f"No data for {ticker}, skipping...")
                continue

            records_to_insert = []

            for _, row in stock_data.iterrows():
                records_to_insert.append((
                    row['date'].date(),
                    row['name'],
                    safe_cast_to_float(row['open']),
                    safe_cast_to_float(row['high']),
                    safe_cast_to_float(row['low']),
                    safe_cast_to_float(row['close']),
                    safe_cast_to_int(row['volume']),
                ))

            with conn_worker.cursor() as cursor:
                cursor.executemany(insert_query, records_to_insert)
                conn_worker.commit()

            print(f"Successfully loaded raw data for {ticker}.")
        except Exception as e:
            if conn_worker:
                conn_worker.rollback()
            print(f"Error loading raw data for {ticker}: {e}")
        finally:
            if conn_worker:
                connection_pool.putconn(conn_worker)

def transform_historical_market_indicators(connection_pool, update=True):
    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT FullDate, DateKey
                              FROM dev.DimDate;""")
            date_map = {fulldate: datekey for fulldate, datekey in cursor.fetchall()}

            if not update:
                cursor.execute("""SELECT RawDate, Ticker, Close FROM staging.RawMarketIndicators WHERE Ticker IN ('^VIX', '^TNX', 'CL=F', 'DX-Y.NYB');""")
            else:
                query_limit_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
                cursor.execute("""
                               SELECT RawDate, Ticker, Close
                               FROM staging.RawMarketIndicators
                               WHERE Ticker IN ('^VIX', '^TNX', 'CL=F', 'DX-Y.NYB')
                                 AND RawDate >= %s
                               ORDER BY RawDate ASC;
                               """, (query_limit_date,))
            raw_data = cursor.fetchall()

        if not raw_data:
            return

        df_raw = pd.DataFrame(raw_data, columns=['date', 'ticker', 'close'])

        df_pivot = df_raw.pivot(index='date', columns='ticker', values='close')

        rename_map = {
            '^VIX': 'VIX_Close',
            '^TNX': 'TNX_Close',
            'CL=F': 'Oil_Close',
            'DX-Y.NYB': 'USD_Close',
        }
        df_pivot.rename(columns=rename_map, inplace=True)

        df_pivot = df_pivot.ffill()

        df_pivot['Oil_Change'] = df_pivot['Oil_Close'].pct_change()
        df_pivot['USD_Change'] = df_pivot['USD_Close'].pct_change()
        df_pivot['VIX_Change'] = df_pivot['VIX_Close'].pct_change()
        df_pivot['TNX_Change'] = df_pivot['TNX_Close'].pct_change()

        records_to_insert = []
        for date_val, row in df_pivot.iterrows():
            date_key = date_map.get(date_val)
            if date_key is None:
                continue

            records_to_insert.append((
                date_key,
                safe_cast_to_float(row.get('VIX_Close')),
                safe_cast_to_float(row.get('TNX_Close')),
                safe_cast_to_float(row.get('Oil_Close')),
                safe_cast_to_float(row.get('USD_Close')),
                safe_cast_to_float(row.get('USD_Change')),
                safe_cast_to_float(row.get('VIX_Change')),
                safe_cast_to_float(row.get('TNX_Change')),
                safe_cast_to_float(row.get('Oil_Change')),
            ))

        insert_query = """
                       INSERT INTO dev.FactMarketIndicators (DateKey, VIX_Close, TNX_Close, Oil_Close, USD_Close, USD_Change, VIX_Change, TNX_Change, Oil_Change)
                       VALUES %s
                       ON CONFLICT (DateKey) DO UPDATE SET
                           VIX_Close = EXCLUDED.VIX_Close,
                           TNX_Close = EXCLUDED.TNX_Close,
                            Oil_Close = EXCLUDED.Oil_Close,
                            USD_Close = EXCLUDED.USD_Close,
                            USD_Change = EXCLUDED.USD_Change,
                            VIX_Change = EXCLUDED.VIX_Change,
                            TNX_Change = EXCLUDED.TNX_Change,
                            Oil_Change = EXCLUDED.Oil_Change;
                       """
        with conn.cursor() as cursor:
            extras.execute_values(cursor, insert_query, records_to_insert)
            conn.commit()
        print("Historical market indicators transformed successfully.")

    except Exception as e:
        print(f"Error fetching raw market indicator data: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)


def transform_historical_stock_prices(connection_pool, update=True):
    print("Starting transformation pipeline...")
    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT StockKey, TickerSymbol, Sector
                              FROM dev.DimStock;""")
            stock_keys = cursor.fetchall()

            if not update:
                cursor.execute("""SELECT DateKey, VIX_Close, TNX_Close, Oil_Close, USD_Close, Oil_Change, USD_Change, VIX_Change, TNX_Change
                                  FROM dev.FactMarketIndicators;""")
            else:
                limit_date_obj = datetime.date.today() - datetime.timedelta(days=365)
                limit_date_key = int(limit_date_obj.strftime('%Y%m%d'))  # Kết quả: 20231204

                cursor.execute("""
                               SELECT DateKey, VIX_Close, TNX_Close, Oil_Close, USD_Close, 
                                      Oil_Change, USD_Change, VIX_Change, TNX_Change
                               FROM dev.FactMarketIndicators
                               WHERE DateKey >= %s
                               ORDER BY DateKey ASC;
                               """, (limit_date_key,))
            market_data = cursor.fetchall()

            cursor.execute("""SELECT FullDate, DateKey
                              FROM dev.DimDate;""")
            date_map = {fulldate: datekey for fulldate, datekey in cursor.fetchall()}
    finally:
        if conn:
            connection_pool.putconn(conn)

    try:
        df_market = pd.DataFrame(market_data, columns=['DateKey', 'VIX_Close', 'TNX_Close', 'Oil_Close', 'USD_Close', 'Oil_Change', 'USD_Change', 'VIX_Change', 'TNX_Change'])

        df_market = df_market.sort_values('DateKey')

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for stock_key, ticker, sector in stock_keys:
                f = executor.submit(
                    etl.process_one_stock_transformation,
                    connection_pool,
                    stock_key,
                    ticker,
                    sector,
                    df_market,
                    date_map,
                    update
                )
                futures.append(f)

            for future in futures:
                future.result()
        print("All historical stock prices transformed and loaded successfully.")
    except Exception as e:
        print(f"Error transforming historical stock prices: {e}")
        raise


def train_all_stocks(connection_pool):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled Memory Growth for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)

    conn = None
    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT StockKey, TickerSymbol FROM dev.DimStock;""")
            stock_list = cursor.fetchall()

            cursor.execute("""SELECT DateKey, Oil_Change, USD_Change, VIX_Change, TNX_Change
                              FROM dev.FactMarketIndicators
                              ORDER BY DateKey ASC""")
            market_data = cursor.fetchall()
            market_columns = ['datekey', 'oil_change', 'usd_change', 'vix_change', 'tnx_change']
            df_market = pd.DataFrame(market_data, columns=market_columns)

            df_market['datekey'] = df_market['datekey'].astype(int)

    except Exception as e:
        print(f"Error fetching stock list for training: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

    total_stocks = len(stock_list)
    print(f"Found {total_stocks} stocks to train.")

    today = datetime.date.today()
    training_date_key = int(today.strftime('%Y%m%d'))

    for i, (stock_key, ticker) in enumerate(stock_list):
        print(f"\n[{i + 1}/{total_stocks}] Processing {ticker}...")

        etl.train_one_stock_model(connection_pool, stock_key, ticker, df_market, training_date_key)

        import tensorflow.keras.backend as K
        import gc
        K.clear_session()
        gc.collect()

def predict_all_stocks(connection_pool):
    conn = None

    today = datetime.date.today()
    prediction_date_key = int(today.strftime('%Y%m%d'))

    try:
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("""SELECT StockKey, TickerSymbol FROM dev.DimStock;""")
            stock_list = cursor.fetchall()

            cursor.execute("""SELECT DateKey, Oil_Change, USD_Change, VIX_Change, TNX_Change
                              FROM dev.FactMarketIndicators
                              ORDER BY DateKey ASC""")
            market_data = cursor.fetchall()
            market_columns = ['datekey', 'oil_change', 'usd_change', 'vix_change', 'tnx_change']
            df_market = pd.DataFrame(market_data, columns=market_columns)
            df_market['datekey'] = df_market['datekey'].astype(int)
    except Exception as e:
        print(f"Error fetching stock list for prediction: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

    total_stocks = len(stock_list)
    print(f"Found {total_stocks} stocks to predict.")

    for i, (stock_key, ticker) in enumerate(stock_list):
        print(f"\n[{i + 1}/{total_stocks}] Predicting for {ticker}...")

        etl.predict_one_stock(connection_pool, stock_key, ticker, df_market, prediction_date_key)
