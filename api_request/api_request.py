import yfinance as yf
import pandas as pd

def fetch_data():
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df = pd.read_csv(nasdaq_url, sep='|')
    tickers = df['Symbol'].tolist()

    return tickers

def get_all_tickers():
    df = pd.read_csv('all_stock_name.csv')
    return df['name'].unique().tolist()

def download_and_format_stock_data(ticker, start_date='1990-01-01', retries=3):
    for i in range(retries):
        try:
            stock_data = yf.download(ticker, start=start_date, auto_adjust=True)
            if not stock_data.empty:
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)
                stock_data.reset_index(inplace=True)
                stock_data['Name'] = ticker
                stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name']]
                stock_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'name']

                return stock_data
        except Exception as e:
            print(f"Attempt {i+1} failed for {ticker}: {e}")
    return pd.DataFrame()

def get_full_stock_dimensions(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)

        info = ticker_obj.info

        dim_data = {
            'TickerSymbol': ticker,
            'CompanyName': info.get('longName') or 'N/A',
            'Industry': info.get('industry') or 'Unknown',
            'Sector': info.get('sector') or 'Unknown',
            'Country': info.get('country') or 'Unknown',
            'MarketCap': info.get('marketCap'),
            'forwardPE': info.get('forwardPE'),
            'trailingPE': info.get('trailingPE'),
            'priceToBook': info.get('priceToBook'),
            'dividendYield': info.get('dividendYield'),
            'beta': info.get('beta')
        }

        return pd.DataFrame([dim_data])

    except Exception as e:
        return pd.DataFrame()