import psycopg2
from psycopg2 import pool
import streamlit as st
import pandas as pd

# Import cấu hình từ file config
try:
    from config import (
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA
    )
except ImportError:
    # Cấu hình mặc định nếu không tìm thấy file config
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "dev"
    DB_USER = "db_user"
    DB_PASSWORD = "db_password"
    DB_SCHEMA = "dev"

# Database configuration
# Sử dụng cloudflared tunnel: cloudflared access tcp --hostname timescaledb.tranvienduyhung.id.vn --url localhost:5432
DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "options": f"-c search_path={DB_SCHEMA}"
}


def update_db_config(host=None, port=None, database=None, user=None, password=None, schema=None):
    """Cập nhật cấu hình database"""
    global DB_CONFIG
    if host:
        DB_CONFIG["host"] = host
    if port:
        DB_CONFIG["port"] = int(port)
    if database:
        DB_CONFIG["database"] = database
    if user:
        DB_CONFIG["user"] = user
    if password:
        DB_CONFIG["password"] = password
    if schema:
        DB_CONFIG["schema"] = schema


@st.cache_resource
def init_connection_pool():
    """Khởi tạo connection pool"""
    try:
        connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            **DB_CONFIG
        )
        return connection_pool
    except Exception as e:
        st.error(f"Không thể tạo connection pool: {e}")
        return None


def get_connection(host=None, port=None, database=None, user=None, password=None):
    """Lấy connection - có thể truyền tham số hoặc dùng config mặc định"""
    try:
        config = {
            "host": host or DB_CONFIG["host"],
            "port": int(port) if port else DB_CONFIG["port"],
            "database": database or DB_CONFIG["database"],
            "user": user or DB_CONFIG["user"],
            "password": password or DB_CONFIG["password"],
            "options": "-c search_path=dev",  # Set schema dev
        }
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        st.error(f"Lỗi kết nối database: {e}")
        return None


def close_connection(conn):
    """Đóng connection"""
    if conn:
        conn.close()


def execute_query(query, params=None):
    """Thực thi query và trả về kết quả"""
    conn = get_connection()
    if not conn:
        return {"error": "Không thể kết nối database"}

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)

        # Nếu là SELECT query, lấy kết quả
        if query.strip().upper().startswith("SELECT"):
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            close_connection(conn)
            return {"columns": columns, "data": results}
        else:
            conn.commit()
            cursor.close()
            close_connection(conn)
            return {"status": "success"}

    except Exception as e:
        close_connection(conn)
        return {"error": str(e)}


def get_tables():
    """Lấy danh sách tables trong schema dev"""
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'dev'
        ORDER BY table_name;
    """
    result = execute_query(query)
    if result and "data" in result:
        return [row[0] for row in result["data"]]
    return []


def get_table_data(table_name, limit=100):
    """Lấy dữ liệu từ một table trong schema dev"""
    query = f"SELECT * FROM dev.{table_name} LIMIT %s"
    return execute_query(query, (limit,))


def test_connection():
    """Kiểm tra kết nối database"""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            close_connection(conn)
            return True
        except:
            close_connection(conn)
            return False
    return False


# ==================== STOCK DATA QUERIES ====================

def get_all_stocks():
    """Lấy danh sách tất cả cổ phiếu từ dimstock"""
    query = """
        SELECT stockkey, tickersymbol, companyname, industry, sector, country
        FROM dev.dimstock
        ORDER BY tickersymbol;
    """
    result = execute_query(query)
    if result and "data" in result:
        return pd.DataFrame(result["data"], columns=result["columns"])
    return pd.DataFrame()


def get_stock_price(stock_key, start_date=None, end_date=None, limit=365):
    """Lấy giá cổ phiếu từ factstockprice"""
    # Convert numpy.int64 to Python int
    stock_key = int(stock_key)

    query = """
        SELECT 
            sp.datekey,
            d.fulldate as date,
            sp.open,
            sp.high,
            sp.low,
            sp.close,
            sp.volume
        FROM dev.factstockprice sp
        JOIN dev.dimdate d ON sp.datekey = d.datekey
        WHERE sp.stockkey = %s
    """
    params = [stock_key]

    if start_date:
        query += " AND d.fulldate >= %s"
        params.append(start_date)
    if end_date:
        query += " AND d.fulldate <= %s"
        params.append(end_date)

    query += " ORDER BY d.fulldate DESC LIMIT %s"
    params.append(limit)

    result = execute_query(query, tuple(params))

    # Debug: hiển thị lỗi nếu có
    if result and "error" in result:
        st.error(f"❌ Query error: {result['error']}")
        return pd.DataFrame()

    if result and "data" in result:
        df = pd.DataFrame(result["data"], columns=result["columns"])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    return pd.DataFrame()


def get_stock_predictions(stock_key):
    """Lấy dự đoán giá cổ phiếu từ factstockprediction"""
    stock_key = int(stock_key)
    query = """
        SELECT 
            sp.predictiondatekey,
            d.fulldate as prediction_date,
            sp.forecastprices,
            sp.createdat
        FROM dev.factstockprediction sp
        JOIN dev.dimdate d ON sp.predictiondatekey = d.datekey
        WHERE sp.stockkey = %s
        ORDER BY sp.createdat DESC
        LIMIT 1;
    """
    result = execute_query(query, (stock_key,))
    if result and "data" in result:
        return pd.DataFrame(result["data"], columns=result["columns"])
    return pd.DataFrame()


def get_stock_fundamentals(stock_key):
    """Lấy thông tin cơ bản từ factfundamentals"""
    stock_key = int(stock_key)
    query = """
        SELECT 
            marketcap, forwardpe, trailingpe, pricetobook, dividendyield, beta
        FROM dev.factfundamentals
        WHERE stockkey = %s;
    """
    result = execute_query(query, (stock_key,))
    if result and "data" in result:
        return pd.DataFrame(result["data"], columns=result["columns"])
    return pd.DataFrame()


def get_model_performance(stock_key):
    """Lấy thông tin hiệu suất mô hình từ factmodelperformance"""
    stock_key = int(stock_key)
    query = """
        SELECT 
            performanceid, trainingdatekey, mae, rmse, mape, 
            lookbackwindow, forecasthorizon, createdat
        FROM dev.factmodelperformance
        WHERE stockkey = %s
        ORDER BY createdat DESC;
    """
    result = execute_query(query, (stock_key,))
    if result and "data" in result:
        return pd.DataFrame(result["data"], columns=result["columns"])
    return pd.DataFrame()


def get_market_indicators(start_date=None, end_date=None, limit=365):
    """Lấy các chỉ số thị trường từ factmarketindicators"""
    query = """
        SELECT 
            mi.datekey,
            d.fulldate as date,
            mi.oilprice,
            mi.vixindex,
            mi.usdindex,
            mi.tnxrate
        FROM dev.factmarketindicators mi
        JOIN dev.dimdate d ON mi.datekey = d.datekey
    """
    params = []

    if start_date:
        query += " WHERE d.fulldate >= %s"
        params.append(start_date)
        if end_date:
            query += " AND d.fulldate <= %s"
            params.append(end_date)
    elif end_date:
        query += " WHERE d.fulldate <= %s"
        params.append(end_date)

    query += " ORDER BY d.fulldate DESC LIMIT %s"
    params.append(limit)

    result = execute_query(query, tuple(params) if params else (limit,))
    if result and "data" in result:
        df = pd.DataFrame(result["data"], columns=result["columns"])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    return pd.DataFrame()


