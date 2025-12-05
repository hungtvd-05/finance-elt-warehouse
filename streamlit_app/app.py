import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Thêm thư mục gốc vào sys.path để import từ models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import cấu hình
try:
    from config import (
        DEFAULT_DATA_DAYS, DEFAULT_PREDICTION_DAYS,
        CURRENCY_SYMBOL, COLOR_UP, COLOR_DOWN
    )
except ImportError:
    DEFAULT_DATA_DAYS = 365
    DEFAULT_PREDICTION_DAYS = 14
    CURRENCY_SYMBOL = "$"
    COLOR_UP = "#26a69a"
    COLOR_DOWN = "#ef5350"

# Import database với xử lý lỗi
try:
    from database import (
        get_connection, close_connection, update_db_config, test_connection,
        get_all_stocks, get_stock_price, get_stock_predictions,
        get_stock_fundamentals, get_model_performance, get_market_indicators
    )
    DB_AVAILABLE = True
except Exception as e:
    DB_AVAILABLE = False
    DB_ERROR = str(e)

# Import models cho prediction từ thư mục models chính
try:
    from models import predict_linear_regression, predict_arima, predict_random_forest
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    MODELS_ERROR = str(e)

    # Tạo fallback functions khi import thất bại
    def predict_linear_regression(df, days_to_predict=14):
        raise ImportError(f"Không thể import models: {MODELS_ERROR}")

    def predict_arima(df, days_to_predict=14):
        raise ImportError(f"Không thể import models: {MODELS_ERROR}")

    def predict_random_forest(df, days_to_predict=14):
        raise ImportError(f"Không thể import models: {MODELS_ERROR}")

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích và dự đoán giá cổ phiếu",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện chuyên nghiệp
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: bold;
    }
    
    /* Card container */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Success/Error boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.4);
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def get_stock_data_from_db(stock_key, days=180):
    """Lấy dữ liệu stock từ database"""
    if DB_AVAILABLE and stock_key:
        try:
            df = get_stock_price(stock_key, limit=days)
            if not df.empty:
                return df
        except Exception as e:
            st.error(f"❌ Lỗi kết nối database: {e}")

    # Trả về DataFrame rỗng nếu không có dữ liệu
    return pd.DataFrame()


def get_fundamentals_from_db(stock_key):
    """Lấy thông tin fundamentals từ database"""
    if DB_AVAILABLE and stock_key:
        try:
            return get_stock_fundamentals(stock_key)
        except:
            pass
    return pd.DataFrame()


def get_predictions_from_db(stock_key):
    """Lấy dự đoán từ database"""
    if DB_AVAILABLE and stock_key:
        try:
            return get_stock_predictions(stock_key)
        except:
            pass
    return pd.DataFrame()


def calculate_moving_averages(df, periods=[5, 10, 20, 50]):
    """Tính các đường trung bình động"""
    for period in periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_ema(df, periods=[12, 26]):
    """Tính Exponential Moving Average"""
    for period in periods:
        df[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df


def calculate_rsi(df, period=14):
    """Tính RSI (Relative Strength Index)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df):
    """Tính MACD"""
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Tính Bollinger Bands"""
    df['BB_Middle'] = df['close'].rolling(window=period).mean()
    df['BB_Std'] = df['close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)
    return df




def main():
    # Header
    st.markdown('<h1 class="main-header">Hệ thống phân tích và dự đoán giá cổ phiếu</h1>', unsafe_allow_html=True)
    # Sidebar
    with st.sidebar:
        st.markdown("### Điều hướng")

        page = st.radio(
            "",
            ["Dashboard", "Phân tích kỹ thuật", "Biểu đồ nâng cao", "Dự đoán AI", "Cài đặt"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stock selector - Lấy từ database
        st.markdown("### Chọn cổ phiếu")

        # Lấy danh sách cổ phiếu từ database
        stocks_df = pd.DataFrame()
        if DB_AVAILABLE:
            try:
                stocks_df = get_all_stocks()
            except:
                pass

        if not stocks_df.empty:
            # Tạo dict mapping ticker -> stockkey
            stock_options = stocks_df['tickersymbol'].tolist()
            st.session_state['stocks_df'] = stocks_df
            selected_stock = st.selectbox(
                "Mã cổ phiếu:",
                stock_options,
                format_func=lambda x: f"{x} - {stocks_df[stocks_df['tickersymbol']==x]['companyname'].values[0][:30]}..."
                    if len(stocks_df[stocks_df['tickersymbol']==x]['companyname'].values[0]) > 30
                    else f"{x} - {stocks_df[stocks_df['tickersymbol']==x]['companyname'].values[0]}"
            )
            # Lưu stockkey
            st.session_state['selected_stockkey'] = stocks_df[stocks_df['tickersymbol']==selected_stock]['stockkey'].values[0]
        else:
            st.error("Không thể lấy danh sách cổ phiếu. Vui lòng kiểm tra kết nối database!")
            selected_stock = None
            st.session_state['selected_stockkey'] = None

        st.markdown("---")
        
        # Database status
        st.markdown("### Cơ sở dữ liệu")
        if DB_AVAILABLE:
            try:
                if test_connection():
                    st.success("Đã kết nối")
                else:
                    st.warning("Module OK, chưa kết nối")
            except:
                st.warning("Module OK, chưa kết nối")
        else:
            st.error("Chưa cài module")

        st.markdown("---")
        st.markdown("**VKU Finance Project**")
        st.caption("Version 1.0.0")

    # Store selected stock in session state
    st.session_state['selected_stock'] = selected_stock

    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard_page()
    elif page == "Phân tích kỹ thuật":
        show_technical_analysis_page()
    elif page == "Biểu đồ nâng cao":
        show_advanced_charts_page()
    elif page == "Dự đoán AI":
        show_prediction_page()
    elif page == "Cài đặt":
        show_settings_page()


def show_dashboard_page():
    selected_stock = st.session_state.get('selected_stock', None)
    stock_key = st.session_state.get('selected_stockkey', None)

    if not stock_key:
        st.error("Vui lòng chọn cổ phiếu từ sidebar. Kiểm tra kết nối database!")
        return

    # Lấy dữ liệu từ database
    df = get_stock_data_from_db(stock_key, 365)

    if df.empty:
        st.error("Không có dữ liệu cho cổ phiếu này. Vui lòng kiểm tra kết nối database!")
        return

    st.caption(f"Dữ liệu từ Database - Schema: dev - {len(df)} bản ghi")

    df = calculate_moving_averages(df)
    
    # Top metrics
    st.markdown("### Tổng quan thị trường")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0

    with col1:
        st.metric(
            label=f"{selected_stock}",
            value=f"${current_price:,.2f}",
            delta=f"{change_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Cao nhất (52W)",
            value=f"${df['high'].max():,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Thấp nhất (52W)",
            value=f"${df['low'].min():,.2f}",
            delta=None
        )
    
    with col4:
        avg_volume = df['volume'].mean()
        st.metric(
            label="KL Trung bình",
            value=f"{avg_volume/1000000:.2f}M",
            delta=None
        )
    
    with col5:
        volatility = df['close'].pct_change().std() * 100
        st.metric(
            label="Biến động",
            value=f"{volatility:.2f}%",
            delta=None
        )
    
    st.markdown("---")
    
    # Main chart and info
    col_chart, col_info = st.columns([2, 1])
    
    with col_chart:
        st.markdown("### Biểu đồ giá & Trung bình động")

        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Moving averages
        colors = {'MA5': '#FF6B6B', 'MA10': '#4ECDC4', 'MA20': '#45B7D1', 'MA50': '#96CEB4'}
        for ma in ['MA5', 'MA10', 'MA20']:
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[ma],
                    mode='lines',
                    name=ma,
                    line=dict(color=colors[ma], width=1.5)
                ))
        
        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_info:
        st.markdown("### Thông tin cổ phiếu")
        
        # Lấy thông tin company từ stocks_df
        stocks_df = st.session_state.get('stocks_df', pd.DataFrame())
        company_name = selected_stock
        industry = "N/A"
        sector = "N/A"

        if not stocks_df.empty:
            stock_info = stocks_df[stocks_df['tickersymbol'] == selected_stock]
            if not stock_info.empty:
                company_name = stock_info['companyname'].values[0]
                industry = stock_info['industry'].values[0]
                sector = stock_info['sector'].values[0]

        # Stock info card
        st.markdown(f"""
        <div class="info-card">
            <h4>{selected_stock}</h4>
            <p><strong>Công ty:</strong> {company_name[:40]}...</p>
            <p><strong>Ngành:</strong> {industry}</p>
            <p><strong>Lĩnh vực:</strong> {sector}</p>
            <p><strong>Giá hiện tại:</strong> ${current_price:,.2f}</p>
            <p><strong>Thay đổi:</strong> <span style="color: {'green' if change >= 0 else 'red'}">${change:+,.2f} ({change_pct:+.2f}%)</span></p>
            <p><strong>Khối lượng:</strong> {df['volume'].iloc[-1]:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Hiển thị Fundamentals từ database
        fundamentals = get_fundamentals_from_db(stock_key)
        if not fundamentals.empty:
            st.markdown("#### Chỉ số cơ bản")
            fund = fundamentals.iloc[0]

            market_cap = fund.get('marketcap', 0)
            if market_cap:
                if market_cap >= 1e12:
                    mc_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_str = f"${market_cap/1e9:.2f}B"
                else:
                    mc_str = f"${market_cap/1e6:.2f}M"
            else:
                mc_str = "N/A"

            pe_forward = f"{fund.get('forwardpe'):.2f}" if fund.get('forwardpe') else 'N/A'
            pe_trailing = f"{fund.get('trailingpe'):.2f}" if fund.get('trailingpe') else 'N/A'
            pb = f"{fund.get('pricetobook'):.2f}" if fund.get('pricetobook') else 'N/A'
            div_yield = f"{fund.get('dividendyield'):.2f}%" if fund.get('dividendyield') else 'N/A'
            beta_val = f"{fund.get('beta'):.3f}" if fund.get('beta') else 'N/A'

            st.markdown(f"""
            - **Vốn hóa:** {mc_str}
            - **P/E (Dự phóng):** {pe_forward}
            - **P/E (Hiện tại):** {pe_trailing}
            - **P/B:** {pb}
            - **Tỷ suất cổ tức:** {div_yield}
            - **Beta:** {beta_val}
            """, unsafe_allow_html=True)

        st.markdown("#### Chỉ số kỹ thuật")

        # Calculate indicators
        df = calculate_rsi(df)
        rsi_value = df['RSI'].iloc[-1]
        
        rsi_color = "green" if 30 < rsi_value < 70 else ("red" if rsi_value >= 70 else "blue")
        rsi_signal = "Trung tính" if 30 < rsi_value < 70 else ("Quá mua" if rsi_value >= 70 else "Quá bán")
        
        st.markdown(f"""
        - **RSI (14):** <span style="color: {rsi_color}">{rsi_value:.1f} ({rsi_signal})</span>
        - **MA5:** ${df['MA5'].iloc[-1]:,.2f}
        - **MA10:** ${df['MA10'].iloc[-1]:,.2f}
        - **MA20:** ${df['MA20'].iloc[-1]:,.2f}
        """, unsafe_allow_html=True)
        
        # Signal
        ma5 = df['MA5'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        
        if current_price > ma5 > ma20:
            signal = "MUA"
            signal_color = "#28a745"
        elif current_price < ma5 < ma20:
            signal = "BÁN"
            signal_color = "#dc3545"
        else:
            signal = "GIỮ"
            signal_color = "#ffc107"
        
        st.markdown(f"""
        <div style="background-color: {signal_color}20; border-left: 4px solid {signal_color}; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <h4 style="margin: 0;">Tín hiệu: {signal}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Volume chart
    st.markdown("### Biểu đồ khối lượng giao dịch")

    fig_volume = go.Figure()
    
    colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
              for i in range(len(df))]
    
    fig_volume.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        marker_color=colors,
        name='Khối lượng'
    ))
    
    fig_volume.update_layout(
        height=250,
        template='plotly_white',
        margin=dict(l=0, r=0, t=10, b=0)
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)


def show_technical_analysis_page():
    selected_stock = st.session_state.get('selected_stock', None)
    stock_key = st.session_state.get('selected_stockkey', None)

    st.markdown(f"### Phân tích kỹ thuật - {selected_stock}")

    if not stock_key:
        st.error("Vui lòng chọn cổ phiếu từ sidebar!")
        return

    # Lấy dữ liệu từ database
    df = get_stock_data_from_db(stock_key, 365)

    if df.empty:
        st.error("Không có dữ liệu cho cổ phiếu này!")
        return

    st.caption(f"Dữ liệu từ Database - {len(df)} bản ghi")

    df = calculate_moving_averages(df, [5, 10, 20, 50])
    df = calculate_ema(df, [12, 26])
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Trung bình động", "RSI & MACD", "Dải Bollinger", "Tổng hợp"])

    with tab1:
        st.markdown("#### Biểu đồ trung bình động (Moving Averages)")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("**Chọn MA hiển thị:**")
            show_ma5 = st.checkbox("MA5 (5 ngày)", value=True)
            show_ma10 = st.checkbox("MA10 (10 ngày)", value=True)
            show_ma20 = st.checkbox("MA20 (20 ngày)", value=True)
            show_ma50 = st.checkbox("MA50 (50 ngày)", value=False)
            show_ema12 = st.checkbox("EMA12", value=False)
            show_ema26 = st.checkbox("EMA26", value=False)
        
        with col1:
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['close'],
                mode='lines', name='Giá đóng cửa',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Moving averages
            ma_config = {
                'MA5': (show_ma5, '#FF6B6B'),
                'MA10': (show_ma10, '#4ECDC4'),
                'MA20': (show_ma20, '#45B7D1'),
                'MA50': (show_ma50, '#96CEB4'),
                'EMA12': (show_ema12, '#FF9F43'),
                'EMA26': (show_ema26, '#A55EEA')
            }
            
            for ma, (show, color) in ma_config.items():
                if show and ma in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df[ma],
                        mode='lines', name=ma,
                        line=dict(color=color, width=1.5, dash='dash' if 'EMA' in ma else 'solid')
                    ))
            
            fig.update_layout(
                height=500,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Ngày",
                yaxis_title="Giá (USD)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.info("""
        **Giải thích Moving Averages:**
        - **MA (Simple Moving Average):** Trung bình cộng giá đóng cửa trong N ngày
        - **EMA (Exponential Moving Average):** Trung bình có trọng số, ưu tiên giá gần đây hơn
        - **Tín hiệu MUA:** Giá cắt lên trên MA / MA ngắn hạn cắt lên MA dài hạn
        - **Tín hiệu BÁN:** Giá cắt xuống dưới MA / MA ngắn hạn cắt xuống MA dài hạn
        """)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RSI (Relative Strength Index)")
            
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=df['date'], y=df['RSI'],
                mode='lines', name='RSI',
                line=dict(color='#8B5CF6', width=2)
            ))
            
            # Overbought/Oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá mua (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá bán (30)")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            
            fig_rsi.update_layout(
                height=300,
                template='plotly_white',
                yaxis=dict(range=[0, 100]),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            current_rsi = df['RSI'].iloc[-1]
            if current_rsi >= 70:
                st.error(f"RSI = {current_rsi:.1f} - Vùng QUÁ MUA, cân nhắc chốt lời")
            elif current_rsi <= 30:
                st.success(f"RSI = {current_rsi:.1f} - Vùng QUÁ BÁN, cơ hội mua vào")
            else:
                st.info(f"RSI = {current_rsi:.1f} - Vùng TRUNG TÍNH")

        with col2:
            st.markdown("#### MACD (Moving Average Convergence Divergence)")
            
            fig_macd = make_subplots(rows=1, cols=1)
            
            # MACD line
            fig_macd.add_trace(go.Scatter(
                x=df['date'], y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color='#3B82F6', width=2)
            ))
            
            # Signal line
            fig_macd.add_trace(go.Scatter(
                x=df['date'], y=df['Signal'],
                mode='lines', name='Signal',
                line=dict(color='#EF4444', width=2)
            ))
            
            # Histogram
            colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Histogram']]
            fig_macd.add_trace(go.Bar(
                x=df['date'], y=df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors
            ))
            
            fig_macd.update_layout(
                height=300,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
            
            macd_val = df['MACD'].iloc[-1]
            signal_val = df['Signal'].iloc[-1]
            if macd_val > signal_val:
                st.success(f"MACD ({macd_val:.2f}) > Signal ({signal_val:.2f}) - Xu hướng TĂNG")
            else:
                st.error(f"MACD ({macd_val:.2f}) < Signal ({signal_val:.2f}) - Xu hướng GIẢM")

    with tab3:
        st.markdown("#### Dải Bollinger")

        fig_bb = go.Figure()
        
        # Upper band
        fig_bb.add_trace(go.Scatter(
            x=df['date'], y=df['BB_Upper'],
            mode='lines', name='Dải trên',
            line=dict(color='rgba(250, 128, 114, 0.7)', width=1)
        ))
        
        # Lower band
        fig_bb.add_trace(go.Scatter(
            x=df['date'], y=df['BB_Lower'],
            mode='lines', name='Dải dưới',
            line=dict(color='rgba(250, 128, 114, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(250, 128, 114, 0.1)'
        ))
        
        # Middle band (SMA20)
        fig_bb.add_trace(go.Scatter(
            x=df['date'], y=df['BB_Middle'],
            mode='lines', name='Dải giữa (SMA20)',
            line=dict(color='#FF6B6B', width=1.5, dash='dash')
        ))
        
        # Price
        fig_bb.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines', name='Giá đóng cửa',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_bb.update_layout(
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Ngày",
            yaxis_title="Giá (USD)"
        )
        
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Analysis
        current_price = df['close'].iloc[-1]
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_middle = df['BB_Middle'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dải trên", f"${bb_upper:,.2f}")
        with col2:
            st.metric("Dải giữa", f"${bb_middle:,.2f}")
        with col3:
            st.metric("Dải dưới", f"${bb_lower:,.2f}")

        if current_price > bb_upper:
            st.warning("Giá đang ở TRÊN dải Bollinger - Có thể quá mua")
        elif current_price < bb_lower:
            st.success("Giá đang ở DƯỚI dải Bollinger - Có thể quá bán")
        else:
            st.info("Giá đang ở TRONG dải Bollinger - Biến động bình thường")

    with tab4:
        st.markdown("#### Bảng tổng hợp chỉ số kỹ thuật")

        # Summary table
        indicators = {
            "Chỉ số": ["Giá hiện tại", "MA5", "MA10", "MA20", "MA50", "EMA12", "EMA26", "RSI(14)", "MACD", "BB Upper", "BB Lower"],
            "Giá trị": [
                f"${df['close'].iloc[-1]:,.2f}",
                f"${df['MA5'].iloc[-1]:,.2f}",
                f"${df['MA10'].iloc[-1]:,.2f}",
                f"${df['MA20'].iloc[-1]:,.2f}",
                f"${df['MA50'].iloc[-1]:,.2f}" if not pd.isna(df['MA50'].iloc[-1]) else "N/A",
                f"${df['EMA12'].iloc[-1]:,.2f}",
                f"${df['EMA26'].iloc[-1]:,.2f}",
                f"{df['RSI'].iloc[-1]:.1f}",
                f"{df['MACD'].iloc[-1]:,.2f}",
                f"${df['BB_Upper'].iloc[-1]:,.2f}",
                f"${df['BB_Lower'].iloc[-1]:,.2f}"
            ],
            "Tín hiệu": [
                "—",
                "MUA" if df['close'].iloc[-1] > df['MA5'].iloc[-1] else "BÁN",
                "MUA" if df['close'].iloc[-1] > df['MA10'].iloc[-1] else "BÁN",
                "MUA" if df['close'].iloc[-1] > df['MA20'].iloc[-1] else "BÁN",
                "MUA" if df['close'].iloc[-1] > df['MA50'].iloc[-1] else "BÁN" if not pd.isna(df['MA50'].iloc[-1]) else "N/A",
                "MUA" if df['close'].iloc[-1] > df['EMA12'].iloc[-1] else "BÁN",
                "MUA" if df['close'].iloc[-1] > df['EMA26'].iloc[-1] else "BÁN",
                "Quá mua" if df['RSI'].iloc[-1] > 70 else ("Quá bán" if df['RSI'].iloc[-1] < 30 else "Trung tính"),
                "MUA" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "BÁN",
                "—",
                "—"
            ]
        }
        
        st.dataframe(pd.DataFrame(indicators), use_container_width=True, hide_index=True)


def show_advanced_charts_page():
    selected_stock = st.session_state.get('selected_stock', None)
    stock_key = st.session_state.get('selected_stockkey', None)

    st.markdown(f"### Biểu đồ nâng cao - {selected_stock}")

    if not stock_key:
        st.error("Vui lòng chọn cổ phiếu từ sidebar!")
        return

    # Lấy dữ liệu từ database
    df = get_stock_data_from_db(stock_key, 365)

    if df.empty:
        st.error("Không có dữ liệu cho cổ phiếu này!")
        return

    st.caption(f"Dữ liệu từ Database - {len(df)} bản ghi")

    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    
    tab1, tab2, tab3 = st.tabs(["Biểu đồ nến", "So sánh cổ phiếu", "Phân tích biến động"])

    with tab1:
        st.markdown("#### Biểu đồ nến Nhật Bản")

        # Combined chart with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Giá & MA', 'Khối lượng', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # MA lines
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='#FF6B6B', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='#45B7D1', width=1)), row=1, col=1)
        
        # Volume
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Khối lượng', marker_color=colors), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='#8B5CF6', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        fig.update_yaxes(title_text="Giá (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### So sánh hiệu suất các cổ phiếu")
        
        # Lấy danh sách cổ phiếu từ database
        stocks_df = st.session_state.get('stocks_df', pd.DataFrame())
        if not stocks_df.empty:
            stock_options = stocks_df['tickersymbol'].tolist()
        if stocks_df.empty:
            st.error("❌ Không thể lấy danh sách cổ phiếu từ database!")
            return

        stock_options = stocks_df['tickersymbol'].tolist()

        stocks_to_compare = st.multiselect(
            "Chọn cổ phiếu để so sánh:",
            stock_options,
            default=stock_options[:3] if len(stock_options) >= 3 else stock_options
        )
        
        if stocks_to_compare:
            fig_compare = go.Figure()
            
            for stock in stocks_to_compare:
                # Lấy stockkey từ ticker
                stock_info = stocks_df[stocks_df['tickersymbol'] == stock]
                if not stock_info.empty:
                    sk = stock_info['stockkey'].values[0]
                    stock_df = get_stock_data_from_db(sk, 180)
                    
                    if stock_df.empty:
                        continue

                    # Normalize to percentage change from first day
                    normalized = (stock_df['close'] / stock_df['close'].iloc[0] - 1) * 100
                    
                    fig_compare.add_trace(go.Scatter(
                        x=stock_df['date'],
                        y=normalized,
                        mode='lines',
                        name=stock,
                        line=dict(width=2)
                    ))
            
            fig_compare.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_compare.update_layout(
                height=500,
                template='plotly_white',
                title="So sánh hiệu suất (% thay đổi)",
                xaxis_title="Ngày",
                yaxis_title="% Thay đổi",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
    
    with tab3:
        st.markdown("#### Phân tích biến động giá")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns distribution
            df['returns'] = df['close'].pct_change() * 100
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df['returns'].dropna(),
                nbinsx=50,
                name='Phân phối lợi nhuận',
                marker_color='#3B82F6'
            ))
            
            fig_hist.update_layout(
                height=400,
                template='plotly_white',
                title="Phân phối lợi nhuận hàng ngày (%)",
                xaxis_title="% Thay đổi",
                yaxis_title="Tần suất"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Rolling volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=df['date'],
                y=df['volatility'],
                mode='lines',
                fill='tozeroy',
                name='Biến động 20 ngày',
                line=dict(color='#EF4444')
            ))
            
            fig_vol.update_layout(
                height=400,
                template='plotly_white',
                title="Biến động giá (Rolling 20 ngày)",
                xaxis_title="Ngày",
                yaxis_title="Độ lệch chuẩn (%)"
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Statistics
        st.markdown("#### Thống kê")

        col1, col2, col3, col4 = st.columns(4)
        
        returns = df['returns'].dropna()
        
        with col1:
            st.metric("Lợi nhuận TB/ngày", f"{returns.mean():.3f}%")
        with col2:
            st.metric("Độ lệch chuẩn", f"{returns.std():.3f}%")
        with col3:
            st.metric("Tỷ lệ Sharpe", f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}")
        with col4:
            st.metric("Sụt giảm tối đa", f"{((df['close'] / df['close'].cummax() - 1).min() * 100):.2f}%")


def show_prediction_page():
    selected_stock = st.session_state.get('selected_stock', None)
    stock_key = st.session_state.get('selected_stockkey', None)

    st.markdown(f"### Dự đoán giá cổ phiếu - {selected_stock}")

    if not stock_key:
        st.error("Vui lòng chọn cổ phiếu từ sidebar!")
        return

    # Lấy dữ liệu từ database
    df = get_stock_data_from_db(stock_key, 365)

    if df.empty:
        st.error("Không có dữ liệu cho cổ phiếu này!")
        return

    st.caption(f"Dữ liệu từ Database - {len(df)} bản ghi")

    df = calculate_moving_averages(df)
    
    # ===== CHỌN MÔ HÌNH =====
    st.markdown("#### Chọn mô hình dự đoán")
    col_model, col_days = st.columns([2, 1])

    with col_model:
        model_choice = st.selectbox(
            "Mô hình:",
            ["Hybrid (LSTM) - Từ CSDL", "Linear Regression", "Auto ARIMA", "Random Forest"],
            index=0,
            help="Hybrid lấy từ cơ sở dữ liệu, các mô hình khác tính toán thời gian thực"
        )

    with col_days:
        if "Hybrid" not in model_choice:
            days_to_predict = st.slider("Số ngày dự đoán:", 7, 30, 14)
        else:
            days_to_predict = 14  # Hybrid cố định 14 ngày
            st.info("Hybrid: 14 ngày")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Biểu đồ dự đoán")

        last_price = df['close'].iloc[-1]
        last_date = df['date'].iloc[-1]

        # Biến lưu metrics của model
        model_metrics = None
        model_name = ""

        # ===== XỬ LÝ THEO MÔ HÌNH =====
        if "Hybrid" in model_choice:
            # Lấy dự đoán từ database (LSTM đã train sẵn)
            predictions_df = get_predictions_from_db(stock_key)

            if not predictions_df.empty:
                forecast_data = predictions_df['forecastprices'].iloc[0]

                if isinstance(forecast_data, list):
                    if len(forecast_data) > 0 and isinstance(forecast_data[0], list):
                        predicted_prices = [float(x) for x in forecast_data[0]]
                    else:
                        predicted_prices = [float(x) for x in forecast_data]
                elif isinstance(forecast_data, str):
                    forecast_str = forecast_data.replace('{{', '').replace('}}', '').replace('{', '').replace('}', '')
                    predicted_prices = [float(x) for x in forecast_str.split(',')]
                else:
                    predicted_prices = [float(x) for x in list(forecast_data)]

                days_to_predict = len(predicted_prices)
                model_name = "Hybrid (LSTM)"
                st.info(f"Dự đoán từ mô hình Hybrid (LSTM) - {days_to_predict} ngày - Từ cơ sở dữ liệu")

                # Lấy metrics từ database
                model_perf = pd.DataFrame()
                if DB_AVAILABLE:
                    try:
                        model_perf = get_model_performance(stock_key)
                    except:
                        pass
                if not model_perf.empty:
                    perf = model_perf.iloc[0]
                    model_metrics = {
                        'mae': perf.get('mae', 0),
                        'rmse': perf.get('rmse', 0),
                        'mape': perf.get('mape', 0),
                        'model_name': 'Hybrid (LSTM)'
                    }
            else:
                st.warning("Chưa có dự đoán Hybrid cho cổ phiếu này trong database!")
                predicted_prices = [last_price] * days_to_predict
                model_name = "Hybrid (LSTM) - N/A"

        elif "Linear Regression" in model_choice:
            # Tính toán Linear Regression realtime
            with st.spinner("Đang tính toán Linear Regression..."):
                predicted_prices, model_metrics = predict_linear_regression(df, days_to_predict)
                model_name = "Linear Regression"
                st.info(f"Dự đoán Linear Regression - {days_to_predict} ngày - Tính toán thời gian thực")

        elif "ARIMA" in model_choice:
            # Tính toán ARIMA realtime
            with st.spinner("Đang tính toán Auto ARIMA..."):
                predicted_prices, model_metrics = predict_arima(df, days_to_predict)
                model_name = model_metrics.get('model_name', 'Auto ARIMA')
                st.info(f"Dự đoán {model_name} - {days_to_predict} ngày - Tính toán thời gian thực")

        elif "Random Forest" in model_choice:
            # Tính toán Random Forest realtime
            with st.spinner("Đang tính toán Random Forest..."):
                predicted_prices, model_metrics = predict_random_forest(df, days_to_predict)
                model_name = "Random Forest"
                st.info(f"Dự đoán Random Forest - {days_to_predict} ngày - Tính toán thời gian thực")

        # Tạo future_dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict, freq='D')

        # Confidence interval
        std = df['close'].pct_change().std() * last_price
        upper_bound = [p + 2 * std * np.sqrt(i+1) for i, p in enumerate(predicted_prices)]
        lower_bound = [p - 2 * std * np.sqrt(i+1) for i, p in enumerate(predicted_prices)]

        # ===== VẼ BIỂU ĐỒ =====
        fig = go.Figure()
        
        # Chỉ lấy 30 ngày gần nhất
        df_recent = df.tail(30).copy()

        # Historical data
        fig.add_trace(go.Scatter(
            x=df_recent['date'],
            y=df_recent['close'],
            mode='lines+markers',
            name='Giá lịch sử',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Kết nối
        fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]],
            y=[last_price, predicted_prices[0]],
            mode='lines',
            line=dict(color='#888888', width=1, dash='dot'),
            showlegend=False
        ))

        # Prediction line với markers rõ ràng
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_prices,
            mode='lines+markers+text',
            name=f'Dự đoán ({days_to_predict} ngày)',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10, symbol='circle', color='#FF6B6B',
                       line=dict(color='white', width=2)),
            text=[f'${p:.2f}' for p in predicted_prices],
            textposition='top center',
            textfont=dict(size=9, color='#FF6B6B')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.15)',
            line=dict(color='rgba(255,107,107,0.3)'),
            name='Khoảng tin cậy 95%',
            hoverinfo='skip'
        ))
        
        # Thêm đường dọc phân cách giữa lịch sử và dự đoán
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash")
        )

        # Thêm annotation cho đường phân cách
        fig.add_annotation(
            x=last_date,
            y=1.05,
            yref="paper",
            text="Hôm nay",
            showarrow=False,
            font=dict(color="green", size=12)
        )

        # Cập nhật layout để zoom vào vùng quan trọng
        fig.update_layout(
            height=550,
            template='plotly_white',
            title=dict(
                text=f"📈 Dự đoán giá {selected_stock} - {days_to_predict} ngày tới",
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Ngày",
                tickformat='%d/%m/%Y',
                tickangle=45,
                dtick='D1',  # Hiển thị mỗi ngày
                range=[df_recent['date'].iloc[0], future_dates[-1] + timedelta(days=1)]
            ),
            yaxis=dict(
                title="Giá (USD)",
                tickformat='$,.2f'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Bảng chi tiết dự đoán từng ngày
        st.markdown("#### Chi tiết dự đoán từng ngày")
        prediction_detail = pd.DataFrame({
            'Ngày': [d.strftime('%d/%m/%Y (%a)') for d in future_dates],
            'Giá dự đoán': [f'${p:.2f}' for p in predicted_prices],
            'Thay đổi từ hôm nay': [f'{((p - last_price) / last_price * 100):+.2f}%' for p in predicted_prices],
            'Khoảng tin cậy': [f'${l:.2f} - ${u:.2f}' for l, u in zip(lower_bound, upper_bound)]
        })
        st.dataframe(prediction_detail, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Tổng quan dự đoán")

        # Tính toán các chỉ số
        predicted_change = ((predicted_prices[-1] - last_price) / last_price) * 100
        max_predicted = max(predicted_prices)
        min_predicted = min(predicted_prices)
        avg_predicted = sum(predicted_prices) / len(predicted_prices)

        # Metrics chính
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Giá hiện tại",
                value=f"${last_price:.2f}"
            )
        with col_m2:
            st.metric(
                label="Giá dự đoán (cuối)",
                value=f"${predicted_prices[-1]:.2f}",
                delta=f"{predicted_change:+.2f}%"
            )

        st.markdown("---")

        # Thống kê dự đoán
        st.markdown(f"##### Thống kê {days_to_predict} ngày tới")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Cao nhất", f"${max_predicted:.2f}")
            st.metric("Thấp nhất", f"${min_predicted:.2f}")
        with col_s2:
            st.metric("Trung bình", f"${avg_predicted:.2f}")
            st.metric("Biên độ", f"${max_predicted - min_predicted:.2f}")

        st.markdown("---")

        # Xu hướng
        trend = "TĂNG" if predicted_change > 1 else ("GIẢM" if predicted_change < -1 else "ĐI NGANG")
        trend_color = "green" if predicted_change > 1 else ("red" if predicted_change < -1 else "orange")
        st.markdown(f"##### Xu hướng dự đoán: <span style='color:{trend_color}'>{trend}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Hiển thị Model Performance
        st.markdown(f"#### Độ chính xác: {model_name}")

        if model_metrics:
            st.metric("MAE", f"{model_metrics.get('mae', 0):.4f}")
            st.metric("RMSE", f"{model_metrics.get('rmse', 0):.4f}")
            st.metric("MAPE", f"{model_metrics.get('mape', 0):.2f}%")
        else:
            st.metric("MAE", "N/A")
            st.metric("RMSE", "N/A")
            st.metric("MAPE", "N/A")

    st.markdown("---")
    
    st.warning("""
    **Lưu ý:** Đây chỉ là dự đoán dựa trên mô hình thống kê. 
    Không nên sử dụng kết quả này làm căn cứ duy nhất cho quyết định đầu tư.
    """)


def show_settings_page():
    st.markdown("### Cài đặt hệ thống")

    tab1, tab2, tab3 = st.tabs(["Cơ sở dữ liệu", "Hiển thị", "Thông tin"])

    with tab1:
        st.markdown("#### Cấu hình kết nối cơ sở dữ liệu")

        with st.form("db_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                db_host = st.text_input("Host", value="localhost")
                db_port = st.text_input("Port", value="5432")
                db_name = st.text_input("Database", value="dev")

            with col2:
                db_user = st.text_input("Tên đăng nhập", value="db_user")
                db_password = st.text_input("Mật khẩu", value="db_password", type="password")
                db_schema = st.text_input("Schema", value="public")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Lưu cấu hình", type="primary")
            with col2:
                test_conn = st.form_submit_button("Test kết nối")

            if submitted:
                if DB_AVAILABLE:
                    update_db_config(
                        host=db_host,
                        port=db_port,
                        database=db_name,
                        user=db_user,
                        password=db_password,
                        schema=db_schema
                    )
                    st.success("Đã lưu cấu hình")
                else:
                    st.error("Module database chưa sẵn sàng")

            if test_conn:
                with st.spinner("Đang kiểm tra kết nối..."):
                    if DB_AVAILABLE:
                        try:
                            conn = get_connection(
                                host=db_host,
                                port=db_port,
                                database=db_name,
                                user=db_user,
                                password=db_password
                            )
                            if conn:
                                st.success("Kết nối database thành công!")
                                close_connection(conn)
                            else:
                                st.error("Không thể kết nối database")
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
                    else:
                        st.error("Module database chưa sẵn sàng")

    with tab2:
        st.markdown("#### Cài đặt hiển thị")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Giao diện", ["Sáng", "Tối", "Tự động"])
            st.selectbox("Ngôn ngữ", ["Tiếng Việt", "English"])
            st.number_input("Số điểm dữ liệu mặc định", min_value=30, max_value=365, value=180)
        
        with col2:
            st.multiselect("MA mặc định", ["MA5", "MA10", "MA20", "MA50"], default=["MA5", "MA20"])
            st.selectbox("Định dạng số", ["1,000,000", "1.000.000", "1 000 000"])
            st.checkbox("Hiển thị khối lượng mặc định", value=True)

    with tab3:
        st.markdown("#### Thông tin hệ thống")
        
        st.markdown("""
        <div class="info-card">
            <h4>Hệ thống phân tích và dự đoán giá cổ phiếu</h4>
            <p><strong>Phiên bản:</strong> 1.0.0</p>
            <p><strong>Framework:</strong> Streamlit + Plotly</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Công nghệ sử dụng")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Backend:**
            - Python 3.10+
            - PostgreSQL
            - Apache Airflow
            """)
        
        with col2:
            st.markdown("""
            **Frontend:**
            - Streamlit
            - Plotly
            - Pandas
            """)
        
        with col3:
            st.markdown("""
            **ML/AI:**
            - Scikit-learn
            - TensorFlow/Keras
            - Prophet
            """)


if __name__ == "__main__":
    main()
