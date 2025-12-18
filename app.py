"""
🏛️ Institutional Commodities Analytics Platform - Streamlit Cloud Optimized
Advanced GARCH, Regime Detection, Portfolio Analysis & Risk Management
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
from arch import arch_model
import warnings
import time
import os
import json
from typing import Dict, List, Optional, Tuple
import concurrent.futures
from scipy import stats
import statsmodels.api as sm

# ============================================================================
# Configuration & Setup
# ============================================================================

# Performance optimization
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Commodities Analytics Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Institutional Commodities Analytics Platform"
    }
)

# ============================================================================
# Custom CSS for Professional Interface
# ============================================================================

st.markdown("""
<style>
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        right: -50%;
        bottom: -50%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 8s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0.5rem 0;
        font-family: 'Arial', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding: 0.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-weight: 700;
        color: #6c757d;
        background: white;
        border-radius: 10px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #667eea;
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Professional Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Enhanced Tables */
    .dataframe {
        border: none !important;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-collapse: separate;
        border-spacing: 0;
        margin: 1rem 0;
    }
    
    .dataframe thead {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .dataframe thead th {
        background: transparent !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 1.2rem 1rem !important;
        text-align: center;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dataframe tbody tr {
        transition: all 0.3s ease;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e3f2fd !important;
        transform: scale(1.005);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 25px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading Animation */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255,255,255,0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Professional Cards */
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .professional-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    /* Enhanced Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-weight: 700;
        color: #2c3e50;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%);
    }
    
    /* Professional Footer */
    .professional-footer {
        background: linear-gradient(135deg, #2c3e50 0%, #4a6491 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Grid Layout */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0 1rem;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Commodities Universe & Configuration
# ============================================================================

COMMODITIES = {
    "Precious Metals": {
        "GC=F": {"name": "Gold Futures", "category": "precious", "symbol": "GC=F"},
        "SI=F": {"name": "Silver Futures", "category": "precious", "symbol": "SI=F"},
        "PL=F": {"name": "Platinum Futures", "category": "precious", "symbol": "PL=F"},
        "PA=F": {"name": "Palladium Futures", "category": "precious", "symbol": "PA=F"},
    },
    "Industrial Metals": {
        "HG=F": {"name": "Copper Futures", "category": "industrial", "symbol": "HG=F"},
        "ALI=F": {"name": "Aluminum Futures", "category": "industrial", "symbol": "ALI=F"},
        "ZC=F": {"name": "Corn Futures", "category": "industrial", "symbol": "ZC=F"},
    },
    "Energy": {
        "CL=F": {"name": "Crude Oil WTI", "category": "energy", "symbol": "CL=F"},
        "NG=F": {"name": "Natural Gas", "category": "energy", "symbol": "NG=F"},
        "BZ=F": {"name": "Brent Crude", "category": "energy", "symbol": "BZ=F"},
        "HO=F": {"name": "Heating Oil", "category": "energy", "symbol": "HO=F"},
    },
    "Agriculture": {
        "ZS=F": {"name": "Soybean Futures", "category": "agriculture", "symbol": "ZS=F"},
        "ZW=F": {"name": "Wheat Futures", "category": "agriculture", "symbol": "ZW=F"},
        "KC=F": {"name": "Coffee Futures", "category": "agriculture", "symbol": "KC=F"},
        "SB=F": {"name": "Sugar Futures", "category": "agriculture", "symbol": "SB=F"},
        "CT=F": {"name": "Cotton Futures", "category": "agriculture", "symbol": "CT=F"},
    },
    "Softs": {
        "OJ=F": {"name": "Orange Juice", "category": "softs", "symbol": "OJ=F"},
        "LB=F": {"name": "Lumber", "category": "softs", "symbol": "LB=F"},
    }
}

BENCHMARKS = {
    "SPY": {"name": "S&P 500 ETF", "category": "equity"},
    "DXY": {"name": "US Dollar Index", "category": "currency"},
    "TLT": {"name": "20+ Year Treasury", "category": "bond"},
    "GLD": {"name": "Gold Trust", "category": "commodity"},
    "VIX": {"name": "Volatility Index", "category": "volatility"}
}

# ============================================================================
# Data Management Layer
# ============================================================================

class DataManager:
    """Advanced data management with caching and error handling"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self):
        """Create requests session with retry logic"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    @st.cache_data(ttl=1800, show_spinner=False, max_entries=20)
    def fetch_data(_self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data with enhanced error handling"""
        try:
            ticker = yf.Ticker(symbol)
            ticker.session = _self.session
            
            # Try to fetch data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                timeout=30
            )
            
            if df.empty or len(df) < 50:
                return None
            
            # Calculate returns and indicators
            df = _self._enhance_dataframe(df)
            return df
            
        except Exception as e:
            st.warning(f"Failed to fetch {symbol}: {str(e)[:100]}")
            return None
    
    def _enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated columns to dataframe"""
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Volatility measures
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_50'] = df['Returns'].rolling(window=50).std() * np.sqrt(252)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Daily range
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Range_SMA'] = df['Daily_Range'].rolling(window=20).mean()
        
        return df
    
    @st.cache_data(ttl=1800, show_spinner=False, max_entries=10)
    def bulk_fetch(_self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Bulk fetch data for multiple symbols"""
        data_dict = {}
        
        progress_bar = st.progress(0, text="Loading market data...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = _self.fetch_data(symbol, start_date, end_date)
                if df is not None:
                    data_dict[symbol] = df
            except Exception as e:
                st.warning(f"Error loading {symbol}: {str(e)[:100]}")
            
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress, text=f"Loading {symbol} ({i+1}/{len(symbols)})")
        
        progress_bar.empty()
        return data_dict

# ============================================================================
# Analytics Engine
# ============================================================================

class AnalyticsEngine:
    """Advanced analytics engine for commodities"""
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if df is None or len(df) < 100:
            return {}
        
        returns = df['Returns'].dropna()
        
        metrics = {
            # Basic metrics
            'current_price': df['Close'].iloc[-1],
            'previous_close': df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1],
            'daily_change': df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0,
            'daily_change_pct': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
            
            # Range metrics
            'daily_high': df['High'].iloc[-1],
            'daily_low': df['Low'].iloc[-1],
            'daily_range_pct': ((df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Low'].iloc[-1]) * 100,
            
            # Volume metrics
            'volume': df['Volume'].iloc[-1],
            'avg_volume_20d': df['Volume'].tail(20).mean(),
            'volume_ratio': df['Volume'].iloc[-1] / df['Volume'].tail(20).mean() if df['Volume'].tail(20).mean() > 0 else 1,
            
            # Return metrics
            'total_return': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100,
            'annual_return': ((1 + returns.mean()) ** 252 - 1) * 100,
            'annual_volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            
            # Risk metrics
            'var_95': np.percentile(returns, 5) * 100,
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df) * 100,
            'calmar_ratio': ((1 + returns.mean()) ** 252 - 1) / abs(self._calculate_max_drawdown(df)) if abs(self._calculate_max_drawdown(df)) > 0 else 0,
            
            # Statistical metrics
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera': stats.jarque_bera(returns)[0],
            'jarque_bera_pval': stats.jarque_bera(returns)[1],
            
            # Technical indicators
            'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
            'macd': df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
            'stochastic': df['Stochastic_%K'].iloc[-1] if 'Stochastic_%K' in df.columns else 50,
            'atr': df['ATR'].iloc[-1] if 'ATR' in df.columns else 0,
            'bb_width': df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else 0,
            
            # Volatility metrics
            'volatility_20d': df['Volatility_20'].iloc[-1] * 100 if 'Volatility_20' in df.columns else 0,
            'volatility_50d': df['Volatility_50'].iloc[-1] * 100 if 'Volatility_50' in df.columns else 0,
            'volatility_ratio': (df['Volatility_20'].iloc[-1] / df['Volatility_50'].iloc[-1]) if ('Volatility_20' in df.columns and 'Volatility_50' in df.columns and df['Volatility_50'].iloc[-1] > 0) else 1,
            
            # Trend metrics
            'sma_20': df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else df['Close'].iloc[-1],
            'sma_50': df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else df['Close'].iloc[-1],
            'price_vs_sma20': ((df['Close'].iloc[-1] - df['SMA_20'].iloc[-1]) / df['SMA_20'].iloc[-1] * 100) if 'SMA_20' in df.columns else 0,
            'price_vs_sma50': ((df['Close'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1] * 100) if 'SMA_50' in df.columns else 0,
            'sma_cross': 'Bullish' if (df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]) else 'Bearish' if ('SMA_20' in df.columns and 'SMA_50' in df.columns) else 'Neutral',
        }
        
        # Win/Loss metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        metrics['win_rate'] = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        metrics['avg_win'] = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
        metrics['avg_loss'] = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
        metrics['profit_factor'] = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')
        
        return metrics
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(df) < 20:
            return 0
        
        cumulative = (1 + df['Returns'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        returns_df = pd.DataFrame()
        
        for symbol, df in data_dict.items():
            if 'Returns' in df.columns:
                returns_df[symbol] = df['Returns']
        
        if len(returns_df.columns) < 2:
            return pd.DataFrame()
        
        # Align dates and drop NaN
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 50:
            return pd.DataFrame()
        
        return returns_df.corr()
    
    def calculate_rolling_beta(self, asset_returns: pd.Series, 
                              benchmark_returns: pd.Series, 
                              window: int = 63) -> pd.Series:
        """Calculate rolling beta against benchmark"""
        aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < window:
            return pd.Series()
        
        betas = []
        dates = []
        
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            X = sm.add_constant(window_data.iloc[:, 1])  # Benchmark
            y = window_data.iloc[:, 0]  # Asset
            
            try:
                model = sm.OLS(y, X).fit()
                beta = model.params[1] if len(model.params) > 1 else 0
                betas.append(beta)
                dates.append(aligned.index[i])
            except:
                betas.append(np.nan)
                dates.append(aligned.index[i])
        
        return pd.Series(betas, index=dates)
    
    def detect_regimes(self, returns: pd.Series, n_regimes: int = 3) -> Dict:
        """Detect market regimes using statistical methods"""
        if len(returns) < 100:
            return {}
        
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=20).mean() * 252
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Simple regime detection based on volatility
        vol_threshold = rolling_vol.median()
        
        regimes = pd.Series(1, index=returns.index)  # Default: Normal regime
        
        # High volatility regime
        regimes[rolling_vol > vol_threshold * 1.5] = 2
        
        # Low volatility regime
        regimes[rolling_vol < vol_threshold * 0.7] = 0
        
        # Crisis regime (high volatility + negative returns)
        crisis_condition = (rolling_vol > vol_threshold * 1.5) & (rolling_mean < -0.1)
        regimes[crisis_condition] = 3
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in sorted(regimes.unique()):
            regime_returns = returns[regimes == regime]
            if len(regime_returns) > 10:
                regime_stats[int(regime)] = {
                    'count': len(regime_returns),
                    'proportion': len(regime_returns) / len(returns),
                    'mean_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
                }
        
        # Label regimes
        regime_labels = {
            0: 'Low Volatility',
            1: 'Normal',
            2: 'High Volatility',
            3: 'Crisis'
        }
        
        return {
            'regimes': regimes,
            'regime_stats': regime_stats,
            'regime_labels': regime_labels,
            'rolling_mean': rolling_mean,
            'rolling_vol': rolling_vol
        }
    
    def fit_garch_model(self, returns: pd.Series, p: int = 1, q: int = 1) -> Optional[Dict]:
        """Fit GARCH model to returns"""
        if len(returns) < 200:
            return None
        
        try:
            # Scale returns for better convergence
            returns_scaled = returns.dropna() * 100
            
            # Fit GARCH model
            model = arch_model(
                returns_scaled,
                mean='Constant',
                vol='GARCH',
                p=p,
                q=q,
                dist='t'
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Calculate conditional volatility
            cond_vol = result.conditional_volatility / 100  # Rescale back
            
            return {
                'model': result,
                'params': dict(result.params),
                'aic': result.aic,
                'bic': result.bic,
                'cond_volatility': cond_vol,
                'residuals': result.resid / 100,  # Rescale back
                'converged': result.convergence_flag == 0
            }
            
        except Exception as e:
            st.warning(f"GARCH model fitting failed: {str(e)[:100]}")
            return None
    
    def forecast_volatility(self, garch_model: Dict, steps: int = 30) -> np.ndarray:
        """Forecast volatility using GARCH model"""
        if garch_model is None or 'model' not in garch_model:
            return np.array([])
        
        try:
            forecast = garch_model['model'].forecast(horizon=steps)
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].values) / 100  # Rescale
            return forecast_vol
        except:
            return np.array([])
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate advanced risk metrics"""
        if len(returns) < 100:
            return {}
        
        # Historical VaR
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        max_dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # Tail Risk Metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calculate tail dependence with normal distribution
        z_scores = (returns - returns.mean()) / returns.std()
        extreme_negative = (z_scores < -2.5).sum() / len(returns) * 100
        extreme_positive = (z_scores > 2.5).sum() / len(returns) * 100
        
        # Risk-adjusted returns
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'extreme_negative_pct': extreme_negative,
            'extreme_positive_pct': extreme_positive,
            'calmar_ratio': returns.mean() * 252 / abs(max_dd/100) if abs(max_dd) > 0 else 0,
            'omega_ratio': self._calculate_omega_ratio(returns),
            'tail_ratio': abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(drawdown) == 0:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        return gains / losses if losses > 0 else float('inf')

# ============================================================================
# Visualization Engine
# ============================================================================

class VisualizationEngine:
    """Advanced visualization engine"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#3b82f6',
            'dark': '#1f2937',
            'light': '#f3f4f6'
        }
    
    def create_price_chart(self, df: pd.DataFrame, title: str, 
                          show_indicators: bool = True) -> go.Figure:
        """Create comprehensive price chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{title} - Price Action', 'Volume', 'Technical Indicators')
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Price',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color=self.color_palette['warning'], width=1.5, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color=self.color_palette['success'], width=1.5, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color=self.color_palette['info'], width=1, dash='dot'),
                opacity=0.5,
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color=self.color_palette['info'], width=1, dash='dot'),
                opacity=0.5,
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Volume with color coding
        colors = [self.color_palette['success'] if close >= open_ else self.color_palette['danger'] 
                 for close, open_ in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add volume moving average
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volume_SMA'],
                name='Volume SMA 20',
                line=dict(color=self.color_palette['dark'], width=1.5),
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Technical Indicators
        if show_indicators:
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color=self.color_palette['primary'], width=1.5)
                ),
                row=3, col=1
            )
            
            # Add RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color=self.color_palette['danger'], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=self.color_palette['success'], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color=self.color_palette['dark'], 
                         opacity=0.3, row=3, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color=self.color_palette['warning'], width=1.5),
                    yaxis='y2'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color=self.color_palette['secondary'], width=1.5, dash='dash'),
                    yaxis='y2'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=self.color_palette['dark'])
            ),
            height=800,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        if show_indicators:
            fig.update_yaxes(title_text="MACD", row=3, col=1, secondary_y=True)
        
        return fig
    
    def create_volatility_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create volatility chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Historical Volatility', 'Daily Returns Distribution')
        )
        
        # Historical volatility
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volatility_20'] * 100,
                name='20-Day Volatility',
                line=dict(color=self.color_palette['primary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volatility_50'] * 100,
                name='50-Day Volatility',
                line=dict(color=self.color_palette['secondary'], width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Returns distribution
        returns = df['Returns'].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                name='Returns Distribution',
                nbinsx=50,
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                histnorm='probability density'
            ),
            row=2, col=1
        )
        
        # Add normal distribution overlay
        if len(returns) > 0:
            x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            y_norm = stats.norm.pdf(x_norm, returns.mean() * 100, returns.std() * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    name='Normal Distribution',
                    line=dict(color=self.color_palette['danger'], width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=dict(
                text=f'{title} - Volatility Analysis',
                x=0.5,
                xanchor='center'
            ),
            height=700,
            template='plotly_white',
            showlegend=True,
            bargap=0.05
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_xaxes(title_text="Daily Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        
        return fig
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str) -> go.Figure:
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(
                title="Correlation",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color=self.color_palette['dark'])
            ),
            height=600,
            template='plotly_white',
            xaxis_title="Assets",
            yaxis_title="Assets",
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_regime_chart(self, df: pd.DataFrame, regimes: pd.Series, 
                           regime_stats: Dict, regime_labels: Dict, 
                           title: str) -> go.Figure:
        """Create regime detection chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price with Regimes', 'Rolling Returns', 'Rolling Volatility')
        )
        
        # Price with regime shading
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Price',
                line=dict(color=self.color_palette['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add regime shading
        regime_colors = {
            0: 'rgba(16, 185, 129, 0.2)',  # Green for low vol
            1: 'rgba(59, 130, 246, 0.2)',   # Blue for normal
            2: 'rgba(245, 158, 11, 0.2)',   # Yellow for high vol
            3: 'rgba(239, 68, 68, 0.2)'     # Red for crisis
        }
        
        # Find regime change points
        regime_changes = regimes.diff().fillna(0)
        change_points = regime_changes[regime_changes != 0].index
        
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            regime = regimes.loc[start]
            
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=regime_colors.get(regime, 'rgba(128, 128, 128, 0.2)'),
                opacity=0.3,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Rolling returns
        rolling_returns = df['Returns'].rolling(window=20).mean() * 252 * 100
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rolling_returns,
                name='20-Day Rolling Return',
                line=dict(color=self.color_palette['success'], width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # Rolling volatility
        rolling_vol = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rolling_vol,
                name='20-Day Rolling Vol',
                line=dict(color=self.color_palette['danger'], width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=dict(
                text=f'{title} - Market Regimes',
                x=0.5,
                xanchor='center'
            ),
            height=800,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Volatility (%)", row=3, col=1)
        
        return fig
    
    def create_garch_chart(self, df: pd.DataFrame, garch_results: Dict, 
                          forecast: np.ndarray, title: str) -> go.Figure:
        """Create GARCH model results chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Conditional Volatility',
                'Volatility Forecast',
                'Standardized Residuals',
                'QQ Plot of Residuals'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Conditional volatility
        if garch_results and 'cond_volatility' in garch_results:
            cond_vol = garch_results['cond_volatility'] * 100  # Convert to percentage
            
            fig.add_trace(
                go.Scatter(
                    x=df.index[-len(cond_vol):],
                    y=cond_vol,
                    name='GARCH Volatility',
                    line=dict(color=self.color_palette['primary'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ),
                row=1, col=1
            )
        
        # Volatility forecast
        if len(forecast) > 0:
            forecast_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast),
                freq='B'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast * 100,
                    name='Forecast',
                    line=dict(color=self.color_palette['danger'], width=2),
                    mode='lines+markers',
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[df['Volatility_20'].iloc[-1] * 100] * len(forecast),
                    name='Current Vol',
                    line=dict(color=self.color_palette['dark'], width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=2
            )
        
        # Standardized residuals
        if garch_results and 'residuals' in garch_results:
            residuals = garch_results['residuals']
            cond_vol = garch_results['cond_volatility']
            std_residuals = residuals / cond_vol
            
            fig.add_trace(
                go.Scatter(
                    x=df.index[-len(std_residuals):],
                    y=std_residuals,
                    name='Std Residuals',
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=self.color_palette['warning'],
                        opacity=0.6
                    )
                ),
                row=2, col=1
            )
            
            # Add confidence bands
            fig.add_hline(y=2, line_dash="dash", line_color=self.color_palette['danger'], 
                         opacity=0.3, row=2, col=1)
            fig.add_hline(y=-2, line_dash="dash", line_color=self.color_palette['danger'], 
                         opacity=0.3, row=2, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)
        
        # QQ Plot
        if garch_results and 'residuals' in garch_results:
            residuals = garch_results['residuals'].dropna()
            
            if len(residuals) > 0:
                qq = stats.probplot(residuals, dist="norm")
                theoretical = qq[0][1]
                sample = qq[0][0]
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=sample,
                        mode='markers',
                        name='QQ Plot',
                        marker=dict(
                            size=6,
                            color=self.color_palette['info']
                        )
                    ),
                    row=2, col=2
                )
                
                # Add 45-degree line
                min_val = min(theoretical.min(), sample.min())
                max_val = max(theoretical.max(), sample.max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Normal Line',
                        line=dict(
                            color=self.color_palette['danger'],
                            width=2,
                            dash='dash'
                        )
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text=f'{title} - GARCH Analysis',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Forecast Vol (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Std Residuals", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        return fig
    
    def create_risk_metrics_chart(self, risk_metrics: Dict, title: str) -> go.Figure:
        """Create risk metrics visualization"""
        
        # Prepare data for radar chart
        categories = ['VaR/CVaR', 'Drawdown', 'Risk-Adjusted', 'Tail Risk', 'Performance']
        
        # Normalize metrics for radar chart
        normalized_metrics = {
            'VaR/CVaR': (abs(risk_metrics.get('var_95', 0)) + abs(risk_metrics.get('cvar_95', 0))) / 2 / 10,
            'Drawdown': abs(risk_metrics.get('max_drawdown', 0)) / 10,
            'Risk-Adjusted': (risk_metrics.get('sharpe_ratio', 0) + risk_metrics.get('sortino_ratio', 0)) / 2 / 2,
            'Tail Risk': (abs(risk_metrics.get('skewness', 0)) + risk_metrics.get('kurtosis', 0)) / 10,
            'Performance': risk_metrics.get('calmar_ratio', 0) / 2
        }
        
        # Cap values for radar chart
        for key in normalized_metrics:
            normalized_metrics[key] = min(max(normalized_metrics[key], 0), 1)
        
        values = [normalized_metrics[cat] for cat in categories]
        
        # Create radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # Close the shape
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color=self.color_palette['primary'], width=2),
            marker=dict(size=8, color=self.color_palette['primary'])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title=dict(
                text=f'{title} - Risk Profile',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig

# ============================================================================
# Main Application Class
# ============================================================================

class CommoditiesAnalyticsPlatform:
    """Main application class for commodities analytics"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analytics = AnalyticsEngine()
        self.viz = VisualizationEngine()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'selected_assets' not in st.session_state:
            st.session_state.selected_assets = []
        if 'asset_data' not in st.session_state:
            st.session_state.asset_data = {}
        if 'benchmark_data' not in st.session_state:
            st.session_state.benchmark_data = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def display_header(self):
        """Display application header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; padding:0; font-size:2.8rem; font-weight:800;">🏛️ INSTITUTIONAL COMMODITIES ANALYTICS PLATFORM</h1>
            <p style="margin:0; padding-top:1rem; font-size:1.3rem; opacity:0.95; line-height:1.6;">
                Advanced GARCH Modeling • Regime Detection • Portfolio Analytics • Risk Management
            </p>
            <div style="margin-top:1.5rem; display:flex; gap:1rem; flex-wrap:wrap;">
                <span class="status-badge status-success">Real-time Data</span>
                <span class="status-badge status-info">Institutional Grade</span>
                <span class="status-badge status-warning">Risk Analytics</span>
                <span class="status-badge status-danger">Stress Testing</span>
                <span class="status-badge status-success">Streamlit Cloud</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats bar
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Available Commodities", "25+", "Global Coverage")
        with col2:
            st.metric("Historical Data", "20+ Years", "Daily Frequency")
        with col3:
            st.metric("Analytics Models", "15+", "GARCH • Regime • Beta")
        with col4:
            st.metric("Platform Status", "🟢 Online", "Streamlit Cloud")
    
    def setup_sidebar(self):
        """Setup the application sidebar"""
        with st.sidebar:
            # Header
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            st.markdown("## ⚙️ ANALYSIS CONFIGURATION")
            st.markdown("Configure your institutional analysis")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Date Range
            st.subheader("📅 Date Range")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)  # Default 3 years
            
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start Date", start_date, key="start_date")
            with col2:
                end = st.date_input("End Date", end_date, key="end_date")
            
            if start >= end:
                st.error("Start date must be before end date")
                return None, None, [], []
            
            # Asset Selection
            st.subheader("📊 Commodity Selection")
            
            selected_assets = []
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category} ({len(assets)} assets)", expanded=category == "Precious Metals"):
                    for symbol, info in assets.items():
                        if st.checkbox(
                            f"{info['name']}",
                            value=symbol in ["GC=F", "CL=F", "SI=F"],  # Default selections
                            key=f"asset_{symbol}"
                        ):
                            selected_assets.append(symbol)
            
            # Benchmark Selection
            st.subheader("📈 Benchmark Selection")
            
            selected_benchmarks = []
            for symbol, info in BENCHMARKS.items():
                if st.checkbox(
                    f"{info['name']} ({symbol})",
                    value=symbol in ["SPY", "DXY"],
                    key=f"bench_{symbol}"
                ):
                    selected_benchmarks.append(symbol)
            
            # Analysis Settings
            st.subheader("🔬 Analysis Settings")
            
            with st.expander("Advanced Configuration", expanded=False):
                # GARCH Settings
                st.markdown("**GARCH Model**")
                garch_p = st.slider("ARCH Order (p)", 1, 3, 1, key="garch_p")
                garch_q = st.slider("GARCH Order (q)", 1, 3, 1, key="garch_q")
                forecast_days = st.slider("Forecast Horizon (days)", 5, 60, 30, key="forecast_days")
                
                # Regime Detection
                st.markdown("**Regime Detection**")
                n_regimes = st.slider("Number of Regimes", 2, 4, 3, key="n_regimes")
                
                # Display Options
                st.markdown("**Display Options**")
                show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=True, key="show_advanced")
                show_technical_indicators = st.checkbox("Show Technical Indicators", value=True, key="show_tech")
            
            # Action Buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                load_button = st.button("🚀 Load Data", type="primary", use_container_width=True)
            
            with col2:
                if st.session_state.data_loaded:
                    clear_button = st.button("🗑️ Clear Cache", type="secondary", use_container_width=True)
                    if clear_button:
                        st.cache_data.clear()
                        st.session_state.data_loaded = False
                        st.session_state.asset_data = {}
                        st.session_state.benchmark_data = {}
                        st.rerun()
            
            # Information Panel
            with st.expander("ℹ️ Platform Information", expanded=False):
                st.markdown("""
                ### About This Platform
                
                **Features:**
                - Real-time commodity prices from Yahoo Finance
                - Advanced GARCH volatility modeling
                - Market regime detection
                - Comprehensive risk analytics
                - Portfolio correlation analysis
                - Institutional-grade reporting
                
                **Data Sources:**
                - Yahoo Finance API
                - 20+ years historical data
                - 25+ global commodities
                - Daily frequency updates
                
                **Methodology:**
                - Returns calculated as log returns
                - Annualized metrics where applicable
                - Robust error handling
                - Cloud-optimized performance
                """)
            
            return start, end, selected_assets, selected_benchmarks, load_button
    
    def load_data(self, start_date, end_date, selected_assets, selected_benchmarks):
        """Load data for selected assets and benchmarks"""
        
        # Show loading animation
        with st.spinner("📥 Loading market data from Yahoo Finance..."):
            progress_bar = st.progress(0, text="Initializing data load...")
            
            # Load asset data
            if selected_assets:
                st.session_state.asset_data = self.data_manager.bulk_fetch(
                    selected_assets, 
                    start_date, 
                    end_date
                )
                progress_bar.progress(50, text="Loading commodities data...")
            
            # Load benchmark data
            if selected_benchmarks:
                st.session_state.benchmark_data = self.data_manager.bulk_fetch(
                    selected_benchmarks,
                    start_date,
                    end_date
                )
                progress_bar.progress(100, text="Loading benchmark data...")
            
            time.sleep(0.5)  # Small delay for smooth transition
            progress_bar.empty()
            
            # Update session state
            st.session_state.data_loaded = True
            st.session_state.selected_assets = selected_assets
            
            # Show success message
            if st.session_state.asset_data:
                loaded_count = len(st.session_state.asset_data)
                total_count = len(selected_assets)
                success_rate = loaded_count / total_count * 100
                
                st.success(f"""
                ✅ Successfully loaded {loaded_count} out of {total_count} assets ({success_rate:.0f}%)
                
                **Loaded Assets:** {', '.join(list(st.session_state.asset_data.keys())[:5])}{'...' if len(st.session_state.asset_data) > 5 else ''}
                """)
            else:
                st.error("No data was loaded. Please check your selections and try again.")
    
    def display_dashboard(self):
        """Display main analytics dashboard"""
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "📊 Overview Dashboard",
            "📈 Price Analytics",
            "⚡ Volatility Analysis",
            "🔍 GARCH Modeling",
            "🎯 Regime Detection",
            "📉 Risk Metrics",
            "🔄 Correlation Matrix",
            "🏛️ Portfolio Analytics"
        ])
        
        # Tab 1: Overview Dashboard
        with tabs[0]:
            self.display_overview_dashboard()
        
        # Tab 2: Price Analytics
        with tabs[1]:
            self.display_price_analytics()
        
        # Tab 3: Volatility Analysis
        with tabs[2]:
            self.display_volatility_analysis()
        
        # Tab 4: GARCH Modeling
        with tabs[3]:
            self.display_garch_modeling()
        
        # Tab 5: Regime Detection
        with tabs[4]:
            self.display_regime_detection()
        
        # Tab 6: Risk Metrics
        with tabs[5]:
            self.display_risk_metrics()
        
        # Tab 7: Correlation Matrix
        with tabs[6]:
            self.display_correlation_matrix()
        
        # Tab 8: Portfolio Analytics
        with tabs[7]:
            self.display_portfolio_analytics()
    
    def display_overview_dashboard(self):
        """Display overview dashboard with key metrics"""
        st.subheader("📊 Institutional Dashboard")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Create metrics grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        for symbol, df in st.session_state.asset_data.items():
            if df is not None and len(df) > 0:
                metrics = self.analytics.calculate_performance_metrics(df)
                
                if metrics:
                    # Determine status based on metrics
                    daily_change = metrics['daily_change_pct']
                    volatility = metrics['volatility_20d']
                    rsi = metrics['rsi']
                    
                    if daily_change >= 1:
                        status_color = "#10b981"  # Green
                        status_text = "🟢 Strong"
                    elif daily_change >= 0:
                        status_color = "#3b82f6"  # Blue
                        status_text = "🔵 Moderate"
                    elif daily_change >= -1:
                        status_color = "#f59e0b"  # Yellow
                        status_text = "🟡 Weak"
                    else:
                        status_color = "#ef4444"  # Red
                        status_text = "🔴 Bearish"
                    
                    # Create metric card
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{symbol}</div>
                            <div class="metric-value">${metrics['current_price']:,.2f}</div>
                            <div style="font-size:0.9rem; margin:0.5rem 0;">
                                <div>Change: <span style="color:{status_color}; font-weight:bold;">{metrics['daily_change_pct']:+.2f}%</span></div>
                                <div>Vol: {metrics['volatility_20d']:.1f}% | RSI: {metrics['rsi']:.1f}</div>
                                <div>Return: {metrics['total_return']:+.1f}% | Sharpe: {metrics['sharpe_ratio']:.2f}</div>
                            </div>
                            <div style="font-size:0.8rem; color:#666;">
                                {status_text} • {COMMODITIES.get(symbol, {}).get('name', 'Unknown')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Quick chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index[-30:],
                            y=df['Close'].iloc[-30:],
                            mode='lines',
                            line=dict(color=status_color, width=2),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showticklabels=False),
                            yaxis=dict(showticklabels=False)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market summary
        st.subheader("📈 Market Summary")
        
        if st.session_state.asset_data:
            # Calculate market statistics
            all_metrics = []
            for symbol, df in st.session_state.asset_data.items():
                metrics = self.analytics.calculate_performance_metrics(df)
                if metrics:
                    metrics['symbol'] = symbol
                    all_metrics.append(metrics)
            
            if all_metrics:
                summary_df = pd.DataFrame(all_metrics)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_return = summary_df['daily_change_pct'].mean()
                    st.metric("Avg Daily Return", f"{avg_return:+.2f}%")
                
                with col2:
                    avg_vol = summary_df['volatility_20d'].mean()
                    st.metric("Avg Volatility", f"{avg_vol:.1f}%")
                
                with col3:
                    bullish_count = (summary_df['daily_change_pct'] > 0).sum()
                    total_count = len(summary_df)
                    st.metric("Bullish Assets", f"{bullish_count}/{total_count}")
                
                with col4:
                    high_vol_count = (summary_df['volatility_20d'] > 30).sum()
                    st.metric("High Vol Assets", high_vol_count)
                
                # Performance comparison
                st.subheader("Performance Comparison")
                
                perf_data = []
                for metrics in all_metrics:
                    perf_data.append({
                        'Asset': metrics['symbol'],
                        'Daily Return': metrics['daily_change_pct'],
                        'Total Return': metrics['total_return'],
                        'Volatility': metrics['volatility_20d'],
                        'Sharpe': metrics['sharpe_ratio'],
                        'RSI': metrics['rsi']
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(
                    perf_df.sort_values('Daily Return', ascending=False).style.format({
                        'Daily Return': '{:+.2f}%',
                        'Total Return': '{:+.1f}%',
                        'Volatility': '{:.1f}%',
                        'Sharpe': '{:.2f}',
                        'RSI': '{:.1f}'
                    }).background_gradient(subset=['Daily Return'], cmap='RdYlGn'),
                    use_container_width=True
                )
    
    def display_price_analytics(self):
        """Display price analytics for selected assets"""
        st.subheader("📈 Price Analytics")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset for Detailed Analysis",
            list(st.session_state.asset_data.keys()),
            format_func=lambda x: f"{x} - {COMMODITIES.get(x, {}).get('name', 'Unknown')}"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) == 0:
            st.error(f"No data available for {selected_asset}")
            return
        
        # Display key metrics
        metrics = self.analytics.calculate_performance_metrics(df)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics['current_price']:,.2f}",
                    f"{metrics['daily_change_pct']:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Daily Range",
                    f"${metrics['daily_low']:,.2f} - ${metrics['daily_high']:,.2f}",
                    f"{metrics['daily_range_pct']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Volume",
                    f"{metrics['volume']:,.0f}",
                    f"{metrics['volume_ratio']:.1f}x avg"
                )
            
            with col4:
                st.metric(
                    "Total Return",
                    f"{metrics['total_return']:+.1f}%",
                    f"{metrics['annual_return']:+.1f}% annual"
                )
        
        # Price chart
        asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
        fig = self.viz.create_price_chart(df, f"{asset_name} ({selected_asset})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators table
        st.subheader("Technical Indicators")
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                #### Trend Indicators
                """)
                
                trend_data = {
                    'SMA 20': f"${metrics['sma_20']:,.2f}",
                    'Price vs SMA20': f"{metrics['price_vs_sma20']:+.1f}%",
                    'SMA 50': f"${metrics['sma_50']:,.2f}",
                    'Price vs SMA50': f"{metrics['price_vs_sma50']:+.1f}%",
                    'SMA Cross': metrics['sma_cross'],
                    'MACD': f"{metrics['macd']:.4f}"
                }
                
                for key, value in trend_data.items():
                    st.metric(key, value)
            
            with col2:
                st.markdown("""
                #### Momentum & Volatility
                """)
                
                momentum_data = {
                    'RSI': f"{metrics['rsi']:.1f}",
                    'Stochastic %K': f"{metrics['stochastic']:.1f}",
                    'ATR': f"{metrics['atr']:.3f}",
                    'BB Width': f"{metrics['bb_width']:.3f}",
                    '20D Volatility': f"{metrics['volatility_20d']:.1f}%",
                    'Volatility Ratio': f"{metrics['volatility_ratio']:.2f}"
                }
                
                for key, value in momentum_data.items():
                    st.metric(key, value)
    
    def display_volatility_analysis(self):
        """Display volatility analysis"""
        st.subheader("⚡ Volatility Analysis")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset for Volatility Analysis",
            list(st.session_state.asset_data.keys()),
            key="vol_asset_selector"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) == 0:
            st.error(f"No data available for {selected_asset}")
            return
        
        # Volatility metrics
        metrics = self.analytics.calculate_performance_metrics(df)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("20-Day Volatility", f"{metrics['volatility_20d']:.1f}%")
            
            with col2:
                st.metric("50-Day Volatility", f"{metrics['volatility_50d']:.1f}%")
            
            with col3:
                vol_ratio = metrics['volatility_ratio']
                status = "🟢 Decreasing" if vol_ratio < 1 else "🔴 Increasing"
                st.metric("Volatility Trend", f"{vol_ratio:.2f}", status)
            
            with col4:
                atr = metrics['atr']
                st.metric("Average True Range", f"{atr:.3f}")
        
        # Volatility chart
        asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
        fig = self.viz.create_volatility_chart(df, f"{asset_name} ({selected_asset})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical volatility analysis
        st.subheader("Historical Volatility Analysis")
        
        if len(df) > 100:
            # Calculate rolling volatility statistics
            rolling_stats = []
            windows = [5, 10, 20, 50, 100, 252]
            
            for window in windows:
                if len(df) >= window:
                    rolling_vol = df['Returns'].rolling(window=window).std() * np.sqrt(252) * 100
                    current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
                    avg_vol = rolling_vol.mean() if not rolling_vol.empty else 0
                    max_vol = rolling_vol.max() if not rolling_vol.empty else 0
                    min_vol = rolling_vol.min() if not rolling_vol.empty else 0
                    
                    rolling_stats.append({
                        'Window': f"{window}D",
                        'Current': f"{current_vol:.1f}%",
                        'Average': f"{avg_vol:.1f}%",
                        'Maximum': f"{max_vol:.1f}%",
                        'Minimum': f"{min_vol:.1f}%",
                        'Percentile': f"{(rolling_vol.rank(pct=True).iloc[-1] * 100):.0f}%"
                    })
            
            if rolling_stats:
                stats_df = pd.DataFrame(rolling_stats)
                st.dataframe(stats_df, use_container_width=True)
                
                # Volatility term structure visualization
                fig = go.Figure()
                
                windows = [int(s['Window'].replace('D', '')) for s in rolling_stats]
                current_vols = [float(s['Current'].replace('%', '')) for s in rolling_stats]
                avg_vols = [float(s['Average'].replace('%', '')) for s in rolling_stats]
                
                fig.add_trace(go.Scatter(
                    x=windows,
                    y=current_vols,
                    name='Current Volatility',
                    mode='lines+markers',
                    line=dict(color=self.viz.color_palette['primary'], width=3),
                    marker=dict(size=10)
                ))
                
                fig.add_trace(go.Scatter(
                    x=windows,
                    y=avg_vols,
                    name='Historical Average',
                    mode='lines',
                    line=dict(color=self.viz.color_palette['dark'], width=2, dash='dash'),
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title="Volatility Term Structure",
                    xaxis_title="Window (Days)",
                    yaxis_title="Annualized Volatility (%)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def display_garch_modeling(self):
        """Display GARCH modeling interface"""
        st.subheader("🔍 GARCH Volatility Modeling")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset for GARCH Modeling",
            list(st.session_state.asset_data.keys()),
            key="garch_asset_selector"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) < 200:
            st.warning(f"Insufficient data for GARCH modeling. Need at least 200 observations, got {len(df) if df is not None else 0}.")
            return
        
        # GARCH parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p = st.slider("ARCH Order (p)", 1, 3, 1, key="garch_p_slider")
        
        with col2:
            q = st.slider("GARCH Order (q)", 1, 3, 1, key="garch_q_slider")
        
        with col3:
            forecast_days = st.slider("Forecast Days", 5, 60, 30, key="forecast_days_slider")
        
        # Run GARCH analysis
        if st.button("Run GARCH Analysis", type="primary"):
            with st.spinner(f"Fitting GARCH({p},{q}) model..."):
                # Fit GARCH model
                garch_results = self.analytics.fit_garch_model(
                    df['Returns'],
                    p=p,
                    q=q
                )
                
                if garch_results and garch_results['converged']:
                    st.success("✅ GARCH model converged successfully!")
                    
                    # Forecast volatility
                    forecast = self.analytics.forecast_volatility(
                        garch_results,
                        steps=forecast_days
                    )
                    
                    # Display model parameters
                    st.subheader("Model Parameters")
                    
                    params = garch_results['params']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ω (Omega)", f"{params.get('omega', 0):.6f}")
                    
                    with col2:
                        st.metric("α (Alpha)", f"{params.get('alpha[1]', 0):.4f}")
                    
                    with col3:
                        st.metric("β (Beta)", f"{params.get('beta[1]', 0):.4f}")
                    
                    with col4:
                        alpha_beta = params.get('alpha[1]', 0) + params.get('beta[1]', 0)
                        status = "✅ Stationary" if alpha_beta < 1 else "⚠️ Non-stationary"
                        st.metric("α + β", f"{alpha_beta:.4f}", status)
                    
                    # Model diagnostics
                    st.subheader("Model Diagnostics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("AIC", f"{garch_results['aic']:.1f}")
                        st.metric("BIC", f"{garch_results['bic']:.1f}")
                        st.metric("Log Likelihood", f"{garch_results['model'].loglikelihood:.1f}")
                    
                    with col2:
                        # Calculate Ljung-Box test
                        residuals = garch_results['residuals'].dropna()
                        if len(residuals) > 0:
                            from statsmodels.stats.diagnostic import acorr_ljungbox
                            
                            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
                            lb_pvalue = lb_test['lb_pvalue'].iloc[0]
                            
                            status = "✅ Pass" if lb_pvalue > 0.05 else "⚠️ Fail"
                            st.metric("Ljung-Box Test", status, f"p={lb_pvalue:.4f}")
                        
                        # Jarque-Bera test
                        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
                        status = "✅ Pass" if jb_pvalue > 0.05 else "⚠️ Fail"
                        st.metric("Jarque-Bera Test", status, f"p={jb_pvalue:.4f}")
                    
                    # Display GARCH chart
                    asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
                    fig = self.viz.create_garch_chart(
                        df,
                        garch_results,
                        forecast,
                        f"{asset_name} ({selected_asset})"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast details
                    if len(forecast) > 0:
                        st.subheader("Volatility Forecast")
                        
                        forecast_df = pd.DataFrame({
                            'Day': range(1, len(forecast) + 1),
                            'Forecasted Volatility (%)': forecast * 100
                        })
                        
                        st.dataframe(
                            forecast_df.style.format({'Forecasted Volatility (%)': '{:.2f}%'}),
                            use_container_width=True
                        )
                        
                        # Forecast statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Volatility",
                                f"{df['Volatility_20'].iloc[-1] * 100:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Avg Forecast",
                                f"{forecast.mean() * 100:.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Max Forecast",
                                f"{forecast.max() * 100:.1f}%"
                            )
                        
                        with col4:
                            vol_change = ((forecast.mean() / df['Volatility_20'].iloc[-1]) - 1) * 100
                            st.metric(
                                "Volatility Change",
                                f"{vol_change:+.1f}%"
                            )
                
                else:
                    st.error("❌ GARCH model failed to converge. Try different parameters.")
    
    def display_regime_detection(self):
        """Display regime detection analysis"""
        st.subheader("🎯 Market Regime Detection")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            list(st.session_state.asset_data.keys()),
            key="regime_asset_selector"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) < 100:
            st.warning(f"Insufficient data for regime detection. Need at least 100 observations.")
            return
        
        # Regime detection parameters
        n_regimes = st.slider("Number of Regimes", 2, 4, 3, key="n_regimes_slider")
        
        # Run regime detection
        if st.button("Detect Market Regimes", type="primary"):
            with st.spinner("Analyzing market regimes..."):
                # Detect regimes
                regime_results = self.analytics.detect_regimes(
                    df['Returns'],
                    n_regimes=n_regimes
                )
                
                if regime_results and 'regimes' in regime_results:
                    st.success("✅ Regime detection completed!")
                    
                    # Display regime statistics
                    st.subheader("Regime Statistics")
                    
                    regime_stats = regime_results['regime_stats']
                    regime_labels = regime_results['regime_labels']
                    
                    stats_data = []
                    for regime, stats in regime_stats.items():
                        stats_data.append({
                            'Regime': regime_labels.get(regime, f"Regime {regime}"),
                            'Count': stats['count'],
                            'Proportion': f"{stats['proportion']:.1%}",
                            'Mean Return': f"{stats['mean_return']:.2%}",
                            'Volatility': f"{stats['volatility']:.2%}",
                            'Sharpe Ratio': f"{stats['sharpe']:.2f}",
                            'Label': self._get_regime_label(stats['mean_return'], stats['volatility'])
                        })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                    
                    # Display regime chart
                    asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
                    fig = self.viz.create_regime_chart(
                        df,
                        regime_results['regimes'],
                        regime_results['regime_stats'],
                        regime_results['regime_labels'],
                        f"{asset_name} ({selected_asset})"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Regime transitions
                    st.subheader("Regime Transitions")
                    
                    regimes = regime_results['regimes']
                    transitions = regimes.diff().fillna(0)
                    n_transitions = (transitions != 0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Regime Changes", n_transitions)
                    
                    with col2:
                        avg_duration = len(regimes) / (n_transitions + 1) if n_transitions > 0 else len(regimes)
                        st.metric("Avg Regime Duration", f"{avg_duration:.0f} days")
                    
                    with col3:
                        current_regime = regimes.iloc[-1]
                        current_label = regime_labels.get(current_regime, f"Regime {current_regime}")
                        st.metric("Current Regime", current_label)
                    
                    # Regime risk metrics
                    st.subheader("Regime-Specific Risk Metrics")
                    
                    for regime, stats in regime_stats.items():
                        with st.expander(f"{regime_labels.get(regime, f'Regime {regime}')} Analysis"):
                            regime_returns = df['Returns'][regimes == regime]
                            
                            if len(regime_returns) > 10:
                                risk_metrics = self.analytics.calculate_risk_metrics(regime_returns)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2f}%")
                                    st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2f}%")
                                
                                with col2:
                                    st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2f}%")
                                    st.metric("Drawdown Duration", f"{risk_metrics.get('max_dd_duration', 0)} days")
                                
                                with col3:
                                    st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                                    st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
                                
                                with col4:
                                    st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
                                    st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
    
    def _get_regime_label(self, mean_return: float, volatility: float) -> str:
        """Get descriptive label for regime based on return and volatility"""
        if mean_return > 0.1 and volatility < 0.2:
            return "Bull Market"
        elif mean_return > 0.1 and volatility >= 0.2:
            return "Rally"
        elif mean_return <= 0 and volatility > 0.3:
            return "Crisis"
        elif mean_return <= 0 and volatility <= 0.3:
            return "Bear Market"
        elif volatility > 0.25:
            return "High Volatility"
        elif volatility < 0.15:
            return "Low Volatility"
        else:
            return "Normal"
    
    def display_risk_metrics(self):
        """Display comprehensive risk metrics"""
        st.subheader("📉 Comprehensive Risk Analytics")
        
        if not st.session_state.asset_data:
            st.info("Please load data first using the sidebar configuration.")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset for Risk Analysis",
            list(st.session_state.asset_data.keys()),
            key="risk_asset_selector"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) < 100:
            st.warning(f"Insufficient data for risk analysis. Need at least 100 observations.")
            return
        
        # Calculate risk metrics
        returns = df['Returns'].dropna()
        risk_metrics = self.analytics.calculate_risk_metrics(returns)
        
        if not risk_metrics:
            st.error("Failed to calculate risk metrics")
            return
        
        # Display key risk metrics
        st.subheader("Key Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2f}%")
            st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2f}%")
        
        with col2:
            st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2f}%")
            st.metric("Drawdown Duration", f"{risk_metrics.get('max_dd_duration', 0)} days")
        
        with col3:
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
        
        with col4:
            st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")
            st.metric("Omega Ratio", f"{risk_metrics.get('omega_ratio', 0):.2f}")
        
        # Tail risk metrics
        st.subheader("Tail Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
            if risk_metrics.get('skewness', 0) < -0.5:
                st.warning("Negative skew indicates higher left-tail risk")
            elif risk_metrics.get('skewness', 0) > 0.5:
                st.info("Positive skew indicates higher right-tail potential")
        
        with col2:
            st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
            if risk_metrics.get('kurtosis', 0) > 3:
                st.warning("High kurtosis indicates fat tails (more extreme events)")
        
        with col3:
            st.metric("Tail Ratio", f"{risk_metrics.get('tail_ratio', 0):.2f}")
            st.metric("Extreme Events", f"{risk_metrics.get('extreme_negative_pct', 0):.1f}%")
        
        # Risk visualization
        asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
        fig = self.viz.create_risk_metrics_chart(risk_metrics, f"{asset_name} ({selected_asset})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical drawdown analysis
        st.subheader("Historical Drawdown Analysis")
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.3)',
            line=dict(color=self.viz.color_palette['danger'], width=2)
        ))
        
        fig.update_layout(
            title="Historical Drawdowns",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        if len(drawdown) > 0:
            drawdown_stats = {
                'Max Drawdown': f"{drawdown.min():.2f}%",
                'Avg Drawdown': f"{drawdown[drawdown < 0].mean():.2f}%" if len(drawdown[drawdown < 0]) > 0 else "0%",
                'Drawdown Frequency': f"{(drawdown < 0).sum() / len(drawdown) * 100:.1f}%",
                'Recovery Time (Avg)': "N/A"  # Would need more complex calculation
            }
            
            stats_df = pd.DataFrame(list(drawdown_stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
    
    def display_correlation_matrix(self):
        """Display correlation matrix"""
        st.subheader("🔄 Correlation Matrix")
        
        if not st.session_state.asset_data or len(st.session_state.asset_data) < 2:
            st.info("Please load at least 2 assets to view correlation matrix.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.analytics.calculate_correlation_matrix(st.session_state.asset_data)
        
        if corr_matrix.empty:
            st.warning("Insufficient data to calculate correlations.")
            return
        
        # Display correlation matrix
        fig = self.viz.create_correlation_heatmap(
            corr_matrix,
            "Commodities Returns Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.subheader("Correlation Insights")
        
        # Find strongest correlations
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                corr_values.append({
                    'Asset 1': asset1,
                    'Asset 2': asset2,
                    'Correlation': correlation,
                    'Abs Correlation': abs(correlation)
                })
        
        if corr_values:
            corr_df = pd.DataFrame(corr_values)
            
            # Strongest positive correlations
            st.markdown("#### Strongest Positive Correlations")
            top_positive = corr_df.nlargest(5, 'Correlation')
            st.dataframe(
                top_positive[['Asset 1', 'Asset 2', 'Correlation']].style.format({'Correlation': '{:.3f}'}),
                use_container_width=True
            )
            
            # Strongest negative correlations
            st.markdown("#### Strongest Negative Correlations")
            top_negative = corr_df.nsmallest(5, 'Correlation')
            st.dataframe(
                top_negative[['Asset 1', 'Asset 2', 'Correlation']].style.format({'Correlation': '{:.3f}'}),
                use_container_width=True
            )
            
            # Average correlation
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            st.metric("Average Correlation", f"{avg_corr:.3f}")
            
            # Correlation distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=corr_df['Correlation'],
                nbinsx=20,
                marker_color=self.viz.color_palette['primary'],
                opacity=0.7,
                name='Correlation Distribution'
            ))
            
            fig.update_layout(
                title="Correlation Distribution",
                xaxis_title="Correlation Coefficient",
                yaxis_title="Frequency",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_portfolio_analytics(self):
        """Display portfolio analytics"""
        st.subheader("🏛️ Portfolio Analytics")
        
        if not st.session_state.asset_data or len(st.session_state.asset_data) < 2:
            st.info("Please load at least 2 assets for portfolio analysis.")
            return
        
        # Portfolio construction
        st.markdown("### Portfolio Construction")
        
        assets = list(st.session_state.asset_data.keys())
        weights = {}
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Asset Weights**")
            for asset in assets:
                weight = st.slider(
                    f"{asset} Weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0 / len(assets),
                    step=1.0,
                    key=f"weight_{asset}"
                )
                weights[asset] = weight / 100.0  # Convert to decimal
        
        with col2:
            st.markdown("**Portfolio Summary**")
            total_weight = sum(weights.values())
            st.metric("Total Weight", f"{total_weight*100:.1f}%")
            
            if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
                st.warning(f"Weights sum to {total_weight*100:.1f}%. Normalizing to 100%.")
                # Normalize weights
                weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate portfolio returns
        if st.button("Analyze Portfolio", type="primary"):
            with st.spinner("Calculating portfolio metrics..."):
                # Get aligned returns
                returns_df = pd.DataFrame()
                for asset, df in st.session_state.asset_data.items():
                    if asset in weights and weights[asset] > 0:
                        returns_df[asset] = df['Returns']
                
                # Align and drop NaN
                returns_df = returns_df.dropna()
                
                if len(returns_df) < 50:
                    st.error("Insufficient overlapping data for portfolio analysis")
                    return
                
                # Calculate weighted portfolio returns
                portfolio_returns = pd.Series(0, index=returns_df.index)
                for asset in weights:
                    if asset in returns_df.columns:
                        portfolio_returns += returns_df[asset] * weights[asset]
                
                # Calculate portfolio metrics
                portfolio_metrics = self.analytics.calculate_risk_metrics(portfolio_returns)
                
                # Display portfolio metrics
                st.subheader("Portfolio Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_return = ((1 + portfolio_returns).prod() - 1) * 100
                    annual_return = ((1 + portfolio_returns.mean()) ** 252 - 1) * 100
                    st.metric("Total Return", f"{total_return:.1f}%")
                    st.metric("Annual Return", f"{annual_return:.1f}%")
                
                with col2:
                    annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    st.metric("Annual Volatility", f"{annual_vol:.1f}%")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{portfolio_metrics.get('max_drawdown', 0):.1f}%")
                    st.metric("VaR (95%)", f"{portfolio_metrics.get('var_95', 0):.2f}%")
                
                with col4:
                    st.metric("Sortino Ratio", f"{portfolio_metrics.get('sortino_ratio', 0):.2f}")
                    st.metric("Calmar Ratio", f"{portfolio_metrics.get('calmar_ratio', 0):.2f}")
                
                # Portfolio composition
                st.subheader("Portfolio Composition")
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(weights.keys()),
                    values=[weights[k] * 100 for k in weights.keys()],
                    hole=0.4,
                    marker_colors=[self.viz.color_palette['primary'], 
                                  self.viz.color_palette['secondary'],
                                  self.viz.color_palette['success'],
                                  self.viz.color_palette['warning'],
                                  self.viz.color_palette['danger'],
                                  self.viz.color_palette['info']]
                )])
                
                fig.update_layout(
                    title="Portfolio Allocation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Portfolio returns chart
                st.subheader("Portfolio Cumulative Returns")
                
                cumulative_returns = (1 + portfolio_returns).cumprod()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns * 100,
                    name='Portfolio',
                    line=dict(color=self.viz.color_palette['primary'], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                # Add individual assets for comparison
                for asset in weights:
                    if asset in returns_df.columns and weights[asset] > 0.1:  # Only show significant holdings
                        asset_cumulative = (1 + returns_df[asset]).cumprod() * 100
                        fig.add_trace(go.Scatter(
                            x=asset_cumulative.index,
                            y=asset_cumulative,
                            name=asset,
                            line=dict(width=1, dash='dot'),
                            opacity=0.5
                        ))
                
                fig.update_layout(
                    title="Cumulative Returns Comparison",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk contribution analysis
                st.subheader("Risk Contribution Analysis")
                
                # Calculate covariance matrix
                cov_matrix = returns_df.cov() * 252  # Annualize
                
                # Calculate portfolio variance
                weights_array = np.array([weights[asset] for asset in returns_df.columns])
                portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate marginal contributions
                marginal_contrib = (cov_matrix.values @ weights_array) / portfolio_volatility
                contrib_to_risk = weights_array * marginal_contrib
                percent_contrib = contrib_to_risk / portfolio_volatility * 100
                
                # Create risk contribution dataframe
                risk_contrib_data = []
                for i, asset in enumerate(returns_df.columns):
                    risk_contrib_data.append({
                        'Asset': asset,
                        'Weight': f"{weights[asset]*100:.1f}%",
                        'Marginal Contribution': f"{marginal_contrib[i]:.4f}",
                        'Contribution to Risk': f"{contrib_to_risk[i]:.4f}",
                        'Percent Contribution': f"{percent_contrib[i]:.1f}%"
                    })
                
                risk_contrib_df = pd.DataFrame(risk_contrib_data)
                st.dataframe(risk_contrib_df, use_container_width=True)
                
                # Benchmark comparison (if available)
                if 'SPY' in st.session_state.benchmark_data:
                    st.subheader("Benchmark Comparison (vs S&P 500)")
                    
                    spy_returns = st.session_state.benchmark_data['SPY']['Returns']
                    
                    # Align dates
                    aligned = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
                    aligned.columns = ['Portfolio', 'SPY']
                    
                    if len(aligned) > 50:
                        # Calculate beta and alpha
                        X = sm.add_constant(aligned['SPY'])
                        y = aligned['Portfolio']
                        model = sm.OLS(y, X).fit()
                        
                        beta = model.params['SPY']
                        alpha = model.params['const'] * 252 * 100  # Annualized percentage
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Beta (vs SPY)", f"{beta:.2f}")
                        
                        with col2:
                            st.metric("Alpha (Annual)", f"{alpha:.2f}%")
                        
                        with col3:
                            tracking_error = (aligned['Portfolio'] - aligned['SPY']).std() * np.sqrt(252) * 100
                            st.metric("Tracking Error", f"{tracking_error:.2f}%")
                        
                        with col4:
                            information_ratio = (aligned['Portfolio'].mean() - aligned['SPY'].mean()) * np.sqrt(252) / tracking_error * 100 if tracking_error > 0 else 0
                            st.metric("Information Ratio", f"{information_ratio:.2f}")
                        
                        # Cumulative returns comparison
                        cum_portfolio = (1 + aligned['Portfolio']).cumprod()
                        cum_spy = (1 + aligned['SPY']).cumprod()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=cum_portfolio.index,
                            y=cum_portfolio * 100,
                            name='Portfolio',
                            line=dict(color=self.viz.color_palette['primary'], width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=cum_spy.index,
                            y=cum_spy * 100,
                            name='S&P 500',
                            line=dict(color=self.viz.color_palette['dark'], width=3)
                        ))
                        
                        fig.update_layout(
                            title="Portfolio vs S&P 500",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return (%)",
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def display_footer(self):
        """Display application footer"""
        st.markdown("""
        <div class="professional-footer">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="text-align: left;">
                    <h3 style="margin: 0; color: white;">🏛️ Institutional Commodities Analytics</h3>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">
                        Advanced analytics platform for institutional investors
                    </p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; color: rgba(255,255,255,0.8);">
                        Data Source: Yahoo Finance<br>
                        Last Update: {timestamp}<br>
                        Version: 3.0.0 | Streamlit Cloud
                    </p>
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 1.5rem 0;">
            <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
                <p>
                    © 2024 Institutional Commodities Analytics. For professional use only.<br>
                    Past performance is not indicative of future results. All investments involve risk.
                </p>
            </div>
        </div>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize platform
    platform = CommoditiesAnalyticsPlatform()
    
    # Display header
    platform.display_header()
    
    # Setup sidebar and get configuration
    config = platform.setup_sidebar()
    
    if config:
        start_date, end_date, selected_assets, selected_benchmarks, load_button = config
        
        # Load data if button clicked
        if load_button and selected_assets:
            platform.load_data(start_date, end_date, selected_assets, selected_benchmarks)
        
        # Display analytics if data is loaded
        if st.session_state.data_loaded and st.session_state.asset_data:
            platform.display_dashboard()
        elif load_button and not selected_assets:
            st.warning("⚠️ Please select at least one commodity to analyze.")
        elif not st.session_state.data_loaded:
            st.info("👈 Configure your analysis using the sidebar and click 'Load Data' to begin.")
    
    # Display footer
    platform.display_footer()

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")
        
        # Show error details in expander
        with st.expander("Technical Details"):
            import traceback
            st.code(traceback.format_exc())
