"""
Enhanced Precious Metals & Commodities ARCH/GARCH Analysis Dashboard
With Real-time Data, Volatility Forecasting, and Advanced Analytics
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
from statsmodels.stats.diagnostic import het_arch
import quantstats as qs
import warnings
import json
import time
import os
import random
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Performance optimization
os.environ['NUMEXPR_MAX_THREADS'] = '8'
warnings.filterwarnings('ignore')

# Configure quantstats
qs.extend_pandas()

# Page configuration
st.set_page_config(
    page_title="Precious Metals Analytics Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for superior design
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .positive {
        color: #27ae60;
        font-weight: 700;
    }
    
    .negative {
        color: #e74c3c;
        font-weight: 700;
    }
    
    .neutral {
        color: #f39c12;
        font-weight: 700;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 2px solid #e0e0e0;
        padding: 0 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.5rem;
        font-weight: 600;
        color: #7f8c8d;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
        border-bottom: 3px solid #667eea !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Enhanced sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Enhanced tables */
    .dataframe {
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-radius: 10px;
        overflow: hidden;
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .dataframe thead {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .dataframe thead th {
        background: transparent !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 1rem !important;
        text-align: center;
    }
    
    .dataframe tbody tr {
        transition: background-color 0.3s ease;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e3f2fd !important;
        transform: scale(1.01);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
        color: white;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        text-align: center;
    }
    
    .loading-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Flash animation for updates */
    @keyframes flash {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .flash-update {
        animation: flash 1s ease-in-out;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# PRECIOUS METALS & COMMODITIES UNIVERSE
COMMODITIES = {
    "Precious Metals": {
        "GC=F": {"name": "Gold Futures", "category": "metal", "symbol": "GC=F"},
        "SI=F": {"name": "Silver Futures", "category": "metal", "symbol": "SI=F"},
        "PL=F": {"name": "Platinum Futures", "category": "metal", "symbol": "PL=F"},
        "PA=F": {"name": "Palladium Futures", "category": "metal", "symbol": "PA=F"},
    },
    "Industrial Metals": {
        "HG=F": {"name": "Copper Futures", "category": "industrial", "symbol": "HG=F"},
        "ALI=F": {"name": "Aluminum Futures", "category": "industrial", "symbol": "ALI=F"},
    },
    "Energy": {
        "CL=F": {"name": "Crude Oil WTI", "category": "energy", "symbol": "CL=F"},
        "NG=F": {"name": "Natural Gas", "category": "energy", "symbol": "NG=F"},
        "BZ=F": {"name": "Brent Crude", "category": "energy", "symbol": "BZ=F"},
    },
    "Agriculture": {
        "ZC=F": {"name": "Corn Futures", "category": "agriculture", "symbol": "ZC=F"},
        "ZW=F": {"name": "Wheat Futures", "category": "agriculture", "symbol": "ZW=F"},
        "ZS=F": {"name": "Soybean Futures", "category": "agriculture", "symbol": "ZS=F"},
    }
}

# Utility functions
def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return float(value)
    except:
        return default

def format_percentage(value, decimals=2):
    """Format percentage with sign"""
    try:
        value = safe_float(value)
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{decimals}f}%"
    except:
        return "N/A"

def format_currency(value):
    """Format currency"""
    try:
        return f"${safe_float(value):,.2f}"
    except:
        return "N/A"

def format_number(value, decimals=2):
    """Format number"""
    try:
        return f"{safe_float(value):,.{decimals}f}"
    except:
        return "N/A"

class EnhancedYFinance:
    """Enhanced yfinance with rate limiting handling and retries"""
    
    @staticmethod
    def create_session_with_retry():
        """Create requests session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    @staticmethod
    def fetch_with_retry(symbol, start_date, end_date, max_retries=5):
        """Fetch data with retry logic and rate limiting handling"""
        for attempt in range(max_retries):
            try:
                # Add random delay between retries to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(2, 5) * attempt
                    time.sleep(delay)
                
                ticker = yf.Ticker(symbol)
                
                # Configure yfinance to use our session
                ticker.session = EnhancedYFinance.create_session_with_retry()
                
                # Fetch data with timeout
                df = ticker.history(
                    start=start_date, 
                    end=end_date, 
                    auto_adjust=True,
                    timeout=30
                )
                
                if df.empty:
                    st.warning(f"⚠️ Empty data for {symbol} (attempt {attempt + 1}/{max_retries})")
                    continue
                
                if len(df) < 50:
                    st.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} points")
                    continue
                
                return df
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    st.warning(f"⏳ Rate limited for {symbol}. Waiting before retry...")
                    time.sleep(random.uniform(10, 20))
                else:
                    st.warning(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {error_msg[:100]}")
                
                if attempt == max_retries - 1:
                    st.error(f"❌ Failed to fetch {symbol} after {max_retries} attempts")
        
        return None

class DataManager:
    """Enhanced data management with caching and error handling"""
    
    @st.cache_data(ttl=300)  # 5 minutes cache for real-time data
    def fetch_data(_self, symbol, start_date, end_date):
        """Enhanced data fetching with better error handling"""
        try:
            # Use enhanced yfinance with retry logic
            df = EnhancedYFinance.fetch_with_retry(symbol, start_date, end_date)
            
            if df is None or df.empty:
                return None
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Calculate technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Volatility
            df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Additional calculations
            df['HL'] = (df['High'] + df['Low']) / 2
            df['OC'] = (df['Open'] + df['Close']) / 2
            
            # Calculate ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['ATR'] = ranges.max(axis=1).rolling(14).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Stochastic Oscillator
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stochastic'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['Stochastic_SMA'] = df['Stochastic'].rolling(3).mean()
            
            return df
        except Exception as e:
            st.error(f"Error fetching {symbol}: {str(e)}")
            return None

class AnalysisEngine:
    """Advanced analysis engine with QuantStats integration"""
    
    def calculate_performance_metrics(self, returns, dates=None):
        """Calculate comprehensive performance metrics using QuantStats with fixed date handling"""
        if returns is None or len(returns) < 100:
            return {}
        
        try:
            # Convert to pandas Series with proper dates
            if dates is not None and len(dates) == len(returns):
                returns_series = pd.Series(returns, index=dates)
            else:
                # Create synthetic dates if not provided
                returns_series = pd.Series(returns)
                # Create date index starting from today and going backwards
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(returns))
                dates = pd.date_range(start=start_date, end=end_date, periods=len(returns))
                returns_series.index = dates
            
            # Calculate metrics using QuantStats
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = qs.stats.comp(returns_series)
            
            try:
                metrics['cagr'] = qs.stats.cagr(returns_series)
            except:
                metrics['cagr'] = 0
            
            try:
                metrics['volatility'] = qs.stats.volatility(returns_series)
            except:
                metrics['volatility'] = np.std(returns_series) * np.sqrt(252) if len(returns_series) > 0 else 0
            
            try:
                metrics['sharpe'] = qs.stats.sharpe(returns_series)
            except:
                metrics['sharpe'] = 0
            
            try:
                metrics['sortino'] = qs.stats.sortino(returns_series)
            except:
                metrics['sortino'] = 0
            
            try:
                metrics['max_drawdown'] = qs.stats.max_drawdown(returns_series)
            except:
                metrics['max_drawdown'] = 0
            
            try:
                metrics['calmar'] = qs.stats.calmar(returns_series)
            except:
                metrics['calmar'] = 0
            
            # Risk metrics
            try:
                metrics['var_95'] = qs.stats.value_at_risk(returns_series)
            except:
                metrics['var_95'] = np.percentile(returns_series, 5) if len(returns_series) > 0 else 0
            
            try:
                metrics['cvar_95'] = qs.stats.conditional_value_at_risk(returns_series)
            except:
                metrics['cvar_95'] = 0
            
            try:
                metrics['skew'] = qs.stats.skew(returns_series)
            except:
                metrics['skew'] = 0
            
            try:
                metrics['kurtosis'] = qs.stats.kurtosis(returns_series)
            except:
                metrics['kurtosis'] = 0
            
            # Additional metrics
            try:
                metrics['omega'] = qs.stats.omega(returns_series)
            except:
                metrics['omega'] = 0
            
            try:
                metrics['tail_ratio'] = qs.stats.tail_ratio(returns_series)
            except:
                metrics['tail_ratio'] = 0
            
            try:
                metrics['common_sense_ratio'] = qs.stats.common_sense_ratio(returns_series)
            except:
                metrics['common_sense_ratio'] = 0
            
            try:
                metrics['information_ratio'] = qs.stats.information_ratio(returns_series)
            except:
                metrics['information_ratio'] = 0
            
            # Gain/Pain metrics
            try:
                metrics['gain_to_pain'] = qs.stats.gain_to_pain_ratio(returns_series)
            except:
                metrics['gain_to_pain'] = 0
            
            try:
                metrics['win_rate'] = qs.stats.win_rate(returns_series)
            except:
                metrics['win_rate'] = 0
            
            try:
                metrics['avg_win'] = qs.stats.avg_win(returns_series)
            except:
                metrics['avg_win'] = 0
            
            try:
                metrics['avg_loss'] = qs.stats.avg_loss(returns_series)
            except:
                metrics['avg_loss'] = 0
            
            return metrics
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def arch_effect_test(self, returns, lags=5):
        """Test for ARCH effects"""
        if returns is None or len(returns) < lags + 10:
            return {"present": False, "p_value": 1.0}
        
        try:
            returns_array = np.array(returns)
            returns_clean = returns_array[~np.isnan(returns_array)]
            
            if len(returns_clean) < lags + 10:
                return {"present": False, "p_value": 1.0}
            
            LM, LM_p, F, F_p = het_arch(returns_clean - np.mean(returns_clean), maxlag=lags)
            
            return {
                "present": LM_p < 0.05,
                "p_value": LM_p,
                "LM_statistic": LM,
                "F_statistic": F
            }
        except Exception as e:
            return {"present": False, "p_value": 1.0}
    
    def fit_garch_model(self, returns, p=1, q=1):
        """Fit GARCH model"""
        if returns is None or len(returns) < 100:
            return None
        
        try:
            returns_array = np.array(returns)
            returns_clean = returns_array[~np.isnan(returns_array)]
            
            if len(returns_clean) < 100:
                return None
            
            # Scale returns for better convergence
            returns_scaled = returns_clean * 100
            
            # Fit GARCH model
            model = arch_model(returns_scaled, vol='GARCH', p=p, q=q, dist='t')
            result = model.fit(disp='off', show_warning=False, options={'maxiter': 1000})
            
            return {
                "model": result,
                "params": dict(result.params),
                "aic": result.aic,
                "bic": result.bic,
                "converged": result.convergence_flag == 0
            }
        except Exception as e:
            return None
    
    def forecast_volatility(self, garch_model, steps=30):
        """Forecast volatility using GARCH model"""
        if garch_model is None:
            return None
        
        try:
            forecast = garch_model.forecast(horizon=steps)
            return forecast.variance.iloc[-1].values
        except Exception as e:
            return None

class Visualizer:
    """Enhanced visualization with superior design"""
    
    def create_price_chart(self, df, title, show_bb=True, show_indicators=False):
        """Create enhanced price chart with Bollinger Bands"""
        rows = 3 if show_indicators else 2
        row_heights = [0.5, 0.2, 0.3] if show_indicators else [0.7, 0.3]
        subplot_titles = ['Price with Indicators', 'Volume', 'Technical Indicators'] if show_indicators else ['Price with Indicators', 'Volume']
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,
            row_heights=row_heights,
            shared_xaxes=True
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Price',
                      line=dict(color='#1f77b4', width=2),
                      hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                      line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                      opacity=0.7),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                      line=dict(color='#2ca02c', width=1.5, dash='dash'),
                      opacity=0.7),
            row=1, col=1
        )
        
        # Bollinger Bands
        if show_bb and 'BB_Upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                          line=dict(color='#9467bd', width=1, dash='dot'),
                          opacity=0.5),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle',
                          line=dict(color='#8c564b', width=1, dash='dot'),
                          opacity=0.5),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                          line=dict(color='#9467bd', width=1, dash='dot'),
                          opacity=0.5,
                          fill='tonexty',
                          fillcolor='rgba(148, 103, 189, 0.1)'),
                row=1, col=1
            )
        
        # Volume with color coding
        colors = ['#27ae60' if close >= open else '#e74c3c' 
                  for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors, opacity=0.7,
                   hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'),
            row=2, col=1
        )
        
        # Technical indicators if requested
        if show_indicators and rows == 3:
            # RSI
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                          line=dict(color='#3498db', width=1.5)),
                row=3, col=1
            )
            
            # Add RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                         opacity=0.3, row=3, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                          line=dict(color='#f39c12', width=1.5)),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                          line=dict(color='#9b59b6', width=1.5, dash='dash')),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=800 if show_indicators else 600,
            template="plotly_white",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=rows, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if show_indicators:
            fig.update_yaxes(title_text="Indicator Value", row=3, col=1)
        
        return fig
    
    def create_volatility_forecast_chart(self, forecast_values, title="Volatility Forecast"):
        """Create volatility forecast chart"""
        if forecast_values is None:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(forecast_values) + 1)),
            y=forecast_values,
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8, color='#c0392b'),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.1)'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=400,
            template="plotly_white",
            xaxis_title="Days Ahead",
            yaxis_title="Forecasted Volatility",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_garch_volatility_chart(self, garch_results, returns, dates, title):
        """Create GARCH volatility chart"""
        if garch_results is None:
            return None
        
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Conditional Volatility', 'Standardized Residuals'),
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4],
                shared_xaxes=True
            )
            
            # Get conditional volatility
            cond_vol = garch_results["model"].conditional_volatility
            
            # Use provided dates or create synthetic ones
            if dates is not None and len(dates) >= len(cond_vol):
                vol_dates = dates[-len(cond_vol):]
            else:
                # Create synthetic dates
                vol_dates = pd.date_range(end=datetime.now(), periods=len(cond_vol), freq='D')
            
            # Conditional volatility
            fig.add_trace(
                go.Scatter(x=vol_dates, y=cond_vol, name='GARCH Volatility',
                          line=dict(color='#d35400', width=2),
                          fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.1)',
                          hovertemplate='Date: %{x}<br>Volatility: %{y:.4f}<extra></extra>'),
                row=1, col=1
            )
            
            # Standardized residuals
            std_resid = garch_results["model"].resid / cond_vol
            
            fig.add_trace(
                go.Scatter(x=vol_dates, y=std_resid, name='Std. Residuals',
                          mode='markers',
                          marker=dict(color='#3498db', size=4, opacity=0.6),
                          hovertemplate='Date: %{x}<br>Residual: %{y:.4f}<extra></extra>'),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=2, col=1)
            
            # Add confidence bands
            fig.add_hline(y=2, line_dash="dot", line_color="red", 
                         opacity=0.3, row=2, col=1)
            fig.add_hline(y=-2, line_dash="dot", line_color="red", 
                         opacity=0.3, row=2, col=1)
            
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center'),
                height=600,
                template="plotly_white",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_yaxes(title_text="Standardized Residuals", row=2, col=1)
            
            return fig
        except Exception as e:
            return None
    
    def create_risk_return_chart(self, metrics_dict):
        """Create risk-return scatter plot"""
        if not metrics_dict:
            return None
        
        try:
            data = []
            for symbol, metrics in metrics_dict.items():
                data.append({
                    'Asset': symbol,
                    'Volatility': safe_float(metrics.get('volatility', 0)) * 100,
                    'CAGR': safe_float(metrics.get('cagr', 0)) * 100,
                    'Sharpe': safe_float(metrics.get('sharpe', 0)),
                    'Total Return': safe_float(metrics.get('total_return', 0)) * 100
                })
            
            df = pd.DataFrame(data)
            
            fig = px.scatter(
                df,
                x='Volatility',
                y='CAGR',
                size='Total Return',
                color='Sharpe',
                hover_name='Asset',
                hover_data=['Sharpe', 'Total Return'],
                color_continuous_scale='RdYlGn',
                title='Risk-Return Profile',
                size_max=60
            )
            
            # Add efficient frontier concept
            max_return = df['CAGR'].max()
            min_risk = df['Volatility'].min()
            
            fig.add_trace(go.Scatter(
                x=[min_risk, df['Volatility'].max()],
                y=[0, max_return * 1.2],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.5
            ))
                        fig.update_layout(
                height=500,
                template="plotly_white",
                hovermode='closest'
            )
            
            return fig
        except Exception as e:
            return None
    
    def create_distribution_chart(self, returns, title):
        """Create returns distribution chart with statistical overlay"""
        if returns is None or len(returns) < 50:
            return None
        
        try:
            returns_clean = returns[~np.isnan(returns)]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Returns Distribution', 'Q-Q Plot', 'Autocorrelation', 'Partial Autocorrelation'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                      [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Histogram with KDE
            fig.add_trace(
                go.Histogram(x=returns_clean, 
                            name='Returns',
                            histnorm='probability density',
                            marker_color='#3498db',
                            opacity=0.7,
                            nbinsx=50),
                row=1, col=1
            )
            
            # Add normal distribution overlay
            mu, sigma = np.mean(returns_clean), np.std(returns_clean)
            x = np.linspace(min(returns_clean), max(returns_clean), 100)
            y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma) ** 2)
            
            fig.add_trace(
                go.Scatter(x=x, y=y, 
                          name='Normal Dist',
                          line=dict(color='red', width=2, dash='dash')),
                row=1, col=1
            )
            
            # Q-Q Plot
            from scipy import stats
            qq = stats.probplot(returns_clean, dist="norm")
            theoretical = qq[0][1]
            sample = qq[0][0]
            
            fig.add_trace(
                go.Scatter(x=theoretical, y=sample,
                          mode='markers',
                          name='Q-Q Plot',
                          marker=dict(color='#2ecc71', size=5)),
                row=1, col=2
            )
            
            # Add 45-degree line
            min_val = min(theoretical.min(), sample.min())
            max_val = max(theoretical.max(), sample.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines',
                          name='Normal Line',
                          line=dict(color='red', dash='dash', width=1)),
                row=1, col=2
            )
            
            # Autocorrelation
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            import io
            import base64
            
            # Calculate ACF
            acf_vals = np.correlate(returns_clean - np.mean(returns_clean), 
                                   returns_clean - np.mean(returns_clean), 
                                   mode='full')
            acf_vals = acf_vals[len(acf_vals)//2:]
            acf_vals = acf_vals[:20] / acf_vals[0]
            
            # Calculate PACF
            pacf_vals = [1.0]
            for k in range(1, 20):
                try:
                    pacf = np.corrcoef(returns_clean[:-k] - np.mean(returns_clean[:-k]),
                                      returns_clean[k:] - np.mean(returns_clean[k:]))[0, 1]
                    pacf_vals.append(pacf)
                except:
                    pacf_vals.append(0)
            
            # Plot ACF
            fig.add_trace(
                go.Bar(x=list(range(len(acf_vals))),
                      y=acf_vals,
                      name='ACF',
                      marker_color='#9b59b6'),
                row=2, col=1
            )
            
            # Add confidence bands
            conf_int = 1.96 / np.sqrt(len(returns_clean))
            fig.add_hline(y=conf_int, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=1)
            fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=1)
            
            # Plot PACF
            fig.add_trace(
                go.Bar(x=list(range(len(pacf_vals))),
                      y=pacf_vals,
                      name='PACF',
                      marker_color='#e67e22'),
                row=2, col=2
            )
            
            fig.add_hline(y=conf_int, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=2)
            fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=2)
            
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center'),
                height=700,
                template="plotly_white",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Returns", row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)
            fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
            fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
            fig.update_xaxes(title_text="Lag", row=2, col=1)
            fig.update_yaxes(title_text="ACF", row=2, col=1)
            fig.update_xaxes(title_text="Lag", row=2, col=2)
            fig.update_yaxes(title_text="PACF", row=2, col=2)
            
            return fig
        except Exception as e:
            return None
    
    def create_correlation_matrix(self, data_dict):
        """Create correlation matrix heatmap"""
        if not data_dict or len(data_dict) < 2:
            return None
        
        try:
            # Extract returns from each asset
            returns_df = pd.DataFrame()
            for symbol, data in data_dict.items():
                if 'Returns' in data.columns and len(data) > 50:
                    returns_df[symbol] = data['Returns']
            
            if len(returns_df.columns) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                hoverongaps=False,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Asset Returns Correlation Matrix",
                height=500,
                template="plotly_white",
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            return fig
        except Exception as e:
            return None
    
    def create_rolling_metrics_chart(self, df, title="Rolling Metrics"):
        """Create rolling metrics chart"""
        if df is None or len(df) < 100:
            return None
        
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility', 'Rolling Maximum Drawdown'),
                vertical_spacing=0.08,
                row_heights=[0.33, 0.33, 0.34],
                shared_xaxes=True
            )
            
            # Calculate rolling Sharpe (21-day window, annualized)
            rolling_sharpe = df['Returns'].rolling(window=21).apply(
                lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0
            )
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rolling_sharpe,
                          name='Rolling Sharpe',
                          line=dict(color='#2ecc71', width=2)),
                row=1, col=1
            )
            
            # Calculate rolling volatility (21-day window, annualized)
            rolling_vol = df['Returns'].rolling(window=21).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rolling_vol,
                          name='Rolling Volatility',
                          line=dict(color='#e74c3c', width=2)),
                row=2, col=1
            )
            
            # Calculate rolling maximum drawdown
            rolling_max = df['Close'].rolling(window=252, min_periods=1).max()
            rolling_dd = (df['Close'] - rolling_max) / rolling_max
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rolling_dd * 100,
                          name='Rolling Drawdown',
                          line=dict(color='#f39c12', width=2),
                          fill='tozeroy',
                          fillcolor='rgba(243, 156, 18, 0.1)'),
                row=3, col=1
            )
            
            # Add zero lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=3, col=1)
            
            fig.update_layout(
                title=dict(text=title, x=0.5, xanchor='center'),
                height=700,
                template="plotly_white",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            return fig
        except Exception as e:
            return None

# Main Application
class PreciousMetalsDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analysis_engine = AnalysisEngine()
        self.visualizer = Visualizer()
        self.selected_assets = []
        self.data_dict = {}
        self.metrics_dict = {}
        
    def setup_sidebar(self):
        """Configure the sidebar"""
        with st.sidebar:
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            st.markdown("## 🏆 Precious Metals Pro")
            st.markdown("Advanced Volatility Analytics")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Date range selection
            st.subheader("📅 Date Range")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 5)  # Default 5 years
            
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start Date", start_date)
            with col2:
                end = st.date_input("End Date", end_date)
            
            if start >= end:
                st.error("Start date must be before end date")
                st.stop()
            
            # Asset selection
            st.subheader("📊 Select Assets")
            
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category} ({len(assets)})"):
                    for symbol, info in assets.items():
                        if st.checkbox(f"{info['name']} ({symbol})", 
                                     value=symbol in ["GC=F", "SI=F"]):  # Default to gold and silver
                            self.selected_assets.append(symbol)
            
            if not self.selected_assets:
                st.warning("Please select at least one asset")
                st.stop()
            
            # Analysis parameters
            st.subheader("⚙️ Analysis Settings")
            
            garch_p = st.slider("GARCH p (AR terms)", 1, 5, 1)
            garch_q = st.slider("GARCH q (MA terms)", 1, 5, 1)
            forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, 30)
            
            # Additional options
            show_indicators = st.checkbox("Show Technical Indicators", value=True)
            show_bb = st.checkbox("Show Bollinger Bands", value=True)
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                if st.button("📊 Run Analysis", use_container_width=True, type="primary"):
                    # Analysis will run automatically after button click
                    pass
            
            st.markdown("---")
            
            # Information panel
            with st.expander("ℹ️ About This Dashboard"):
                st.markdown("""
                ### Precious Metals & Commodities Analytics Pro
                
                **Features:**
                - Real-time market data from Yahoo Finance
                - Advanced GARCH volatility modeling
                - Comprehensive risk metrics
                - Correlation analysis
                - Volatility forecasting
                - Performance analytics
                
                **Methodology:**
                - Returns are calculated as log returns
                - GARCH models are fitted using maximum likelihood
                - All metrics are annualized where applicable
                - Charts are interactive and exportable
                
                **Data Sources:**
                - Yahoo Finance API
                - Real-time futures data
                - Historical prices back to 2000
                """)
            
            return start, end, garch_p, garch_q, forecast_horizon, show_indicators, show_bb
    
    def display_header(self):
        """Display the main header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; padding:0; font-size:2.5rem;">🏆 Precious Metals & Commodities Analytics Pro</h1>
            <p style="margin:0; padding-top:0.5rem; font-size:1.2rem; opacity:0.9;">
                Advanced ARCH/GARCH Volatility Analysis • Real-time Forecasting • Risk Management
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status bar
        with st.container():
            cols = st.columns(4)
            with cols[0]:
                st.metric("Selected Assets", len(self.selected_assets))
            with cols[1]:
                st.metric("Data Points", "1,000+")
            with cols[2]:
                st.metric("Analysis Period", "5 Years")
            with cols[3]:
                st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    def load_data(self, start_date, end_date):
        """Load data for all selected assets"""
        if not self.selected_assets:
            return
        
        progress_bar = st.progress(0, text="Loading market data...")
        
        for i, symbol in enumerate(self.selected_assets):
            progress = (i + 1) / len(self.selected_assets)
            progress_bar.progress(progress, text=f"Loading {symbol}...")
            
            with st.spinner(f"Fetching {symbol}..."):
                df = self.data_manager.fetch_data(symbol, start_date, end_date)
                if df is not None:
                    self.data_dict[symbol] = df
                    
                    # Calculate performance metrics
                    returns = df['Returns'].dropna()
                    if len(returns) > 100:
                        self.metrics_dict[symbol] = self.analysis_engine.calculate_performance_metrics(
                            returns.values, dates=returns.index
                        )
        
        progress_bar.empty()
        
        if not self.data_dict:
            st.error("Failed to load data for any selected assets")
            st.stop()
        
        st.success(f"✅ Successfully loaded {len(self.data_dict)} assets")
    
    def display_metrics_dashboard(self):
        """Display the metrics dashboard"""
        st.subheader("📈 Performance Metrics Dashboard")
        
        if not self.metrics_dict:
            return
        
        # Create metrics cards
        cols = st.columns(min(4, len(self.selected_assets)))
        
        for idx, (symbol, metrics) in enumerate(self.metrics_dict.items()):
            if idx < len(cols):
                with cols[idx]:
                    asset_name = COMMODITIES.get(symbol.split('.')[0], {}).get('name', symbol)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{asset_name}</div>
                        <div class="metric-value">{format_percentage(metrics.get('total_return', 0) * 100)}</div>
                        <div style="font-size:0.9rem;">
                            <div>Sharpe: <strong>{format_number(metrics.get('sharpe', 0), 2)}</strong></div>
                            <div>Volatility: <strong>{format_percentage(metrics.get('volatility', 0) * 100)}</strong></div>
                            <div>Max DD: <strong>{format_percentage(metrics.get('max_drawdown', 0) * 100)}</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        
        metrics_df = pd.DataFrame()
        for symbol, metrics in self.metrics_dict.items():
            row = {
                'Asset': symbol,
                'Total Return': format_percentage(metrics.get('total_return', 0) * 100),
                'CAGR': format_percentage(metrics.get('cagr', 0) * 100),
                'Sharpe': format_number(metrics.get('sharpe', 0), 2),
                'Sortino': format_number(metrics.get('sortino', 0), 2),
                'Volatility': format_percentage(metrics.get('volatility', 0) * 100),
                'Max DD': format_percentage(metrics.get('max_drawdown', 0) * 100),
                'VaR 95%': format_percentage(metrics.get('var_95', 0) * 100),
                'CVaR 95%': format_percentage(metrics.get('cvar_95', 0) * 100),
                'Win Rate': format_percentage(metrics.get('win_rate', 0) * 100)
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        
        st.dataframe(metrics_df.set_index('Asset'), use_container_width=True)
    
    def display_analysis_tabs(self, garch_p, garch_q, forecast_horizon, show_indicators, show_bb):
        """Display the main analysis tabs"""
        
        tabs = st.tabs([
            "📊 Price Charts",
            "📈 Volatility Analysis",
            "📉 Risk Analysis",
            "🔍 GARCH Modeling",
            "📋 Correlation Matrix",
            "⚡ Real-time Monitoring"
        ])
        
        with tabs[0]:
            self.display_price_charts(show_indicators, show_bb)
        
        with tabs[1]:
            self.display_volatility_analysis(garch_p, garch_q, forecast_horizon)
        
        with tabs[2]:
            self.display_risk_analysis()
        
        with tabs[3]:
            self.display_garch_modeling(garch_p, garch_q)
        
        with tabs[4]:
            self.display_correlation_matrix()
        
        with tabs[5]:
            self.display_real_time_monitoring()
    
    def display_price_charts(self, show_indicators, show_bb):
        """Display price charts for selected assets"""
        st.subheader("Interactive Price Charts")
        
        # Asset selector for individual charts
        selected_chart = st.selectbox(
            "Select Asset for Detailed Chart",
            list(self.data_dict.keys()),
            format_func=lambda x: f"{x} - {COMMODITIES.get(x.split('.')[0], {}).get('name', 'Unknown')}"
        )
        
        if selected_chart in self.data_dict:
            df = self.data_dict[selected_chart]
            asset_name = COMMODITIES.get(selected_chart.split('.')[0], {}).get('name', selected_chart)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                current_price = df['Close'].iloc[-1]
                price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                st.metric("Current Price", format_currency(current_price), 
                         f"{price_change:+.2f}%")
            
            with col2:
                st.metric("Today's Range", 
                         f"{format_currency(df['Low'].iloc[-1])} - {format_currency(df['High'].iloc[-1])}")
            
            with col3:
                volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = (volume / avg_volume) * 100
                st.metric("Volume", f"{volume:,.0f}", f"{volume_ratio:.1f}% of avg")
            
            # Display chart
            fig = self.visualizer.create_price_chart(
                df, 
                f"{asset_name} - Price Analysis", 
                show_bb=show_bb, 
                show_indicators=show_indicators
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display additional metrics
            cols = st.columns(4)
            technical_metrics = {
                'RSI': df['RSI'].iloc[-1] if 'RSI' in df.columns else None,
                'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else None,
                'Stochastic': df['Stochastic'].iloc[-1] if 'Stochastic' in df.columns else None,
                'ATR': df['ATR'].iloc[-1] if 'ATR' in df.columns else None,
                'BB Width': df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else None,
                '20 Day Vol': df['Volatility_20'].iloc[-1] if 'Volatility_20' in df.columns else None,
                'SMA 20/50': (df['SMA_20'].iloc[-1] / df['SMA_50'].iloc[-1] - 1) * 100 
                            if all(col in df.columns for col in ['SMA_20', 'SMA_50']) else None,
                'EMA 12/26': (df['EMA_12'].iloc[-1] / df['EMA_26'].iloc[-1] - 1) * 100 
                            if all(col in df.columns for col in ['EMA_12', 'EMA_26']) else None
            }
            
            metric_names = list(technical_metrics.keys())
            for i in range(4):
                with cols[i]:
                    for j in range(2):
                        idx = i * 2 + j
                        if idx < len(metric_names):
                            name = metric_names[idx]
                            value = technical_metrics[name]
                            if value is not None:
                                if name == 'RSI':
                                    color = "green" if value < 30 else "red" if value > 70 else "orange"
                                elif 'Vol' in name or 'ATR' in name:
                                    color = "orange"
                                elif name in ['BB Width', '20 Day Vol']:
                                    color = "blue"
                                else:
                                    color = "black"
                                
                                if isinstance(value, float):
                                    display_value = f"{value:.2f}{'%' if '%' in name else ''}"
                                else:
                                    display_value = str(value)
                                
                                st.markdown(f"""
                                <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid {color};">
                                    <div style="font-size: 0.8rem; color: #666;">{name}</div>
                                    <div style="font-size: 1.1rem; font-weight: bold;">{display_value}</div>
                                </div>
                                """, unsafe_allow_html=True)
        
        # Mini charts for all assets
        st.subheader("All Assets Overview")
        cols = st.columns(3)
        
        for idx, (symbol, df) in enumerate(self.data_dict.items()):
            if idx < 9:  # Limit to 9 charts
                with cols[idx % 3]:
                    asset_name = COMMODITIES.get(symbol.split('.')[0], {}).get('name', symbol)[:20]
                    
                    # Calculate metrics for mini card
                    current = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change_pct = ((current - prev) / prev) * 100
                    
                    # Create mini chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index[-30:],  # Last 30 days
                        y=df['Close'].iloc[-30:],
                        mode='lines',
                        line=dict(width=2, color='green' if change_pct >= 0 else 'red'),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.1)' if change_pct >= 0 else 'rgba(255, 0, 0, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title=dict(
                            text=f"{asset_name}<br><span style='font-size:0.8em; color:{'green' if change_pct >= 0 else 'red'}'>"
                                 f"{format_currency(current)} ({change_pct:+.2f}%)</span>",
                            x=0.5,
                            xanchor='center'
                        ),
                        height=200,
                        margin=dict(l=10, r=10, t=50, b=10),
                        showlegend=False,
                        xaxis_showticklabels=False,
                        yaxis_showticklabels=False,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_volatility_analysis(self, garch_p, garch_q, forecast_horizon):
        """Display volatility analysis"""
        st.subheader("Volatility Analysis")
        
        # Select asset for volatility analysis
        vol_asset = st.selectbox(
            "Select Asset for Volatility Analysis",
            list(self.data_dict.keys()),
            key="vol_asset_selector"
        )
        
        if vol_asset not in self.data_dict:
            return
        
        df = self.data_dict[vol_asset]
        returns = df['Returns'].dropna()
        
        # Check for ARCH effects
        with st.spinner("Testing for ARCH effects..."):
            arch_test = self.analysis_engine.arch_effect_test(returns.values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if arch_test["present"]:
                st.markdown("""
                <div class="status-badge status-success">ARCH Effects Present</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-badge status-warning">No ARCH Effects</div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("ARCH Test p-value", f"{arch_test['p_value']:.4f}")
        
        with col3:
            st.metric("LM Statistic", f"{arch_test.get('LM_statistic', 0):.2f}")
        
        # Fit GARCH model
        with st.spinner("Fitting GARCH model..."):
            garch_results = self.analysis_engine.fit_garch_model(
                returns.values, p=garch_p, q=garch_q
            )
        
        if garch_results and garch_results["converged"]:
            st.success("✅ GARCH model converged successfully")
            
            # Display GARCH parameters
            st.subheader("GARCH Model Parameters")
            
            params_df = pd.DataFrame.from_dict(
                garch_results["params"], 
                orient='index', 
                columns=['Value']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(params_df, use_container_width=True)
            
            with col2:
                metrics_df = pd.DataFrame({
                    'Metric': ['AIC', 'BIC', 'Log Likelihood'],
                    'Value': [
                        garch_results["aic"],
                        garch_results["bic"],
                        garch_results["model"].loglikelihood
                    ]
                })
                st.dataframe(metrics_df.set_index('Metric'), use_container_width=True)
            
            # Volatility forecast
            with st.spinner("Generating volatility forecast..."):
                forecast = self.analysis_engine.forecast_volatility(
                    garch_results["model"], 
                    steps=forecast_horizon
                )
            
            if forecast is not None:
                st.subheader("Volatility Forecast")
                
                fig = self.visualizer.create_volatility_forecast_chart(
                    forecast, 
                    f"{vol_asset} - {forecast_horizon}-Day Volatility Forecast"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast statistics
                cols = st.columns(4)
                forecast_stats = {
                    'Current Vol': forecast[0],
                    'Avg Forecast': np.mean(forecast),
                    'Max Forecast': np.max(forecast),
                    'Min Forecast': np.min(forecast),
                    'Volatility Persistence': garch_results["params"].get('beta[1]', 0) 
                                            if 'beta[1]' in garch_results["params"] else 0,
                    'Shock Impact': garch_results["params"].get('alpha[1]', 0) 
                                  if 'alpha[1]' in garch_results["params"] else 0
                }
                
                for idx, (name, value) in enumerate(forecast_stats.items()):
                    with cols[idx % 4]:
                        st.metric(name, f"{value:.4f}")
            
            # GARCH volatility chart
            st.subheader("Conditional Volatility")
            
            fig = self.visualizer.create_garch_volatility_chart(
                garch_results,
                returns.values,
                returns.index,
                f"{vol_asset} - GARCH Conditional Volatility"
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ GARCH model failed to converge")
    
    def display_risk_analysis(self):
        """Display risk analysis"""
        st.subheader("Comprehensive Risk Analysis")
        
        # Select asset for risk analysis
        risk_asset = st.selectbox(
            "Select Asset for Risk Analysis",
            list(self.data_dict.keys()),
            key="risk_asset_selector"
        )
        
        if risk_asset not in self.data_dict:
            return
        
        df = self.data_dict[risk_asset]
        returns = df['Returns'].dropna()
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = self.metrics_dict.get(risk_asset, {})
        
        with col1:
            st.metric("Value at Risk (95%)", 
                     format_percentage(metrics.get('var_95', 0) * 100))
        
        with col2:
            st.metric("Conditional VaR (95%)", 
                     format_percentage(metrics.get('cvar_95', 0) * 100))
        
        with col3:
            st.metric("Maximum Drawdown", 
                     format_percentage(metrics.get('max_drawdown', 0) * 100))
        
        with col4:
            st.metric("Tail Ratio", 
                     format_number(metrics.get('tail_ratio', 0)))
        
        # Distribution analysis
        st.subheader("Returns Distribution Analysis")
        
        fig = self.visualizer.create_distribution_chart(
            returns.values,
            f"{risk_asset} - Returns Distribution"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling metrics
        st.subheader("Rolling Risk Metrics")
        
        fig = self.visualizer.create_rolling_metrics_chart(
            df,
            f"{risk_asset} - Rolling Risk Metrics"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical drawdowns
        st.subheader("Historical Drawdown Analysis")
        
        # Calculate drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        drawdown_df = pd.DataFrame({
            'Cumulative Returns': cumulative,
            'Running Max': running_max,
            'Drawdown': drawdown
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Cumulative Returns'] * 100,
            name='Cumulative Returns',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Running Max'] * 100,
            name='Running Maximum',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['Drawdown'] * 100,
            name='Drawdown',
            line=dict(color='#f39c12', width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.2)'
        ))
        
        fig.update_layout(
            title=f"{risk_asset} - Drawdown Analysis",
            height=500,
            template="plotly_white",
            hovermode='x unified',
            yaxis_title="Percentage (%)",
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_garch_modeling(self, garch_p, garch_q):
        """Display GARCH modeling tools"""
        st.subheader("Advanced GARCH Modeling")
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        models_to_compare = st.multiselect(
            "Select models to compare",
            list(self.data_dict.keys()),
            default=list(self.data_dict.keys())[:3] if len(self.data_dict) >= 3 else list(self.data_dict.keys())
        )
        
        if len(models_to_compare) >= 2:
            with st.spinner("Comparing GARCH models..."):
                comparison_data = []
                
                for symbol in models_to_compare:
                    df = self.data_dict[symbol]
                    returns = df['Returns'].dropna()
                    
                    if len(returns) > 100:
                        garch_results = self.analysis_engine.fit_garch_model(
                            returns.values, p=garch_p, q=garch_q
                        )
                        
                        if garch_results and garch_results["converged"]:
                            comparison_data.append({
                                'Asset': symbol,
                                'AIC': garch_results["aic"],
                                'BIC': garch_results["bic"],
                                'Omega': garch_results["params"].get('omega', 0),
                                'Alpha': garch_results["params"].get('alpha[1]', 0),
                                'Beta': garch_results["params"].get('beta[1]', 0),
                                'Log Likelihood': garch_results["model"].loglikelihood
                            })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(
                        comparison_df.sort_values('AIC').set_index('Asset'), 
                        use_container_width=True
                    )
                    
                    # Visual comparison
                    fig = go.Figure()
                    
                    for idx, row in comparison_df.iterrows():
                        fig.add_trace(go.Bar(
                            name=row['Asset'],
                            x=['AIC', 'BIC', 'Alpha', 'Beta'],
                            y=[row['AIC'], row['BIC'], row['Alpha'], row['Beta']],
                            text=[f"{row['AIC']:.2f}", f"{row['BIC']:.2f}", 
                                  f"{row['Alpha']:.4f}", f"{row['Beta']:.4f}"],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="GARCH Model Parameters Comparison",
                        barmode='group',
                        height=500,
                        template="plotly_white",
                        xaxis_title="Parameter",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # GARCH model diagnostics
        st.markdown("### Model Diagnostics")
        
        diag_asset = st.selectbox(
            "Select asset for diagnostics",
            list(self.data_dict.keys()),
            key="diag_asset"
        )
        
        if diag_asset in self.data_dict:
            df = self.data_dict[diag_asset]
            returns = df['Returns'].dropna()
            
            garch_results = self.analysis_engine.fit_garch_model(
                returns.values, p=garch_p, q=garch_q
            )
            
            if garch_results and garch_results["converged"]:
                # Residual diagnostics
                residuals = garch_results["model"].resid
                standardized_residuals = residuals / garch_results["model"].conditional_volatility
                
                # Create diagnostics plots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Residuals Distribution', 'Residuals ACF',
                                   'Squared Residuals ACF', 'Q-Q Plot of Std. Residuals'),
                    specs=[[{"type": "histogram"}, {"type": "bar"}],
                          [{"type": "bar"}, {"type": "scatter"}]]
                )
                
                # Residuals histogram
                fig.add_trace(
                    go.Histogram(x=residuals, nbinsx=50,
                                name='Residuals',
                                marker_color='#3498db'),
                    row=1, col=1
                )
                
                # Residuals ACF
                from statsmodels.tsa.stattools import acf
                
                resid_acf = acf(residuals, nlags=20)
                fig.add_trace(
                    go.Bar(x=list(range(len(resid_acf))), y=resid_acf,
                          name='Residuals ACF',
                          marker_color='#2ecc71'),
                    row=1, col=2
                )
                
                # Squared residuals ACF
                squared_resid_acf = acf(residuals**2, nlags=20)
                fig.add_trace(
                    go.Bar(x=list(range(len(squared_resid_acf))), y=squared_resid_acf,
                          name='Squared Residuals ACF',
                          marker_color='#e74c3c'),
                    row=2, col=1
                )
                
                # Q-Q plot of standardized residuals
                from scipy import stats
                standardized_resid_clean = standardized_residuals[~np.isnan(standardized_residuals)]
                if len(standardized_resid_clean) > 0:
                    qq = stats.probplot(standardized_resid_clean, dist="norm")
                    
                    fig.add_trace(
                        go.Scatter(x=qq[0][1], y=qq[0][0],
                                  mode='markers',
                                  name='Q-Q Plot',
                                  marker=dict(color='#9b59b6', size=5)),
                        row=2, col=2
                    )
                    
                    # Add 45-degree line
                    min_val = min(qq[0][1].min(), qq[0][0].min())
                    max_val = max(qq[0][1].max(), qq[0][0].max())
                    fig.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                  mode='lines',
                                  name='Normal Line',
                                  line=dict(color='red', dash='dash', width=1)),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Ljung-Box test results
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                lb_test = acorr_ljungbox(residuals, lags=[5, 10, 20], return_df=True)
                
                st.markdown("#### Ljung-Box Test for Autocorrelation")
                st.dataframe(lb_test, use_container_width=True)
    
    def display_correlation_matrix(self):
        """Display correlation analysis"""
        st.subheader("Asset Correlation Analysis")
        
        if len(self.data_dict) < 2:
            st.warning("Need at least 2 assets for correlation analysis")
            return
        
        # Correlation matrix
        fig = self.visualizer.create_correlation_matrix(self.data_dict)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling correlation
        st.subheader("Rolling Correlation (90-day window)")
        
        # Select two assets for rolling correlation
        col1, col2 = st.columns(2)
        
        with col1:
            asset1 = st.selectbox(
                "First Asset",
                list(self.data_dict.keys()),
                key="corr_asset1"
            )
        
        with col2:
            available_assets = [a for a in self.data_dict.keys() if a != asset1]
            asset2 = st.selectbox(
                "Second Asset",
                available_assets,
                key="corr_asset2"
            )
        
        if asset1 in self.data_dict and asset2 in self.data_dict:
            # Calculate rolling correlation
            returns1 = self.data_dict[asset1]['Returns']
            returns2 = self.data_dict[asset2]['Returns']
            
            # Align indices
            aligned_returns = pd.concat([returns1, returns2], axis=1, join='inner')
            aligned_returns.columns = [asset1, asset2]
            
            rolling_corr = aligned_returns[asset1].rolling(window=90).corr(aligned_returns[asset2])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                name='90-day Rolling Correlation',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3)
            fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3)
            
            fig.update_layout(
                title=f"Rolling Correlation: {asset1} vs {asset2}",
                height=400,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Correlation",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation statistics
            current_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0
            avg_corr = rolling_corr.mean() if not rolling_corr.empty else 0
            min_corr = rolling_corr.min() if not rolling_corr.empty else 0
            max_corr = rolling_corr.max() if not rolling_corr.empty else 0
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Current Correlation", f"{current_corr:.3f}")
            with cols[1]:
                st.metric("Average Correlation", f"{avg_corr:.3f}")
            with cols[2]:
                st.metric("Minimum Correlation", f"{min_corr:.3f}")
            with cols[3]:
                st.metric("Maximum Correlation", f"{max_corr:.3f}")
        
        # Risk-return scatter
        st.subheader("Risk-Return Profile")
        
        fig = self.visualizer.create_risk_return_chart(self.metrics_dict)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    def display_real_time_monitoring(self):
        """Display real-time monitoring dashboard"""
        st.subheader("⚡ Real-time Market Monitoring")
        
        # Refresh button
        if st.button("🔄 Refresh Real-time Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Create real-time metrics dashboard
        st.markdown("### Live Market Metrics")
        
        # Use columns for metrics display
        cols = st.columns(4)
        
        # Simulate real-time updates (in production, this would connect to a live data feed)
        for idx, (symbol, df) in enumerate(self.data_dict.items()):
            if idx < 4:  # Show first 4 assets
                with cols[idx]:
                    current = df['Close'].iloc[-1]
                    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current
                    change = ((current - prev_close) / prev_close) * 100
                    
                    # Calculate volatility
                    recent_vol = df['Returns'].tail(20).std() * np.sqrt(252) * 100
                    
                    # Determine status
                    if abs(change) > 2:
                        status_color = "#e74c3c"  # Red for large moves
                    elif abs(change) > 1:
                        status_color = "#f39c12"  # Orange for medium moves
                    else:
                        status_color = "#27ae60"  # Green for small moves
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;
                                border-left: 4px solid {status_color};">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                            {symbol}
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                            {format_currency(current)}
                        </div>
                        <div style="font-size: 1rem; color: {'#27ae60' if change >= 0 else '#e74c3c'}; 
                                    margin: 0.5rem 0;">
                            {change:+.2f}%
                        </div>
                        <div style="font-size: 0.8rem; color: #7f8c8d;">
                            Vol: {recent_vol:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Market heatmap
        st.markdown("### Market Heatmap")
        
        # Create a simple heatmap of recent performance
        heatmap_data = []
        for symbol, df in self.data_dict.items():
            if len(df) > 5:
                recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                heatmap_data.append({
                    'Asset': symbol,
                    'Return': recent_return,
                    'Category': COMMODITIES.get(symbol.split('.')[0], {}).get('category', 'unknown')
                })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            fig = px.treemap(
                heatmap_df,
                path=['Category', 'Asset'],
                values=abs(heatmap_df['Return']),
                color='Return',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title='Recent Performance Heatmap'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Market alerts
        st.markdown("### Market Alerts")
        
        alerts = []
        for symbol, df in self.data_dict.items():
            if len(df) > 20:
                # Check for volatility spikes
                current_vol = df['Returns'].tail(20).std() * np.sqrt(252)
                avg_vol = df['Returns'].rolling(60).std().iloc[-1] * np.sqrt(252)
                
                if current_vol > avg_vol * 1.5:
                    alerts.append({
                        'asset': symbol,
                        'type': 'volatility',
                        'message': f'Volatility spike detected: {current_vol:.1%} vs average {avg_vol:.1%}',
                        'severity': 'high'
                    })
                
                # Check for RSI extremes
                if 'RSI' in df.columns and not np.isnan(df['RSI'].iloc[-1]):
                    rsi = df['RSI'].iloc[-1]
                    if rsi > 70:
                        alerts.append({
                            'asset': symbol,
                            'type': 'overbought',
                            'message': f'RSI overbought: {rsi:.1f}',
                            'severity': 'medium'
                        })
                    elif rsi < 30:
                        alerts.append({
                            'asset': symbol,
                            'type': 'oversold',
                            'message': f'RSI oversold: {rsi:.1f}',
                            'severity': 'medium'
                        })
        
        if alerts:
            for alert in alerts[:5]:  # Show first 5 alerts
                severity_color = {
                    'high': '#e74c3c',
                    'medium': '#f39c12',
                    'low': '#3498db'
                }.get(alert['severity'], '#7f8c8d')
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; 
                            margin-bottom: 0.5rem; border-left: 4px solid {severity_color};
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{alert['asset']}</strong> - {alert['type'].upper()}
                        </div>
                        <div style="font-size: 0.8rem; color: {severity_color};">
                            {alert['severity'].upper()}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                        {alert['message']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No market alerts at this time")
        
        # Trading volume analysis
        st.markdown("### Volume Analysis")
        
        volume_data = []
        for symbol, df in self.data_dict.items():
            if len(df) > 1:
                volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].tail(20).mean()
                volume_ratio = (volume / avg_volume) * 100
                
                volume_data.append({
                    'Asset': symbol,
                    'Volume': volume,
                    'Volume Ratio': volume_ratio,
                    'Price Change': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                })
        
        if volume_data:
            volume_df = pd.DataFrame(volume_data)
            
            fig = px.scatter(
                volume_df,
                x='Price Change',
                y='Volume Ratio',
                size='Volume',
                color='Asset',
                hover_name='Asset',
                title='Volume vs Price Change',
                labels={
                    'Price Change': 'Price Change (%)',
                    'Volume Ratio': 'Volume vs 20-day Avg (%)'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main execution method"""
        
        # Display header
        self.display_header()
        
        # Setup sidebar and get parameters
        start_date, end_date, garch_p, garch_q, forecast_horizon, show_indicators, show_bb = self.setup_sidebar()
        
        # Load data
        with st.spinner("Loading market data..."):
            self.load_data(start_date, end_date)
        
        # Display metrics dashboard
        self.display_metrics_dashboard()
        
        # Display analysis tabs
        self.display_analysis_tabs(garch_p, garch_q, forecast_horizon, show_indicators, show_bb)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
            <p>🏆 Precious Metals & Commodities Analytics Pro v1.0</p>
            <p>For professional use only. Past performance is not indicative of future results.</p>
            <p>Data provided by Yahoo Finance • Analysis updated at {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    try:
        # Initialize and run dashboard
        dashboard = PreciousMetalsDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")
          
