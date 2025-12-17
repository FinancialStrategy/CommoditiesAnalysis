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
from io import BytesIO

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

class DataManager:
    """Enhanced data management with caching and error handling"""
    
    @st.cache_data(ttl=300)  # 5 minutes cache for real-time data
    def fetch_data(_self, symbol, start_date, end_date):
        """Enhanced data fetching with better error handling"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty or len(df) < 50:
                st.warning(f"⚠️ Insufficient data for {symbol}")
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
    
    def calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics using QuantStats"""
        if returns is None or len(returns) < 100:
            return {}
        
        try:
            # Convert to pandas Series
            returns_series = pd.Series(returns)
            
            # Calculate metrics using QuantStats
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = qs.stats.comp(returns_series)
            metrics['cagr'] = qs.stats.cagr(returns_series)
            metrics['volatility'] = qs.stats.volatility(returns_series)
            metrics['sharpe'] = qs.stats.sharpe(returns_series)
            metrics['sortino'] = qs.stats.sortino(returns_series)
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns_series)
            metrics['calmar'] = qs.stats.calmar(returns_series)
            
            # Risk metrics
            metrics['var_95'] = qs.stats.value_at_risk(returns_series)
            metrics['cvar_95'] = qs.stats.conditional_value_at_risk(returns_series)
            metrics['skew'] = qs.stats.skew(returns_series)
            metrics['kurtosis'] = qs.stats.kurtosis(returns_series)
            
            # Additional metrics
            metrics['omega'] = qs.stats.omega(returns_series)
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns_series)
            metrics['common_sense_ratio'] = qs.stats.common_sense_ratio(returns_series)
            metrics['information_ratio'] = qs.stats.information_ratio(returns_series)
            
            # Gain/Pain metrics
            metrics['gain_to_pain'] = qs.stats.gain_to_pain_ratio(returns_series)
            metrics['win_rate'] = qs.stats.win_rate(returns_series)
            metrics['avg_win'] = qs.stats.avg_win(returns_series)
            metrics['avg_loss'] = qs.stats.avg_loss(returns_series)
            
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
            st.error(f"Error in ARCH test: {str(e)}")
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
            st.error(f"Error fitting GARCH model: {str(e)}")
            return None
    
    def forecast_volatility(self, garch_model, steps=30):
        """Forecast volatility using GARCH model"""
        if garch_model is None:
            return None
        
        try:
            forecast = garch_model.forecast(horizon=steps)
            return forecast.variance.iloc[-1].values
        except Exception as e:
            st.error(f"Error forecasting volatility: {str(e)}")
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
    
    def create_garch_volatility_chart(self, garch_results, returns, title):
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
            
            # Align dates
            dates = pd.date_range(end=datetime.now(), periods=len(cond_vol), freq='D')
            
            # Conditional volatility
            fig.add_trace(
                go.Scatter(x=dates, y=cond_vol, name='GARCH Volatility',
                          line=dict(color='#d35400', width=2),
                          fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.1)',
                          hovertemplate='Date: %{x}<br>Volatility: %{y:.4f}<extra></extra>'),
                row=1, col=1
            )
            
            # Standardized residuals
            std_resid = garch_results["model"].resid / cond_vol
            
            fig.add_trace(
                go.Scatter(x=dates, y=std_resid, name='Std. Residuals',
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
            st.error(f"Error creating volatility chart: {str(e)}")
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
            st.error(f"Error creating risk-return chart: {str(e)}")
            return None
    
    def create_technical_indicators_chart(self, df, title):
        """Create technical indicators chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('RSI', 'MACD', 'Stochastic'),
            vertical_spacing=0.08,
            row_heights=[0.33, 0.33, 0.34],
            shared_xaxes=True
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     opacity=0.5, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     opacity=0.5, row=1, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                     opacity=0.3, row=1, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='#f39c12', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                      line=dict(color='#9b59b6', width=2, dash='dash')),
            row=2, col=1
        )
        
        fig.add_bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                   marker_color=['#27ae60' if x >= 0 else '#e74c3c' for x in df['MACD_Histogram']],
                   opacity=0.6, row=2, col=1)
        
        # Stochastic
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stochastic'], name='%K',
                      line=dict(color='#2ecc71', width=2)),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stochastic_SMA'], name='%D',
                      line=dict(color='#e74c3c', width=2, dash='dash')),
            row=3, col=1
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", 
                     opacity=0.5, row=3, col=1)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            height=700,
            template="plotly_white",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig

class Dashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analysis_engine = AnalysisEngine()
        self.visualizer = Visualizer()
        self.last_update = datetime.now()
        
    def render_header(self):
        """Render dashboard header"""
        st.markdown(f"""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.8rem;">🏆 Precious Metals & Commodities Analytics Pro</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Real-time Analysis with ARCH/GARCH Modeling & Advanced Risk Metrics
            </p>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.7;">
                Last Updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Analysis Settings")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real-time update option
            st.markdown("#### 🔄 Real-time Updates")
            auto_refresh = st.checkbox("Auto Refresh (5 min intervals)", value=False)
            
            if auto_refresh:
                refresh_time = st.slider("Refresh interval (seconds)", 60, 600, 300, 30)
                if st.button("🔄 Refresh Now"):
                    st.rerun()
            
            st.markdown("---")
            
            # Date range selection
            st.markdown("#### 📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*3),
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            
            # Commodity category selection
            st.markdown("---")
            st.markdown("#### 🏆 Commodity Selection")
            
            category = st.selectbox(
                "Select Category",
                list(COMMODITIES.keys())
            )
            
            # Individual commodity selection
            selected_symbols = st.multiselect(
                "Select Commodities",
                options=list(COMMODITIES[category].keys()),
                default=list(COMMODITIES[category].keys())[:3],
                format_func=lambda x: COMMODITIES[category][x]["name"]
            )
            
            # Analysis parameters
            st.markdown("---")
            st.markdown("#### 📊 Analysis Parameters")
            
            show_technical = st.checkbox("Show Technical Indicators", value=True)
            show_bb = st.checkbox("Show Bollinger Bands", value=True)
            arch_lags = st.slider("ARCH Test Lags", 1, 20, 5)
            garch_p = st.slider("GARCH(p)", 1, 3, 1)
            garch_q = st.slider("GARCH(q)", 1, 3, 1)
            forecast_steps = st.slider("Volatility Forecast Steps", 5, 90, 30)
            
            # Cache timestamp
            st.markdown("---")
            st.caption(f"Data cached until: {(datetime.now() + timedelta(minutes=5)).strftime('%H:%M:%S')}")
            
            return {
                "auto_refresh": auto_refresh,
                "refresh_time": refresh_time if 'refresh_time' in locals() else 300,
                "start_date": start_date,
                "end_date": end_date,
                "category": category,
                "symbols": selected_symbols,
                "show_technical": show_technical,
                "show_bb": show_bb,
                "arch_lags": arch_lags,
                "garch_p": garch_p,
                "garch_q": garch_q,
                "forecast_steps": forecast_steps
            }
    
    def run(self):
        """Main dashboard execution"""
        
        # Render header
        self.render_header()
        
        # Render sidebar and get parameters
        params = self.render_sidebar()
        
        if not params["symbols"]:
            st.warning("⚠️ Please select at least one commodity to analyze.")
            return
        
        # Check for auto-refresh
        if params["auto_refresh"]:
            time.sleep(params["refresh_time"])
            st.rerun()
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch and analyze data
        all_data = {}
        all_metrics = {}
        all_garch_results = {}
        all_returns = {}
        all_forecasts = {}
        
        total_symbols = len(params["symbols"])
        
        for i, symbol in enumerate(params["symbols"]):
            progress = (i / total_symbols) * 100
            progress_bar.progress(int(progress))
            
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            status_text.text(f"📥 Fetching data for {commodity_name}...")
            
            # Fetch data
            df = self.data_manager.fetch_data(
                symbol,
                params["start_date"],
                params["end_date"]
            )
            
            if df is None or df.empty:
                st.warning(f"⚠️ No data available for {commodity_name}")
                continue
            
            all_data[symbol] = df
            
            # Calculate returns
            returns = df['Returns'].dropna().values
            all_returns[symbol] = returns
            
            # Calculate performance metrics
            status_text.text(f"📊 Analyzing performance for {commodity_name}...")
            metrics = self.analysis_engine.calculate_performance_metrics(returns)
            all_metrics[symbol] = metrics
            
            # Test for ARCH effects
            arch_test = self.analysis_engine.arch_effect_test(returns, params["arch_lags"])
            
            # Fit GARCH model if ARCH effects present
            if arch_test["present"]:
                status_text.text(f"🔧 Fitting GARCH model for {commodity_name}...")
                garch_results = self.analysis_engine.fit_garch_model(
                    returns, 
                    p=params["garch_p"], 
                    q=params["garch_q"]
                )
                all_garch_results[symbol] = {
                    "garch": garch_results,
                    "arch_test": arch_test
                }
                
                # Generate volatility forecast
                if garch_results:
                    forecast = self.analysis_engine.forecast_volatility(
                        garch_results["model"],
                        steps=params["forecast_steps"]
                    )
                    all_forecasts[symbol] = forecast
            else:
                all_garch_results[symbol] = {
                    "garch": None,
                    "arch_test": arch_test
                }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        self.last_update = datetime.now()
        
        if not all_data:
            st.error("❌ No data available for analysis. Please try different commodities or date range.")
            return
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "📈 Price Analysis", 
            "📊 Performance", 
            "🔍 Volatility Models",
            "📉 Technical Analysis",
            "📋 Comparison", 
            "📄 Reports",
            "💾 Export"
        ])
        
        with tabs[0]:
            self.render_price_analysis_tab(all_data, params)
        
        with tabs[1]:
            self.render_performance_tab(all_metrics, all_returns)
        
        with tabs[2]:
            self.render_volatility_tab(all_data, all_garch_results, all_forecasts, params)
        
        with tabs[3]:
            self.render_technical_analysis_tab(all_data, params)
        
        with tabs[4]:
            self.render_comparison_tab(all_data, all_metrics, all_returns)
        
        with tabs[5]:
            self.render_reports_tab(all_metrics, all_garch_results, all_returns)
        
        with tabs[6]:
            self.render_export_tab(all_data, all_metrics, all_garch_results)
    
    def render_price_analysis_tab(self, all_data, params):
        """Render price analysis tab"""
        st.markdown("### 📈 Price Charts & Technical Analysis")
        
        for symbol, df in all_data.items():
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            
            st.markdown(f"#### {commodity_name}")
            
            # Price chart
            fig = self.visualizer.create_price_chart(
                df, 
                f"{commodity_name} - Price Analysis",
                show_bb=params["show_bb"],
                show_indicators=params["show_technical"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = df['Close'].iloc[-1]
                price_change_pct = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                delta_color = "normal" if price_change_pct >= 0 else "inverse"
                st.metric("Current Price", format_currency(current_price), 
                         delta=f"{price_change_pct:.2f}%", delta_color=delta_color)
            
            with col2:
                volume_today = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].mean()
                volume_ratio = (volume_today / avg_volume - 1) * 100
                st.metric("Volume Today", f"{volume_today:,.0f}", 
                         delta=f"{volume_ratio:.1f}% vs avg")
            
            with col3:
                volatility = df['Returns'].std() * np.sqrt(252) * 100
                atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
                st.metric("Risk Metrics", 
                         f"Vol: {volatility:.1f}%",
                         delta=f"ATR: ${atr:.2f}")
            
            with col4:
                bb_position = (df['Close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]) * 100
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                st.metric("Technical", 
                         f"RSI: {rsi:.1f}",
                         delta=f"BB%: {bb_position:.1f}%")
            
            st.markdown("---")
    
    def render_performance_tab(self, all_metrics, all_returns):
        """Render performance metrics tab"""
        st.markdown("### 📊 Performance Analytics")
        
        # Risk-Return analysis
        if all_metrics:
            fig = self.visualizer.create_risk_return_chart(all_metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        self.render_performance_table(all_metrics)
        
        # Returns distribution for selected asset
        if all_returns:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_asset = st.selectbox(
                    "Select asset for detailed returns analysis",
                    list(all_returns.keys()),
                    key="returns_asset"
                )
            
            if selected_asset in all_returns:
                returns = all_returns[selected_asset]
                asset_name = COMMODITIES[[cat for cat in COMMODITIES if selected_asset in COMMODITIES[cat]][0]][selected_asset]["name"]
                
                # Display key metrics
                metrics = all_metrics.get(selected_asset, {})
                cols = st.columns(4)
                metric_items = [
                    ("Total Return", metrics.get('total_return', 0)),
                    ("Sharpe Ratio", metrics.get('sharpe', 0)),
                    ("Max Drawdown", metrics.get('max_drawdown', 0)),
                    ("Win Rate", metrics.get('win_rate', 0))
                ]
                
                for idx, (label, value) in enumerate(metric_items):
                    with cols[idx]:
                        st.metric(label, 
                                 format_percentage(value * 100) if 'Return' in label or 'Drawdown' in label else format_number(value, 3))
    
    def render_volatility_tab(self, all_data, all_garch_results, all_forecasts, params):
        """Render volatility modeling tab"""
        st.markdown("### 🔍 Volatility Modeling (ARCH/GARCH)")
        
        for symbol, results in all_garch_results.items():
            if symbol not in all_data:
                continue
                
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            
            st.markdown(f"#### {commodity_name}")
            
            cols = st.columns(4)
            
            with cols[0]:
                arch_present = results["arch_test"]["present"]
                arch_p_value = results["arch_test"]["p_value"]
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">ARCH Effects</div>', unsafe_allow_html=True)
                status_class = "status-success" if arch_present else "status-danger"
                status_text = "✅ Present" if arch_present else "❌ Not Present"
                st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)
                st.markdown(f"**p-value:** {arch_p_value:.4f}")
                if arch_p_value < 0.01:
                    st.markdown("**Significance:** ★★★")
                elif arch_p_value < 0.05:
                    st.markdown("**Significance:** ★★")
                elif arch_p_value < 0.1:
                    st.markdown("**Significance:** ★")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                if results["garch"]:
                    garch_aic = results["garch"]["aic"]
                    garch_converged = results["garch"]["converged"]
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">GARCH Model</div>', unsafe_allow_html=True)
                    status_class = "status-success" if garch_converged else "status-danger"
                    status_text = "✅ Converged" if garch_converged else "❌ Failed"
                    st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)
                    st.markdown(f"**AIC:** {garch_aic:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[2]:
                if results["garch"]:
                    omega = results["garch"]["params"].get("omega", 0)
                    alpha = results["garch"]["params"].get("alpha[1]", 0)
                    beta = results["garch"]["params"].get("beta[1]", 0)
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">GARCH Parameters</div>', unsafe_allow_html=True)
                    st.markdown(f"**ω (Long-term):** {omega:.6f}")
                    st.markdown(f"**α (ARCH):** {alpha:.4f}")
                    st.markdown(f"**β (GARCH):** {beta:.4f}")
                    st.markdown(f"**Persistence:** {alpha + beta:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[3]:
                if symbol in all_forecasts and all_forecasts[symbol] is not None:
                    forecast_values = all_forecasts[symbol]
                    current_vol = np.sqrt(forecast_values[0]) if len(forecast_values) > 0 else 0
                    avg_forecast = np.mean(np.sqrt(forecast_values)) if len(forecast_values) > 0 else 0
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Volatility Forecast</div>', unsafe_allow_html=True)
                    st.markdown(f"**Current:** {current_vol:.4f}")
                    st.markdown(f"**30-day Avg:** {avg_forecast:.4f}")
                    trend = "↗️ Increasing" if avg_forecast > current_vol else "↘️ Decreasing"
                    st.markdown(f"**Trend:** {trend}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # GARCH volatility chart
            if results["garch"] and symbol in all_data:
                returns = all_data[symbol]['Returns'].dropna().values
                fig = self.visualizer.create_garch_volatility_chart(
                    results["garch"],
                    returns,
                    f"{commodity_name} - GARCH Volatility"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Volatility forecast chart
            if symbol in all_forecasts and all_forecasts[symbol] is not None:
                fig = self.visualizer.create_volatility_forecast_chart(
                    np.sqrt(all_forecasts[symbol]),  # Convert variance to volatility
                    f"{commodity_name} - {params['forecast_steps']}-Day Volatility Forecast"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    def render_technical_analysis_tab(self, all_data, params):
        """Render technical analysis tab"""
        st.markdown("### 📉 Technical Indicators")
        
        selected_indicators = st.multiselect(
            "Select indicators to display",
            ["RSI", "MACD", "Stochastic", "ATR", "Bollinger Bands"],
            default=["RSI", "MACD"]
        )
        
        for symbol, df in all_data.items():
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            
            st.markdown(f"#### {commodity_name}")
            
            # Technical indicators chart
            if "RSI" in selected_indicators or "MACD" in selected_indicators or "Stochastic" in selected_indicators:
                fig = self.visualizer.create_technical_indicators_chart(
                    df,
                    f"{commodity_name} - Technical Indicators"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Current indicator values
            cols = st.columns(len(selected_indicators))
            for idx, indicator in enumerate(selected_indicators):
                with cols[idx]:
                    if indicator == "RSI" and 'RSI' in df.columns:
                        rsi_value = df['RSI'].iloc[-1]
                        status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        color = "red" if rsi_value > 70 else "green" if rsi_value < 30 else "orange"
                        st.metric("RSI", f"{rsi_value:.1f}", delta=status, delta_color=color)
                    
                    elif indicator == "MACD" and 'MACD' in df.columns:
                        macd_value = df['MACD'].iloc[-1]
                        signal_value = df['MACD_Signal'].iloc[-1]
                        signal = "Bullish" if macd_value > signal_value else "Bearish"
                        st.metric("MACD", f"{macd_value:.3f}", delta=signal)
                    
                    elif indicator == "ATR" and 'ATR' in df.columns:
                        atr_value = df['ATR'].iloc[-1]
                        st.metric("ATR", f"${atr_value:.2f}")
            
            st.markdown("---")
    
    def render_comparison_tab(self, all_data, all_metrics, all_returns):
        """Render comparison tab"""
        st.markdown("### 📋 Multi-Asset Comparison")
        
        if len(all_data) < 2:
            st.warning("Select at least 2 assets for comparison")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation matrix
            if len(all_returns) > 1:
                # Align returns
                min_length = min(len(r) for r in all_returns.values())
                aligned_returns = {k: v[-min_length:] for k, v in all_returns.items()}
                
                corr_df = pd.DataFrame(aligned_returns).corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_df.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Returns Correlation Matrix',
                    height=500,
                    template="plotly_white",
                    xaxis_title="Assets",
                    yaxis_title="Assets"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cumulative returns comparison
            cum_returns_data = {}
            for symbol, df in all_data.items():
                cum_returns = (1 + df['Returns'].fillna(0)).cumprod()
                cum_returns_data[symbol] = cum_returns
            
            cum_returns_df = pd.DataFrame(cum_returns_data)
            
            fig = go.Figure()
            for column in cum_returns_df.columns:
                commodity_name = COMMODITIES[[cat for cat in COMMODITIES if column in COMMODITIES[cat]][0]][column]["name"]
                fig.add_trace(
                    go.Scatter(
                        x=cum_returns_df.index,
                        y=cum_returns_df[column],
                        name=commodity_name,
                        mode='lines',
                        line=dict(width=2)
                    )
                )
            
            fig.update_layout(
                title="Cumulative Returns Over Time",
                height=500,
                template="plotly_white",
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_reports_tab(self, all_metrics, all_garch_results, all_returns):
        """Render reports tab with improved QuantStats"""
        st.markdown("### 📄 Analysis Reports")
        
        # Asset selection for detailed report
        if not all_returns:
            st.warning("No returns data available for reports")
            return
        
        selected_asset = st.selectbox(
            "Select asset for detailed QuantStats report",
            list(all_returns.keys()),
            key="reports_asset"
        )
        
        if selected_asset and selected_asset in all_returns:
            returns_series = pd.Series(all_returns[selected_asset])
            returns_series.index = pd.date_range(
                end=datetime.now(), 
                periods=len(returns_series), 
                freq='D'
            )
            
            asset_name = COMMODITIES[[cat for cat in COMMODITIES if selected_asset in COMMODITIES[cat]][0]][selected_asset]["name"]
            
            st.markdown(f"#### 📊 {asset_name} - QuantStats Reports")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Generate Returns Report", use_container_width=True, key="returns_report"):
                    with st.spinner("Generating returns report..."):
                        fig = qs.plots.returns(returns_series, benchmark=None, show=False)
                        st.pyplot(fig)
            
            with col2:
                if st.button("📉 Generate Drawdown Report", use_container_width=True, key="drawdown_report"):
                    with st.spinner("Generating drawdown report..."):
                        fig = qs.plots.drawdown(returns_series, show=False)
                        st.pyplot(fig)
            
            with col3:
                if st.button("📊 Generate Metrics Report", use_container_width=True, key="metrics_report"):
                    with st.spinner("Generating metrics report..."):
                        fig = qs.plots.metrics(returns_series, show=False)
                        st.pyplot(fig)
            
            # Additional reports
            col4, col5, col6 = st.columns(3)
            
            with col4:
                if st.button("📈 Monthly Returns", use_container_width=True, key="monthly_returns"):
                    with st.spinner("Generating monthly returns heatmap..."):
                        fig = qs.plots.monthly_heatmap(returns_series, show=False)
                        st.pyplot(fig)
            
            with col5:
                if st.button("📅 Yearly Returns", use_container_width=True, key="yearly_returns"):
                    with st.spinner("Generating yearly returns..."):
                        fig = qs.plots.yearly_returns(returns_series, show=False)
                        st.pyplot(fig)
            
            with col6:
                if st.button("📋 Full Report", use_container_width=True, key="full_report"):
                    with st.spinner("Generating full QuantStats report..."):
                        # Create HTML report
                        html_report = qs.reports.html(returns_series, output=None, download_filename=None)
                        st.components.v1.html(html_report, height=800, scrolling=True)
    
    def render_export_tab(self, all_data, all_metrics, all_garch_results):
        """Render export tab"""
        st.markdown("### 💾 Export Data & Reports")
        
        # Create export data
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": all_metrics,
            "garch_results": {}
        }
        
        # Prepare GARCH results for export
        for symbol, results in all_garch_results.items():
            if results["garch"]:
                export_data["garch_results"][symbol] = {
                    "parameters": results["garch"]["params"],
                    "aic": float(results["garch"]["aic"]),
                    "bic": float(results["garch"]["bic"]),
                    "converged": results["garch"]["converged"],
                    "arch_test": results["arch_test"]
                }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Export JSON
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"commodities_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export CSV metrics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics).T
                csv_data = metrics_df.to_csv()
                st.download_button(
                    label="📥 Download CSV Metrics",
                    data=csv_data,
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Export price data
            if all_data:
                # Combine close prices
                close_prices = {}
                for symbol, df in all_data.items():
                    close_prices[symbol] = df['Close']
                
                prices_df = pd.DataFrame(close_prices)
                prices_csv = prices_df.to_csv()
                st.download_button(
                    label="📥 Download Price Data",
                    data=prices_csv,
                    file_name=f"prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col4:
            # Export Excel report
            if st.button("📊 Generate Excel Report", use_container_width=True):
                with st.spinner("Creating Excel report..."):
                    try:
                        # Create Excel writer
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Write metrics
                            if all_metrics:
                                metrics_df = pd.DataFrame(all_metrics).T
                                metrics_df.to_excel(writer, sheet_name='Performance Metrics')
                            
                            # Write price data
                            if all_data:
                                price_data = {}
                                for symbol, df in all_data.items():
                                    price_data[symbol] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                                
                                for symbol, df in price_data.items():
                                    df.to_excel(writer, sheet_name=f'{symbol}_Prices')
                        
                        # Get the Excel data
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="excel_download"
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel report: {str(e)}")
        
        # Display preview of export data
        with st.expander("📋 Preview Export Data"):
            if all_metrics:
                preview_df = pd.DataFrame(all_metrics).T.head()
                st.dataframe(preview_df, use_container_width=True)
    
    def render_performance_table(self, metrics_dict):
        """Render comprehensive performance table"""
        if not metrics_dict:
            return
        
        st.markdown("### 📋 Detailed Performance Metrics")
        
        # Prepare data for table
        metrics_list = []
        for symbol, metrics in metrics_dict.items():
            row = {"Asset": symbol}
            for key, value in metrics.items():
                if key in ['total_return', 'cagr', 'max_drawdown', 'volatility']:
                    row[key] = format_percentage(safe_float(value) * 100, 2)
                elif key in ['sharpe', 'sortino', 'calmar', 'skew', 'kurtosis']:
                    row[key] = format_number(safe_float(value), 3)
                elif key in ['var_95', 'cvar_95']:
                    row[key] = format_percentage(safe_float(value) * 100, 2)
                elif key in ['win_rate', 'avg_win', 'avg_loss']:
                    row[key] = format_percentage(safe_float(value) * 100, 1)
                else:
                    row[key] = format_number(safe_float(value), 4)
            metrics_list.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(metrics_list)
        
        # Apply styling
        def color_cells(val):
            try:
                if '%' in str(val):
                    num_val = float(str(val).replace('%', '').replace('+', ''))
                    if 'drawdown' in val.lower() or 'var' in val.lower():
                        if num_val < -5:
                            return 'background-color: #ffcccc; color: #000000; font-weight: bold'
                        elif num_val < -2:
                            return 'background-color: #ffe6cc; color: #000000; font-weight: bold'
                    else:
                        if num_val > 10:
                            return 'background-color: #ccffcc; color: #000000; font-weight: bold'
                        elif num_val > 5:
                            return 'background-color: #e6ffcc; color: #000000; font-weight: bold'
            except:
                pass
            return ''
        
        st.dataframe(
            df.style.applymap(color_cells),
            use_container_width=True,
            height=400
        )

def main():
    """Main application entry point"""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or try again later.")

if __name__ == "__main__":
    main()
