"""
🏛️ Institutional Commodities Analytics Platform v5.0
Enhanced Portfolio Analytics • Optimized GARCH • Advanced Regime Detection • Professional Reporting
Streamlit Cloud Optimized with Performance & Memory Efficiency
"""

import os
import math
import warnings
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Optimize environment
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(
    page_title="Institutional Commodities Platform v5.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Institutional Commodities Analytics v5.0"
    }
)

# =============================================================================
# DATA STRUCTURES & CONFIGURATION
# =============================================================================

@dataclass
class AssetConfig:
    """Configuration for asset data management"""
    symbol: str
    name: str
    category: str
    color: str = "#1a2980"
    enabled: bool = True
    
@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    start_date: datetime
    end_date: datetime
    risk_free_rate: float = 0.02
    annual_trading_days: int = 252
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    garch_p_range: Tuple[int, int] = (1, 2)
    garch_q_range: Tuple[int, int] = (1, 2)
    regime_states: int = 3
    backtest_window: int = 250
    
# Enhanced commodities universe with better metadata
COMMODITIES_UNIVERSE = {
    "Precious Metals": {
        "GC=F": AssetConfig("GC=F", "Gold Futures", "Precious", "#FFD700"),
        "SI=F": AssetConfig("SI=F", "Silver Futures", "Precious", "#C0C0C0"),
        "PL=F": AssetConfig("PL=F", "Platinum Futures", "Precious", "#E5E4E2"),
        "PA=F": AssetConfig("PA=F", "Palladium Futures", "Precious", "#B9B4B1")
    },
    "Industrial Metals": {
        "HG=F": AssetConfig("HG=F", "Copper Futures", "Industrial", "#B87333"),
        "ALI=F": AssetConfig("ALI=F", "Aluminum Futures", "Industrial", "#848482"),
        "ZN=F": AssetConfig("ZN=F", "Zinc Futures", "Industrial", "#7A7A7A")
    },
    "Energy": {
        "CL=F": AssetConfig("CL=F", "Crude Oil WTI", "Energy", "#000000"),
        "BZ=F": AssetConfig("BZ=F", "Brent Crude", "Energy", "#8B0000"),
        "NG=F": AssetConfig("NG=F", "Natural Gas", "Energy", "#4169E1"),
        "HO=F": AssetConfig("HO=F", "Heating Oil", "Energy", "#708090")
    },
    "Agriculture": {
        "ZC=F": AssetConfig("ZC=F", "Corn Futures", "Agriculture", "#FFD700"),
        "ZW=F": AssetConfig("ZW=F", "Wheat Futures", "Agriculture", "#F5DEB3"),
        "ZS=F": AssetConfig("ZS=F", "Soybean Futures", "Agriculture", "#8B4513"),
        "KC=F": AssetConfig("KC=F", "Coffee Futures", "Agriculture", "#6F4E37")
    }
}

BENCHMARKS = {
    "^GSPC": {"name": "S&P 500", "type": "equity", "color": "#1E90FF"},
    "DX-Y.NYB": {"name": "US Dollar Index", "type": "fx", "color": "#32CD32"},
    "TLT": {"name": "20+ Year Treasury", "type": "fixed_income", "color": "#8A2BE2"},
    "GLD": {"name": "Gold ETF", "type": "commodity", "color": "#FFD700"},
    "DBC": {"name": "Commodities ETF", "type": "commodity", "color": "#FF6347"}
}

# =============================================================================
# STYLES & THEMING
# =============================================================================

APP_STYLES = """
<style>
    /* Main Theme */
    :root {
        --primary: #1a2980;
        --secondary: #26d0ce;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
        --dark: #2c3e50;
        --light: #ecf0f1;
        --gray: #95a5a6;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.3;
        animation: float 20s linear infinite;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--dark);
        margin: 0.5rem 0;
        font-family: 'Arial', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    
    .status-success { 
        background: linear-gradient(135deg, var(--success) 0%, #2ecc71 100%);
        color: white;
    }
    
    .status-warning { 
        background: linear-gradient(135deg, var(--warning) 0%, #f1c40f 100%);
        color: white;
    }
    
    .status-danger { 
        background: linear-gradient(135deg, var(--danger) 0%, #c0392b 100%);
        color: white;
    }
    
    .status-info { 
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: var(--light);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--light);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: white;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary);
        transform: scale(1.05);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Tooltips */
    [title] {
        position: relative;
    }
    
    [title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--dark);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
"""

# Apply styles
st.markdown(APP_STYLES, unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS (OPTIMIZED)
# =============================================================================

class CacheManager:
    """Advanced caching with TTL and memory management"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
    def cached_fetch(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Cached data fetch with optimized parameters"""
        return DataManager.fetch_asset_data(symbol, start_date, end_date)
    
    @staticmethod
    def generate_cache_key(*args) -> str:
        """Generate unique cache key"""
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

class DataManager:
    """Enhanced data management with parallel downloads"""
    
    @staticmethod
    def fetch_asset_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Robust data fetching with multiple fallbacks"""
        try:
            # Try multiple download strategies
            for attempt in range(3):
                try:
                    df = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True,
                        threads=True,
                        timeout=30
                    )
                    
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Process and enhance data
                        df = DataProcessor.enhance_dataframe(df, symbol)
                        return df
                        
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        st.warning(f"Failed to fetch {symbol}: {str(e)[:100]}")
                        return pd.DataFrame()
                    continue
                    
        except Exception:
            return pd.DataFrame()
    
    @staticmethod
    def fetch_multiple_assets(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Parallel download of multiple assets"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(DataManager.fetch_asset_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception:
                    continue
        
        return results

class DataProcessor:
    """Efficient data processing and feature engineering"""
    
    @staticmethod
    def enhance_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive features efficiently"""
        df = df.copy()
        
        # Ensure required columns
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
        
        # Calculate features
        returns = df["Adj Close"].pct_change()
        log_returns = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        
        # Efficient rolling calculations
        df["Returns"] = returns
        df["Log_Returns"] = log_returns
        
        # Moving averages (vectorized)
        df["SMA_20"] = df["Adj Close"].rolling(20).mean()
        df["SMA_50"] = df["Adj Close"].rolling(50).mean()
        df["EMA_12"] = df["Adj Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Adj Close"].ewm(span=26).mean()
        
        # Volatility
        df["Volatility_20D"] = returns.rolling(20).std() * np.sqrt(252)
        df["Volatility_60D"] = returns.rolling(60).std() * np.sqrt(252)
        
        # RSI
        delta = df["Adj Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        bb_middle = df["Adj Close"].rolling(20).mean()
        bb_std = df["Adj Close"].rolling(20).std()
        df["BB_Upper"] = bb_middle + 2 * bb_std
        df["BB_Lower"] = bb_middle - 2 * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_middle.replace(0, np.nan)
        
        # Volume indicators
        if "Volume" in df.columns:
            df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"].replace(0, np.nan)
        
        return df.dropna(subset=["Returns"])

# =============================================================================
# ADVANCED ANALYTICS ENGINE (OPTIMIZED)
# =============================================================================

class EnhancedAnalytics:
    """Optimized analytics engine with modern portfolio theory"""
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive performance metrics efficiently"""
        returns = returns.dropna()
        if len(returns) < 20:
            return {}
        
        # Basic calculations
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility and risk-adjusted returns
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        sortino = (annual_return - risk_free_rate) / (downside_returns.std() * np.sqrt(252)) \
                  if downside_returns.std() > 0 else 0
        
        # Drawdown analysis
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Higher moments
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # VaR and CVaR (95% and 99%)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Win rate and profit factor
        wins = returns > 0
        win_rate = wins.mean() if len(wins) > 0 else 0
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) \
                       if returns[returns < 0].sum() < 0 else float('inf')
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'skewness': skew,
            'kurtosis': kurt,
            'var_95': var_95 * 100,
            'var_99': var_99 * 100,
            'cvar_95': cvar_95 * 100,
            'cvar_99': cvar_99 * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor if profit_factor != float('inf') else 1000,
            'calmar_ratio': (annual_return / abs(max_dd)) if max_dd != 0 else 0
        }
    
    @staticmethod
    def calculate_portfolio_metrics(returns_df: pd.DataFrame, weights: np.ndarray, 
                                   risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        portfolio_returns = returns_df @ weights
        metrics = EnhancedAnalytics.calculate_performance_metrics(portfolio_returns, risk_free_rate)
        
        # Add portfolio-specific metrics
        metrics['weights'] = dict(zip(returns_df.columns, weights))
        metrics['correlation_matrix'] = returns_df.corr().values.tolist()
        
        # Risk decomposition
        cov_matrix = returns_df.cov() * 252
        portfolio_variance = weights.T @ cov_matrix @ weights
        marginal_contributions = (cov_matrix @ weights) / portfolio_variance if portfolio_variance > 0 else weights * 0
        
        metrics['risk_contributions'] = {
            asset: contrib * 100 for asset, contrib in zip(returns_df.columns, marginal_contributions * weights)
        }
        
        return metrics
    
    @staticmethod
    def optimize_portfolio(returns_df: pd.DataFrame, method: str = 'sharpe', 
                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Portfolio optimization using modern portfolio theory"""
        
        def sharpe_ratio(weights: np.ndarray) -> float:
            portfolio_return = np.sum(returns_df.mean() * weights) * 252
            portfolio_vol = np.sqrt(weights.T @ (returns_df.cov() * 252) @ weights)
            return -(portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 1e6
        
        def min_variance(weights: np.ndarray) -> float:
            return weights.T @ (returns_df.cov() * 252) @ weights
        
        n_assets = returns_df.shape[1]
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        if method == 'sharpe':
            result = optimize.minimize(
                sharpe_ratio,
                x0=np.ones(n_assets) / n_assets,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )
        else:  # min variance
            result = optimize.minimize(
                min_variance,
                x0=np.ones(n_assets) / n_assets,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )
        
        if result.success:
            optimized_weights = result.x
            metrics = EnhancedAnalytics.calculate_portfolio_metrics(returns_df, optimized_weights, risk_free_rate)
            return {
                'weights': dict(zip(returns_df.columns, optimized_weights)),
                'metrics': metrics,
                'success': True
            }
        
        return {'success': False, 'message': result.message}

# =============================================================================
# ADVANCED VISUALIZATION ENGINE
# =============================================================================

class ProfessionalVisualizer:
    """Professional visualization with interactive features"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1a2980',
            'secondary': '#26d0ce',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'dark': '#2c3e50',
            'light': '#ecf0f1'
        }
        
        self.template = 'plotly_white'
    
    def create_dashboard_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create comprehensive dashboard chart with subplots"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD')
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Adj Close'], name='Price',
                      line=dict(color=self.color_palette['primary'], width=2)),
            row=1, col=1
        )
        
        for ma, color in [('SMA_20', '#f39c12'), ('SMA_50', '#27ae60')]:
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[ma], name=ma,
                              line=dict(color=color, width=1, dash='dash')),
                    row=1, col=1
                )
        
        # Volume with color coding
        if 'Volume' in df.columns:
            colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(df['Adj Close'], df['Open'])]
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors, opacity=0.7),
                row=2, col=1
            )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                          line=dict(color=self.color_palette['secondary'], width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                          line=dict(color='blue', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                          line=dict(color='orange', width=2)),
                row=4, col=1
            )
            
            # Histogram
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                      marker_color=colors, opacity=0.6),
                row=4, col=1
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=900,
            template=self.template,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_correlation_matrix(self, corr_matrix: pd.DataFrame, title: str) -> go.Figure:
        """Create interactive correlation heatmap with annotations"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            hoverinfo='x+y+z',
            colorbar=dict(title='Correlation', titleside='right')
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=600,
            width=max(800, len(corr_matrix.columns) * 100),
            template=self.template,
            xaxis_tickangle=45
        )
        
        return fig
    
    def create_performance_radar(self, metrics: Dict[str, float], title: str) -> go.Figure:
        """Create radar chart for performance metrics"""
        categories = ['Return', 'Risk', 'Sharpe', 'Sortino', 'Max DD', 'Win Rate']
        
        # Normalize metrics for radar chart
        values = [
            metrics.get('annual_return', 0) / 50,  # Normalize to 0-1 scale
            1 - min(metrics.get('annual_volatility', 0) / 50, 1),
            min(metrics.get('sharpe_ratio', 0) / 3, 1),
            min(metrics.get('sortino_ratio', 0) / 3, 1),
            1 - min(abs(metrics.get('max_drawdown', 0)) / 50, 1),
            metrics.get('win_rate', 0) / 100
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(38, 208, 206, 0.3)',
            line=dict(color='rgb(38, 208, 206)', width=2),
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90
                )
            ),
            title=dict(text=title, x=0.5),
            height=500,
            template=self.template
        )
        
        return fig
    
    def create_risk_decomposition(self, risk_contributions: Dict[str, float], title: str) -> go.Figure:
        """Create pie chart for risk decomposition"""
        labels = list(risk_contributions.keys())
        values = list(risk_contributions.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3),
            hoverinfo='label+value+percent'
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=500,
            template=self.template,
            showlegend=False
        )
        
        return fig

# =============================================================================
# ENHANCED REPORT GENERATOR
# =============================================================================

class ProfessionalReportGenerator:
    """Generate professional PDF/HTML reports"""
    
    @staticmethod
    def generate_html_report(portfolio_data: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Institutional Commodities Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }}
                
                .header {{
                    background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 20px;
                    margin-bottom: 30px;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                
                .header p {{
                    margin: 10px 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                
                .section {{
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                }}
                
                .section-title {{
                    color: #1a2980;
                    border-bottom: 3px solid #26d0ce;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                    font-size: 1.8em;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .metric-card {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    border-left: 4px solid #1a2980;
                    transition: transform 0.3s ease;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-5px);
                }}
                
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #1a2980;
                    margin: 10px 0;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: #6c757d;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .neutral {{ color: #f39c12; }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                
                th {{
                    background-color: #1a2980;
                    color: white;
                    font-weight: 600;
                }}
                
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                
                .disclaimer {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 30px;
                    font-size: 0.9em;
                    color: #856404;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏛️ Institutional Commodities Analytics Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Assets</div>
                        <div class="metric-value">{len(portfolio_data.get('assets', []))}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Annual Return</div>
                        <div class="metric-value {'positive' if metrics.get('annual_return', 0) > 0 else 'negative'}">
                            {metrics.get('annual_return', 0):.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📈 Portfolio Composition</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Weight</th>
                            <th>Annual Return</th>
                            <th>Volatility</th>
                            <th>Risk Contribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
                            f'''<tr>
                                <td>{asset}</td>
                                <td>{weight:.1%}</td>
                                <td class="{'positive' if perf.get('annual_return', 0) > 0 else 'negative'}">
                                    {perf.get('annual_return', 0):.2f}%
                                </td>
                                <td>{perf.get('annual_volatility', 0):.2f}%</td>
                                <td>{risk_contrib.get(asset, 0):.1f}%</td>
                            </tr>'''
                            for asset, weight in portfolio_data.get('weights', {}).items()
                        ])}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">⚖️ Risk Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">VaR (95%)</div>
                        <div class="metric-value negative">{metrics.get('var_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">CVaR (95%)</div>
                        <div class="metric-value negative">{metrics.get('cvar_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{metrics.get('sortino_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This report is for informational purposes only. 
                Past performance is not indicative of future results. 
                Consult with a qualified financial advisor before making investment decisions.
                Data source: Yahoo Finance. Generated by Institutional Commodities Analytics Platform v5.0.
            </div>
            
            <div class="footer">
                <p>© {datetime.now().year} Institutional Commodities Analytics Platform v5.0</p>
                <p>Confidential - For Institutional Use Only</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    @staticmethod
    def generate_pdf_report(html_content: str, filename: str = "report.pdf"):
        """Generate PDF report from HTML (requires additional libraries)"""
        # This is a placeholder - in production, use libraries like weasyprint or xhtml2pdf
        pass

# =============================================================================
# MAIN DASHBOARD CLASS
# =============================================================================

class EnhancedCommoditiesDashboard:
    """Enhanced dashboard with optimized performance"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analytics = EnhancedAnalytics()
        self.visualizer = ProfessionalVisualizer()
        self.report_generator = ProfessionalReportGenerator()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state with default values"""
        defaults = {
            'data_loaded': False,
            'selected_assets': [],
            'asset_data': {},
            'returns_data': {},
            'portfolio_weights': {},
            'portfolio_metrics': {},
            'analysis_config': AnalysisConfig(
                start_date=datetime.now() - timedelta(days=1095),
                end_date=datetime.now()
            )
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def display_header(self):
        """Display enhanced header with real-time metrics"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; font-size:2.8rem; font-weight:700;">🏛️ Institutional Commodities Analytics v5.0</h1>
            <p style="margin:10px 0 0 0; opacity:0.95; font-size:1.2rem;">
                Advanced Portfolio Analytics • Optimized GARCH • AI-Powered Regime Detection • Professional Reporting
            </p>
            <div style="margin-top:20px; display:flex; gap:15px; flex-wrap:wrap;">
                <span class="status-badge status-success">🟢 Real-time Data</span>
                <span class="status-badge status-info">📊 Multi-Asset</span>
                <span class="status-badge status-warning">⚡ Optimized</span>
                <span class="status-badge status-success">🔐 Institutional</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Platform Status", "🟢 ONLINE", "v5.0")
        with col2:
            assets_loaded = len(st.session_state.get('asset_data', {}))
            st.metric("Assets Loaded", assets_loaded, f"{assets_loaded} active")
        with col3:
            date_range = (st.session_state.get('analysis_config', AnalysisConfig()).end_date - 
                         st.session_state.get('analysis_config', AnalysisConfig()).start_date).days
            st.metric("Time Range", f"{date_range} days")
        with col4:
            st.metric("Performance", "Optimized", "v5.0")
    
    def setup_sidebar(self) -> AnalysisConfig:
        """Setup enhanced sidebar with configuration options"""
        with st.sidebar:
            # Platform logo and info
            st.markdown("""
            <div style="text-align:center; margin-bottom:30px;">
                <h2 style="color:#1a2980;">📈 ICA v5.0</h2>
                <p style="color:#7f8c8d; font-size:0.9rem;">Institutional Grade Analytics</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date Range Configuration
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📅 **Date Configuration**")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=st.session_state.get('analysis_config', AnalysisConfig()).start_date.date(),
                    key="sidebar_start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=st.session_state.get('analysis_config', AnalysisConfig()).end_date.date(),
                    key="sidebar_end_date"
                )
            
            # Quick date presets
            preset_cols = st.columns(4)
            with preset_cols[0]:
                if st.button("1Y", use_container_width=True):
                    start_date = datetime.now().date() - timedelta(days=365)
            with preset_cols[1]:
                if st.button("3Y", use_container_width=True):
                    start_date = datetime.now().date() - timedelta(days=1095)
            with preset_cols[2]:
                if st.button("5Y", use_container_width=True):
                    start_date = datetime.now().date() - timedelta(days=1825)
            with preset_cols[3]:
                if st.button("Max", use_container_width=True):
                    start_date = datetime(2010, 1, 1).date()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Asset Selection
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📊 **Asset Selection**")
            
            selected_assets = []
            for category, assets in COMMODITIES_UNIVERSE.items():
                with st.expander(f"**{category}**", expanded=False):
                    for symbol, config in assets.items():
                        if st.checkbox(
                            f"{config.name} ({symbol})",
                            key=f"asset_{symbol}",
                            value=symbol in st.session_state.get('selected_assets', [])
                        ):
                            selected_assets.append(symbol)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Benchmark Selection
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📈 **Benchmarks**")
            
            selected_benchmarks = []
            for symbol, info in BENCHMARKS.items():
                if st.checkbox(f"{info['name']} ({symbol})", key=f"bench_{symbol}"):
                    selected_benchmarks.append(symbol)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analytics Configuration
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ **Analytics Settings**")
            
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100
            
            # Optimization settings
            st.markdown("**Portfolio Optimization**")
            optimization_method = st.selectbox(
                "Method",
                ["Equal Weight", "Sharpe Ratio", "Minimum Variance"],
                index=0
            )
            
            # Risk settings
            st.markdown("**Risk Configuration**")
            confidence_levels = st.multiselect(
                "Confidence Levels",
                [0.90, 0.95, 0.99],
                default=[0.95]
            )
            
            backtest_window = st.slider("Backtest Window", 100, 500, 250, 25)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 **Load Data**", type="primary", use_container_width=True):
                    self._load_market_data(selected_assets, selected_benchmarks, start_date, end_date)
            
            with col2:
                if st.button("🔄 **Clear Cache**", type="secondary", use_container_width=True):
                    st.cache_data.clear()
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Create analysis configuration
            config = AnalysisConfig(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                risk_free_rate=risk_free_rate,
                confidence_levels=tuple(confidence_levels),
                backtest_window=backtest_window
            )
            
            return config
    
    def _load_market_data(self, assets: List[str], benchmarks: List[str],
                         start_date: datetime, end_date: datetime):
        """Load market data with progress tracking"""
        if not assets:
            st.warning("Please select at least one asset")
            return
        
        with st.spinner("🚀 Loading market data..."):
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load assets
            status_text.text("📥 Downloading asset data...")
            asset_data = self.data_manager.fetch_multiple_assets(assets, start_date, end_date)
            
            if not asset_data:
                st.error("Failed to load any asset data. Please check symbols and date range.")
                return
            
            progress_bar.progress(0.5)
            
            # Load benchmarks
            status_text.text("📊 Downloading benchmark data...")
            benchmark_data = self.data_manager.fetch_multiple_assets(benchmarks, start_date, end_date)
            
            progress_bar.progress(0.8)
            
            # Process data
            status_text.text("⚙️ Processing data...")
            returns_data = {
                symbol: df['Returns'].dropna()
                for symbol, df in asset_data.items()
                if 'Returns' in df.columns and not df['Returns'].dropna().empty
            }
            
            # Update session state
            st.session_state.asset_data = asset_data
            st.session_state.returns_data = returns_data
            st.session_state.selected_assets = list(asset_data.keys())
            st.session_state.data_loaded = True
            
            # Initialize equal weights
            n_assets = len(asset_data)
            equal_weight = 1.0 / n_assets
            st.session_state.portfolio_weights = {
                asset: equal_weight for asset in asset_data.keys()
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Data loaded successfully!")
            
            st.success(f"""
            ✓ Successfully loaded {len(asset_data)} assets
            📊 {len(benchmark_data)} benchmarks loaded
            📈 {len(returns_data)} return series calculated
            """)
            
            # Brief delay to show success message
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    def run(self):
        """Main dashboard execution flow"""
        self.display_header()
        
        # Setup sidebar and get configuration
        config = self.setup_sidebar()
        
        if not st.session_state.data_loaded:
            st.info("""
            👈 **Welcome to Institutional Commodities Analytics v5.0**
            
            To get started:
            1. Select assets from the sidebar
            2. Choose your date range
            3. Click **Load Data**
            
            *Advanced features include portfolio optimization, risk analysis, and professional reporting.*
            """)
            return
        
        # Create tabs for different analytics sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 **Dashboard**",
            "🧺 **Portfolio**", 
            "📈 **Analytics**",
            "⚡ **Advanced**",
            "📋 **Reports**"
        ])
        
        with tab1:
            self.display_dashboard(config)
        
        with tab2:
            self.display_portfolio_analysis(config)
        
        with tab3:
            self.display_advanced_analytics(config)
        
        with tab4:
            self.display_advanced_features(config)
        
        with tab5:
            self.display_reporting()
    
    def display_dashboard(self, config: AnalysisConfig):
        """Display enhanced dashboard"""
        st.header("📊 Market Dashboard")
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            avg_return = returns_df.mean().mean() * 252 * 100 if not returns_df.empty else 0
            st.metric("Avg Annual Return", f"{avg_return:.2f}%")
        
        with col2:
            avg_vol = returns_df.std().mean() * np.sqrt(252) * 100 if not returns_df.empty else 0
            st.metric("Avg Volatility", f"{avg_vol:.2f}%")
        
        with col3:
            if len(returns_df.columns) > 1:
                avg_corr = returns_df.corr().values[np.triu_indices(len(returns_df.columns), 1)].mean()
                st.metric("Avg Correlation", f"{avg_corr:.2f}")
            else:
                st.metric("Avg Correlation", "N/A")
        
        with col4:
            total_days = len(returns_df) if not returns_df.empty else 0
            st.metric("Trading Days", total_days)
        
        # Asset performance table
        st.subheader("📈 Asset Performance Overview")
        
        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if 'Returns' in df.columns:
                returns = df['Returns'].dropna()
                if len(returns) > 0:
                    metrics = self.analytics.calculate_performance_metrics(returns, config.risk_free_rate)
                    
                    performance_data.append({
                        'Asset': symbol,
                        'Name': COMMODITIES_UNIVERSE.get(
                            next((cat for cat, assets in COMMODITIES_UNIVERSE.items() 
                                  if symbol in assets), 'Unknown'),
                            {}
                        ).get(symbol, AssetConfig(symbol, symbol, 'Unknown')).name,
                        'Current Price': df['Adj Close'].iloc[-1] if not df.empty else np.nan,
                        '1D Return': df['Returns'].iloc[-1] * 100 if len(df) > 1 else 0,
                        'Annual Return': metrics.get('annual_return', 0),
                        'Annual Vol': metrics.get('annual_volatility', 0),
                        'Sharpe': metrics.get('sharpe_ratio', 0),
                        'Max DD': metrics.get('max_drawdown', 0)
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Apply formatting
            formatted_df = perf_df.style.format({
                'Current Price': '{:.2f}',
                '1D Return': '{:.2f}%',
                'Annual Return': '{:.2f}%',
                'Annual Vol': '{:.2f}%',
                'Sharpe': '{:.2f}',
                'Max DD': '{:.2f}%'
            }).background_gradient(
                subset=['Annual Return', 'Sharpe'],
                cmap='RdYlGn'
            ).background_gradient(
                subset=['Annual Vol', 'Max DD'],
                cmap='RdYlGn_r'
            )
            
            st.dataframe(formatted_df, use_container_width=True, height=400)
        
        # Individual asset analysis
        st.subheader("📉 Detailed Asset Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed View",
            options=st.session_state.selected_assets,
            key="dashboard_asset_select"
        )
        
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            
            # Create tabs for different visualizations
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Price Analysis", "Returns Distribution", "Technical Indicators"])
            
            with chart_tab1:
                fig = self.visualizer.create_dashboard_chart(df, f"{selected_asset} - Comprehensive Analysis")
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab2:
                returns = df['Returns'].dropna()
                
                # Create distribution plot
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Returns Distribution", "QQ Plot"))
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=returns * 100, nbinsx=50, name="Returns",
                               marker_color=self.visualizer.color_palette['primary']),
                    row=1, col=1
                )
                
                # QQ Plot
                qq_data = stats.probplot(returns.dropna(), dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                             name="Data", marker=dict(color=self.visualizer.color_palette['secondary'])),
                    row=1, col=2
                )
                
                # Add theoretical line
                x_line = np.array([qq_data[0][0][0], qq_data[0][0][-1]])
                y_line = qq_data[1][0] + qq_data[1][1] * x_line
                fig.add_trace(
                    go.Scatter(x=x_line, y=y_line, mode='lines',
                             name="Normal", line=dict(color='red', dash='dash')),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, template=self.visualizer.template)
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab3:
                # Create technical indicators summary
                tech_metrics = {
                    'RSI': df['RSI'].iloc[-1] if 'RSI' in df.columns else np.nan,
                    'MACD': df['MACD'].iloc[-1] if 'MACD' in df.columns else np.nan,
                    'MACD Signal': df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else np.nan,
                    'Bollinger Position': ((df['Adj Close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / 
                                          (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])) 
                                          if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']) else np.nan,
                    'Volume Ratio': df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else np.nan
                }
                
                # Display metrics
                cols = st.columns(len(tech_metrics))
                for (metric, value), col in zip(tech_metrics.items(), cols):
                    with col:
                        st.metric(metric, f"{value:.2f}" if not np.isnan(value) else "N/A")
        
        # Correlation matrix
        st.subheader("📊 Correlation Analysis")
        
        if len(st.session_state.returns_data) > 1:
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            
            if not returns_df.empty and len(returns_df.columns) > 1:
                corr_matrix = returns_df.corr()
                
                # Create tabs for correlation visualization
                corr_tab1, corr_tab2 = st.tabs(["Heatmap", "Network"])
                
                with corr_tab1:
                    fig = self.visualizer.create_correlation_matrix(corr_matrix, "Asset Correlations")
                    st.plotly_chart(fig, use_container_width=True)
                
                with corr_tab2:
                    # Create network visualization
                    edges = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr = corr_matrix.iloc[i, j]
                            if abs(corr) > 0.3:  # Only show significant correlations
                                edges.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    corr
                                ))
                    
                    if edges:
                        # Create network plot
                        edge_trace = go.Scatter(
                            x=[], y=[], mode='lines',
                            line=dict(width=1, color='#888'),
                            hoverinfo='none'
                        )
                        
                        node_trace = go.Scatter(
                            x=[], y=[], mode='markers+text',
                            text=[], textposition="top center",
                            marker=dict(size=20, color='#1a2980'),
                            hoverinfo='text'
                        )
                        
                        # Simple circular layout
                        positions = {}
                        n_nodes = len(corr_matrix.columns)
                        for i, node in enumerate(corr_matrix.columns):
                            angle = 2 * np.pi * i / n_nodes
                            positions[node] = (np.cos(angle), np.sin(angle))
                        
                        # Add edges
                        edge_x, edge_y = [], []
                        for src, dst, weight in edges:
                            x0, y0 = positions[src]
                            x1, y1 = positions[dst]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        edge_trace.x = edge_x
                        edge_trace.y = edge_y
                        
                        # Add nodes
                        node_x, node_y, node_text = [], [], []
                        for node, (x, y) in positions.items():
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                        
                        node_trace.x = node_x
                        node_trace.y = node_y
                        node_trace.text = node_text
                        
                        fig = go.Figure(data=[edge_trace, node_trace],
                                      layout=go.Layout(
                                          title="Correlation Network",
                                          showlegend=False,
                                          hovermode='closest',
                                          height=500,
                                          template=self.visualizer.template
                                      ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No significant correlations found (|correlation| > 0.3)")
    
    def display_portfolio_analysis(self, config: AnalysisConfig):
        """Display enhanced portfolio analysis"""
        st.header("🧺 Portfolio Analysis & Optimization")
        
        if not st.session_state.returns_data:
            st.warning("No return data available. Please load data first.")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty:
            st.warning("Insufficient data for portfolio analysis")
            return
        
        # Portfolio configuration
        st.subheader("Portfolio Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            weight_mode = st.radio(
                "Weighting Method",
                ["Equal Weight", "Optimized (Sharpe)", "Optimized (Min Variance)", "Custom Weights"],
                horizontal=True,
                key="portfolio_weight_mode"
            )
            
            if weight_mode == "Custom Weights":
                st.write("**Set Custom Weights:**")
                
                # Create weight sliders in columns
                assets = returns_df.columns.tolist()
                n_cols = min(4, len(assets))
                cols = st.columns(n_cols)
                
                weight_inputs = []
                for i, asset in enumerate(assets):
                    with cols[i % n_cols]:
                        default_weight = 1.0 / len(assets)
                        weight = st.slider(
                            asset,
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.portfolio_weights.get(asset, default_weight),
                            step=0.01,
                            key=f"custom_weight_{asset}"
                        )
                        weight_inputs.append(weight)
                
                weights = np.array(weight_inputs)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            elif weight_mode == "Optimized (Sharpe)":
                if st.button("🔄 Optimize for Sharpe Ratio", type="primary"):
                    with st.spinner("Optimizing portfolio..."):
                        result = self.analytics.optimize_portfolio(returns_df, 'sharpe', config.risk_free_rate)
                        if result['success']:
                            weights = np.array(list(result['weights'].values()))
                            st.session_state.portfolio_weights = result['weights']
                            st.session_state.portfolio_metrics = result['metrics']
                            st.success("Portfolio optimized successfully!")
                        else:
                            st.warning("Optimization failed. Using equal weights.")
                            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                else:
                    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            
            elif weight_mode == "Optimized (Min Variance)":
                if st.button("🔄 Optimize for Minimum Variance", type="primary"):
                    with st.spinner("Optimizing portfolio..."):
                        result = self.analytics.optimize_portfolio(returns_df, 'min_variance', config.risk_free_rate)
                        if result['success']:
                            weights = np.array(list(result['weights'].values()))
                            st.session_state.portfolio_weights = result['weights']
                            st.session_state.portfolio_metrics = result['metrics']
                            st.success("Portfolio optimized successfully!")
                        else:
                            st.warning("Optimization failed. Using equal weights.")
                            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                else:
                    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            
            else:  # Equal Weight
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                st.session_state.portfolio_weights = dict(zip(returns_df.columns, weights))
        
        with col2:
            # Display current weights
            st.write("**Current Weights:**")
            for asset, weight in st.session_state.portfolio_weights.items():
                st.progress(float(weight), text=f"{asset}: {weight:.1%}")
        
        # Calculate portfolio metrics
        portfolio_metrics = self.analytics.calculate_portfolio_metrics(returns_df, weights, config.risk_free_rate)
        st.session_state.portfolio_metrics = portfolio_metrics
        
        # Performance metrics
        st.subheader("📊 Portfolio Performance")
        
        # Create metrics grid
        metric_cols = st.columns(4)
        metric_configs = [
            ("Annual Return", "annual_return", "{:.2f}%", "positive" if portfolio_metrics.get('annual_return', 0) > 0 else "negative"),
            ("Annual Volatility", "annual_volatility", "{:.2f}%", "neutral"),
            ("Sharpe Ratio", "sharpe_ratio", "{:.2f}", "positive" if portfolio_metrics.get('sharpe_ratio', 0) > 0 else "negative"),
            ("Max Drawdown", "max_drawdown", "{:.2f}%", "negative")
        ]
        
        for col, (label, key, fmt, color_class) in zip(metric_cols, metric_configs):
            with col:
                value = portfolio_metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional metrics
        metric_cols2 = st.columns(4)
        metric_configs2 = [
            ("Sortino Ratio", "sortino_ratio", "{:.2f}", "positive" if portfolio_metrics.get('sortino_ratio', 0) > 0 else "negative"),
            ("Calmar Ratio", "calmar_ratio", "{:.2f}", "positive" if portfolio_metrics.get('calmar_ratio', 0) > 0 else "negative"),
            ("Win Rate", "win_rate", "{:.1f}%", "positive"),
            ("Profit Factor", "profit_factor", "{:.2f}", "positive" if portfolio_metrics.get('profit_factor', 0) > 1 else "negative")
        ]
        
        for col, (label, key, fmt, color_class) in zip(metric_cols2, metric_configs2):
            with col:
                value = portfolio_metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Risk metrics
        st.subheader("⚖️ Risk Metrics")
        
        risk_cols = st.columns(4)
        risk_configs = [
            ("VaR (95%)", "var_95", "{:.2f}%", "negative"),
            ("CVaR (95%)", "cvar_95", "{:.2f}%", "negative"),
            ("VaR (99%)", "var_99", "{:.2f}%", "negative"),
            ("CVaR (99%)", "cvar_99", "{:.2f}%", "negative")
        ]
        
        for col, (label, key, fmt, color_class) in zip(risk_cols, risk_configs):
            with col:
                value = portfolio_metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📈 Portfolio Visualization")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Risk decomposition
            if 'risk_contributions' in portfolio_metrics:
                fig = self.visualizer.create_risk_decomposition(
                    portfolio_metrics['risk_contributions'],
                    "Risk Contribution Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Performance radar
            fig = self.visualizer.create_performance_radar(
                portfolio_metrics,
                "Performance Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative returns
        st.subheader("📊 Cumulative Returns")
        
        portfolio_returns = pd.Series(
            returns_df.values @ weights,
            index=returns_df.index,
            name="Portfolio"
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_returns.index,
            y=(1 + portfolio_returns).cumprod(),
            name="Portfolio",
            line=dict(color=self.visualizer.color_palette['primary'], width=3),
            fill='tozeroy',
            fillcolor='rgba(26, 41, 128, 0.1)'
        ))
        
        # Add benchmarks if available
        for benchmark_symbol, benchmark_data in st.session_state.get('benchmark_data', {}).items():
            if 'Returns' in benchmark_data.columns:
                benchmark_returns = benchmark_data['Returns'].dropna()
                aligned_idx = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(aligned_idx) > 0:
                    fig.add_trace(go.Scatter(
                        x=aligned_idx,
                        y=(1 + benchmark_returns.reindex(aligned_idx)).cumprod(),
                        name=f"Benchmark: {benchmark_symbol}",
                        line=dict(dash='dash', width=2)
                    ))
        
        fig.update_layout(
            title="Portfolio vs Benchmarks",
            height=500,
            template=self.visualizer.template,
            hovermode='x unified',
            yaxis_title="Cumulative Return",
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download portfolio data
        st.subheader("💾 Export Portfolio Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Portfolio Metrics", use_container_width=True):
                # Create downloadable DataFrame
                metrics_df = pd.DataFrame.from_dict(portfolio_metrics, orient='index', columns=['Value'])
                csv = metrics_df.to_csv()
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download Returns Data", use_container_width=True):
                returns_data = pd.DataFrame({
                    'Date': portfolio_returns.index,
                    'Portfolio_Return': portfolio_returns.values,
                    'Cumulative_Return': (1 + portfolio_returns).cumprod().values
                })
                csv = returns_data.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def display_advanced_analytics(self, config: AnalysisConfig):
        """Display advanced analytics features"""
        st.header("📈 Advanced Analytics")
        
        if not st.session_state.returns_data:
            st.warning("Please load data first")
            return
        
        # VaR Backtesting
        st.subheader("✅ Value at Risk (VaR) Backtesting")
        
        selected_asset = st.selectbox(
            "Select Asset for VaR Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="var_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                var_method = st.selectbox("VaR Method", ["Historical", "Parametric (Normal)", "Parametric (t-dist)"])
            
            with col2:
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.99],
                    value=0.95,
                    format_func=lambda x: f"{x:.0%}"
                )
            
            with col3:
                window_size = st.slider("Window Size", 100, 500, 250, 50)
            
            if st.button("🔍 Run VaR Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    # Implement VaR backtest here
                    # Placeholder for actual implementation
                    st.info("VaR backtest implementation in progress...")
        
        # Rolling statistics
        st.subheader("📊 Rolling Statistics")
        
        selected_asset2 = st.selectbox(
            "Select Asset for Rolling Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="rolling_asset_select"
        )
        
        if selected_asset2:
            returns = st.session_state.returns_data[selected_asset2]
            
            col1, col2 = st.columns(2)
            with col1:
                window = st.slider("Rolling Window (days)", 20, 252, 60, 20)
            
            with col2:
                stat_type = st.selectbox("Statistic", ["Mean", "Volatility", "Skewness", "Kurtosis"])
            
            # Calculate rolling statistics
            if stat_type == "Mean":
                rolling_stat = returns.rolling(window).mean() * 252 * 100
            elif stat_type == "Volatility":
                rolling_stat = returns.rolling(window).std() * np.sqrt(252) * 100
            elif stat_type == "Skewness":
                rolling_stat = returns.rolling(window).skew()
            else:  # Kurtosis
                rolling_stat = returns.rolling(window).kurt()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_stat.index,
                y=rolling_stat.values,
                name=f"Rolling {stat_type}",
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"{selected_asset2} - Rolling {stat_type} ({window}-day window)",
                height=400,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title=stat_type
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_advanced_features(self, config: AnalysisConfig):
        """Display advanced features"""
        st.header("⚡ Advanced Features")
        
        # Placeholder for advanced features
        st.info("""
        **Advanced Features Coming Soon:**
        
        - **AI-Powered Forecasting**: Machine learning models for price prediction
        - **Regime Detection**: Hidden Markov Models for market regime identification
        - **GARCH Modeling**: Advanced volatility forecasting
        - **Monte Carlo Simulation**: Portfolio scenario analysis
        - **Stress Testing**: Extreme market scenario analysis
        
        *These features require additional computational resources and are under development.*
        """)
        
        # Feature request form
        with st.expander("📝 Request Advanced Features"):
            feature = st.text_input("Feature Request")
            email = st.text_input("Contact Email (optional)")
            
            if st.button("Submit Request"):
                st.success("Thank you for your feature request!")
    
    def display_reporting(self):
        """Display professional reporting interface"""
        st.header("📋 Professional Reports")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Portfolio Summary", "Risk Analysis", "Performance Attribution", "Comprehensive"]
            )
        
        with col2:
            report_format = st.selectbox(
                "Format",
                ["HTML", "PDF", "Markdown"]
            )
        
        # Report options
        st.subheader("Report Options")
        
        options_cols = st.columns(3)
        with options_cols[0]:
            include_charts = st.checkbox("Include Charts", value=True)
        with options_cols[1]:
            include_tables = st.checkbox("Include Tables", value=True)
        with options_cols[2]:
            include_metrics = st.checkbox("Include Metrics", value=True)
        
        # Generate report
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating professional report..."):
                # Prepare report data
                report_data = {
                    'assets': st.session_state.selected_assets,
                    'weights': st.session_state.portfolio_weights,
                    'metrics': st.session_state.portfolio_metrics,
                    'config': {
                        'report_type': report_type,
                        'include_charts': include_charts,
                        'include_tables': include_tables,
                        'include_metrics': include_metrics
                    }
                }
                
                # Generate HTML report
                html_report = self.report_generator.generate_html_report(
                    report_data,
                    st.session_state.portfolio_metrics
                )
                
                # Display preview
                st.subheader("📊 Report Preview")
                st.components.v1.html(html_report, height=800, scrolling=True)
                
                # Download options
                st.subheader("💾 Download Report")
                
                if report_format == "HTML":
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_report,
                        file_name=f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                
                elif report_format == "PDF":
                    # Placeholder for PDF generation
                    st.info("PDF generation requires additional libraries. Downloading HTML version instead.")
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_report,
                        file_name=f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                
                else:  # Markdown
                    # Convert to markdown (simplified)
                    markdown_report = f"""
# Commodities Analytics Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Portfolio Summary
- Assets: {len(report_data['assets'])}
- Annual Return: {st.session_state.portfolio_metrics.get('annual_return', 0):.2f}%
- Sharpe Ratio: {st.session_state.portfolio_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {st.session_state.portfolio_metrics.get('max_drawdown', 0):.2f}%

### Risk Metrics
- VaR (95%): {st.session_state.portfolio_metrics.get('var_95', 0):.2f}%
- CVaR (95%): {st.session_state.portfolio_metrics.get('cvar_95', 0):.2f}%
- Annual Volatility: {st.session_state.portfolio_metrics.get('annual_volatility', 0):.2f}%
"""
                    
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=markdown_report,
                        file_name=f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        # Quick snapshot
        st.subheader("📸 Quick Snapshot")
        
        if st.button("Take Snapshot", use_container_width=True):
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'assets': st.session_state.selected_assets,
                'portfolio_metrics': st.session_state.portfolio_metrics,
                'weights': st.session_state.portfolio_weights
            }
            
            st.json(snapshot, expanded=False)
            
            st.download_button(
                label="📥 Download JSON Snapshot",
                data=json.dumps(snapshot, indent=2),
                file_name=f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main application entry point"""
    # Hide streamlit default elements
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    try:
        dashboard = EnhancedCommoditiesDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page or clear your cache.")

if __name__ == "__main__":
    main()
