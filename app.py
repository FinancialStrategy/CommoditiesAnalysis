"""
🏛️ Institutional Commodities Analytics Platform
Advanced GARCH, Regime Detection, Portfolio Analysis & Risk Management
Streamlit Cloud Optimized
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Professional CSS (Minimal Colors)
# ============================================================================

st.markdown("""
<style>
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #1a2980;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 1.5rem;
        font-weight: 600;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a2980;
        color: white !important;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-primary {
        background: #1a2980;
        color: white;
    }
    
    .status-success {
        background: #28a745;
        color: white;
    }
    
    .status-warning {
        background: #ffc107;
        color: #212529;
    }
    
    .status-danger {
        background: #dc3545;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 41, 128, 0.2);
    }
    
    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Commodities Universe
# ============================================================================

COMMODITIES = {
    "Precious Metals": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "PL=F": "Platinum Futures",
        "PA=F": "Palladium Futures",
    },
    "Industrial Metals": {
        "HG=F": "Copper Futures",
        "ALI=F": "Aluminum Futures",
    },
    "Energy": {
        "CL=F": "Crude Oil WTI",
        "NG=F": "Natural Gas",
        "BZ=F": "Brent Crude",
    },
    "Agriculture": {
        "ZC=F": "Corn Futures",
        "ZW=F": "Wheat Futures",
        "ZS=F": "Soybean Futures",
    }
}

BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "DXY": "US Dollar Index",
    "TLT": "20+ Year Treasury",
    "GLD": "Gold ETF"
}

# ============================================================================
# Data Manager
# ============================================================================

class DataManager:
    """Data management with caching"""
    
    def __init__(self):
        self.session = None
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                timeout=30
            )
            
            if df.empty or len(df) < 50:
                return None
            
            # Calculate basic indicators
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            return df
            
        except Exception as e:
            st.warning(f"Failed to fetch {symbol}: {str(e)[:100]}")
            return None
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def bulk_fetch(_self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Bulk fetch data for multiple symbols"""
        data_dict = {}
        
        for symbol in symbols:
            df = _self.fetch_data(symbol, start_date, end_date)
            if df is not None:
                data_dict[symbol] = df
        
        return data_dict

# ============================================================================
# Analytics Engine
# ============================================================================

class AnalyticsEngine:
    """Advanced analytics engine"""
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if df is None or len(df) < 20:
            return {}
        
        returns = df['Returns'].dropna()
        
        metrics = {
            # Basic metrics
            'current_price': df['Close'].iloc[-1],
            'daily_change': df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0,
            'daily_change_pct': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
            
            # Return metrics
            'total_return': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100,
            'annual_return': ((1 + returns.mean()) ** 252 - 1) * 100,
            'annual_volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            
            # Risk metrics
            'var_95': np.percentile(returns, 5) * 100,
            'max_drawdown': self._calculate_max_drawdown(df) * 100,
            
            # Technical indicators
            'sma_20': df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else df['Close'].iloc[-1],
            'sma_50': df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else df['Close'].iloc[-1],
            'volatility_20d': df['Volatility_20'].iloc[-1] * 100 if 'Volatility_20' in df.columns else 0,
        }
        
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
        
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 50:
            return pd.DataFrame()
        
        return returns_df.corr()
    
    def fit_garch_model(self, returns: pd.Series, p: int = 1, q: int = 1) -> Optional[Dict]:
        """Fit GARCH model to returns"""
        if len(returns) < 200:
            return None
        
        try:
            returns_scaled = returns.dropna() * 100
            
            model = arch_model(
                returns_scaled,
                mean='Constant',
                vol='GARCH',
                p=p,
                q=q,
                dist='t'
            )
            
            result = model.fit(disp='off', show_warning=False)
            cond_vol = result.conditional_volatility / 100
            
            return {
                'model': result,
                'params': dict(result.params),
                'aic': result.aic,
                'bic': result.bic,
                'cond_volatility': cond_vol,
                'converged': result.convergence_flag == 0
            }
            
        except Exception:
            return None
    
    def forecast_volatility(self, garch_model: Dict, steps: int = 30) -> np.ndarray:
        """Forecast volatility using GARCH model"""
        if garch_model is None or 'model' not in garch_model:
            return np.array([])
        
        try:
            forecast = garch_model['model'].forecast(horizon=steps)
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].values) / 100
            return forecast_vol
        except:
            return np.array([])
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate advanced risk metrics"""
        if len(returns) < 100:
            return {}
        
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # Risk-adjusted returns
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }

# ============================================================================
# Visualization Engine
# ============================================================================

class VisualizationEngine:
    """Professional visualization engine"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1a2980',
            'secondary': '#26d0ce',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'dark': '#343a40',
            'light': '#f8f9fa'
        }
    
    def create_price_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create professional price chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{title} - Price Action', 'Volume')
        )
        
        # Price
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
        
        # Moving averages
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
        
        # Volume
        colors = [self.color_palette['success'] if close >= open_ else self.color_palette['danger'] 
                 for close, open_ in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color=self.color_palette['dark'])
            ),
            height=600,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_volatility_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create volatility chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Volatility_20'] * 100,
            name='20-Day Volatility',
            line=dict(color=self.color_palette['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(26, 41, 128, 0.1)'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{title} - Historical Volatility',
                x=0.5,
                xanchor='center'
            ),
            height=400,
            template='plotly_white',
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str) -> go.Figure:
        """Create correlation heatmap - FIXED VERSION"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            hoverongaps=False,
            colorbar=dict(
                title="Correlation"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color=self.color_palette['dark'])
            ),
            height=500,
            template='plotly_white',
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        
        return fig
    
    def create_garch_chart(self, df: pd.DataFrame, garch_results: Dict, 
                          forecast: np.ndarray, title: str) -> go.Figure:
        """Create GARCH model results chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Conditional Volatility', 'Volatility Forecast')
        )
        
        # Conditional volatility
        if garch_results and 'cond_volatility' in garch_results:
            cond_vol = garch_results['cond_volatility'] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index[-len(cond_vol):],
                    y=cond_vol,
                    name='GARCH Volatility',
                    line=dict(color=self.color_palette['primary'], width=2)
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
                    mode='lines+markers'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=dict(
                text=f'{title} - GARCH Analysis',
                x=0.5,
                xanchor='center'
            ),
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Forecast Volatility (%)", row=2, col=1)
        
        return fig

# ============================================================================
# Main Application
# ============================================================================

class CommoditiesAnalyticsPlatform:
    """Main application class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analytics = AnalyticsEngine()
        self.viz = VisualizationEngine()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'asset_data' not in st.session_state:
            st.session_state.asset_data = {}
    
    def display_header(self):
        """Display professional header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; padding:0; font-size:2.2rem; font-weight:700;">Institutional Commodities Analytics Platform</h1>
            <p style="margin:0; padding-top:1rem; font-size:1.1rem; opacity:0.95;">
                Advanced GARCH Modeling • Risk Management • Portfolio Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        cols = st.columns(4)
        with cols[0]:
            st.metric("Platform Status", "🟢 Online", "Streamlit Cloud")
        with cols[1]:
            st.metric("Data Coverage", "20+ Years", "Daily Frequency")
        with cols[2]:
            st.metric("Commodities", "15+", "Global Markets")
        with cols[3]:
            st.metric("Analytics", "6 Modules", "Institutional Grade")
    
    def setup_sidebar(self):
        """Setup the sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Configuration")
            
            # Date Range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start Date", start_date)
            with col2:
                end = st.date_input("End Date", end_date)
            
            if start >= end:
                st.error("Start date must be before end date")
                return None, None, []
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Asset Selection
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Commodity Selection")
            
            selected_assets = []
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category}"):
                    for symbol, name in assets.items():
                        if st.checkbox(
                            f"{name} ({symbol})",
                            value=symbol in ["GC=F", "CL=F", "SI=F"],
                            key=f"asset_{symbol}"
                        ):
                            selected_assets.append(symbol)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis Settings
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 🔬 Analysis Settings")
            
            with st.expander("GARCH Parameters"):
                garch_p = st.slider("ARCH Order (p)", 1, 3, 1)
                garch_q = st.slider("GARCH Order (q)", 1, 3, 1)
                forecast_days = st.slider("Forecast Days", 5, 60, 30)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                load_button = st.button("📥 Load Data", type="primary", use_container_width=True)
            with col2:
                if st.session_state.data_loaded:
                    clear_button = st.button("🔄 Clear", type="secondary", use_container_width=True)
                    if clear_button:
                        st.cache_data.clear()
                        st.session_state.data_loaded = False
                        st.session_state.asset_data = {}
                        st.rerun()
            
            return start, end, selected_assets, load_button, garch_p, garch_q, forecast_days
    
    def load_data(self, start_date, end_date, selected_assets):
        """Load data for selected assets"""
        if not selected_assets:
            st.warning("Please select at least one commodity")
            return
        
        with st.spinner("Loading market data..."):
            progress_bar = st.progress(0, text="Initializing...")
            
            # Load asset data
            st.session_state.asset_data = self.data_manager.bulk_fetch(
                selected_assets, 
                start_date, 
                end_date
            )
            
            progress_bar.progress(100, text="Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            # Update session state
            st.session_state.data_loaded = True
            
            # Show success message
            if st.session_state.asset_data:
                loaded_count = len(st.session_state.asset_data)
                st.success(f"✅ Successfully loaded {loaded_count} assets")
            else:
                st.error("No data was loaded. Please check your selections.")
    
    def display_dashboard(self):
        """Display main analytics dashboard"""
        # Create tabs
        tabs = st.tabs([
            "📊 Overview",
            "📈 Price Analysis",
            "⚡ Volatility",
            "🔍 GARCH Model",
            "📉 Risk Metrics",
            "🔄 Correlation"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            self.display_overview()
        
        # Tab 2: Price Analysis
        with tabs[1]:
            self.display_price_analysis()
        
        # Tab 3: Volatility
        with tabs[2]:
            self.display_volatility_analysis()
        
        # Tab 4: GARCH Model
        with tabs[3]:
            self.display_garch_modeling()
        
        # Tab 5: Risk Metrics
        with tabs[4]:
            self.display_risk_metrics()
        
        # Tab 6: Correlation
        with tabs[5]:
            self.display_correlation_matrix()
    
    def display_overview(self):
        """Display overview dashboard"""
        st.subheader("Market Overview")
        
        if not st.session_state.asset_data:
            st.info("Please load data first")
            return
        
        # Create metrics grid
        cols = st.columns(min(3, len(st.session_state.asset_data)))
        
        for idx, (symbol, df) in enumerate(st.session_state.asset_data.items()):
            if idx < len(cols):
                with cols[idx]:
                    metrics = self.analytics.calculate_performance_metrics(df)
                    
                    if metrics:
                        change_color = "#28a745" if metrics['daily_change_pct'] >= 0 else "#dc3545"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{symbol}</div>
                            <div class="metric-value">${metrics['current_price']:,.2f}</div>
                            <div style="font-size:0.9rem; margin:0.5rem 0;">
                                <div>Change: <span style="color:{change_color}; font-weight:bold;">{metrics['daily_change_pct']:+.2f}%</span></div>
                                <div>Vol: {metrics['volatility_20d']:.1f}% | Return: {metrics['total_return']:+.1f}%</div>
                                <div>Sharpe: {metrics['sharpe_ratio']:.2f} | VaR: {metrics['var_95']:.1f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Market summary
        st.subheader("Market Summary")
        
        if st.session_state.asset_data:
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
    
    def display_price_analysis(self):
        """Display price analysis"""
        st.subheader("Price Analysis")
        
        if not st.session_state.asset_data:
            st.info("Please load data first")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset",
            list(st.session_state.asset_data.keys())
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None:
            st.error("No data available")
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
                    "Total Return",
                    f"{metrics['total_return']:+.1f}%",
                    f"{metrics['annual_return']:+.1f}% annual"
                )
            
            with col3:
                st.metric(
                    "Annual Volatility",
                    f"{metrics['annual_volatility']:.1f}%",
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}"
                )
            
            with col4:
                st.metric(
                    "Risk Metrics",
                    f"VaR: {metrics['var_95']:.1f}%",
                    f"Max DD: {metrics['max_drawdown']:.1f}%"
                )
        
        # Price chart
        asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
        fig = self.viz.create_price_chart(df, f"{asset_name} ({selected_asset})")
        st.plotly_chart(fig, use_container_width=True)
    
    def display_volatility_analysis(self):
        """Display volatility analysis"""
        st.subheader("Volatility Analysis")
        
        if not st.session_state.asset_data:
            st.info("Please load data first")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset",
            list(st.session_state.asset_data.keys()),
            key="vol_asset"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None:
            st.error("No data available")
            return
        
        # Volatility chart
        asset_name = COMMODITIES.get(selected_asset, {}).get('name', selected_asset)
        fig = self.viz.create_volatility_chart(df, f"{asset_name} ({selected_asset})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical volatility statistics
        if len(df) > 100:
            st.subheader("Historical Volatility Statistics")
            
            windows = [5, 10, 20, 50, 100]
            stats_data = []
            
            for window in windows:
                if len(df) >= window:
                    rolling_vol = df['Returns'].rolling(window=window).std() * np.sqrt(252) * 100
                    current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
                    avg_vol = rolling_vol.mean() if not rolling_vol.empty else 0
                    max_vol = rolling_vol.max() if not rolling_vol.empty else 0
                    
                    stats_data.append({
                        'Window': f"{window}D",
                        'Current': f"{current_vol:.1f}%",
                        'Average': f"{avg_vol:.1f}%",
                        'Maximum': f"{max_vol:.1f}%",
                        'Percentile': f"{(rolling_vol.rank(pct=True).iloc[-1] * 100):.0f}%"
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    def display_garch_modeling(self):
        """Display GARCH modeling"""
        st.subheader("GARCH Volatility Modeling")
        
        if not st.session_state.asset_data:
            st.info("Please load data first")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset",
            list(st.session_state.asset_data.keys()),
            key="garch_asset"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) < 200:
            st.warning(f"Insufficient data for GARCH modeling. Need at least 200 observations.")
            return
        
        # Get parameters from sidebar
        _, _, _, _, garch_p, garch_q, forecast_days = self.setup_sidebar()
        
        # Run GARCH analysis
        if st.button("Run GARCH Analysis", type="primary"):
            with st.spinner(f"Fitting GARCH({garch_p},{garch_q}) model..."):
                garch_results = self.analytics.fit_garch_model(
                    df['Returns'],
                    p=garch_p,
                    q=garch_q
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
                        status = "Stationary" if alpha_beta < 1 else "Non-stationary"
                        st.metric("α + β", f"{alpha_beta:.4f}", status)
                    
                    # Model diagnostics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("AIC", f"{garch_results['aic']:.1f}")
                        st.metric("BIC", f"{garch_results['bic']:.1f}")
                    
                    with col2:
                        st.metric("Log Likelihood", f"{garch_results['model'].loglikelihood:.1f}")
                    
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
                
                else:
                    st.error("❌ GARCH model failed to converge. Try different parameters.")
    
    def display_risk_metrics(self):
        """Display risk metrics"""
        st.subheader("Risk Metrics")
        
        if not st.session_state.asset_data:
            st.info("Please load data first")
            return
        
        # Asset selector
        selected_asset = st.selectbox(
            "Select Asset",
            list(st.session_state.asset_data.keys()),
            key="risk_asset"
        )
        
        if selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        
        if df is None or len(df) < 100:
            st.warning("Insufficient data for risk analysis")
            return
        
        # Calculate risk metrics
        returns = df['Returns'].dropna()
        risk_metrics = self.analytics.calculate_risk_metrics(returns)
        
        if not risk_metrics:
            st.error("Failed to calculate risk metrics")
            return
        
        # Display key risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2f}%")
            st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2f}%")
        
        with col2:
            st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2f}%")
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
        
        with col3:
            st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
            st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
        
        with col4:
            st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
            st.metric("VaR (99%)", f"{risk_metrics.get('var_99', 0):.2f}%")
        
        # Drawdown chart
        st.subheader("Historical Drawdowns")
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.2)',
            line=dict(color='#dc3545', width=2)
        ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_correlation_matrix(self):
        """Display correlation matrix"""
        st.subheader("Correlation Matrix")
        
        if not st.session_state.asset_data or len(st.session_state.asset_data) < 2:
            st.info("Please load at least 2 assets")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.analytics.calculate_correlation_matrix(st.session_state.asset_data)
        
        if corr_matrix.empty:
            st.warning("Insufficient data to calculate correlations")
            return
        
        # Display correlation matrix
        fig = self.viz.create_correlation_heatmap(
            corr_matrix,
            "Commodities Correlation Matrix"
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
                    'Correlation': correlation
                })
        
        if corr_values:
            corr_df = pd.DataFrame(corr_values)
            
            # Strongest positive correlations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strongest Positive**")
                top_positive = corr_df.nlargest(5, 'Correlation')
                st.dataframe(
                    top_positive.style.format({'Correlation': '{:.3f}'}),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Strongest Negative**")
                top_negative = corr_df.nsmallest(5, 'Correlation')
                st.dataframe(
                    top_negative.style.format({'Correlation': '{:.3f}'}),
                    use_container_width=True
                )
            
            # Average correlation
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            st.metric("Average Correlation", f"{avg_corr:.3f}")
    
    def display_footer(self):
        """Display footer"""
        st.markdown("""
        <div class="footer">
            <p>© 2024 Institutional Commodities Analytics | Data: Yahoo Finance | Version: 2.0.0</p>
            <p style="font-size:0.8rem; color:#6c757d;">
                For professional use only. Past performance is not indicative of future results.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# Main Application Entry
# ============================================================================

def main():
    """Main application"""
    
    # Initialize platform
    platform = CommoditiesAnalyticsPlatform()
    
    # Display header
    platform.display_header()
    
    # Setup sidebar
    config = platform.setup_sidebar()
    
    if config:
        start_date, end_date, selected_assets, load_button, garch_p, garch_q, forecast_days = config
        
        # Load data if button clicked
        if load_button:
            platform.load_data(start_date, end_date, selected_assets)
        
        # Display analytics if data is loaded
        if st.session_state.data_loaded:
            platform.display_dashboard()
        elif load_button and not selected_assets:
            st.warning("Please select at least one commodity")
        elif not st.session_state.data_loaded:
            st.info("Configure analysis and click 'Load Data' to begin")
    
    # Display footer
    platform.display_footer()

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please try refreshing the page or contact support.")
