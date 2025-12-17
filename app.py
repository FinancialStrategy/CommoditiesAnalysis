"""
Precious Metals & Commodities ARCH/GARCH Analysis Dashboard
Enhanced with QuantStats and Superior Visualization
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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Precious Metals Analytics",
    page_icon="📈",
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
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
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
    }
    
    .positive {
        color: #27ae60;
        font-weight: 600;
    }
    
    .negative {
        color: #e74c3c;
        font-weight: 600;
    }
    
    .neutral {
        color: #f39c12;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-weight: 600;
        color: #7f8c8d;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom: 3px solid #667eea;
        background-color: transparent;
    }
    
    /* Custom sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Enhanced tables */
    .dataframe {
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e3f2fd !important;
        transition: background-color 0.3s ease;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
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
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
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
    """Enhanced data management with caching"""
    
    @st.cache_data(ttl=3600)
    def fetch_data(_self, symbol, start_date, end_date):
        """Fetch data from Yahoo Finance with caching"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                return None
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Calculate technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

class AnalysisEngine:
    """Advanced analysis engine with QuantStats integration"""
    
    def calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics using QuantStats"""
        if returns is None or len(returns) < 100:
            return {}
        
        try:
            # Convert to pandas Series
            returns_series = pd.Series(returns, index=pd.date_range(end=datetime.now(), periods=len(returns), freq='D'))
            
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
            result = model.fit(disp='off', show_warning=False)
            
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

class Visualizer:
    """Enhanced visualization with superior design"""
    
    def create_price_chart(self, df, title, show_bb=True):
        """Create enhanced price chart with Bollinger Bands"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price with Indicators', 'Volume'),
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            shared_xaxes=True
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Price',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                      line=dict(color='#ff7f0e', width=1, dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                      line=dict(color='#2ca02c', width=1, dash='dash')),
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
        
        # Volume
        colors = ['#27ae60' if close >= open else '#e74c3c' 
                  for close, open in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=700,
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
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_returns_distribution(self, returns, title):
        """Create returns distribution chart"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot'),
            column_widths=[0.7, 0.3]
        )
        
        # Histogram with KDE
        fig.add_trace(
            go.Histogram(x=returns, name='Returns', nbinsx=50,
                        histnorm='probability density',
                        marker_color='#3498db', opacity=0.7),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x = np.linspace(min(returns), max(returns), 100)
        y = (1/(np.std(returns)*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - np.mean(returns))/np.std(returns))**2)
        
        fig.add_trace(
            go.Scatter(x=x, y=y, name='Normal Distribution',
                      line=dict(color='#e74c3c', width=2)),
            row=1, col=1
        )
        
        # Q-Q Plot
        from scipy import stats
        qq = stats.probplot(returns, dist="norm", fit=False)
        
        fig.add_trace(
            go.Scatter(x=qq[0], y=qq[1], mode='markers',
                      name='Q-Q Points', marker=dict(color='#2ecc71', size=5)),
            row=1, col=2
        )
        
        # Add 45-degree line
        min_val = min(min(qq[0]), min(qq[1]))
        max_val = max(max(qq[0]), max(qq[1]))
        
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='45° Line',
                      line=dict(color='#e74c3c', dash='dash', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=500,
            template="plotly_white",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
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
                          fill='tozeroy', fillcolor='rgba(211, 84, 0, 0.1)'),
                row=1, col=1
            )
            
            # Standardized residuals
            std_resid = garch_results["model"].resid / cond_vol
            
            fig.add_trace(
                go.Scatter(x=dates, y=std_resid, name='Std. Residuals',
                          mode='markers',
                          marker=dict(color='#3498db', size=3, opacity=0.5)),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=2, col=1)
            
            fig.update_layout(
                title=title,
                height=600,
                template="plotly_white",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            
            return fig
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")
            return None
    
    def create_performance_heatmap(self, metrics_dict):
        """Create performance metrics heatmap"""
        if not metrics_dict:
            return None
        
        # Prepare data
        assets = list(metrics_dict.keys())
        metrics = ['total_return', 'cagr', 'volatility', 'sharpe', 
                  'sortino', 'max_drawdown', 'calmar']
        
        data = []
        for asset in assets:
            row = []
            for metric in metrics:
                value = metrics_dict[asset].get(metric, 0)
                row.append(safe_float(value))
            data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=metrics,
            y=assets,
            colorscale='RdYlGn',
            zmid=0,
            text=[[format_percentage(val*100) if 'return' in metric or 'drawdown' in metric else 
                   format_number(val, 3) for val, metric in zip(row, metrics)] for row in data],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Performance Metrics Heatmap',
            height=400,
            template="plotly_white",
            xaxis_title="Metrics",
            yaxis_title="Assets"
        )
        
        return fig
    
    def create_correlation_matrix(self, returns_data):
        """Create correlation matrix heatmap"""
        if not returns_data:
            return None
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(returns_data)
            corr_matrix = df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
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
            
            return fig
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")
            return None

class Dashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analysis_engine = AnalysisEngine()
        self.visualizer = Visualizer()
        
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.8rem;">🏆 Precious Metals & Commodities Analytics</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Advanced ARCH/GARCH Analysis with QuantStats Performance Metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Analysis Settings")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Date range selection
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
            category = st.selectbox(
                "Select Commodity Category",
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
            st.markdown("### 📊 Analysis Parameters")
            
            show_bb = st.checkbox("Show Bollinger Bands", value=True)
            arch_lags = st.slider("ARCH Test Lags", 1, 20, 5)
            garch_p = st.slider("GARCH(p)", 1, 3, 1)
            garch_q = st.slider("GARCH(q)", 1, 3, 1)
            
            # Add refresh button
            st.markdown("---")
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                st.rerun()
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "category": category,
                "symbols": selected_symbols,
                "show_bb": show_bb,
                "arch_lags": arch_lags,
                "garch_p": garch_p,
                "garch_q": garch_q
            }
    
    def render_metrics_summary(self, metrics_dict):
        """Render metrics summary cards"""
        if not metrics_dict:
            return
        
        st.markdown("### 📈 Performance Summary")
        
        # Select first asset for summary
        first_asset = list(metrics_dict.keys())[0] if metrics_dict else None
        if not first_asset:
            return
        
        metrics = metrics_dict[first_asset]
        
        cols = st.columns(4)
        with cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Return</div>', unsafe_allow_html=True)
            value = safe_float(metrics.get('total_return', 0)) * 100
            color_class = "positive" if value > 0 else "negative"
            st.markdown(f'<div class="metric-value {color_class}">{format_percentage(value)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe Ratio</div>', unsafe_allow_html=True)
            value = safe_float(metrics.get('sharpe', 0))
            color_class = "positive" if value > 1 else "warning" if value > 0 else "negative"
            st.markdown(f'<div class="metric-value {color_class}">{format_number(value, 3)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Max Drawdown</div>', unsafe_allow_html=True)
            value = safe_float(metrics.get('max_drawdown', 0)) * 100
            color_class = "negative"
            st.markdown(f'<div class="metric-value {color_class}">{format_percentage(value)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
            value = safe_float(metrics.get('volatility', 0)) * 100
            color_class = "warning" if value > 20 else "positive"
            st.markdown(f'<div class="metric-value {color_class}">{format_percentage(value, 1)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
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
                else:
                    row[key] = format_number(safe_float(value), 4)
            metrics_list.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(metrics_list)
        
        # Apply styling
        def color_negative_red(val):
            try:
                if '%' in str(val):
                    num_val = float(str(val).replace('%', '').replace('+', ''))
                    if num_val < 0:
                        return 'color: #e74c3c; font-weight: bold'
                    elif num_val > 0:
                        return 'color: #27ae60; font-weight: bold'
            except:
                pass
            return ''
        
        st.dataframe(
            df.style.applymap(color_negative_red),
            use_container_width=True,
            height=400
        )
    
    def run(self):
        """Main dashboard execution"""
        
        # Render header
        self.render_header()
        
        # Render sidebar and get parameters
        params = self.render_sidebar()
        
        if not params["symbols"]:
            st.warning("⚠️ Please select at least one commodity to analyze.")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch and analyze data
        all_data = {}
        all_metrics = {}
        all_garch_results = {}
        all_returns = {}
        
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
            else:
                all_garch_results[symbol] = {
                    "garch": None,
                    "arch_test": arch_test
                }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        if not all_data:
            st.error("❌ No data available for analysis. Please try different commodities or date range.")
            return
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price Analysis", 
            "📊 Performance", 
            "🔍 Volatility Models", 
            "📋 Comparison", 
            "📄 Reports"
        ])
        
        with tab1:
            self.render_price_analysis_tab(all_data, params)
        
        with tab2:
            self.render_performance_tab(all_metrics, all_returns)
        
        with tab3:
            self.render_volatility_tab(all_data, all_garch_results, params)
        
        with tab4:
            self.render_comparison_tab(all_data, all_metrics, all_returns)
        
        with tab5:
            self.render_reports_tab(all_metrics, all_garch_results)
    
    def render_price_analysis_tab(self, all_data, params):
        """Render price analysis tab"""
        st.markdown("### 📈 Price Charts & Technical Analysis")
        
        # Create columns for charts
        for symbol, df in all_data.items():
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            
            st.markdown(f"#### {commodity_name}")
            
            # Price chart
            fig = self.visualizer.create_price_chart(
                df, 
                f"{commodity_name} - Price Analysis",
                show_bb=params["show_bb"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_price = df['Close'].iloc[-1]
                price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                st.metric("Current Price", format_currency(current_price), 
                         delta=f"{price_change:.2f}%")
            
            with col2:
                avg_volume = df['Volume'].mean()
                st.metric("Avg Daily Volume", f"{avg_volume:,.0f}")
            
            with col3:
                volatility = df['Returns'].std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.2f}%")
            
            with col4:
                sma_ratio = (df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1) * 100
                st.metric("vs SMA 50", f"{sma_ratio:.2f}%")
            
            st.markdown("---")
    
    def render_performance_tab(self, all_metrics, all_returns):
        """Render performance metrics tab"""
        st.markdown("### 📊 Performance Analytics")
        
        # Summary metrics
        self.render_metrics_summary(all_metrics)
        
        # Detailed table
        self.render_performance_table(all_metrics)
        
        # Returns distribution for selected asset
        if all_returns:
            selected_asset = st.selectbox(
                "Select asset for detailed returns analysis",
                list(all_returns.keys()),
                format_func=lambda x: COMMODITIES[[cat for cat in COMMODITIES if x in COMMODITIES[cat]][0]][x]["name"]
            )
            
            if selected_asset in all_returns:
                returns = all_returns[selected_asset]
                asset_name = COMMODITIES[[cat for cat in COMMODITIES if selected_asset in COMMODITIES[cat]][0]][selected_asset]["name"]
                
                fig = self.visualizer.create_returns_distribution(
                    returns, 
                    f"{asset_name} - Returns Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_volatility_tab(self, all_data, all_garch_results, params):
        """Render volatility modeling tab"""
        st.markdown("### 🔍 Volatility Modeling (ARCH/GARCH)")
        
        for symbol, results in all_garch_results.items():
            commodity_name = COMMODITIES[params["category"]][symbol]["name"]
            
            st.markdown(f"#### {commodity_name}")
            
            cols = st.columns(3)
            
            with cols[0]:
                arch_present = results["arch_test"]["present"]
                arch_p_value = results["arch_test"]["p_value"]
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">ARCH Effects</div>', unsafe_allow_html=True)
                status_class = "status-success" if arch_present else "status-danger"
                status_text = "Present" if arch_present else "Not Present"
                st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)
                st.markdown(f"**p-value:** {arch_p_value:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                if results["garch"]:
                    garch_aic = results["garch"]["aic"]
                    garch_converged = results["garch"]["converged"]
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">GARCH Model</div>', unsafe_allow_html=True)
                    status_class = "status-success" if garch_converged else "status-danger"
                    status_text = "Converged" if garch_converged else "Failed"
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
                    st.markdown(f"**ω:** {omega:.6f}")
                    st.markdown(f"**α:** {alpha:.4f}")
                    st.markdown(f"**β:** {beta:.4f}")
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
            
            st.markdown("---")
    
    def render_comparison_tab(self, all_data, all_metrics, all_returns):
        """Render comparison tab"""
        st.markdown("### 📋 Multi-Asset Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance heatmap
            if all_metrics:
                fig = self.visualizer.create_performance_heatmap(all_metrics)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation matrix
            if len(all_returns) > 1:
                # Convert returns to same length
                min_length = min(len(r) for r in all_returns.values())
                aligned_returns = {k: v[:min_length] for k, v in all_returns.items()}
                
                fig = self.visualizer.create_correlation_matrix(aligned_returns)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative returns comparison
        st.markdown("#### Cumulative Returns Comparison")
        
        if all_data:
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
                        mode='lines'
                    )
                )
            
            fig.update_layout(
                title="Cumulative Returns Over Time",
                height=500,
                template="plotly_white",
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_reports_tab(self, all_metrics, all_garch_results):
        """Render reports tab"""
        st.markdown("### 📄 Analysis Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Performance Summary")
            
            # Create summary DataFrame
            summary_data = []
            for symbol, metrics in all_metrics.items():
                summary_data.append({
                    "Asset": symbol,
                    "Total Return": format_percentage(safe_float(metrics.get('total_return', 0)) * 100),
                    "CAGR": format_percentage(safe_float(metrics.get('cagr', 0)) * 100),
                    "Sharpe Ratio": format_number(safe_float(metrics.get('sharpe', 0)), 3),
                    "Max DD": format_percentage(safe_float(metrics.get('max_drawdown', 0)) * 100),
                    "Volatility": format_percentage(safe_float(metrics.get('volatility', 0)) * 100, 1)
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button for summary
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=csv,
                file_name="commodities_summary.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("#### 📈 Volatility Models Summary")
            
            # GARCH models summary
            garch_summary = []
            for symbol, results in all_garch_results.items():
                arch_present = results["arch_test"]["present"]
                garch_info = results["garch"]
                
                row = {
                    "Asset": symbol,
                    "ARCH Effects": "Yes" if arch_present else "No",
                    "GARCH Fitted": "Yes" if garch_info else "No"
                }
                
                if garch_info:
                    row["AIC"] = format_number(garch_info.get("aic", 0), 2)
                    row["Converged"] = "Yes" if garch_info.get("converged", False) else "No"
                
                garch_summary.append(row)
            
            garch_df = pd.DataFrame(garch_summary)
            st.dataframe(garch_df, use_container_width=True)
        
        # Generate QuantStats report
        st.markdown("#### 📊 QuantStats Full Report")
        
        if st.button("Generate Full QuantStats Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                # Select first asset for detailed report
                if all_metrics:
                    first_symbol = list(all_metrics.keys())[0]
                    returns = pd.Series(all_returns[first_symbol])
                    
                    # Create QuantStats HTML report
                    try:
                        qs.reports.html(
                            returns,
                            output='quantstats_report.html',
                            title=f"{first_symbol} QuantStats Report"
                        )
                        
                        # Read and display
                        with open('quantstats_report.html', 'r') as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=800, scrolling=True)
                        
                        # Download button for report
                        with open('quantstats_report.html', 'rb') as f:
                            report_data = f.read()
                        
                        st.download_button(
                            label="📥 Download Full Report (HTML)",
                            data=report_data,
                            file_name=f"{first_symbol}_quantstats_report.html",
                            mime="text/html"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

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
