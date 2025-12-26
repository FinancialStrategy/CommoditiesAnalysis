"""
🏛️ Institutional Commodities Analytics Platform v6.1
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • Machine Learning • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance
"""

import os
import math
import warnings
import textwrap
import json
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# yfinance download compatibility helper (Streamlit Cloud safe)
# -----------------------------------------------------------------------------
def yf_download_safe(params: Dict[str, Any]) -> pd.DataFrame:
    """Call yfinance.download with fallbacks for version/arg compatibility."""
    try:
        return yf.download(**params)
    except TypeError:
        # Some yfinance versions don't accept these args
        p = dict(params)
        p.pop("threads", None)
        p.pop("timeout", None)
        # Backward compatibility: if someone accidentally uses 'symbol'
        if "tickers" not in p and "symbol" in p:
            p["tickers"] = p.pop("symbol")
        return yf.download(**p)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, signal
# Optional dependency (used only for some diagnostic plots)
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None
from io import BytesIO, StringIO
import base64

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Environment optimization
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(
    page_title="Institutional Commodities Platform v6.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/institutional-commodities',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v6.0
                    Advanced analytics platform for institutional commodity trading
                    © 2024 Institutional Trading Analytics"""
    }
)

# =============================================================================
# DATA STRUCTURES & CONFIGURATION
# =============================================================================

class AssetCategory(Enum):
    """Asset categories for classification"""
    PRECIOUS_METALS = "Precious Metals"
    INDUSTRIAL_METALS = "Industrial Metals"
    ENERGY = "Energy"
    AGRICULTURE = "Agriculture"
    BENCHMARK = "Benchmark"

@dataclass
class AssetMetadata:
    """Enhanced metadata for assets"""
    symbol: str
    name: str
    category: AssetCategory
    color: str
    description: str = ""
    exchange: str = "CME"
    contract_size: str = "Standard"
    margin_requirement: float = 0.05
    tick_size: float = 0.01
    enabled: bool = True
    risk_level: str = "Medium"  # Low, Medium, High
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisConfiguration:
    """Comprehensive analysis configuration"""
    start_date: datetime= field(default_factory=lambda: (datetime.now() - timedelta(days=1095)))
    end_date: datetime= field(default_factory=lambda: datetime.now())
    risk_free_rate: float = 0.02
    annual_trading_days: int = 252
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    garch_p_range: Tuple[int, int] = (1, 3)
    garch_q_range: Tuple[int, int] = (1, 3)
    regime_states: int = 3
    backtest_window: int = 250
    rolling_window: int = 60
    volatility_window: int = 20
    monte_carlo_simulations: int = 10000
    optimization_method: str = "sharpe"  # sharpe, min_var, max_ret
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.start_date >= self.end_date:
            return False
        if not (0 <= self.risk_free_rate <= 1):
            return False
        if not all(0.5 <= cl <= 0.999 for cl in self.confidence_levels):
            return False
        return True

# Enhanced commodities universe with comprehensive metadata
COMMODITIES_UNIVERSE = {
    AssetCategory.PRECIOUS_METALS.value: {
        "GC=F": AssetMetadata(
            symbol="GC=F",
            name="Gold Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#FFD700",
            description="COMEX Gold Futures (100 troy ounces)",
            exchange="COMEX",
            contract_size="100 troy oz",
            margin_requirement=0.045,
            tick_size=0.10,
            risk_level="Low"
        ),
        "SI=F": AssetMetadata(
            symbol="SI=F",
            name="Silver Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#C0C0C0",
            description="COMEX Silver Futures (5,000 troy ounces)",
            exchange="COMEX",
            contract_size="5,000 troy oz",
            margin_requirement=0.065,
            tick_size=0.005,
            risk_level="Medium"
        ),
        "PL=F": AssetMetadata(
            symbol="PL=F",
            name="Platinum Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#E5E4E2",
            description="NYMEX Platinum Futures (50 troy ounces)",
            exchange="NYMEX",
            contract_size="50 troy oz",
            margin_requirement=0.075,
            tick_size=0.10,
            risk_level="High"
        ),
    },
    AssetCategory.INDUSTRIAL_METALS.value: {
        "HG=F": AssetMetadata(
            symbol="HG=F",
            name="Copper Futures",
            category=AssetCategory.INDUSTRIAL_METALS,
            color="#B87333",
            description="COMEX Copper Futures (25,000 pounds)",
            exchange="COMEX",
            contract_size="25,000 lbs",
            margin_requirement=0.085,
            tick_size=0.0005,
            risk_level="Medium"
        ),
        "ALI=F": AssetMetadata(
            symbol="ALI=F",
            name="Aluminum Futures",
            category=AssetCategory.INDUSTRIAL_METALS,
            color="#848482",
            description="COMEX Aluminum Futures (44,000 pounds)",
            exchange="COMEX",
            contract_size="44,000 lbs",
            margin_requirement=0.095,
            tick_size=0.0001,
            risk_level="High"
        ),
    },
    AssetCategory.ENERGY.value: {
        "CL=F": AssetMetadata(
            symbol="CL=F",
            name="Crude Oil WTI",
            category=AssetCategory.ENERGY,
            color="#000000",
            description="NYMEX Light Sweet Crude Oil (1,000 barrels)",
            exchange="NYMEX",
            contract_size="1,000 barrels",
            margin_requirement=0.085,
            tick_size=0.01,
            risk_level="High"
        ),
        "NG=F": AssetMetadata(
            symbol="NG=F",
            name="Natural Gas",
            category=AssetCategory.ENERGY,
            color="#4169E1",
            description="NYMEX Natural Gas (10,000 MMBtu)",
            exchange="NYMEX",
            contract_size="10,000 MMBtu",
            margin_requirement=0.095,
            tick_size=0.001,
            risk_level="High"
        ),
    },
    AssetCategory.AGRICULTURE.value: {
        "ZC=F": AssetMetadata(
            symbol="ZC=F",
            name="Corn Futures",
            category=AssetCategory.AGRICULTURE,
            color="#FFD700",
            description="CBOT Corn Futures (5,000 bushels)",
            exchange="CBOT",
            contract_size="5,000 bushels",
            margin_requirement=0.065,
            tick_size=0.0025,
            risk_level="Medium"
        ),
        "ZW=F": AssetMetadata(
            symbol="ZW=F",
            name="Wheat Futures",
            category=AssetCategory.AGRICULTURE,
            color="#F5DEB3",
            description="CBOT Wheat Futures (5,000 bushels)",
            exchange="CBOT",
            contract_size="5,000 bushels",
            margin_requirement=0.075,
            tick_size=0.0025,
            risk_level="Medium"
        ),
    }
}

BENCHMARKS = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "type": "equity",
        "color": "#1E90FF",
        "description": "S&P 500 Equity Index"
    },
    "DX-Y.NYB": {
        "name": "US Dollar Index",
        "type": "currency",
        "color": "#32CD32",
        "description": "US Dollar Currency Index"
    },
    "TLT": {
        "name": "20+ Year Treasury ETF",
        "type": "fixed_income",
        "color": "#8A2BE2",
        "description": "Long-term US Treasury Bonds"
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "type": "commodity",
        "color": "#FFD700",
        "description": "Gold-backed ETF"
    },
    "DBC": {
        "name": "Invesco DB Commodity Index",
        "type": "commodity",
        "color": "#FF6347",
        "description": "Broad Commodities ETF"
    }
}

# =============================================================================
# ADVANCED STYLES & THEMING
# =============================================================================

class ThemeManager:
    """Manage application theming and styling"""
    
    THEMES = {
        "default": {
            "primary": "#1a2980",
            "secondary": "#26d0ce",
            "accent": "#7c3aed",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "dark": "#1f2937",
            "light": "#f3f4f6",
            "gray": "#6b7280",
            "background": "#ffffff"
        },
        "dark": {
            "primary": "#3b82f6",
            "secondary": "#06b6d4",
            "accent": "#8b5cf6",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "dark": "#111827",
            "light": "#374151",
            "gray": "#9ca3af",
            "background": "#1f2937"
        }
    }
    
    @staticmethod
    def get_styles(theme: str = "default") -> str:
        """Get CSS styles for selected theme"""
        colors = ThemeManager.THEMES.get(theme, ThemeManager.THEMES["default"])
        
        return f"""
        <style>
            :root {{
                --primary: {colors['primary']};
                --secondary: {colors['secondary']};
                --accent: {colors['accent']};
                --success: {colors['success']};
                --warning: {colors['warning']};
                --danger: {colors['danger']};
                --dark: {colors['dark']};
                --light: {colors['light']};
                --gray: {colors['gray']};
                --background: {colors['background']};
                --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                --shadow-md: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
                --shadow-lg: 0 10px 25px rgba(0,0,0,0.15), 0 5px 10px rgba(0,0,0,0.05);
                --shadow-xl: 0 20px 40px rgba(0,0,0,0.2), 0 10px 20px rgba(0,0,0,0.1);
                --radius-sm: 6px;
                --radius-md: 10px;
                --radius-lg: 16px;
                --radius-xl: 24px;
                --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            /* Main Header */
            .main-header {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                padding: 2.5rem;
                border-radius: var(--radius-lg);
                color: white;
                margin-bottom: 2rem;
                box-shadow: var(--shadow-xl);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .main-header::before {{
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                background-size: 30px 30px;
                opacity: 0.4;
                animation: float 25s linear infinite;
            }}
            
            @keyframes float {{
                0% {{ transform: translate(0, 0) rotate(0deg); }}
                100% {{ transform: translate(-30px, -30px) rotate(360deg); }}
            }}
            
            /* Cards */
            .metric-card {{
                background: var(--background);
                padding: 1.75rem;
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-md);
                border-left: 5px solid var(--primary);
                margin-bottom: 1.5rem;
                transition: var(--transition);
                border: 1px solid rgba(0,0,0,0.05);
            }}
            
            .metric-card:hover {{
                transform: translateY(-8px);
                box-shadow: var(--shadow-lg);
                border-color: var(--primary);
            }}
            
            .metric-card.glow {{
                animation: pulse-glow 2s infinite;
            }}
            
            @keyframes pulse-glow {{
                0%, 100% {{ box-shadow: 0 0 20px rgba(26, 41, 128, 0.2); }}
                50% {{ box-shadow: 0 0 40px rgba(26, 41, 128, 0.4); }}
            }}
            
            .metric-value {{
                font-size: 2.4rem;
                font-weight: 800;
                color: var(--dark);
                margin: 0.75rem 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .metric-label {{
                font-size: 0.85rem;
                color: var(--gray);
                text-transform: uppercase;
                letter-spacing: 1.2px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            /* Badges */
            .status-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1.25rem;
                border-radius: 50px;
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                transition: var(--transition);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .status-success {{
                background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
                color: white;
            }}
            
            .status-warning {{
                background: linear-gradient(135deg, var(--warning) 0%, #d97706 100%);
                color: white;
            }}
            
            .status-danger {{
                background: linear-gradient(135deg, var(--danger) 0%, #dc2626 100%);
                color: white;
            }}
            
            .status-info {{
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
            }}
            
            .status-badge:hover {{
                transform: scale(1.05);
                box-shadow: var(--shadow-md);
            }}
            
            /* Sidebar */
            .sidebar-section {{
                background: var(--light);
                padding: 1.75rem;
                border-radius: var(--radius-md);
                margin-bottom: 1.5rem;
                border-left: 4px solid var(--primary);
                transition: var(--transition);
                box-shadow: var(--shadow-sm);
            }}
            
            .sidebar-section:hover {{
                background: var(--background);
                box-shadow: var(--shadow-md);
                transform: translateX(5px);
            }}
            
            /* Tabs Enhancement */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 12px;
                background-color: var(--light);
                padding: 12px;
                border-radius: var(--radius-lg);
                margin-bottom: 2rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                border-radius: var(--radius-md);
                padding: 12px 24px;
                background-color: var(--background);
                border: 2px solid transparent;
                transition: var(--transition);
                font-weight: 600;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border-color: var(--primary);
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }}
            
            /* Dataframe Styling */
            .dataframe {{
                border-radius: var(--radius-md);
                overflow: hidden;
                border: 1px solid var(--light);
                box-shadow: var(--shadow-sm);
            }}
            
            .dataframe thead {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
            }}
            
            /* Loading Animations */
            @keyframes shimmer {{
                0% {{ background-position: -200px 0; }}
                100% {{ background-position: calc(200px + 100%) 0; }}
            }}
            
            .shimmer {{
                background: linear-gradient(90deg, var(--light) 0%, var(--background) 50%, var(--light) 100%);
                background-size: 200px 100%;
                animation: shimmer 1.5s infinite;
            }}
            
            /* Progress Bars */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, var(--primary), var(--secondary));
            }}
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: var(--light);
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(135deg, var(--secondary), var(--primary));
            }}
            
            /* Tooltips */
            .custom-tooltip {{
                position: relative;
                display: inline-block;
                cursor: help;
            }}
            
            .custom-tooltip:hover::after {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                background: var(--dark);
                color: white;
                padding: 0.75rem 1rem;
                border-radius: var(--radius-sm);
                font-size: 0.85rem;
                white-space: nowrap;
                z-index: 1000;
                box-shadow: var(--shadow-lg);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                opacity: 0;
                animation: fadeIn 0.3s forwards;
            }}
            
            @keyframes fadeIn {{
                to {{ opacity: 1; }}
            }}
            
            /* Section Headers */
            .section-header {{
                display: flex;
                align-items: center;
                gap: 1rem;
                margin: 2rem 0 1.5rem;
                padding-bottom: 0.75rem;
                border-bottom: 2px solid var(--primary);
            }}
            
            .section-header h2 {{
                margin: 0;
                color: var(--dark);
                font-size: 1.5rem;
                font-weight: 700;
            }}
            
            /* Grid Layout */
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .metric-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .main-header {{
                    padding: 1.5rem;
                }}
                
                .metric-value {{
                    font-size: 2rem;
                }}
            }}
        </style>
        """

# Apply default theme
st.markdown(ThemeManager.get_styles("default"), unsafe_allow_html=True)

# =============================================================================
# IMPORT MANAGEMENT & DEPENDENCY HANDLING
# =============================================================================

class DependencyManager:
    """Manage optional dependencies with graceful fallbacks"""
    
    def __init__(self):
        self.dependencies = {}
        self._load_dependencies()
    
    def _load_dependencies(self):
        """Load optional dependencies"""
        # statsmodels
        try:
            from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
            import statsmodels.api as sm
            from statsmodels.regression.rolling import RollingOLS
            self.dependencies['statsmodels'] = {
                'available': True,
                'module': sm,
                'het_arch': het_arch,
                'acorr_ljungbox': acorr_ljungbox,
                'RollingOLS': RollingOLS
            }
        except ImportError:
            self.dependencies['statsmodels'] = {'available': False}
            if st.session_state.get('show_system_diagnostics', False):
                st.warning("⚠️ statsmodels not available - some features disabled")
        # arch
        try:
            from arch import arch_model
            self.dependencies['arch'] = {
                'available': True,
                'arch_model': arch_model
            }
        except ImportError:
            self.dependencies['arch'] = {'available': False}
            if st.session_state.get('show_system_diagnostics', False):
                st.warning("⚠️ arch not available - GARCH features disabled")
        # hmmlearn & sklearn
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            self.dependencies['hmmlearn'] = {
                'available': True,
                'GaussianHMM': GaussianHMM,
                'StandardScaler': StandardScaler,
                'KMeans': KMeans,
                'PCA': PCA
            }
        except ImportError:
            self.dependencies['hmmlearn'] = {'available': False}
            st.info("ℹ️ hmmlearn/scikit-learn not available - regime detection disabled")
        
        # quantstats
        try:
            import quantstats as qs
            self.dependencies['quantstats'] = {
                'available': True,
                'module': qs
            }
        except ImportError:
            self.dependencies['quantstats'] = {'available': False}
        
        # ta (technical analysis)
        try:
            import ta
            self.dependencies['ta'] = {
                'available': True,
                'module': ta
            }
        except ImportError:
            self.dependencies['ta'] = {'available': False}
    
    def is_available(self, dependency: str) -> bool:
        """Check if dependency is available"""
        return self.dependencies.get(dependency, {}).get('available', False)
    
    def get_module(self, dependency: str):
        """Get dependency module if available"""
        dep = self.dependencies.get(dependency, {})
        return dep.get('module') if dep.get('available') else None

# Initialize dependency manager
dep_manager = DependencyManager()

# =============================================================================
# ADVANCED CACHING SYSTEM
# =============================================================================

class SmartCache:
    """Advanced caching with memory management, TTL, and persistence"""
    
    def __init__(self, max_entries: int = 100, ttl_hours: int = 24):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_hours * 3600
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, (datetime, pd.Timestamp)):
                key_parts.append(arg.isoformat())
            elif isinstance(arg, pd.DataFrame):
                # Create hash from DataFrame content
                content_hash = hashlib.md5(
                    pd.util.hash_pandas_object(arg).values.tobytes()
                ).hexdigest()
                key_parts.append(content_hash)
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    @staticmethod
    def cache_data(ttl: int = 3600, max_entries: int = 50):
        """Decorator for caching data with TTL"""
        def decorator(func):
            @wraps(func)
            @st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=False)
            def wrapper(_arg0, *args, **kwargs):
                try:
                    return func(_arg0, *args, **kwargs)
                except Exception as e:
                    st.warning(f"Cache miss for {func.__name__}: {str(e)[:100]}")
                    # Clear cache for this function on error
                    st.cache_data.clear()
                    return func(_arg0, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def cache_resource(max_entries: int = 20):
        """Decorator for caching resources"""
        def decorator(func):
            @wraps(func)
            @st.cache_resource(max_entries=max_entries)
            def wrapper(_arg0, *args, **kwargs):
                return func(_arg0, *args, **kwargs)
            return wrapper
        return decorator

# =============================================================================
# ENHANCED DATA MANAGER
# =============================================================================

class EnhancedDataManager:
    """Advanced data management with intelligent fetching and preprocessing"""
    
    def __init__(self):
        self.cache = SmartCache()
    
    @SmartCache.cache_data(ttl=7200, max_entries=100)
    def fetch_asset_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        retries: int = 3
    ) -> pd.DataFrame:
        """Fetch and preprocess asset data with intelligent retry logic"""
        cache_key = self.cache.generate_key(
            "fetch_asset", symbol, start_date, end_date, interval
        )
        
        for attempt in range(retries):
            try:
                # Configure yfinance download
                download_params = {
                    'tickers': symbol,
                    'start': start_date,
                    'end': end_date,
                    'interval': interval,
                    'progress': False,
                    'auto_adjust': True,
                    'threads': True,
                    'timeout': 30
                }
                
                # Try different download strategies
                if attempt == 0:
                    # First attempt: standard download
                    df = yf_download_safe(download_params)
                elif attempt == 1:
                    # Second attempt: force direct download
                    download_params['auto_adjust'] = False
                    df = yf_download_safe(download_params)
                else:
                    # Third attempt: try with different parameters
                    download_params['interval'] = "1d"
                    download_params['period'] = "max"
                    df = yf_download_safe(download_params)
                    # Filter by date
                    df = df[df.index >= pd.Timestamp(start_date)]
                    df = df[df.index <= pd.Timestamp(end_date)]
                
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Clean and validate data
                df = self._clean_dataframe(df, symbol)
                
                if len(df) < 20:  # Minimum data points
                    raise ValueError(f"Insufficient data for {symbol}")
                
                return df
                
            except Exception as e:
                if attempt == retries - 1:
                    st.warning(f"Failed to fetch {symbol} after {retries} attempts: {str(e)[:150]}")
                    return pd.DataFrame()
                continue
        
        return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate dataframe"""
        df = df.copy()
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        
        # Map columns
        col_mapping = {}
        for col in required_cols:
            if col not in df.columns:
                # Try to find similar columns
                for actual_col in df.columns:
                    if col.lower() in actual_col.lower():
                        col_mapping[col] = actual_col
                        break
        
        # Create missing columns
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        if 'Close' not in df.columns:
            if 'Adj_Close' in df.columns:
                df['Close'] = df['Adj_Close']
            elif len(df.columns) > 0:
                df['Close'] = df.iloc[:, -1]
            else:
                return pd.DataFrame()
        
        # Fill missing OHLC data
        for col in ['Open', 'High', 'Low']:
            if col not in df.columns:
                df[col] = df['Close']
        
        # Ensure Adj_Close exists (yfinance auto_adjust may remove it)
        
        if 'Adj_Close' not in df.columns:
        
            df['Adj_Close'] = df['Close']

        
        if 'Volume' not in df.columns:
            df['Volume'] = 0.0
        
        # Clean index
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Remove rows with NaN in critical columns
        critical_cols = ['Close', 'Adj_Close']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        return df
    
    @SmartCache.cache_data(ttl=3600, max_entries=50)
    def fetch_multiple_assets(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """Parallel fetch of multiple assets"""
        results = {}
        failed_symbols = []
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            # Create futures
            future_to_symbol = {}
            for symbol in symbols:
                future = executor.submit(
                    self.fetch_asset_data,
                    symbol,
                    start_date,
                    end_date
                )
                future_to_symbol[future] = symbol
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    failed_symbols.append(symbol)
                    continue
        
        # Log failures
        if failed_symbols:
            st.info(f"Failed to load {len(failed_symbols)} symbols: {', '.join(failed_symbols[:5])}")
        
        return results
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical features"""
        df = df.copy()
        
        # Ensure Adj Close exists
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        
        # Returns
        df['Returns'] = df[price_col].pct_change()
        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Price statistics
        df['Price_Range'] = (df['High'] - df['Low']) / df[price_col]
        df['Price_Change'] = df[price_col].diff()
        
        # Moving averages
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            df[f'SMA_{period}'] = df[price_col].rolling(window=period).mean()
            df[f'EMA_{period}'] = df[price_col].ewm(span=period).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_middle = df[price_col].rolling(window=bb_period).mean()
        bb_std = df[price_col].rolling(window=bb_period).std()
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_middle
        df['BB_Position'] = (df[price_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df[price_col].ewm(span=12).mean()
        ema26 = df[price_col].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volatility measures
        df['Volatility_20D'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_60D'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
        df['Realized_Vol'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Volume_Adjusted'] = df['Volume'] * df[price_col]
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df[price_col].shift())
        low_close = np.abs(df['Low'] - df[price_col].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR'] / df[price_col] * 100
        
        # Momentum indicators
        df['Momentum_10D'] = df[price_col].pct_change(periods=10)
        df['Momentum_20D'] = df[price_col].pct_change(periods=20)
        
        # Rate of Change
        df['ROC_10'] = ((df[price_col] - df[price_col].shift(10)) / df[price_col].shift(10)) * 100
        df['ROC_20'] = ((df[price_col] - df[price_col].shift(20)) / df[price_col].shift(20)) * 100
        
        # Williams %R
        period = 14
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        df['Williams_%R'] = ((highest_high - df[price_col]) / (highest_high - lowest_low)) * -100
        
        # Stochastic Oscillator
        df['Stochastic_%K'] = ((df[price_col] - lowest_low) / (highest_high - lowest_low)) * 100
        df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df[price_col]) / 3
        cci_sma = typical_price.rolling(window=20).mean()
        cci_mean_dev = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        df['CCI'] = (typical_price - cci_sma) / (0.015 * cci_mean_dev)
        
        # On Balance Volume
        if 'Volume' in df.columns:
            df['OBV'] = (np.sign(df['Returns'].fillna(0)) * df['Volume']).cumsum()
        
        # Price trends
        df['Trend_Strength'] = df['Returns'].rolling(window=20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # Drop NaN values from feature calculations
        df = df.dropna(subset=['Returns', 'Volatility_20D'])
        
        return df

# =============================================================================
# ADVANCED ANALYTICS ENGINE
# =============================================================================

class InstitutionalAnalytics:
    """Institutional-grade analytics engine with advanced methods"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.annual_trading_days = 252


    # =========================================================================
    # NUMERICAL STABILITY HELPERS (Higham-style PSD / correlation repairs)
    # =========================================================================

    @staticmethod
    def _symmetrize(a: np.ndarray) -> np.ndarray:
        """Force symmetry (numerical hygiene)."""
        a = np.asarray(a, dtype=float)
        return 0.5 * (a + a.T)

    @staticmethod
    def _project_psd(a: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """Projection onto the PSD cone via eigenvalue clipping."""
        a = InstitutionalAnalytics._symmetrize(a)
        vals, vecs = np.linalg.eigh(a)
        vals = np.clip(vals, epsilon, None)
        return InstitutionalAnalytics._symmetrize((vecs * vals) @ vecs.T)

    def _higham_nearest_correlation(
        self,
        corr: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-7,
        epsilon: float = 1e-12,
    ) -> np.ndarray:
        """Higham (2002)-style alternating projections to the nearest correlation matrix.

        This is a defensive routine to prevent hard crashes in downstream routines
        (optimization / Cholesky) when a correlation estimate becomes indefinite
        due to missing data, rounding, or numerical noise.
        """
        a = self._symmetrize(np.asarray(corr, dtype=float))
        # Ensure diagonal starts at 1
        np.fill_diagonal(a, 1.0)

        y = a.copy()
        delta_s = np.zeros_like(y)

        # Frobenius norm scale (avoid divide by 0)
        base = np.linalg.norm(a, ord="fro")
        if not np.isfinite(base) or base <= 0:
            base = 1.0

        for _ in range(int(max_iter)):
            r = y - delta_s
            x = self._project_psd(r, epsilon=epsilon)
            delta_s = x - r

            y = x.copy()
            np.fill_diagonal(y, 1.0)
            y = self._symmetrize(y)

            rel = np.linalg.norm(y - x, ord="fro") / base
            if rel < float(tol):
                break

        # Final PSD polish (rare edge cases)
        y = self._project_psd(y, epsilon=epsilon)
        np.fill_diagonal(y, 1.0)
        return self._symmetrize(y)

    def _ensure_psd_covariance(
        self,
        cov: pd.DataFrame,
        method: str = "higham",
        epsilon: float = 1e-12,
        max_iter: int = 100,
        tol: float = 1e-7,
    ) -> pd.DataFrame:
        """Return a symmetric PSD covariance matrix (defensive; preserves variances).

        Parameters
        ----------
        cov : pd.DataFrame
            Sample covariance estimate (may be indefinite with missing data / noise).
        method : str
            'higham' (default): convert to correlation, apply Higham, convert back.
            'eigen_clip': direct eigenvalue clipping on covariance (fast, less strict).
        """
        if cov is None or cov.empty:
            return cov

        cov_work = cov.copy().astype(float)
        cov_work = cov_work.fillna(0.0)
        cov_work.values[:] = self._symmetrize(cov_work.values)

        # Defensive variance floor
        diag = np.diag(cov_work.values).copy()
        diag = np.where(np.isfinite(diag), diag, 0.0)
        diag = np.maximum(diag, float(epsilon))

        if str(method).lower().strip() == "eigen_clip":
            repaired = self._project_psd(cov_work.values, epsilon=float(epsilon))
            # Keep original variances (important for interpretation)
            np.fill_diagonal(repaired, diag)
            repaired = self._project_psd(repaired, epsilon=float(epsilon))
            repaired_df = pd.DataFrame(repaired, index=cov_work.index, columns=cov_work.columns)
            return repaired_df

        # Higham path: covariance -> correlation -> nearest correlation -> covariance
        d = np.sqrt(diag)
        d = np.where(d > 0, d, np.sqrt(float(epsilon)))
        inv_d = 1.0 / d
        corr = cov_work.values * inv_d[:, None] * inv_d[None, :]
        corr = self._symmetrize(corr)
        np.fill_diagonal(corr, 1.0)

        corr_psd = self._higham_nearest_correlation(
            corr,
            max_iter=int(max_iter),
            tol=float(tol),
            epsilon=float(epsilon),
        )

        cov_psd = corr_psd * d[:, None] * d[None, :]
        cov_psd = self._symmetrize(cov_psd)
        # Ensure variances preserved (numerical)
        np.fill_diagonal(cov_psd, diag)
        cov_psd = self._project_psd(cov_psd, epsilon=float(epsilon))
        np.fill_diagonal(cov_psd, diag)
        cov_psd = self._symmetrize(cov_psd)

        return pd.DataFrame(cov_psd, index=cov_work.index, columns=cov_work.columns)

    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        returns = returns.dropna()
        
        if len(returns) < 20:
            return {}
        
        # Basic calculations
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics
        years = len(returns) / self.annual_trading_days
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility and risk-adjusted returns
        annual_vol = returns.std() * np.sqrt(self.annual_trading_days)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.annual_trading_days) if len(downside_returns) > 1 else 0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_duration = self._calculate_max_dd_duration(drawdown)
        
        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR and CVaR (95% and 99%)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Gain/Loss metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')
        
        # Beta and Alpha (if benchmark provided)
        alpha = beta = treynor = information_ratio = tracking_error = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns
            aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
            if len(aligned) > 20:
                asset_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]
                
                # Beta calculation
                cov_matrix = np.cov(asset_ret, bench_ret)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
                
                # Alpha calculation
                alpha = annual_return - (self.risk_free_rate + beta * (bench_ret.mean() * self.annual_trading_days - self.risk_free_rate))
                
                # Treynor ratio
                treynor = (annual_return - self.risk_free_rate) / beta if beta != 0 else 0
                
                # Information ratio
                tracking_error = (asset_ret - bench_ret).std() * np.sqrt(self.annual_trading_days)
                information_ratio = (annual_return - bench_ret.mean() * self.annual_trading_days) / tracking_error if tracking_error > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd * 100,
            'max_dd_duration': max_dd_duration,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95 * 100,
            'var_99': var_99 * 100,
            'cvar_95': cvar_95 * 100,
            'cvar_99': cvar_99 * 100,
            'win_rate': win_rate * 100,
            'avg_gain': avg_gain * 100,
            'avg_loss': avg_loss * 100,
            'profit_factor': profit_factor if profit_factor != float('inf') else 1000,
            'alpha': alpha * 100,
            'beta': beta,
            'treynor_ratio': treynor,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error * 100,
            'positive_returns': len(positive_returns),
            'negative_returns': len(negative_returns),
            'total_trades': len(returns),
            'years_data': years
        }
    
    def _calculate_max_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        if drawdown.empty:
            return 0
        
        current_duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    # =========================================================================
    # =========================================================================
    # EWMA VOLATILITY RATIO SIGNAL
    # =========================================================================

    def compute_ewma_volatility(
        self,
        returns: pd.Series,
        span: int = 22,
        annualize: bool = False
    ) -> pd.Series:
        """Compute EWMA volatility (std) from returns.

        Uses exponentially-weighted moving average of squared returns with adjust=False.
        Returns a volatility series (same index as input).
        """
        try:
            r = pd.to_numeric(returns, errors="coerce").dropna()
            if r.empty or int(span) <= 1:
                return pd.Series(dtype=float)

            # EWMA variance
            var = (r ** 2).ewm(span=int(span), adjust=False, min_periods=max(5, int(span)//3)).mean()
            vol = np.sqrt(var)
            if annualize:
                vol = vol * np.sqrt(float(self.annual_trading_days))
            vol.name = f"EWMA_VOL_{int(span)}"
            return vol
        except Exception:
            return pd.Series(dtype=float)

    def compute_ewma_volatility_ratio(
        self,
        returns: pd.Series,
        span_fast: int = 22,
        span_mid: int = 33,
        span_slow: int = 99,
        annualize: bool = False
    ) -> pd.DataFrame:
        """Compute the institutional EWMA volatility ratio signal.

        Ratio definition (as requested):
            RATIO = EWMA_VOL(span_fast) / (EWMA_VOL(span_mid) + EWMA_VOL(span_slow))

        Returns a DataFrame with EWMA vols + ratio for charting/reporting.
        """
        try:
            r = pd.to_numeric(returns, errors="coerce").dropna()
            if r.empty:
                return pd.DataFrame()

            v_fast = self.compute_ewma_volatility(r, span=int(span_fast), annualize=annualize)
            v_mid  = self.compute_ewma_volatility(r, span=int(span_mid), annualize=annualize)
            v_slow = self.compute_ewma_volatility(r, span=int(span_slow), annualize=annualize)

            # Align
            df = pd.concat([v_fast, v_mid, v_slow], axis=1).dropna(how="any")
            if df.empty:
                return pd.DataFrame()

            denom = (df[v_mid.name] + df[v_slow.name]).replace(0.0, np.nan)
            ratio = (df[v_fast.name] / denom).rename("EWMA_RATIO")
            out = df.copy()
            out["EWMA_RATIO"] = ratio
            out = out.dropna(how="any")
            return out
        except Exception:
            return pd.DataFrame()

    # PORTFOLIO OPTIMIZATION
    # =========================================================================
    
    def optimize_portfolio(
        self,
        returns_df: pd.DataFrame,
        method: str = 'sharpe',
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> Dict[str, Any]:
        """Advanced portfolio optimization"""
        
        if returns_df.empty or len(returns_df) < 60:
            return {'success': False, 'message': 'Insufficient data'}
        
        n_assets = returns_df.shape[1]
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'sum_to_one': True
            }
        
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n_assets))
        
        # Initial weights
        init_weights = np.ones(n_assets) / n_assets
        
        # Define optimization constraints
        opt_constraints = []
        
        if constraints.get('sum_to_one', True):
            opt_constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        if target_return is not None:
            opt_constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(returns_df.mean() * w) * self.annual_trading_days - target_return
            })
        # Define objective functions
        cov_matrix = returns_df.cov() * self.annual_trading_days
        mean_returns = returns_df.mean() * self.annual_trading_days

        # Defensive covariance repair (prevents hard crashes in sqrt / optimizer due to indefiniteness)
        try:
            cov_matrix = self._ensure_psd_covariance(
                cov_matrix,
                method="higham",
                epsilon=1e-12,
                max_iter=100,
                tol=1e-7,
            )
        except Exception as _psd_e:
            # Fallback to eigen-clip (very fast)
            try:
                cov_matrix = self._ensure_psd_covariance(
                    cov_matrix,
                    method="eigen_clip",
                    epsilon=1e-12,
                    max_iter=50,
                    tol=1e-6,
                )
            except Exception:
                # Last resort: numeric hygiene only
                cov_matrix = cov_matrix.fillna(0.0)
                cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        def portfolio_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 1e6
        
        def portfolio_return(weights):
            return -np.sum(mean_returns * weights)
        
        # Select objective function
        if method == 'sharpe':
            objective = portfolio_sharpe
        elif method == 'min_variance':
            objective = portfolio_variance
        elif method == 'max_return':
            objective = portfolio_return
        else:
            objective = portfolio_sharpe
        
        # Perform optimization
        try:
            result = optimize.minimize(
                objective,
                x0=init_weights,
                bounds=bounds,
                constraints=opt_constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimized_weights = result.x
                optimized_weights = optimized_weights / np.sum(optimized_weights)  # Ensure sum to 1
                
                # Calculate portfolio metrics
                portfolio_returns = returns_df @ optimized_weights
                metrics = self.calculate_performance_metrics(portfolio_returns)
                
                # Calculate risk contributions
                risk_contributions = self._calculate_risk_contributions(
                    returns_df, optimized_weights
                )
                
                # Calculate diversification ratio
                diversification_ratio = self._calculate_diversification_ratio(
                    returns_df, optimized_weights
                )
                
                return {
                    'success': True,
                    'weights': dict(zip(returns_df.columns, optimized_weights)),
                    'metrics': metrics,
                    'risk_contributions': risk_contributions,
                    'diversification_ratio': diversification_ratio,
                    'objective_value': -result.fun if method == 'sharpe' else result.fun,
                    'n_iterations': result.nit
                }
            else:
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _calculate_risk_contributions(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk contributions for each asset"""
        cov_matrix = returns_df.cov() * self.annual_trading_days
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        if portfolio_variance <= 0:
            return {asset: 0 for asset in returns_df.columns}
        
        marginal_contributions = (cov_matrix @ weights) / portfolio_variance
        risk_contributions = marginal_contributions * weights
        
        return dict(zip(returns_df.columns, risk_contributions * 100))
    
    def _calculate_diversification_ratio(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """Calculate diversification ratio"""
        asset_vols = returns_df.std() * np.sqrt(self.annual_trading_days)
        weighted_vol = np.sum(weights * asset_vols)
        portfolio_vol = np.sqrt(weights.T @ (returns_df.cov() * self.annual_trading_days) @ weights)
        
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
    
    # =========================================================================
    # GARCH MODELING
    # =========================================================================
    
    def garch_analysis(
        self,
        returns: pd.Series,
        p_range: Tuple[int, int] = (1, 2),
        q_range: Tuple[int, int] = (1, 2),
        distributions: List[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive GARCH analysis"""
        if not dep_manager.is_available('arch'):
            return {'available': False, 'message': 'ARCH package not available'}
        
        if distributions is None:
            distributions = ['normal', 't', 'skewt']
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 300:
            return {'available': False, 'message': 'Insufficient data for GARCH'}
        
        # Scale returns for better numerical stability
        returns_scaled = returns_clean * 100
        
        results = []
        arch_model = dep_manager.dependencies['arch']['arch_model']
        
        for p in range(p_range[0], p_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                for dist in distributions:
                    try:
                        # Fit GARCH model
                        model = arch_model(
                            returns_scaled,
                            mean='Constant',
                            vol='GARCH',
                            p=p,
                            q=q,
                            dist=dist
                        )
                        fit = model.fit(disp='off', show_warning=False)
                        
                        # Calculate diagnostics
                        std_resid = fit.resid / fit.conditional_volatility
                        
                        # Store results
                        results.append({
                            'p': p,
                            'q': q,
                            'distribution': dist,
                            'aic': fit.aic,
                            'bic': fit.bic,
                            'log_likelihood': fit.loglikelihood,
                            'converged': fit.convergence_flag == 0,
                            'params': dict(fit.params),
                            'conditional_volatility': fit.conditional_volatility / 100
                        })
                        
                    except Exception as e:
                        continue
        
        if not results:
            return {'available': False, 'message': 'No GARCH models converged'}
        
        # Select best model based on BIC
        results_df = pd.DataFrame(results)
        best_model = results_df.loc[results_df['bic'].idxmin()]
        
        return {
            'available': True,
            'best_model': best_model.to_dict(),
            'all_models': results,
            'n_models_tested': len(results),
            'returns': returns_clean
        }
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def detect_regimes(
        self,
        returns: pd.Series,
        n_regimes: int = 3,
        features: List[str] = None
    ) -> Dict[str, Any]:
        """Detect market regimes using HMM"""
        if not dep_manager.is_available('hmmlearn'):
            return {'available': False, 'message': 'HMM package not available'}
        
        if features is None:
            features = ['returns', 'volatility', 'volume']
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 260:
            return {'available': False, 'message': 'Insufficient data for regime detection'}
        
        try:
            # Prepare features
            feature_data = []
            
            if 'returns' in features:
                feature_data.append(returns_clean.values.reshape(-1, 1))
            
            if 'volatility' in features:
                volatility = returns_clean.rolling(window=20).std() * np.sqrt(self.annual_trading_days)
                volatility = volatility.fillna(method='bfill').values.reshape(-1, 1)
                feature_data.append(volatility)
            
            if 'volume' in features and hasattr(returns_clean, 'volume'):
                volume = returns_clean.volume if hasattr(returns_clean, 'volume') else np.ones_like(returns_clean)
                volume = volume.fillna(method='bfill').values.reshape(-1, 1)
                feature_data.append(volume)
            
            # Combine features
            X = np.hstack(feature_data)
            
            # Scale features
            scaler = dep_manager.dependencies['hmmlearn']['StandardScaler']()
            X_scaled = scaler.fit_transform(X)
            
            # Fit HMM
            GaussianHMM = dep_manager.dependencies['hmmlearn']['GaussianHMM']
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type='full',
                n_iter=1000,
                random_state=42,
                tol=1e-6
            )
            model.fit(X_scaled)
            
            # Predict regimes
            regimes = model.predict(X_scaled)
            regime_probs = model.predict_proba(X_scaled)
            
            # Calculate regime statistics
            regime_stats = []
            for i in range(n_regimes):
                mask = regimes == i
                if mask.sum() > 0:
                    regime_returns = returns_clean[mask]
                    stats = {
                        'regime': i,
                        'frequency': mask.mean() * 100,
                        'mean_return': regime_returns.mean() * 100,
                        'volatility': regime_returns.std() * np.sqrt(self.annual_trading_days) * 100,
                        'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(self.annual_trading_days) if regime_returns.std() > 0 else 0,
                        'var_95': np.percentile(regime_returns, 5) * 100
                    }
                    regime_stats.append(stats)
            
            # Label regimes
            if regime_stats:
                stats_df = pd.DataFrame(regime_stats).sort_values('mean_return')
                labels = {}
                colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']
                
                for i, (_, row) in enumerate(stats_df.iterrows()):
                    if i == 0:
                        labels[int(row['regime'])] = {'name': 'Bear', 'color': colors[0]}
                    elif i == len(stats_df) - 1:
                        labels[int(row['regime'])] = {'name': 'Bull', 'color': colors[-1]}
                    else:
                        labels[int(row['regime'])] = {'name': f'Neutral {i}', 'color': colors[i]}
            
            return {
                'available': True,
                'regimes': regimes,
                'regime_probs': regime_probs,
                'regime_stats': regime_stats,
                'regime_labels': labels,
                'model': model,
                'features': X_scaled
            }
            
        except Exception as e:
            return {'available': False, 'message': f'Regime detection failed: {str(e)}'}
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> Dict[str, Any]:
        """Calculate Value at Risk using different methods"""
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 100:
            return {}
        
        if method == 'historical':
            var = np.percentile(returns_clean, (1 - confidence_level) * 100)
        elif method == 'parametric':
            # Normal distribution assumption
            mean = returns_clean.mean()
            std = returns_clean.std()
            var = mean + std * stats.norm.ppf(1 - confidence_level)
        elif method == 'modified':
            # Cornish-Fisher expansion for skewness and kurtosis
            mean = returns_clean.mean()
            std = returns_clean.std()
            skew = returns_clean.skew()
            kurt = returns_clean.kurtosis()
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z + 
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * kurt / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            var = mean + std * z_cf
        else:
            var = np.percentile(returns_clean, (1 - confidence_level) * 100)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = returns_clean[returns_clean <= var].mean()
        
        return {
            'var': var * 100,
            'cvar': cvar * 100,
            'confidence_level': confidence_level,
            'method': method,
            'observations': len(returns_clean)
        }
    
    def stress_test(
        self,
        returns: pd.Series,
        scenarios: List[float] = None
    ) -> Dict[str, Any]:
        """Perform stress testing with historical scenarios"""
        if scenarios is None:
            scenarios = [-0.01, -0.02, -0.05, -0.10]
        
        returns_clean = returns.dropna()
        results = {}
        
        for shock in scenarios:
            # Apply shock to returns
            shocked_returns = returns_clean + shock
            
            # Calculate metrics for shocked returns
            metrics = self.calculate_performance_metrics(shocked_returns)
            
            # Calculate loss metrics
            current_value = 100  # Base value
            shocked_value = current_value * (1 + shocked_returns.sum())
            loss = current_value - shocked_value
            
            results[f'shock_{abs(shock)*100:.0f}%'] = {
                'shock': shock * 100,
                'shocked_return': shocked_returns.mean() * 100,
                'shocked_volatility': shocked_returns.std() * np.sqrt(self.annual_trading_days) * 100,
                'loss': loss,
                'max_drawdown': metrics.get('max_drawdown', 0),
                'var_95': metrics.get('var_95', 0)
            }
        
        return results
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 10000,
        n_days: int = 252
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulation for returns"""
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 60:
            return {}
        
        mean = returns_clean.mean()
        std = returns_clean.std()
        
        # Generate random returns
        np.random.seed(42)
        simulated_returns = np.random.normal(mean, std, (n_simulations, n_days))
        
        # Calculate paths
        paths = 100 * np.cumprod(1 + simulated_returns, axis=1)
        
        # Calculate statistics
        final_values = paths[:, -1]
        max_values = paths.max(axis=1)
        min_values = paths.min(axis=1)
        
        return {
            'paths': paths,
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'var_95_final': np.percentile(final_values, 5),
            'cvar_95_final': final_values[final_values <= np.percentile(final_values, 5)].mean(),
            'probability_loss': (final_values < 100).mean() * 100,
            'expected_max': np.mean(max_values),
            'expected_min': np.mean(min_values)
        }

# =============================================================================
# ADVANCED VISUALIZATION ENGINE
# =============================================================================

class InstitutionalVisualizer:
    """Professional visualization engine for institutional analytics"""
    
    def __init__(self, theme: str = "default"):
        self.theme = theme
        self.colors = ThemeManager.THEMES.get(theme, ThemeManager.THEMES["default"])
        
        # Plotly template
        self.template = go.layout.Template(
            layout=go.Layout(
                font_family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                title_font_size=20,
                title_font_color=self.colors['dark'],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.colors['dark'],
                    font_size=12,
                    font_family="Inter"
                ),
                colorway=[self.colors['primary'], self.colors['secondary'], 
                         self.colors['accent'], self.colors['success'],
                         self.colors['warning'], self.colors['danger']],
                xaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    zerolinecolor='rgba(0,0,0,0.1)',
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    zerolinecolor='rgba(0,0,0,0.1)',
                    zerolinewidth=1
                ),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font_size=12
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
        )
    
    def create_price_chart(
        self,
        df: pd.DataFrame,
        title: str,
        show_indicators: bool = True
    ) -> go.Figure:
        """Create comprehensive price chart with technical indicators"""
        
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        
        # Determine subplot configuration
        if show_indicators:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=(
                    f"{title} - Price Action",
                    "Volume",
                    "RSI",
                    "MACD"
                )
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{title} - Price Action", "Volume")
            )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                name='Price',
                line=dict(color=self.colors['primary'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.colors['primary'][1:3], 16)}, "
                         f"{int(self.colors['primary'][3:5], 16)}, "
                         f"{int(self.colors['primary'][5:7], 16)}, 0.1)"
            ),
            row=1, col=1
        )
        
        # Moving averages
        for period, color in [(20, self.colors['secondary']), (50, self.colors['accent'])]:
            if f'SMA_{period}' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f'SMA_{period}'],
                        name=f'SMA {period}',
                        line=dict(color=color, width=1.5, dash='dash'),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color=self.colors['gray'], width=1, dash='dot'),
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
                    line=dict(color=self.colors['gray'], width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=False,
                    fill='tonexty',
                    fillcolor=f"rgba({int(self.colors['gray'][1:3], 16)}, "
                             f"{int(self.colors['gray'][3:5], 16)}, "
                             f"{int(self.colors['gray'][5:7], 16)}, 0.1)"
                ),
                row=1, col=1
            )
        
        # Volume
        if 'Volume' in df.columns:
            colors = [self.colors['success'] if close >= open_ else self.colors['danger']
                     for close, open_ in zip(df[price_col], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2 if show_indicators else 2, col=1
            )
        
        # RSI
        if show_indicators and 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color=self.colors['accent'], width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors['danger'],
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors['success'],
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color=self.colors['gray'],
                         opacity=0.3, row=3, col=1)
        
        # MACD
        if show_indicators and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                row=4, col=1
            )
            
            # Histogram
            colors = [self.colors['success'] if x >= 0 else self.colors['danger']
                     for x in df['MACD_Histogram']]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=24, color=self.colors['dark'])
            ),
            height=900 if show_indicators else 700,
            template=self.template,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2 if show_indicators else 2, col=1)
        
        if show_indicators:
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig
    
    def create_performance_chart(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Performance Analysis"
    ) -> go.Figure:
        """Create performance visualization with multiple metrics"""
        

        # Robust handling: sometimes callers pass a DataFrame (multi-asset)
        # into this function. Distribution/QQ plots require 1D data, so in that
        # case we return a dedicated cumulative performance figure.
        if isinstance(returns, pd.DataFrame):
            df = returns.copy()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(axis=0, how="all")
            if df.empty:
                fig_empty = go.Figure()
                fig_empty.update_layout(title=title, template=self.template, height=420)
                return fig_empty

            fig_multi = go.Figure()
            for c in df.columns:
                s = pd.Series(df[c]).dropna()
                if s.empty:
                    continue
                cum = (1.0 + s).cumprod()
                fig_multi.add_trace(
                    go.Scatter(
                        x=cum.index,
                        y=cum.values,
                        name=str(c),
                        line=dict(width=2),
                    )
                )
            fig_multi.update_layout(
                title=title,
                height=520,
                xaxis_title="Date",
                yaxis_title="Cumulative Growth of $1",
                template=self.template,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=70, b=40),
            )
            return fig_multi

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Cumulative Returns",
                "Drawdown",
                "Rolling Returns (12M)",
                "Rolling Volatility (12M)",
                "Returns Distribution",
                "QQ Plot"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                name="Portfolio",
                line=dict(color=self.colors['primary'], width=3),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.colors['primary'][1:3], 16)}, "
                         f"{int(self.colors['primary'][3:5], 16)}, "
                         f"{int(self.colors['primary'][5:7], 16)}, 0.2)"
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name="Benchmark",
                    line=dict(color=self.colors['gray'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line=dict(color=self.colors['danger'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.colors['danger'][1:3], 16)}, "
                         f"{int(self.colors['danger'][3:5], 16)}, "
                         f"{int(self.colors['danger'][5:7], 16)}, 0.3)"
            ),
            row=2, col=1
        )
        
        # Rolling returns (12 months)
        rolling_returns = returns.rolling(window=252).mean() * 252 * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns.values,
                name="Rolling Return",
                line=dict(color=self.colors['success'], width=2)
            ),
            row=2, col=2
        )
        
        # Rolling volatility (12 months)
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="Rolling Volatility",
                line=dict(color=self.colors['warning'], width=2)
            ),
            row=3, col=1
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name="Returns",
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # QQ Plot
        if len(returns) > 10:
            qq_data = stats.probplot(returns.dropna(), dist="norm")
            fig.add_trace(
                go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name="Data",
                    marker=dict(color=self.colors['secondary'], size=6)
                ),
                row=3, col=2
            )
            
            # Add theoretical line
            x_line = np.array([qq_data[0][0][0], qq_data[0][0][-1]])
            y_line = qq_data[1][0] + qq_data[1][1] * x_line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name="Normal",
                    line=dict(color=self.colors['danger'], width=2, dash='dash')
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=24)),
            height=1000,
            template=self.template,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes titles
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=2)
        fig.update_yaxes(title_text="Annual Volatility (%)", row=3, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_xaxes(title_text="Return (%)", row=3, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=3, col=2)
        
        return fig
    
    def create_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """Create interactive correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            hoverinfo='x+y+z',
	            # Plotly Heatmap ColorBar does NOT support a top-level `titleside`.
	            # Some snippets online use `titleside`, but it will raise on
	            # Streamlit Cloud's Plotly versions.
	            # Use the supported nested form: colorbar.title.text.
	            colorbar=dict(
	                title=dict(text='Correlation'),
	                tickformat='.2f'
	            )
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=600,
            width=max(800, len(corr_matrix.columns) * 100),
            template=self.template,
            xaxis_tickangle=45,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def create_risk_decomposition(
        self,
        risk_contributions: Dict[str, float],
        title: str = "Risk Contribution Breakdown"
    ) -> go.Figure:
        """Create risk decomposition visualization"""
        
        labels = list(risk_contributions.keys())
        values = list(risk_contributions.values())
        
        fig = go.Figure(data=[go.Sunburst(
            labels=labels,
            parents=[''] * len(labels),
            values=values,
            branchvalues="total",
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Risk Contribution: %{value:.1f}%<br>',
            textinfo='label+percent entry'
        )])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template=self.template,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    def create_regime_chart(
        self,
        price: pd.Series,
        regimes: np.ndarray,
        regime_labels: Dict[int, Dict],
        title: str = "Market Regimes"
    ) -> go.Figure:
        """Create regime visualization"""
        
        fig = go.Figure()
        
        # Plot price
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price.values,
            name='Price',
            line=dict(color=self.colors['gray'], width=1),
            opacity=0.7
        ))
        
        # Add regime highlights
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            mask = regimes == regime
            regime_dates = price.index[mask]
            regime_prices = price.values[mask]
            
            label_info = regime_labels.get(int(regime), {'name': f'Regime {regime}', 'color': self.colors['gray']})
            
            fig.add_trace(go.Scatter(
                x=regime_dates,
                y=regime_prices,
                mode='markers',
                name=label_info['name'],
                marker=dict(
                    size=8,
                    color=label_info['color'],
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                opacity=0.8
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template=self.template,
            hovermode='x unified',
            yaxis_title="Price",
            xaxis_title="Date"
        )
        
        return fig
    
    def create_garch_volatility(
        self,
        returns: pd.Series,
        conditional_vol: np.ndarray,
        forecast_vol: Optional[np.ndarray] = None,
        title: str = "GARCH Volatility Analysis"
    ) -> go.Figure:
        """Create GARCH volatility visualization"""
        
        fig = go.Figure()
        
        # Realized volatility
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        fig.add_trace(go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            name='Realized Vol (20D)',
            line=dict(color=self.colors['gray'], width=2),
            opacity=0.7
        ))
        
        # Conditional volatility
        if conditional_vol is not None:
            cond_vol_series = pd.Series(conditional_vol * 100, index=returns.index[:len(conditional_vol)])
            fig.add_trace(go.Scatter(
                x=cond_vol_series.index,
                y=cond_vol_series.values,
                name='GARCH Conditional Vol',
                line=dict(color=self.colors['primary'], width=3)
            ))
        
        # Forecast volatility
        if forecast_vol is not None:
            forecast_dates = pd.date_range(
                start=returns.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast_vol),
                freq='D'
            )
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_vol * 100,
                name='Volatility Forecast',
                line=dict(color=self.colors['danger'], width=2, dash='dot')
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template=self.template,
            hovermode='x unified',
            yaxis_title="Annualized Volatility (%)",
            xaxis_title="Date"
        )
        
        return fig

    def create_ewma_ratio_signal_chart(
        self,
        ewma_df: pd.DataFrame,
        title: str = "EWMA Volatility Ratio Signal",
        bb_window: int = 20,
        bb_k: float = 2.0,
        green_max: float = 0.35,
        red_min: float = 0.55,
        show_bollinger: bool = True,
        show_threshold_lines: bool = True
    ) -> go.Figure:
        """Create an institutional EWMA ratio chart with Bollinger Bands + alarm zones.

        Zones:
            GREEN  : ratio <= green_max
            ORANGE : green_max < ratio < red_min
            RED    : ratio >= red_min
        """
        df = ewma_df.copy()
        if df.empty or "EWMA_RATIO" not in df.columns:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=520,
                template=self.template
            )
            return fig

        ratio = pd.to_numeric(df["EWMA_RATIO"], errors="coerce").dropna()
        if ratio.empty:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=520,
                template=self.template
            )
            return fig

        # Bollinger on ratio (rolling)
        bb_window = int(max(5, bb_window))
        bb_k = float(bb_k)

        mid = ratio.rolling(window=bb_window, min_periods=max(5, bb_window//2)).mean()
        std = ratio.rolling(window=bb_window, min_periods=max(5, bb_window//2)).std()
        upper = (mid + bb_k * std).rename("BB_UPPER")
        lower = (mid - bb_k * std).rename("BB_LOWER")

        # Determine y-range for colored zones
        y_min = float(max(0.0, np.nanmin([ratio.min(), lower.min() if not lower.dropna().empty else ratio.min()])))
        y_max = float(np.nanmax([ratio.max(), upper.max() if not upper.dropna().empty else ratio.max()]))
        y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.1
        y_top = y_max + y_pad

        x0 = ratio.index.min()
        x1 = ratio.index.max()

        # Zone levels sanity
        green_max = float(green_max)
        red_min = float(red_min)
        if red_min <= green_max:
            red_min = green_max + 1e-6

        fig = go.Figure()

        # Add shaded bands (risk signal)
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=y_min, y1=green_max,
            fillcolor=self.colors.get("success", "#10b981"),
            opacity=0.10,
            line_width=0,
            layer="below"
        )
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=green_max, y1=red_min,
            fillcolor=self.colors.get("warning", "#f59e0b"),
            opacity=0.10,
            line_width=0,
            layer="below"
        )
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=red_min, y1=y_top,
            fillcolor=self.colors.get("danger", "#ef4444"),
            opacity=0.10,
            line_width=0,
            layer="below"
        )

        # Ratio line
        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio.values,
                name="EWMA Ratio",
                mode="lines",
                line=dict(color=self.colors.get("primary", "#1a2980"), width=2.5)
            )
        )

        if show_bollinger:
            fig.add_trace(
                go.Scatter(
                    x=mid.index,
                    y=mid.values,
                    name=f"BB Mid ({bb_window})",
                    mode="lines",
                    line=dict(color=self.colors.get("secondary", "#26d0ce"), width=2, dash="dot"),
                    opacity=0.9
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=upper.index,
                    y=upper.values,
                    name="BB Upper",
                    mode="lines",
                    line=dict(color=self.colors.get("warning", "#f59e0b"), width=2, dash="dash"),
                    opacity=0.9
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=lower.index,
                    y=lower.values,
                    name="BB Lower",
                    mode="lines",
                    line=dict(color=self.colors.get("warning", "#f59e0b"), width=2, dash="dash"),
                    opacity=0.9
                )
            )

        if show_threshold_lines:
            fig.add_hline(
                y=green_max,
                line_dash="dash",
                line_color=self.colors.get("success", "#10b981"),
                opacity=0.7
            )
            fig.add_hline(
                y=red_min,
                line_dash="dash",
                line_color=self.colors.get("danger", "#ef4444"),
                opacity=0.7
            )

        # Latest marker with status color
        last_x = ratio.index[-1]
        last_y = float(ratio.iloc[-1])
        if last_y <= green_max:
            mcol = self.colors.get("success", "#10b981")
            status = "GREEN"
        elif last_y >= red_min:
            mcol = self.colors.get("danger", "#ef4444")
            status = "RED"
        else:
            mcol = self.colors.get("warning", "#f59e0b")
            status = "ORANGE"

        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_y],
                name=f"Latest ({status})",
                mode="markers",
                marker=dict(size=10, color=mcol, symbol="diamond")
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=560,
            template=self.template,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=70, b=50)
        )

        fig.update_yaxes(title_text="Ratio", range=[y_min, y_top])
        fig.update_xaxes(title_text="Date", rangeslider=dict(visible=True))

        return fig

# =============================================================================
# INSTITUTIONAL DASHBOARD
# =============================================================================

class InstitutionalCommoditiesDashboard:
    """Institutional commodities dashboard (Streamlit)."""
    
    def __init__(self):
        # Initialize components
        self.data_manager = EnhancedDataManager()
        self.analytics = InstitutionalAnalytics()
        self.visualizer = InstitutionalVisualizer()
        
        # Initialize session state
        self._init_session_state()
        
        # Performance tracking
        self.start_time = datetime.now()
    
    def _init_session_state(self):
        """Initialize comprehensive session state"""
        defaults = {
            # Data state
            'data_loaded': False,
            'selected_assets': [],
            'selected_benchmarks': [],
            'asset_data': {},
            'benchmark_data': {},
            'returns_data': {},
            'feature_data': {},
            
            # Portfolio state
            'portfolio_weights': {},
            'portfolio_metrics': {},
            'optimization_results': {},
            
            # Analysis state
            'garch_results': {},
            'regime_results': {},
            'risk_results': {},
            'monte_carlo_results': {},
            
            # Configuration
            'analysis_config': AnalysisConfiguration(
                start_date=datetime.now() - timedelta(days=1095),
                end_date=datetime.now()
            ),
            
            # UI state
            'current_tab': 'dashboard',
            'last_update': datetime.now(),
            'error_log': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _log_error(self, error: Exception, context: str = ""):
        """Log errors for debugging"""
        error_entry = {
            'timestamp': datetime.now(),
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        st.session_state.error_log.append(error_entry)


    def _safe_data_points(self, returns_data) -> int:
        """Safely compute number of observations in returns_data (DataFrame/Series/dict/array).

        Streamlit session_state may store returns either as a DataFrame (preferred) or a dict of series/frames.
        This helper avoids ambiguous truth checks and '.values()' call mistakes.
        """
        try:
            if returns_data is None:
                return 0

            # Dict of returns series/frames
            if isinstance(returns_data, dict):
                if len(returns_data) == 0:
                    return 0
                first = next(iter(returns_data.values()), None)
                if first is None:
                    return 0
                if isinstance(first, (pd.DataFrame, pd.Series)):
                    return 0 if first.empty else int(first.shape[0])
                try:
                    return int(len(first))
                except Exception:
                    return 0

            # Pandas objects
            if isinstance(returns_data, pd.DataFrame):
                return 0 if returns_data.empty else int(returns_data.shape[0])
            if isinstance(returns_data, pd.Series):
                return 0 if returns_data.empty else int(returns_data.shape[0])

            # Numpy arrays / lists
            if hasattr(returns_data, "shape") and returns_data.shape is not None:
                shp = returns_data.shape
                return int(shp[0]) if len(shp) >= 1 else 0

            return int(len(returns_data))
        except Exception:
            return 0
    
    # =========================================================================
    # HEADER & SIDEBAR
    # =========================================================================
    

    def display_header(self):
        """Display the app header."""

        st.components.v1.html(f"""
        <div style="
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            padding: 1.6rem 1.8rem;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        ">
            <div style="font-size:2.25rem; font-weight:850; line-height:1.15;">
                🏛️ Institutional Commodities Analytics v6.0
            </div>
        </div>
        """, height=115)




    def _render_sidebar_controls(self):
        """Sidebar: universe/asset selection + dates + load button."""
        with st.sidebar:
            st.markdown("## ⚙️ Controls")

            with st.expander("System", expanded=False):
                st.checkbox(
                    "Show system diagnostics",
                    key="show_system_diagnostics",
                    value=False,
                    help="When enabled, shows optional dependency notices and low-level system warnings."
                )

            # --- Universe / Asset selection ---
            categories = list(COMMODITIES_UNIVERSE.keys())
            # Prefer common defaults if available
            preferred_defaults = [
                AssetCategory.PRECIOUS_METALS.value,
                AssetCategory.ENERGY.value,
            ]
            default_categories = [c for c in preferred_defaults if c in categories] or (categories[:2] if categories else [])
            selected_categories = st.multiselect(
                "Commodity Groups",
                options=categories,
                default=default_categories,
                key="sidebar_groups",
                help="Select one or more commodity groups to populate the asset list."
            )

            ticker_to_label = {}
            for cat in selected_categories:
                for t, meta in COMMODITIES_UNIVERSE.get(cat, {}).items():
                    ticker_to_label[t] = f"{t} — {getattr(meta, 'name', str(t))}"

            asset_options = list(ticker_to_label.keys())
            preferred_assets = ["GC=F", "SI=F", "CL=F", "HG=F"]
            default_assets = [t for t in preferred_assets if t in asset_options]
            if not default_assets and asset_options:
                default_assets = asset_options[: min(4, len(asset_options))]

            selected_assets = st.multiselect(
                "Assets",
                options=asset_options,
                default=default_assets,
                format_func=lambda x: ticker_to_label.get(x, x),
                key="sidebar_assets",
                help="Select the assets to analyze."
            )

            # --- Benchmarks ---
            bench_options = list(BENCHMARKS.keys())
            bench_to_label = {k: f"{k} — { (v.get('name','') if isinstance(v, dict) else getattr(v, 'name', str(v))) }" for k, v in BENCHMARKS.items()}
            preferred_bench = ["SPY", "BCOM", "DBC"]
            default_bench = [b for b in preferred_bench if b in bench_options][:1] or (bench_options[:1] if bench_options else [])
            selected_benchmarks = st.multiselect(
                "Benchmarks",
                options=bench_options,
                default=default_bench,
                format_func=lambda x: bench_to_label.get(x, x),
                key="sidebar_benchmarks",
                help="Select one or more benchmarks for relative metrics."
            )

            st.markdown("---")

            # --- Dates ---
            today = datetime.now().date()
            default_start = today - timedelta(days=365 * 2)

            # Persist dates across reruns
            prev_cfg = st.session_state.get("analysis_config", None)
            prev_start = getattr(prev_cfg, "start_date", None)
            prev_end = getattr(prev_cfg, "end_date", None)

            c1, c2 = st.columns(2)
            start_date = c1.date_input(
                "Start",
                value=(prev_start.date() if prev_start else default_start),
                key="sidebar_start_date"
            )
            end_date = c2.date_input(
                "End",
                value=(prev_end.date() if prev_end else today),
                key="sidebar_end_date"
            )

            # --- Runtime / actions ---
            auto_reload = st.checkbox(
                "Auto-reload on changes",
                value=False,
                key="sidebar_autoreload",
                help="If enabled, any change in selections triggers reloading data automatically."
            )
            load_clicked = st.button("🚀 Load Data", use_container_width=True, key="sidebar_load_btn")
            clear_clicked = st.button("🧹 Clear cached data", use_container_width=True, key="sidebar_clear_cache_btn")

            if clear_clicked:
                try:
                    if hasattr(st, "cache_data"):
                        st.cache_data.clear()
                    if hasattr(st, "cache_resource"):
                        st.cache_resource.clear()
                    st.success("Cache cleared.")
                except Exception as e:
                    self._log_error(e, context="cache_clear")
                    st.warning("Cache clear attempted. If the issue persists, reload the app.")

            return {
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "start_date": start_date,
                "end_date": end_date,
                "auto_reload": auto_reload,
                "load_clicked": load_clicked,
            }

    def _load_sidebar_selection(self, sidebar_state: dict):
        """Load data based on sidebar state and populate session_state."""
        selected_assets = sidebar_state.get("selected_assets", [])
        selected_benchmarks = sidebar_state.get("selected_benchmarks", [])
        start_date = sidebar_state.get("start_date")
        end_date = sidebar_state.get("end_date")

        if not selected_assets:
            st.warning("Please select at least one asset from the sidebar.")
            st.session_state.data_loaded = False
            return

        # Normalize dates
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        if end_dt <= start_dt:
            st.warning("End date must be after the start date.")
            st.session_state.data_loaded = False
            return

        # Hash selections to avoid unnecessary reloads
        selection_fingerprint = json.dumps(
            {
                "assets": selected_assets,
                "benchmarks": selected_benchmarks,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            sort_keys=True,
        )
        selection_hash = hashlib.sha256(selection_fingerprint.encode("utf-8")).hexdigest()

        if st.session_state.get("last_selection_hash") == selection_hash and st.session_state.get("data_loaded", False):
            return

        st.session_state.last_selection_hash = selection_hash
        st.session_state.selected_assets = selected_assets
        st.session_state.selected_benchmarks = selected_benchmarks

        # Update analysis config dates (keep other defaults)
        cfg = st.session_state.get("analysis_config", AnalysisConfiguration(start_date=start_dt, end_date=end_dt))
        cfg.start_date = start_dt
        cfg.end_date = end_dt
        st.session_state.analysis_config = cfg

        with st.spinner("Loading market data..."):
            try:
                raw_assets = self.data_manager.fetch_multiple_assets(selected_assets, start_dt, end_dt, max_workers=4)
                raw_bench = self.data_manager.fetch_multiple_assets(selected_benchmarks, start_dt, end_dt, max_workers=3) if selected_benchmarks else {}

                asset_data = {}
                missing_assets = []
                for sym, df in (raw_assets or {}).items():
                    if df is None or df.empty:
                        missing_assets.append(sym)
                        continue
                    # Ensure Close exists
                    if "Close" not in df.columns and "Adj Close" in df.columns:
                        df["Close"] = df["Adj Close"]
                    df_feat = self.data_manager.calculate_technical_features(df)
                    asset_data[sym] = df_feat

                bench_data = {}
                missing_bench = []
                for sym, df in (raw_bench or {}).items():
                    if df is None or df.empty:
                        missing_bench.append(sym)
                        continue
                    if "Close" not in df.columns and "Adj Close" in df.columns:
                        df["Close"] = df["Adj Close"]
                    df_feat = self.data_manager.calculate_technical_features(df)
                    bench_data[sym] = df_feat

                if not asset_data:
                    st.session_state.data_loaded = False
                    st.error("No valid market data could be loaded for the selected assets. Try a wider date range or fewer tickers.")
                    if missing_assets:
                        st.info("Missing assets: " + ", ".join(missing_assets))
                    return

                # Build returns matrix (aligned)
                returns_df = pd.DataFrame({sym: df["Returns"] for sym, df in asset_data.items() if "Returns" in df.columns})
                returns_df = returns_df.dropna(how="all")

                bench_returns_df = pd.DataFrame({sym: df["Returns"] for sym, df in bench_data.items() if "Returns" in df.columns})
                bench_returns_df = bench_returns_df.dropna(how="all") if not bench_returns_df.empty else bench_returns_df

                st.session_state.asset_data = asset_data
                st.session_state.benchmark_data = bench_data
                st.session_state.returns_data = returns_df
                st.session_state.benchmark_returns_data = bench_returns_df
                st.session_state.data_loaded = True

                # Surface missing data as a soft warning
                if missing_assets:
                    st.sidebar.warning("Some assets returned no data: " + ", ".join(missing_assets))
                if missing_bench:
                    st.sidebar.warning("Some benchmarks returned no data: " + ", ".join(missing_bench))

                st.sidebar.success("Data loaded.")
            except Exception as e:
                self._log_error(e, context="data_load")
                st.session_state.data_loaded = False
                st.error(f"Data load failed: {e}")

    def _display_tracking_error(self, config: 'AnalysisConfiguration'):
        """Interactive Tracking Error analytics with institutional band zones.
        Robust implementation: always available even if earlier patch blocks were misplaced.
        """
        st.markdown("### 🎯 Tracking Error (Institutional Band Monitoring)")
        # --- Load returns
        to_df = getattr(self, "_to_returns_df", None)
        if callable(to_df):
            returns_df = to_df(st.session_state.get("returns_data", None))
            bench_df = to_df(st.session_state.get("benchmark_returns_data", None))
        else:
            returns_df = st.session_state.get("returns_data", None)
            bench_df = st.session_state.get("benchmark_returns_data", None)
            returns_df = returns_df.copy() if isinstance(returns_df, pd.DataFrame) else pd.DataFrame(returns_df) if isinstance(returns_df, dict) else pd.DataFrame()
            bench_df = bench_df.copy() if isinstance(bench_df, pd.DataFrame) else pd.DataFrame(bench_df) if isinstance(bench_df, dict) else pd.DataFrame()

        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        bench_df = bench_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

        if returns_df.empty:
            st.info("Load data first to compute Tracking Error.")
            return
        if bench_df.empty:
            st.warning("No benchmark returns available. Please select at least one benchmark in the sidebar and reload data.")
            return

        key_ns = "te_tab__"

        # --- Controls
        c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])
        with c1:
            scope = st.selectbox(
                "Scope",
                ["Portfolio (Equal Weight)", "Single Asset"],
                index=0,
                key=f"{key_ns}scope",
                help="Compute tracking error for an equal-weight portfolio of selected assets or a single asset.",
            )
        with c2:
            window = st.selectbox(
                "Rolling window (days)",
                [20, 60, 126, 252],
                index=3,
                key=f"{key_ns}window",
            )
        with c3:
            green_thr = st.number_input(
                "Green threshold (TE)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("te_green_thr", 0.04)),
                step=0.005,
                format="%.3f",
                key=f"{key_ns}green",
                help="Default institutional policy: TE < 4% = Green",
            )
        with c4:
            orange_thr = st.number_input(
                "Orange threshold (TE)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("te_orange_thr", 0.08)),
                step=0.005,
                format="%.3f",
                key=f"{key_ns}orange",
                help="Default institutional policy: 4–8% = Orange, >8% = Red",
            )

        st.session_state["te_green_thr"] = float(green_thr)
        st.session_state["te_orange_thr"] = float(orange_thr)

        bcols = list(bench_df.columns)
        bench_col = st.selectbox(
            "Benchmark",
            bcols,
            index=0,
            key=f"{key_ns}bench",
            help="Benchmark series used for Tracking Error.",
        )

        # --- Build portfolio/asset series
        if scope.startswith("Portfolio"):
            assets = list(returns_df.columns)
            default_assets = assets[: min(6, len(assets))]
            sel_assets = st.multiselect(
                "Select assets for equal-weight portfolio",
                assets,
                default=default_assets,
                key=f"{key_ns}assets",
            )
            if not sel_assets:
                st.warning("Select at least 1 asset.")
                return
            port = returns_df[sel_assets].mean(axis=1)
            series_name = "EQW_Portfolio"
        else:
            assets = list(returns_df.columns)
            asset = st.selectbox(
                "Asset",
                assets,
                index=0,
                key=f"{key_ns}asset",
            )
            port = returns_df[asset]
            series_name = str(asset)

        bench = bench_df[bench_col]

        # --- Align / active
        idx = port.dropna().index.intersection(bench.dropna().index)
        if len(idx) < max(60, int(window)):
            st.warning("Not enough overlapping data points to compute robust Tracking Error.")
            return

        port = port.loc[idx].astype(float)
        bench = bench.loc[idx].astype(float)
        active = (port - bench).dropna()

        if active.empty:
            st.warning("Active return series is empty after alignment.")
            return

        # --- Tracking error series (rolling)
        te_roll = active.rolling(int(window)).std(ddof=1) * np.sqrt(252.0)
        te_roll.name = "TrackingError"
        te_last = float(te_roll.dropna().iloc[-1]) if te_roll.dropna().shape[0] else np.nan

        # --- KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current TE (ann.)", f"{te_last:.2%}" if np.isfinite(te_last) else "N/A")
        k2.metric("Avg TE (ann.)", f"{float(te_roll.mean()):.2%}" if te_roll.dropna().shape[0] else "N/A")
        k3.metric("Max TE (ann.)", f"{float(te_roll.max()):.2%}" if te_roll.dropna().shape[0] else "N/A")
        k4.metric("Window", f"{int(window)}d")

        # --- Determine band range
        y_max = float(np.nanmax([te_roll.max(), orange_thr * 1.35, 0.12])) if te_roll.dropna().shape[0] else float(orange_thr * 1.35)
        y_max = max(y_max, orange_thr * 1.35, green_thr * 1.35, 0.05)

        # --- Plot with bands
        fig = go.Figure()

        # Bands (green/orange/red)
        x0 = te_roll.index.min()
        x1 = te_roll.index.max()
        fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=0, y1=green_thr,
                      fillcolor="rgba(0,200,0,0.18)", line_width=0, layer="below")
        fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=green_thr, y1=orange_thr,
                      fillcolor="rgba(255,165,0,0.18)", line_width=0, layer="below")
        fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=orange_thr, y1=y_max,
                      fillcolor="rgba(255,0,0,0.16)", line_width=0, layer="below")

        fig.add_trace(go.Scatter(x=te_roll.index, y=te_roll.values, mode="lines", name="Rolling TE (ann.)"))
        if np.isfinite(te_last):
            fig.add_trace(go.Scatter(x=[te_roll.index[-1]], y=[te_last], mode="markers", name="Current", marker=dict(size=10)))

        fig.update_layout(
            title=f"Tracking Error — {series_name} vs {bench_col} (rolling {int(window)}d)",
            height=460,
            xaxis_title="Date",
            yaxis_title="Tracking Error (annualized)",
            margin=dict(l=10, r=10, t=60, b=10),
            legend_title="Series",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_ns}chart")

        # --- Weekly table (last TE per week)
        st.markdown("#### Weekly Tracking Error Snapshot")
        te_week = te_roll.resample("W-FRI").last().dropna()
        if te_week.empty:
            st.info("Weekly snapshot not available yet.")
        else:
            table = pd.DataFrame({
                "Week": te_week.index.strftime("%Y-%m-%d"),
                "TE": te_week.values,
            })
            def _band(v: float) -> str:
                if not np.isfinite(v):
                    return "N/A"
                if v < green_thr:
                    return "GREEN"
                if v < orange_thr:
                    return "ORANGE"
                return "RED"
            table["Band"] = [_band(v) for v in table["TE"]]
            table["TE"] = table["TE"].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "N/A")
            st.dataframe(table.tail(30), use_container_width=True)

        with st.expander("Method Notes (Institutional)", expanded=False):
            st.markdown(
                """**Tracking Error (TE)** is the annualized standard deviation of **active returns** (Portfolio − Benchmark).\n\n"
                "- Rolling TE uses the selected window and annualizes by √252.\n"
                "- Band thresholds are configurable; typical policy: **<4% green**, **4–8% orange**, **>8% red**.\n"
                "- Portfolio scope here uses **equal weights** for the selected assets (manual optimizer weights are in Portfolio Lab tab)."""
            )

    def run(self):
        """Run the Streamlit app."""
        try:
            self.display_header()

            sidebar_state = self._render_sidebar_controls()

            # Auto reload on changes (optional)
            if sidebar_state.get("auto_reload", False):
                # trigger load if fingerprint changed
                self._load_sidebar_selection(sidebar_state)
            # Explicit load button
            if sidebar_state.get("load_clicked", False):
                self._load_sidebar_selection(sidebar_state)

            # --- Ensure AnalysisConfiguration exists (used by all display tabs) ---


            cfg = st.session_state.get("analysis_config")


            if cfg is None or not isinstance(cfg, AnalysisConfiguration):


                cfg = AnalysisConfiguration()


                st.session_state["analysis_config"] = cfg


            if not st.session_state.get("data_loaded", False):
                self._display_welcome(cfg)
                return

            tab_labels = [
                "📊 Dashboard",
                "🧠 Advanced Analytics",
                "🧮 Risk Analytics",
                "📉 EWMA Ratio Signal",
                "📈 Portfolio",
                "🎯 Tracking Error",
                "β Rolling Beta",
                "📉 Relative VaR/CVaR/ES",
                "🧪 Stress Testing",
                "📑 Reporting",
                "⚙️ Settings",
                "🧰 Portfolio Lab (PyPortfolioOpt)",
            ]
            tabs = st.tabs(tab_labels)

            with tabs[0]:
                self._display_dashboard(cfg)
            with tabs[1]:
                self._display_advanced_analytics(cfg)
            with tabs[2]:
                self._display_risk_analytics(cfg)
            with tabs[3]:
                self._display_ewma_ratio_signal(cfg)
            with tabs[4]:
                self._display_portfolio(cfg)
            with tabs[5]:
                self._display_tracking_error(cfg)
            with tabs[6]:
                self._display_rolling_beta(cfg)
            with tabs[7]:
                self._display_relative_risk(cfg)
            with tabs[8]:
                self._display_stress_testing(cfg)
            with tabs[9]:
                self._display_reporting(cfg)
            with tabs[10]:
                self._display_settings(cfg)

            with tabs[11]:
                self._display_portfolio_lab(cfg)

        except Exception as e:
            self._log_error(e, context="run")
            st.error(f"🚨 Application Error: {e}")
            st.code(traceback.format_exc())

    def _display_welcome(self, config: Optional[AnalysisConfiguration] = None):
        """Display welcome screen (clean)."""

        st.markdown("### 🏛️ Welcome")
        st.write("Select assets and dates from the sidebar, then click **Load Data**.")

        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown(
                """
- Select assets from the sidebar  
- Choose the date range  
- Click **Load Data**  
- Explore: **Dashboard**, **Portfolio**, **GARCH**, **Regimes**, **Analytics**, **Reports**
                """.strip()
            )

    def _display_dashboard(self, config: AnalysisConfiguration):
        """Display main dashboard"""
        st.markdown('<div class="section-header"><h2>📊 Market Dashboard</h2></div>', unsafe_allow_html=True)
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            avg_return = returns_df.mean().mean() * 252 * 100 if not returns_df.empty else 0
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📈 Avg Annual Return</div>
                <div class="metric-value {'positive' if avg_return > 0 else 'negative'}">
                    {avg_return:.2f}%
                </div>
            </div>
            """), unsafe_allow_html=True)
        
        with col2:
            avg_vol = returns_df.std().mean() * np.sqrt(252) * 100 if not returns_df.empty else 0
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📉 Avg Volatility</div>
                <div class="metric-value">{avg_vol:.2f}%</div>
            </div>
            """), unsafe_allow_html=True)
        
        with col3:
            #  reporting removed per user request (including diagnostics & coverage).
            avg_skew = float(returns_df.skew().mean()) if not returns_df.empty else np.nan
            avg_skew_disp = "N/A" if (avg_skew is None or (isinstance(avg_skew, float) and np.isnan(avg_skew))) else f"{avg_skew:.3f}"
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📐 Avg Skewness</div>
                <div class="metric-value">{avg_skew_disp}</div>
            </div>
            """), unsafe_allow_html=True)

        with col4:
            # Widgets removed per user request.
            # Provide a stable informational KPI instead.
            n_assets = int(returns_df.shape[1]) if isinstance(returns_df, pd.DataFrame) and not returns_df.empty else 0
            n_obs = int(returns_df.shape[0]) if isinstance(returns_df, pd.DataFrame) and not returns_df.empty else 0
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📦 Assets / Obs</div>
                <div class="metric-value">{n_assets} / {n_obs}</div>
            </div>
            """), unsafe_allow_html=True)

# =============================================================================
# 🔧 SAFETY BINDERS — Ensure required dashboard methods exist (no AttributeErrors)
# =============================================================================

def _icd__to_returns_df_fallback(self, returns_data):
    """Robustly coerce session_state returns_data into a wide DataFrame."""
    import numpy as np
    import pandas as pd
    if returns_data is None:
        return pd.DataFrame()
    if isinstance(returns_data, pd.DataFrame):
        return returns_data.copy()
    if isinstance(returns_data, pd.Series):
        return returns_data.to_frame()
    if isinstance(returns_data, dict):
        cols = {}
        for k, v in returns_data.items():
            if v is None:
                continue
            if isinstance(v, pd.Series):
                cols[str(k)] = v
            elif isinstance(v, pd.DataFrame):
                if v.shape[1] >= 1:
                    cols[str(k)] = v.iloc[:, 0]
        if not cols:
            return pd.DataFrame()
        df = pd.concat(cols, axis=1)
        return df
    try:
        return pd.DataFrame(returns_data)
    except Exception:
        return pd.DataFrame()

def _icd_display_relative_risk_fallback(self, cfg):
    """
    Relative Risk Dashboard:
    - Relative VaR / CVaR(ES) (Historical) on active returns (Portfolio - Benchmark)
    - Tracking Error (annualized)
    - Interactive bands (green/orange/red) via thresholds
    This is a safe fallback to avoid AttributeError if the method was not merged into the class.
    """
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    st.subheader("📊 Relative Risk (vs Benchmark) — Relative VaR / ES + Bands")

    # Returns universe
    if hasattr(self, "_to_returns_df"):
        returns_df = self._to_returns_df(st.session_state.get("returns_data", None))
    else:
        returns_df = _icd__to_returns_df_fallback(self, st.session_state.get("returns_data", None))

    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    if returns_df.empty:
        st.info("Relative risk cannot be computed: returns data is empty.")
        return

    # Benchmarks
    bench_dict = st.session_state.get("benchmark_returns", None)
    if not isinstance(bench_dict, dict):
        bench_dict = {}

    bench_options = ["(None)"] + list(bench_dict.keys())
    bmk_key = st.selectbox("Benchmark", options=bench_options, index=0, key="relrisk_bmk")
    bench = bench_dict.get(bmk_key) if bmk_key != "(None)" else None

    # Portfolio series
    port_series = st.session_state.get("portfolio_returns", None)
    if isinstance(port_series, pd.Series) and not port_series.dropna().empty:
        portfolio = port_series.dropna()
        st.caption("Using portfolio_returns from session_state.")
    else:
        asset = st.selectbox("Target (proxy portfolio): choose asset", options=list(returns_df.columns), index=0, key="relrisk_asset")
        portfolio = returns_df[asset].dropna()

    if bench is None or not isinstance(bench, pd.Series) or bench.dropna().empty:
        st.info("Select a benchmark to compute relative risk (active series).")
        return

    idx = portfolio.index.intersection(bench.dropna().index)
    if len(idx) < 60:
        st.warning("Insufficient overlap with benchmark (need ~60+ observations).")
        return

    active = (portfolio.loc[idx] - bench.loc[idx]).dropna()
    if active.empty:
        st.warning("Active series is empty after alignment.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha = st.select_slider("α (tail)", options=[0.10, 0.05, 0.025, 0.01], value=0.05, key="relrisk_alpha")
    with c2:
        horizon = st.selectbox("Horizon (days)", options=[1, 5, 10, 21], index=0, key="relrisk_h")
    with c3:
        window = st.selectbox("Rolling window", options=[63, 126, 252, 504], index=2, key="relrisk_win")
    with c4:
        green = st.number_input("Green ≤", value=0.02, step=0.005, format="%.4f", key="relrisk_green")
        orange = st.number_input("Orange ≤", value=0.04, step=0.005, format="%.4f", key="relrisk_orange")

    def _hist_var_es(series, a):
        losses = -series
        var_loss = float(np.quantile(losses, a))
        tail = losses[losses >= var_loss]
        es_loss = float(np.mean(tail)) if len(tail) else float(np.max(losses))
        # Convert back to return-space thresholds (negative for losses)
        return -var_loss, -es_loss

    roll_var = active.rolling(int(window)).apply(lambda s: _hist_var_es(pd.Series(s).dropna(), float(alpha))[0] if len(pd.Series(s).dropna()) else np.nan, raw=False)
    roll_es  = active.rolling(int(window)).apply(lambda s: _hist_var_es(pd.Series(s).dropna(), float(alpha))[1] if len(pd.Series(s).dropna()) else np.nan, raw=False)

    if int(horizon) > 1:
        scale = float(np.sqrt(int(horizon)))
        roll_var = roll_var * scale
        roll_es = roll_es * scale

    latest_var = float(roll_var.dropna().iloc[-1]) if not roll_var.dropna().empty else np.nan
    latest_es  = float(roll_es.dropna().iloc[-1]) if not roll_es.dropna().empty else np.nan
    te = float(active.std(ddof=1) * np.sqrt(252)) if active.dropna().shape[0] > 1 else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("Tracking Error (ann.)", f"{te:.2%}" if np.isfinite(te) else "N/A")
    k2.metric(f"Active VaR (Hist, α={alpha}, h={horizon})", f"{latest_var:.2%}" if np.isfinite(latest_var) else "N/A")
    k3.metric(f"Active ES (Hist, α={alpha}, h={horizon})", f"{latest_es:.2%}" if np.isfinite(latest_es) else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active.index, y=active.values, name="Active Return", mode="lines"))
    fig.add_trace(go.Scatter(x=roll_var.index, y=roll_var.values, name="Rolling Active VaR", mode="lines"))
    fig.add_trace(go.Scatter(x=roll_es.index, y=roll_es.values, name="Rolling Active ES", mode="lines"))

    g = float(green)
    o = float(orange)

    # Green band [-g, g]
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-g, y1=g, opacity=0.15, line_width=0)
    # Orange band [-o, -g] and [g, o]
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-o, y1=-g, opacity=0.12, line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=g, y1=o, opacity=0.12, line_width=0)
    # Reference lines
    fig.add_hline(y=o, line_width=1)
    fig.add_hline(y=-o, line_width=1)

    fig.update_layout(
        title="Active Return vs Rolling Relative VaR/ES (Bands via thresholds)",
        height=520,
        xaxis_title="Date",
        yaxis_title="Active Return / Risk",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, key="relrisk_chart")


# Bind missing methods safely (no crashes)
try:
    if not hasattr(InstitutionalCommoditiesDashboard, "_to_returns_df"):
        InstitutionalCommoditiesDashboard._to_returns_df = _icd__to_returns_df_fallback
    if not hasattr(InstitutionalCommoditiesDashboard, "_display_relative_risk"):
        InstitutionalCommoditiesDashboard._display_relative_risk = _icd_display_relative_risk_fallback
except Exception:
    pass



# =============================================================================
# 🧠 ADDITIVE MODULE: Quantum Sovereign v14.0 (Integrated as a separate mode)
# - Integrated WITHOUT removing existing InstitutionalCommoditiesDashboard features.
# - Runs as an additional "Mode" in the same Streamlit app.
# - Heavy ML deps (xgboost / tensorflow) are optional; the UI will degrade gracefully if missing.
# =============================================================================

def run_quantum_sovereign_v14_terminal():
    """
    Quantum Sovereign v14.0 (Quantum Sovereign) — integrated mode
    Modules: Hybrid LSTM/RNN/XGBoost, ERC Portfolio Optimization, Black-Scholes Greeks,
    Macro Sensitivity, Automated Signals, and Performance Backtesting.
    """
    import os
    import math
    import warnings
    import json
    import hashlib
    import traceback
    import logging
    from datetime import datetime, timedelta
    from typing import Dict, Any, Optional, Tuple, List, Union, Callable
    from dataclasses import dataclass, field, asdict
    from functools import lru_cache, wraps
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    import pandas as pd
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Optional heavy deps — degrade gracefully instead of crashing Streamlit Cloud
    _XGB_OK = False
    _TF_OK = False
    _SKL_OK = False

    try:
        import xgboost as xgb
        _XGB_OK = True
    except Exception:
        xgb = None
        _XGB_OK = False

    try:
        from sklearn.preprocessing import MinMaxScaler
        _SKL_OK = True
    except Exception:
        MinMaxScaler = None
        _SKL_OK = False

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping
        _TF_OK = True
    except Exception:
        Sequential = None
        LSTM = Dense = Dropout = SimpleRNN = BatchNormalization = None
        EarlyStopping = None
        _TF_OK = False

    from scipy import stats, optimize
    from scipy.stats import norm
    from scipy.optimize import minimize

    # =============================================================================
    # 1. ADVANCED METADATA & CONFIGURATION
    # =============================================================================

    @dataclass
    class AssetMetadata:
        symbol: str
        name: str
        category: str
        color: str
        risk_profile: str
        exchange: str

    # NOTE: Industrial Metals extended to include Aluminum (ALI=F) in addition to Copper (HG=F)
    ASSET_UNIVERSE = {
        "Energy": {
            "CL=F": AssetMetadata("CL=F", "Crude Oil WTI", "Energy", "#00d4ff", "High", "NYMEX"),
            "NG=F": AssetMetadata("NG=F", "Natural Gas", "Energy", "#4169E1", "High", "NYMEX"),
        },
        "Precious Metals": {
            "GC=F": AssetMetadata("GC=F", "Gold", "Metals", "#FFD700", "Low", "COMEX"),
            "SI=F": AssetMetadata("SI=F", "Silver", "Metals", "#C0C0C0", "Medium", "COMEX"),
        },
        "Industrial Metals": {
            "HG=F": AssetMetadata("HG=F", "Copper", "Metals", "#B87333", "Medium", "COMEX"),
            "ALI=F": AssetMetadata("ALI=F", "Aluminum", "Metals", "#A9A9A9", "Medium", "COMEX"),
        }
    }

    # =============================================================================
    # 2. DATA ACQUISITION & MANAGEMENT (Multi-threaded & Cached)
    # =============================================================================

    class InstitutionalDataManager:
        def __init__(self):
            self.session_data = {}

        @st.cache_data(ttl=3600)
        def get_data(self, tickers: List[str], period: str = "5y") -> pd.DataFrame:
            """High-performance multi-threaded data fetching."""
            try:
                data = yf.download(tickers, period=period, progress=False, threads=True)
                if isinstance(data.columns, pd.MultiIndex):
                    return data['Adj Close'].dropna()
                return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]}).dropna()
            except Exception as e:
                st.error(f"Data Engine Critical Failure: {e}")
                return pd.DataFrame()

        def get_macro_data(self) -> pd.DataFrame:
            """Global Macro Overlay (DXY, 10Y Yields)."""
            macro = yf.download(["DX-Y.NYB", "^TNX"], period="5y", progress=False)['Adj Close']
            macro.columns = ["DXY", "US10Y"]
            return macro.pct_change().dropna()

    # =============================================================================
    # 3. HYBRID QUANTUM AI ENGINE (LSTM + ELMAN RNN + XGBOOST)
    # =============================================================================

    class QuantumAIEngine:
        def __init__(self, lookback: int = 60):
            self.lookback = lookback
            self.scalers = {}

        def _prepare_data(self, data: pd.Series):
            if not _SKL_OK or MinMaxScaler is None:
                raise RuntimeError("scikit-learn is required for MinMaxScaler.")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
            X, y = [], []
            for i in range(self.lookback, len(scaled_data)):
                X.append(scaled_data[i-self.lookback:i, 0])
                y.append(scaled_data[i, 0])
            return np.array(X), np.array(y), scaled_data, scaler

        def build_lstm(self):
            if not _TF_OK or Sequential is None:
                raise RuntimeError("tensorflow is required for LSTM model.")
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(self.lookback, 1)),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(50, return_sequences=False),
                Dropout(0.3),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        def run_prediction(self, data: pd.Series, steps: int = 15) -> Dict[str, np.ndarray]:
            X, y, scaled_full, scaler = self._prepare_data(data)

            # 1. XGBoost Fit
            if not _XGB_OK or xgb is None:
                raise RuntimeError("xgboost is required for XGBoost model.")
            xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03, subsample=0.8)
            xgb_model.fit(X, y)

            # 2. LSTM Fit
            lstm_model = self.build_lstm()
            early_stop = EarlyStopping(monitor='loss', patience=5) if EarlyStopping is not None else None
            callbacks = [early_stop] if early_stop is not None else []
            lstm_model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=20, batch_size=32, verbose=0, callbacks=callbacks)

            # Recursive Forecasting
            preds_xgb, preds_lstm = [], []
            curr_window_xgb = scaled_full[-self.lookback:].flatten()
            curr_window_lstm = scaled_full[-self.lookback:].reshape(1, self.lookback, 1)

            for _ in range(steps):
                # XGB Prediction
                p_xgb = xgb_model.predict(curr_window_xgb.reshape(1, -1))[0]
                preds_xgb.append(p_xgb)
                curr_window_xgb = np.append(curr_window_xgb[1:], p_xgb)

                # LSTM Prediction
                p_lstm = lstm_model.predict(curr_window_lstm, verbose=0)[0, 0]
                preds_lstm.append(p_lstm)
                curr_window_lstm = np.append(curr_window_lstm[:, 1:, :], [[[p_lstm]]], axis=1)

            return {
                "XGBoost": scaler.inverse_transform(np.array(preds_xgb).reshape(-1, 1)),
                "LSTM": scaler.inverse_transform(np.array(preds_lstm).reshape(-1, 1))
            }

    # =============================================================================
    # 4. SIGNAL & RISK INTELLIGENCE
    # =============================================================================

    class SignalIntelligence:
        @staticmethod
        def generate_trade_parameters(current_p, forecast_df, ann_vol):
            ensemble_forecast = forecast_df.mean(axis=1)
            target = ensemble_forecast.iloc[-1]
            expected_ret = (target - current_p) / current_p

            # Volatility adjusted Stop Loss (2.0x ATR Approximation)
            daily_vol = ann_vol / math.sqrt(252)
            sl_buffer = current_p * daily_vol * 2.0

            if expected_ret > 0.04:
                return {"Action": "STRONG BUY", "Color": "#00ff88", "SL": current_p - sl_buffer, "TP": current_p + (sl_buffer * 3)}
            elif expected_ret < -0.04:
                return {"Action": "STRONG SELL", "Color": "#ff3b3b", "SL": current_p + sl_buffer, "TP": current_p - (sl_buffer * 3)}
            return {"Action": "HOLD / NEUTRAL", "Color": "#888888", "SL": None, "TP": None}

    # =============================================================================
    # 5. DERIVATIVES & PORTFOLIO ENGINE
    # =============================================================================

    class QuantLibrary:
        @staticmethod
        def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = norm.cdf(d1)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = norm.cdf(d1) - 1
            vega = S * norm.pdf(d1) * np.sqrt(T)
            return price, delta, vega

        @staticmethod
        def calculate_erc_weights(returns):
            cov = returns.cov().values * 252
            n = len(returns.columns)

            def objective(w):
                w = w.reshape(-1, 1)
                p_vol = np.sqrt(w.T @ cov @ w)
                rc = (w * (cov @ w)) / p_vol
                return np.sum((rc - p_vol/n)**2)

            res = minimize(objective, np.ones(n)/n, bounds=[(0,1)]*n, constraints={'type':'eq','fun':lambda x: np.sum(x)-1})
            return res.x

    # =============================================================================
    # 6. OMNI-TERMINAL APPLICATION INTERFACE
    # =============================================================================

    class SovereignTerminal:
        def __init__(self):
            # Keep the original intent but prevent Streamlit crash if already configured elsewhere
            try:
                st.set_page_config(page_title="Quantum Sovereign v14", layout="wide", initial_sidebar_state="expanded")
            except Exception:
                pass
            self.dm = InstitutionalDataManager()
            self.ai = QuantumAIEngine()
            self.quant = QuantLibrary()

        def apply_custom_css(self):
            st.markdown("""
            <style>
                .stApp { background-color: #0b0d11; }
                .metric-container { background: #151921; padding: 20px; border-radius: 12px; border: 1px solid #2d343f; }
                .header-text { color: #00d4ff; font-family: 'Inter', sans-serif; font-weight: 800; }
            </style>
            """, unsafe_allow_html=True)

        def _dep_warnings(self):
            missing = []
            if not _XGB_OK:
                missing.append("xgboost")
            if not _TF_OK:
                missing.append("tensorflow")
            if not _SKL_OK:
                missing.append("scikit-learn")
            if missing:
                st.warning("Quantum AI modules require extra packages not found in this environment: " + ", ".join(missing))
                st.code(
                    "requirements.txt suggestions:\n"
                    "xgboost\n"
                    "scikit-learn\n"
                    "tensorflow\n"
                )

        def run(self):
            self.apply_custom_css()
            st.sidebar.markdown("<h1 class='header-text'>🏛️ Quantum Sovereign</h1>", unsafe_allow_html=True)

            self._dep_warnings()

            # Sidebar Universe Selection
            category = st.sidebar.selectbox("Market Segment", list(ASSET_UNIVERSE.keys()), key="qs_category")
            selected_tickers = st.sidebar.multiselect(
                "Active Assets",
                list(ASSET_UNIVERSE[category].keys()),
                default=list(ASSET_UNIVERSE[category].keys())[:2],
                key="qs_assets"
            )

            period = st.sidebar.selectbox("History window", ["1y", "2y", "5y", "10y", "max"], index=2, key="qs_period")

            if st.sidebar.button("INITIALIZE TERMINAL EXECUTION", key="qs_init"):
                with st.spinner("Processing Quantum Models..."):
                    # Data Ingestion
                    price_data = self.dm.get_data(selected_tickers, period=period)
                    if price_data is None or price_data.empty:
                        st.error("No price data returned.")
                        return

                    returns = price_data.pct_change().dropna()
                    macro_data = self.dm.get_macro_data()

                    # Main Dashboard Layout
                    t1, t2, t3, t4, t5 = st.tabs(["📡 Signals", "🧠 AI Forecast", "🌍 Macro & Correlation", "🧮 Portfolio Lab", "🎫 Options"])

                    # Cache forecasts per ticker to avoid undefined variables between tabs
                    if "qs_forecasts" not in st.session_state:
                        st.session_state["qs_forecasts"] = {}

                    with t1:
                        st.markdown("### 📡 Automated Trade Signals (Ensemble Intelligence)")
                        for ticker in selected_tickers:
                            if ticker not in price_data.columns:
                                continue
                            curr_p = float(price_data[ticker].iloc[-1])
                            vol = float(returns[ticker].std(ddof=1) * np.sqrt(252)) if ticker in returns.columns else np.nan

                            forecast_dict = None
                            f_df = None
                            if _XGB_OK and _TF_OK and _SKL_OK:
                                try:
                                    forecast_dict = self.ai.run_prediction(price_data[ticker])
                                    f_df = pd.DataFrame(forecast_dict)
                                    st.session_state["qs_forecasts"][ticker] = f_df
                                except Exception as e:
                                    st.warning(f"AI forecast failed for {ticker}: {e}")
                                    f_df = None
                            else:
                                f_df = None

                            signal = {"Action": "HOLD / NEUTRAL", "Color": "#888888", "SL": None, "TP": None}
                            if f_df is not None and not f_df.empty and np.isfinite(vol):
                                signal = SignalIntelligence.generate_trade_parameters(curr_p, f_df, vol)

                            # UI Card
                            col_sig, col_chart = st.columns([1, 2])
                            with col_sig:
                                st.markdown(f"""
                                <div class='metric-container'>
                                    <h3 style='color:#8892b0'>{ticker}</h3>
                                    <h2 style='color:{signal['Color']}'>{signal['Action']}</h2>
                                    <p>Entry: ${curr_p:.2f}</p>
                                    {f"<p style='color:#00ff88'>TP: ${signal['TP']:.2f}</p><p style='color:#ff3b3b'>SL: ${signal['SL']:.2f}</p>" if signal['SL'] else ""}
                                </div>
                                """, unsafe_allow_html=True)

                            with col_chart:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=price_data[ticker].values[-40:], name="Historical", line=dict(color="#5161f1")))
                                if f_df is not None and not f_df.empty:
                                    fig.add_trace(go.Scatter(
                                        x=np.arange(40, 40 + len(f_df)),
                                        y=f_df.mean(axis=1).values,
                                        name="AI Ensemble",
                                        line=dict(dash='dash', color=signal['Color'])
                                    ))
                                fig.update_layout(template="plotly_dark", height=250, margin=dict(l=0, r=0, t=20, b=0))
                                st.plotly_chart(fig, use_container_width=True, key=f"qs_sig_chart_{ticker}")

                    with t2:
                        st.markdown("### 🧠 Quantum AI Decomposition")
                        st.write("Comparison of LSTM (Deep Learning) vs XGBoost (Gradient Boosting) Paths")

                        if not (_XGB_OK and _TF_OK and _SKL_OK):
                            st.info("Install xgboost + tensorflow + scikit-learn to enable AI forecast comparison charts.")
                        else:
                            pick = st.selectbox("Select asset", selected_tickers, index=0, key="qs_ai_pick")
                            f_df = st.session_state.get("qs_forecasts", {}).get(pick, None)
                            if f_df is None:
                                try:
                                    forecast_dict = self.ai.run_prediction(price_data[pick])
                                    f_df = pd.DataFrame(forecast_dict)
                                    st.session_state["qs_forecasts"][pick] = f_df
                                except Exception as e:
                                    st.error(f"Forecast failed: {e}")
                                    f_df = None
                            if f_df is not None:
                                st.line_chart(f_df)

                    with t3:
                        st.markdown("### 🌍 Macro Sensitivity & Cross-Asset Beta")
                        combined = pd.concat([returns, macro_data], axis=1).dropna()
                        if combined.empty:
                            st.info("Not enough overlapping macro + asset returns.")
                        else:
                            corr_matrix = combined.corr()
                            try:
                                st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark"),
                                                use_container_width=True, key="qs_macro_corr")
                            except Exception:
                                st.plotly_chart(px.imshow(corr_matrix, color_continuous_scale="RdBu_r", template="plotly_dark"),
                                                use_container_width=True, key="qs_macro_corr2")

                    with t4:
                        st.markdown("### 🧮 Institutional Portfolio Allocation")
                        if returns.shape[1] < 2:
                            st.info("Need at least 2 assets to compute ERC weights.")
                        else:
                            weights = self.quant.calculate_erc_weights(returns)
                            weight_df = pd.DataFrame({"Asset": list(returns.columns), "Weight": weights})
                            st.plotly_chart(px.pie(weight_df, values='Weight', names='Asset', hole=0.5,
                                                   title="Equal Risk Contribution (ERC) Allocation", template="plotly_dark"),
                                            use_container_width=True, key="qs_erc_pie")
                            st.dataframe(weight_df.set_index("Asset"), use_container_width=True)

                    with t5:
                        st.markdown("### 🎫 Options Hub (Derivatives Pricing)")
                        selected_opt = st.selectbox("Select Asset for Pricing", selected_tickers, key="qs_opt_asset")
                        S = float(price_data[selected_opt].iloc[-1])
                        K = st.number_input("Strike Price", value=float(S), key="qs_opt_strike")
                        vol_opt = float(returns[selected_opt].std(ddof=1) * np.sqrt(252)) if selected_opt in returns.columns else np.nan
                        T = st.number_input("Time to maturity (years)", value=0.10, step=0.01, key="qs_opt_T")
                        r = st.number_input("Risk-free rate", value=0.04, step=0.005, key="qs_opt_r")
                        opt_type = st.selectbox("Option Type", ["call", "put"], index=0, key="qs_opt_type")

                        if not np.isfinite(vol_opt) or vol_opt <= 0:
                            st.info("Volatility cannot be computed for options pricing (insufficient returns).")
                        else:
                            p, d, v = self.quant.black_scholes_greeks(S, K, float(T), float(r), vol_opt, option_type=opt_type)
                            st.metric(f"Option Price ({opt_type.upper()})", f"${p:.2f}")
                            st.write(f"Delta: {d:.3f} | Vega: {v:.2f}")

            else:
                st.info("Use the sidebar to select assets and click **INITIALIZE TERMINAL EXECUTION**.")

    # Run terminal
    SovereignTerminal().run()


# =============================================================================
# 🧭 APPLICATION ROUTER — Mode selector (Institutional v6.x + Quantum Sovereign v14)
# =============================================================================

def _run_app_router():
    import streamlit as st

    st.sidebar.markdown("### 🧭 Platform Mode")
    mode = st.sidebar.radio(
        "Select application layer",
        options=[
            "🏛️ Institutional Commodities Platform (v6.x)",
            "🧠 Quantum Sovereign Terminal (v14.0)"
        ],
        index=0,
        key="app_mode_selector"
    )

    if mode == "🧠 Quantum Sovereign Terminal (v14.0)":
        run_quantum_sovereign_v14_terminal()
    else:
        # Ensure InstitutionalCommoditiesDashboard exists and is runnable
        try:
            dashboard = InstitutionalCommoditiesDashboard()
            dashboard.run()
        except Exception as e:
            st.error(f"Institutional dashboard failed to start: {e}")
            st.exception(e)


# =============================================================================
# ✅ MERGE PATCH: Missing Dashboard Tabs (Advanced/Risk/EWMA/Portfolio/Beta/Relative/Stress/Reporting/Settings/PyPortfolioOpt)
# - Integrated as *safe binders* to avoid AttributeError while keeping the long single-file layout intact.
# - Uses existing InstitutionalAnalytics + InstitutionalVisualizer implementations already in this file.
# =============================================================================

def _icd__get_returns_df(self):
    """Return returns as a wide DataFrame (robust to dict/Series/DataFrame)."""
    import numpy as np
    import pandas as pd
    import streamlit as st
    rd = st.session_state.get("returns_data", None)
    if hasattr(self, "_to_returns_df"):
        try:
            df = self._to_returns_df(rd)
        except Exception:
            df = None
    else:
        df = None
    if df is None:
        try:
            # fallback defined earlier in this file (SAFETY BINDERS section)
            df = _icd__to_returns_df_fallback(self, rd)
        except Exception:
            df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = pd.DataFrame()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return df


def _icd__get_bench_df():
    """Return benchmark returns as a wide DataFrame."""
    import numpy as np
    import pandas as pd
    import streamlit as st
    b = st.session_state.get("benchmark_returns_data", None)
    if b is None:
        return pd.DataFrame()
    if isinstance(b, pd.DataFrame):
        df = b.copy()
    elif isinstance(b, dict):
        cols = {}
        for k, v in b.items():
            if v is None:
                continue
            if isinstance(v, pd.Series):
                cols[str(k)] = v
            elif isinstance(v, pd.DataFrame) and v.shape[1] >= 1:
                cols[str(k)] = v.iloc[:, 0]
        df = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    else:
        try:
            df = pd.DataFrame(b)
        except Exception:
            df = pd.DataFrame()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return df


def _display_advanced_analytics(self, config: AnalysisConfiguration):
    """Display advanced analytics section (GARCH + Regime Detection)."""
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="section-header"><h2>🧠 Advanced Analytics</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    # ------------------------------
    # GARCH Analysis
    # ------------------------------
    st.subheader("GARCH Volatility Modeling")

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_asset = st.selectbox(
            "Select Asset for GARCH",
            options=list(returns_df.columns),
            key="aa_garch_asset",
        )
    with c2:
        min_obs = st.number_input("Min obs", min_value=60, max_value=1000, value=260, step=10, key="aa_garch_minobs")

    if selected_asset:
        asset_returns = returns_df[selected_asset].dropna()
        if len(asset_returns) >= int(min_obs):
            with st.spinner("Fitting GARCH models..."):
                # respect configuration ranges when present
                try:
                    p_rng = getattr(config, "garch_p_range", (1, 2))
                    q_rng = getattr(config, "garch_q_range", (1, 2))
                except Exception:
                    p_rng, q_rng = (1, 2), (1, 2)

                garch_results = self.analytics.garch_analysis(
                    asset_returns,
                    p_range=tuple(p_rng),
                    q_range=tuple(q_rng),
                )

            if garch_results and garch_results.get("available", False):
                best_model = garch_results.get("best_model", {})
                st.success("GARCH models fitted successfully.")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Best Model", f"GARCH({best_model.get('p','?')},{best_model.get('q','?')})")
                with m2:
                    st.metric("Distribution", f"{best_model.get('distribution','?')}")
                with m3:
                    st.metric("AIC", f"{float(best_model.get('aic', float('nan'))):.2f}" if best_model.get("aic") is not None else "N/A")
                with m4:
                    st.metric("BIC", f"{float(best_model.get('bic', float('nan'))):.2f}" if best_model.get("bic") is not None else "N/A")

                cond_vol = best_model.get("conditional_volatility", None)
                if cond_vol is not None:
                    # Chart 1: existing GARCH visualization (typically returns + volatility overlay, depending on Visualizer)
                    try:
                        fig = self.visualizer.create_garch_volatility(asset_returns, cond_vol)
                        st.plotly_chart(fig, use_container_width=True, key="aa_garch_vol_chart")
                    except Exception as e:
                        st.info(f"Volatility plot unavailable: {e}")

                    # Chart 2 (NEW): conditional volatility series as a separate detailed chart
                    try:
                        import numpy as np
                        import pandas as pd
                        import plotly.graph_objects as go

                        # Normalize conditional volatility to a clean pandas Series aligned to dates
                        if isinstance(cond_vol, pd.Series):
                            cond_vol_ser = cond_vol.copy()
                        elif isinstance(cond_vol, (list, tuple, np.ndarray)):
                            # Align to the *tail* of the returns index if lengths differ
                            n = len(cond_vol)
                            cond_vol_ser = pd.Series(cond_vol, index=asset_returns.index[-n:])
                        else:
                            # best effort conversion
                            cond_vol_ser = pd.Series(cond_vol)

                        cond_vol_ser = cond_vol_ser.dropna()
                        if not cond_vol_ser.empty:
                            # Optional smoothing line (20D MA) for better "institutional" readability
                            ma20 = cond_vol_ser.rolling(20).mean()

                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=cond_vol_ser.index,
                                y=cond_vol_ser.values,
                                name="Conditional Volatility",
                                mode="lines",
                                line=dict(width=2),
                            ))
                            fig2.add_trace(go.Scatter(
                                x=ma20.index,
                                y=ma20.values,
                                name="20D MA",
                                mode="lines",
                                line=dict(width=1, dash="dash"),
                            ))

                            template = getattr(self.visualizer, "template", "plotly_white")
                            fig2.update_layout(
                                title=f"GARCH Conditional Volatility (Detailed) — {selected_asset}",
                                height=460,
                                template=template,
                                hovermode="x unified",
                                xaxis_title="Date",
                                yaxis_title="σ(t)",
                                margin=dict(l=10, r=10, t=55, b=10),
                            )

                            st.plotly_chart(fig2, use_container_width=True, key="aa_garch_condvol_detail")

                            with st.expander("📌 Conditional Volatility (last 30 obs)", expanded=False):
                                tail_df = pd.DataFrame({"cond_vol": cond_vol_ser.tail(30)})
                                st.dataframe(tail_df, use_container_width=True)
                    except Exception as e:
                        st.info(f"Conditional volatility series chart unavailable: {e}")
            else:
                st.warning(garch_results.get("message", "GARCH analysis failed.") if isinstance(garch_results, dict) else "GARCH analysis failed.")
        else:
            st.info(f"Need at least {int(min_obs)} observations for robust GARCH estimation.")

    st.divider()

    # ------------------------------
    # Regime Detection
    # ------------------------------
    st.subheader("Market Regime Detection")

    r1, r2, r3 = st.columns([2, 1, 1])
    with r1:
        regime_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=list(returns_df.columns),
            key="aa_regime_asset",
        )
    with r2:
        n_regimes = st.slider("Number of Regimes", 2, 5, int(getattr(config, "regime_states", 3)), key="aa_n_regimes")
    with r3:
        min_obs_reg = st.number_input("Min obs", min_value=120, max_value=5000, value=520, step=20, key="aa_reg_minobs")

    if regime_asset:
        asset_returns = returns_df[regime_asset].dropna()
        if len(asset_returns) >= int(min_obs_reg):
            with st.spinner("Detecting market regimes..."):
                regime_results = self.analytics.detect_regimes(asset_returns, n_regimes=int(n_regimes))

            if regime_results and regime_results.get("available", False):
                # Get price data for plotting if available
                price_series = None
                try:
                    ad = st.session_state.get("asset_data", {})
                    if isinstance(ad, dict) and regime_asset in ad:
                        dfp = ad[regime_asset]
                        # robust column naming
                        for col in ["Adj Close", "Adj_Close", "Close", "Price"]:
                            if isinstance(dfp, pd.DataFrame) and col in dfp.columns:
                                price_series = dfp[col].dropna()
                                break
                except Exception:
                    price_series = None

                if price_series is not None and not price_series.empty:
                    try:
                        fig = self.visualizer.create_regime_chart(
                            price_series,
                            regime_results.get("regimes", None),
                            regime_results.get("regime_labels", {}) or {},
                        )
                        st.plotly_chart(fig, use_container_width=True, key="aa_regime_chart")
                    except Exception as e:
                        st.info(f"Regime chart unavailable: {e}")
                else:
                    st.caption("Price series not available for chart. Showing regime statistics only.")

                st.subheader("Regime Statistics")
                regime_stats = regime_results.get("regime_stats", [])
                if regime_stats:
                    st.dataframe(pd.DataFrame(regime_stats), use_container_width=True)
            else:
                st.warning(regime_results.get("message", "Regime detection failed.") if isinstance(regime_results, dict) else "Regime detection failed.")
        else:
            st.info(f"Need at least {int(min_obs_reg)} observations for regime detection.")


def _display_risk_analytics(self, config: AnalysisConfiguration):
    """Display risk analytics (VaR/CVaR + Monte Carlo)."""
    import streamlit as st

    st.markdown('<div class="section-header"><h2>🧮 Risk Analytics</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    st.subheader("Value at Risk (VaR) Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        var_asset = st.selectbox("Select Asset", options=list(returns_df.columns), key="aa_var_asset")
    with c2:
        confidence_level = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95, key="aa_var_conf")
    with c3:
        var_method = st.selectbox("Method", options=["historical", "parametric", "modified"], key="aa_var_method")

    if var_asset:
        asset_returns = returns_df[var_asset].dropna()
        if len(asset_returns) >= 100:
            var_results = self.analytics.calculate_var(asset_returns, confidence_level=float(confidence_level), method=str(var_method))
            if var_results:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(f"VaR ({confidence_level*100:.0f}%)", f"{var_results.get('var', float('nan')):.2f}%")
                with m2:
                    st.metric(f"CVaR / ES ({confidence_level*100:.0f}%)", f"{var_results.get('cvar', float('nan')):.2f}%")
                with m3:
                    st.metric("Obs", f"{int(var_results.get('observations', len(asset_returns)))}")
        else:
            st.info("Need at least 100 observations for VaR/CVaR estimates.")

    st.divider()

    st.subheader("Monte Carlo Simulation")

    c1, c2 = st.columns([2, 1])
    with c1:
        mc_asset = st.selectbox("Select Asset for Simulation", options=list(returns_df.columns), key="aa_mc_asset")
    with c2:
        n_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000, key="aa_mc_sims")

    if mc_asset:
        asset_returns = returns_df[mc_asset].dropna()
        if len(asset_returns) >= 60:
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = self.analytics.monte_carlo_simulation(asset_returns, n_simulations=int(n_simulations))
            if mc_results:
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Mean Final Value", f"${mc_results.get('mean_final_value', float('nan')):.2f}")
                with k2:
                    st.metric("Std Final Value", f"${mc_results.get('std_final_value', float('nan')):.2f}")
                with k3:
                    st.metric("VaR 95%", f"${mc_results.get('var_95_final', float('nan')):.2f}")
                with k4:
                    st.metric("Probability of Loss", f"{mc_results.get('probability_loss', float('nan')):.1f}%")
            else:
                st.warning("Monte Carlo simulation returned no results.")
        else:
            st.info("Need at least 60 observations for Monte Carlo simulation.")


def _display_ewma_ratio_signal(self, config: AnalysisConfiguration):
    """Display EWMA Volatility Ratio signal tab (22 / (33+99) style)."""
    import numpy as np
    import streamlit as st

    st.markdown('<div class="section-header"><h2>📉 EWMA Volatility Ratio Signal</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    st.subheader("Institutional EWMA Volatility Ratio")

    c1, c2 = st.columns([2, 1])
    with c1:
        ewma_asset = st.selectbox("Select Asset", options=list(returns_df.columns), key="aa_ewma_asset")
    with c2:
        annualize = st.checkbox("Annualize Volatility", value=True, key="aa_ewma_annualize")

    if not ewma_asset:
        return

    asset_returns = returns_df[ewma_asset].dropna()

    # Parameters
    p1, p2, p3 = st.columns(3)
    with p1:
        span_fast = st.number_input("Fast Span", min_value=5, max_value=120, value=22, key="aa_ewma_fast")
    with p2:
        span_mid = st.number_input("Mid Span", min_value=10, max_value=240, value=33, key="aa_ewma_mid")
    with p3:
        span_slow = st.number_input("Slow Span", min_value=20, max_value=500, value=99, key="aa_ewma_slow")

    # Thresholds (ratio is dimensionless; policy thresholds are user-configurable)
    t1, t2 = st.columns(2)
    with t1:
        green_max = st.number_input("Green Max Threshold", min_value=0.0, max_value=2.0, value=0.35, step=0.01, key="aa_ewma_green")
    with t2:
        red_min = st.number_input("Red Min Threshold", min_value=0.0, max_value=5.0, value=0.55, step=0.01, key="aa_ewma_red")

    if len(asset_returns) <= int(span_slow):
        st.info(f"Need more data than Slow Span ({int(span_slow)}) to compute the ratio.")
        return

    with st.spinner("Calculating EWMA ratio..."):
        ewma_df = self.analytics.compute_ewma_volatility_ratio(
            asset_returns,
            span_fast=int(span_fast),
            span_mid=int(span_mid),
            span_slow=int(span_slow),
            annualize=bool(annualize),
        )

    if ewma_df is None or getattr(ewma_df, "empty", True):
        st.warning("EWMA ratio computation returned no data.")
        return

    try:
        fig = self.visualizer.create_ewma_ratio_signal_chart(
            ewma_df,
            title=f"EWMA Volatility Ratio - {ewma_asset}",
            green_max=float(green_max),
            red_min=float(red_min),
        )
        st.plotly_chart(fig, use_container_width=True, key="aa_ewma_ratio_chart")
    except Exception as e:
        st.info(f"EWMA signal chart unavailable: {e}")

    # Latest values
    latest = ewma_df.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Fast EWMA Vol", f"{float(latest.get('EWMA_VOL_22', np.nan)):.4f}" if np.isfinite(latest.get('EWMA_VOL_22', np.nan)) else "N/A")
    with k2:
        st.metric("Mid EWMA Vol", f"{float(latest.get('EWMA_VOL_33', np.nan)):.4f}" if np.isfinite(latest.get('EWMA_VOL_33', np.nan)) else "N/A")
    with k3:
        st.metric("Slow EWMA Vol", f"{float(latest.get('EWMA_VOL_99', np.nan)):.4f}" if np.isfinite(latest.get('EWMA_VOL_99', np.nan)) else "N/A")
    with k4:
        ratio = float(latest.get("EWMA_RATIO", np.nan))
        if np.isfinite(ratio):
            if ratio <= float(green_max):
                status = "🟢 GREEN"
            elif ratio >= float(red_min):
                status = "🔴 RED"
            else:
                status = "🟡 ORANGE"
            st.metric("Ratio", f"{ratio:.4f}", delta=status, delta_color="off")
        else:
            st.metric("Ratio", "N/A")


def _display_portfolio(self, config: AnalysisConfiguration):
    """Portfolio analysis: manual weights builder + optimization + saved comparisons.

    Design goals:
      - Let the user decide asset weights separately (manual builder).
      - Keep the existing in-file optimizer (no feature removals).
      - Provide a clean institutional layout with charts + metrics + save/compare.
    """
    import re
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="section-header"><h2>📈 Portfolio Analysis</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _compute_metrics(r: pd.Series, annual_days: int, rf: float) -> dict:
        r = pd.Series(r).dropna()
        if r.empty:
            return {}
        n = int(r.shape[0])
        cum = (1.0 + r).cumprod()
        ann_ret = float(cum.iloc[-1] ** (annual_days / max(n, 1)) - 1.0)
        ann_vol = float(r.std(ddof=0) * np.sqrt(annual_days))
        sharpe = float((ann_ret - rf) / ann_vol) if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
        downside = r[r < 0].std(ddof=0) * np.sqrt(annual_days)
        sortino = float((ann_ret - rf) / downside) if downside and np.isfinite(downside) and downside > 0 else np.nan
        dd = cum / cum.cummax() - 1.0
        max_dd = float(dd.min()) if not dd.empty else np.nan
        return {
            "n_obs": n,
            "annual_return": ann_ret * 100.0,
            "annual_volatility": ann_vol * 100.0,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd * 100.0,
        }

    def _risk_contributions(cov: pd.DataFrame, w: pd.Series) -> pd.Series:
        # Percent contribution to variance (sums ~ 1)
        try:
            w = w.astype(float)
            port_var = float(w.T @ cov.values @ w.values)
            if not np.isfinite(port_var) or port_var <= 0:
                return pd.Series(index=w.index, dtype=float)
            mc = cov.values @ w.values
            rc = (w.values * mc) / port_var
            return pd.Series(rc, index=w.index)
        except Exception:
            return pd.Series(index=w.index, dtype=float)

    annual_days = int(getattr(config, "annual_trading_days", 252) or 252)
    rf = float(getattr(config, "risk_free_rate", 0.0) or 0.0)

    # Persistent storage for comparisons
    st.session_state.setdefault("saved_portfolios", [])

    tab_manual, tab_opt, tab_saved = st.tabs(["🧩 Manual Weights Builder", "⚙️ Optimizer", "📚 Saved & Comparison"])

    # =====================================================================
    # TAB 1: MANUAL WEIGHTS BUILDER
    # =====================================================================
    with tab_manual:
        left, right = st.columns([1.15, 1.0], gap="large")

        with left:
            st.subheader("1) Select Assets")
            available_assets = returns_df.columns.tolist()

            selected_assets = st.multiselect(
                "Assets",
                options=available_assets,
                default=available_assets[: min(6, len(available_assets))],
                key="portfolio_manual_assets",
            )

            if not selected_assets:
                st.info("Select at least 1 asset to build a portfolio.")
                return

            base = returns_df[selected_assets].dropna(how="all")
            base = base.replace([np.inf, -np.inf], np.nan).dropna(how="all")

            if base.shape[0] < 30:
                st.warning("Not enough aligned return history for portfolio analytics (need ≥ 30 observations).")
                return

            st.subheader("2) Decide Weights (separately per asset)")
            st.caption("Enter weights in % terms. Optionally auto-normalize so total becomes 100%.")

            auto_norm = st.checkbox("Auto-normalize weights to 100%", value=True, key="portfolio_manual_autonorm")

            # Seed defaults (equal weights), but keep user edits if already set in session
            default_w = 100.0 / max(len(selected_assets), 1)

            cols = st.columns(2, gap="small")
            w_raw = {}
            for i, a in enumerate(selected_assets):
                c = cols[i % 2]
                safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(a))[:50]
                key = f"portfolio_w_{safe}"
                w = c.number_input(
                    f"{a} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get(key, default_w)),
                    step=0.25,
                    key=key,
                )
                w_raw[a] = float(w) / 100.0

            total_raw = float(sum(w_raw.values()))
            st.write(f"**Total entered:** {total_raw*100:.2f}%")

            if total_raw <= 0:
                st.warning("Total weight must be > 0.")
                return

            if auto_norm:
                weights = {k: (v / total_raw) for k, v in w_raw.items()}
            else:
                weights = dict(w_raw)

            w = pd.Series(weights, dtype=float).reindex(selected_assets).fillna(0.0)

            # Optional benchmark selector (for beta / TE)
            bench_df = st.session_state.get("benchmark_returns_data", pd.DataFrame())
            bench_df = bench_df if isinstance(bench_df, pd.DataFrame) else pd.DataFrame()
            bench_opts = ["(None)"] + (bench_df.columns.tolist() if not bench_df.empty else [])
            benchmark_choice = st.selectbox("Benchmark (optional)", options=bench_opts, index=0, key="portfolio_manual_benchmark")

            # Build portfolio (explicit button prevents heavy recompute loops)
            build = st.button("✅ Build / Update Portfolio", key="portfolio_manual_build_btn")

        with right:
            st.subheader("Portfolio Output")
            st.caption("Weights, metrics and charts update when you click **Build / Update Portfolio**.")

            # show preview always
            wdf = pd.DataFrame({"Asset": w.index, "Weight": w.values})
            wdf["Weight %"] = (wdf["Weight"] * 100.0).map(lambda x: f"{x:.2f}%")
            st.dataframe(wdf[["Asset", "Weight %"]].set_index("Asset"), use_container_width=True)

            # Weight chart
            try:
                fig_w = px.pie(wdf, names="Asset", values="Weight", title="Weights (Normalized)" if auto_norm else "Weights")
                fig_w.update_layout(height=360, legend_orientation="h")
                st.plotly_chart(fig_w, use_container_width=True)
            except Exception:
                pass

        if not build:
            st.info("Click **Build / Update Portfolio** to compute portfolio returns + charts + metrics.")
            return

        # -----------------------------------------------------------------
        # Compute portfolio returns
        # -----------------------------------------------------------------
        aligned = base.dropna()
        if aligned.shape[0] < 30:
            st.warning("Not enough fully-aligned observations across chosen assets after dropping NA.")
            return

        port_returns = (aligned.mul(w, axis=1)).sum(axis=1).dropna()
        st.session_state["portfolio_manual_result"] = {
            "assets": selected_assets,
            "weights": w.to_dict(),
            "returns": port_returns,
            "benchmark": benchmark_choice,
        }

        # -----------------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------------
        metrics = _compute_metrics(port_returns, annual_days=annual_days, rf=rf)

        # Beta / Tracking Error vs benchmark (optional)
        te = np.nan
        beta = np.nan
        if benchmark_choice != "(None)" and not bench_df.empty and benchmark_choice in bench_df.columns:
            b = bench_df[benchmark_choice].dropna()
            pair = pd.concat([port_returns, b], axis=1).dropna()
            if pair.shape[0] >= 30:
                pair.columns = ["portfolio", "benchmark"]
                active = pair["portfolio"] - pair["benchmark"]
                te = float(active.std(ddof=0) * np.sqrt(annual_days) * 100.0)
                vb = float(pair["benchmark"].var(ddof=0))
                beta = float(pair["portfolio"].cov(pair["benchmark"]) / vb) if vb and vb > 0 else np.nan

        # -----------------------------------------------------------------
        # Display metrics cards
        # -----------------------------------------------------------------
        st.subheader("3) Performance & Risk Summary")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Annual Return", f"{metrics.get('annual_return', np.nan):.2f}%")
        k2.metric("Annual Vol", f"{metrics.get('annual_volatility', np.nan):.2f}%")
        k3.metric("Sharpe", f"{metrics.get('sharpe_ratio', np.nan):.2f}")
        k4.metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan):.2f}%")
        k5.metric("Tracking Error", f"{te:.2f}%" if np.isfinite(te) else "N/A")

        if np.isfinite(beta):
            st.caption(f"β (portfolio vs {benchmark_choice}): **{beta:.2f}**")

        # -----------------------------------------------------------------
        # Charts: equity curve + drawdown + rolling volatility
        # -----------------------------------------------------------------
        st.subheader("4) Portfolio Charts")
        cum = (1.0 + port_returns).cumprod()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Portfolio Equity Curve", line=dict(width=2)))
        fig_eq.update_layout(
            title="Cumulative Performance (Equity Curve)",
            height=420,
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template=getattr(self.visualizer, "template", "plotly_white"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        dd = cum / cum.cummax() - 1.0
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values * 100.0, name="Drawdown (%)", fill="tozeroy"))
        fig_dd.update_layout(
            title="Drawdown (%)",
            height=280,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=getattr(self.visualizer, "template", "plotly_white"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        win = st.select_slider("Rolling window (days)", options=[20, 60, 126, 252], value=60, key="portfolio_manual_rollwin")
        roll_vol = port_returns.rolling(int(win)).std(ddof=0) * np.sqrt(annual_days) * 100.0
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name=f"Rolling Vol ({win}d)", line=dict(width=2)))
        fig_rv.update_layout(
            title=f"Rolling Annualized Volatility ({win}d)",
            height=320,
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template=getattr(self.visualizer, "template", "plotly_white"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_rv, use_container_width=True)

        # Risk contributions
        try:
            cov = aligned.cov()
            rc = _risk_contributions(cov, w)
            if not rc.empty and rc.notna().any():
                st.subheader("5) Risk Contributions (variance share)")
                rc_df = pd.DataFrame({"Asset": rc.index, "Contribution": rc.values})
                rc_df["Contribution %"] = (rc_df["Contribution"] * 100.0).map(lambda x: f"{x:.2f}%")
                st.dataframe(rc_df[["Asset", "Contribution %"]].set_index("Asset"), use_container_width=True)
        except Exception:
            pass

        # Save snapshot
        st.subheader("6) Save Portfolio Snapshot")
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            name = st.text_input("Portfolio Name", value=f"Manual_{len(st.session_state['saved_portfolios'])+1}", key="portfolio_manual_name")
        with c2:
            save = st.button("💾 Save", key="portfolio_manual_save_btn")

        if save:
            st.session_state["saved_portfolios"].append(
                {
                    "name": str(name).strip() or f"Manual_{len(st.session_state['saved_portfolios'])+1}",
                    "type": "manual",
                    "assets": selected_assets,
                    "weights": w.to_dict(),
                    "benchmark": benchmark_choice,
                    "metrics": {**metrics, "tracking_error": te if np.isfinite(te) else np.nan, "beta": beta},
                    "returns": port_returns,
                }
            )
            st.success("Saved.")

    # =====================================================================
    # TAB 2: OPTIMIZER (existing feature preserved)
    # =====================================================================
    with tab_opt:
        st.subheader("Optimizer")
        st.caption("This keeps your existing in-file optimization flow, with the same API calls and no feature removal.")

        optimization_method = st.selectbox(
            "Optimization Method",
            options=["sharpe", "min_variance", "max_return"],
            key="opt_method",
        )

        available_assets = returns_df.columns.tolist()
        selected_portfolio_assets = st.multiselect(
            "Select Assets for Portfolio",
            options=available_assets,
            default=available_assets[: min(6, len(available_assets))],
            key="portfolio_assets",
        )

        if not selected_portfolio_assets:
            st.info("Select assets to optimize.")
            return

        portfolio_returns = returns_df[selected_portfolio_assets].dropna()
        if portfolio_returns.shape[0] < 60:
            st.warning("Need at least 60 observations for optimization.")
            return

        st.subheader("Constraints")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_weight = st.number_input("Minimum Weight", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="min_weight")
        with c2:
            max_weight = st.number_input("Maximum Weight", min_value=0.1, max_value=1.0, value=1.0, step=0.01, key="max_weight")
        with c3:
            run_opt = st.button("🚀 Optimize Portfolio", key="optimize_btn")

        constraints = {"min_weight": float(min_weight), "max_weight": float(max_weight), "sum_to_one": True}

        if run_opt:
            if not hasattr(self, "analytics") or not hasattr(self.analytics, "optimize_portfolio"):
                st.error("Optimization engine is not available: analytics.optimize_portfolio not found.")
                return

            with st.spinner("Optimizing portfolio..."):
                optimization_results = self.analytics.optimize_portfolio(
                    portfolio_returns,
                    method=str(optimization_method),
                    constraints=constraints,
                )

            if isinstance(optimization_results, dict) and optimization_results.get("success", False):
                st.session_state["optimization_results"] = optimization_results
                st.success("Portfolio optimized successfully!")

                weights = optimization_results.get("weights", {}) or {}
                wser = pd.Series(weights, dtype=float)
                wser = wser[wser > 0].sort_values(ascending=False)

                l, r = st.columns([1.0, 1.0], gap="large")
                with l:
                    st.subheader("Optimal Weights")
                    wdf = pd.DataFrame({"Asset": wser.index, "Weight": wser.values})
                    wdf["Weight %"] = (wdf["Weight"] * 100.0).map(lambda x: f"{x:.2f}%")
                    st.dataframe(wdf[["Asset", "Weight %"]].set_index("Asset"), use_container_width=True)
                with r:
                    try:
                        fig = px.bar(wdf, x="Asset", y="Weight", title="Optimal Weights (bar)")
                        fig.update_layout(height=360, template=getattr(self.visualizer, "template", "plotly_white"))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

                metrics = optimization_results.get("metrics", {}) or {}
                st.subheader("Portfolio Performance (Optimizer Output)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Annual Return", f"{metrics.get('annual_return', 0):.2f}%")
                k2.metric("Annual Volatility", f"{metrics.get('annual_volatility', 0):.2f}%")
                k3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                k4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")

                rc = optimization_results.get("risk_contributions", None)
                if isinstance(rc, dict) and rc:
                    st.subheader("Risk Contributions")
                    rdf = pd.DataFrame({"Asset": list(rc.keys()), "Contribution": list(rc.values())})
                    rdf["Contribution %"] = rdf["Contribution"].map(lambda x: f"{x:.2f}%")
                    st.dataframe(rdf[["Asset", "Contribution %"]].set_index("Asset"), use_container_width=True)

                # Save snapshot
                st.subheader("Save Optimized Portfolio")
                c1, c2 = st.columns([1.2, 0.8])
                with c1:
                    nm = st.text_input("Name", value=f"Optim_{len(st.session_state['saved_portfolios'])+1}", key="portfolio_opt_name")
                with c2:
                    save_opt = st.button("💾 Save", key="portfolio_opt_save_btn")

                if save_opt:
                    # Build a portfolio return series using the optimized weights
                    ww = pd.Series(weights, dtype=float).reindex(portfolio_returns.columns).fillna(0.0)
                    pret = (portfolio_returns.dropna().mul(ww, axis=1)).sum(axis=1).dropna()
                    st.session_state["saved_portfolios"].append(
                        {
                            "name": str(nm).strip() or f"Optim_{len(st.session_state['saved_portfolios'])+1}",
                            "type": "optimized",
                            "assets": selected_portfolio_assets,
                            "weights": ww.to_dict(),
                            "benchmark": "(None)",
                            "metrics": metrics,
                            "returns": pret,
                        }
                    )
                    st.success("Saved.")
            else:
                msg = optimization_results.get("message", "Unknown error") if isinstance(optimization_results, dict) else "Unknown error"
                st.error(f"Optimization failed: {msg}")

    # =====================================================================
    # TAB 3: SAVED PORTFOLIOS & COMPARISON
    # =====================================================================
    with tab_saved:
        st.subheader("Saved Portfolios")
        saved = st.session_state.get("saved_portfolios", [])
        if not saved:
            st.info("No saved portfolios yet. Save one from the Manual Builder or Optimizer tab.")
            return

        # Summary table
        rows = []
        for p in saved:
            met = p.get("metrics", {}) or {}
            rows.append(
                {
                    "Name": p.get("name", ""),
                    "Type": p.get("type", ""),
                    "Assets": len(p.get("assets", []) or []),
                    "Annual Return (%)": met.get("annual_return", np.nan),
                    "Annual Vol (%)": met.get("annual_volatility", np.nan),
                    "Sharpe": met.get("sharpe_ratio", np.nan),
                    "Max DD (%)": met.get("max_drawdown", np.nan),
                    "TE (%)": met.get("tracking_error", np.nan),
                    "Beta": met.get("beta", np.nan),
                }
            )
        sdf = pd.DataFrame(rows)
        st.dataframe(sdf, use_container_width=True)

        names = [p.get("name", f"P{i+1}") for i, p in enumerate(saved)]
        sel = st.multiselect("Select portfolios to compare", options=names, default=names[: min(3, len(names))], key="portfolio_compare_sel")
        if not sel:
            return

        name_to_port = {p.get("name", f"P{i+1}"): p for i, p in enumerate(saved)}
        fig = go.Figure()
        for nm in sel:
            p = name_to_port.get(nm)
            if not p:
                continue
            r = p.get("returns", None)
            if r is None:
                continue
            r = pd.Series(r).dropna()
            if r.empty:
                continue
            cum = (1.0 + r).cumprod()
            fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=str(nm), line=dict(width=2)))

        fig.update_layout(
            title="Comparison: Equity Curves (Growth of $1)",
            height=520,
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template=getattr(self.visualizer, "template", "plotly_white"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional delete
        with st.expander("🗑️ Manage saved portfolios"):
            del_name = st.selectbox("Choose a portfolio to delete", options=["(None)"] + names, key="portfolio_delete_sel")
            if del_name != "(None)" and st.button("Delete", key="portfolio_delete_btn"):
                st.session_state["saved_portfolios"] = [p for p in saved if p.get("name") != del_name]
                st.success("Deleted. Re-run comparison if needed.")
def _display_rolling_beta(self, config: AnalysisConfiguration):
    """Display rolling beta analysis vs selected benchmark."""
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    st.markdown('<div class="section-header"><h2>β Rolling Beta Analysis</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    bench_df = _icd__get_bench_df()

    if returns_df.empty:
        st.warning("No returns data available.")
        return
    if bench_df.empty:
        st.warning("No benchmark data available. Please select benchmarks in the sidebar.")
        return

    st.subheader("Rolling Beta Calculation")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        asset = st.selectbox("Select Asset", options=list(returns_df.columns), key="aa_beta_asset")
    with c2:
        benchmark = st.selectbox("Select Benchmark", options=list(bench_df.columns), key="aa_beta_benchmark")
    with c3:
        window = st.select_slider("Rolling Window", options=[20, 60, 126, 252], value=60, key="aa_beta_window")

    if not asset or not benchmark:
        return

    aligned = pd.concat([returns_df[asset], bench_df[benchmark]], axis=1).dropna()
    aligned.columns = ["asset", "benchmark"]

    if len(aligned) <= int(window):
        st.warning("Not enough overlapping observations for the selected window.")
        return

    # rolling beta
    rb = aligned["asset"].rolling(int(window)).cov(aligned["benchmark"]) / aligned["benchmark"].rolling(int(window)).var()
    rb = rb.replace([np.inf, -np.inf], np.nan).dropna()

    if rb.empty:
        st.warning("Rolling beta series is empty after computation.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rb.index, y=rb.values, name=f"Beta vs {benchmark}", mode="lines"))
    mean_beta = float(rb.mean())
    fig.add_hline(y=mean_beta, line_dash="dash", line_color="gray", annotation_text=f"Mean: {mean_beta:.2f}", annotation_position="bottom right")
    fig.update_layout(
        title=f"Rolling {int(window)}-Day Beta: {asset} vs {benchmark}",
        height=520,
        xaxis_title="Date",
        yaxis_title="Beta",
        template=getattr(self.visualizer, "template", "plotly_white"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="aa_beta_chart")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Current Beta", f"{float(rb.iloc[-1]):.2f}")
    with k2:
        st.metric("Mean Beta", f"{mean_beta:.2f}")
    with k3:
        st.metric("Std Beta", f"{float(rb.std()):.2f}")
    with k4:
        st.metric("Min Beta", f"{float(rb.min()):.2f}")


def _display_relative_risk(self, config: AnalysisConfiguration):
    """Display relative VaR/CVaR/ES vs benchmark (active returns) + bands."""
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    st.markdown('<div class="section-header"><h2>📉 Relative VaR/CVaR/ES Analysis</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    bench_df = _icd__get_bench_df()

    if returns_df.empty:
        st.warning("No returns data available.")
        return
    if bench_df.empty:
        st.warning("No benchmark data available. Please select benchmarks in the sidebar.")
        return

    st.subheader("Active Risk Measures (Asset − Benchmark)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        asset = st.selectbox("Select Asset", options=list(returns_df.columns), key="aa_rel_asset")
    with c2:
        benchmark = st.selectbox("Select Benchmark", options=list(bench_df.columns), key="aa_rel_benchmark")
    with c3:
        confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95, key="aa_rel_conf")
    with c4:
        roll_win = st.selectbox("Rolling Window", options=[63, 126, 252, 504], index=2, key="aa_rel_win")

    if not asset or not benchmark:
        return

    aligned = pd.concat([returns_df[asset], bench_df[benchmark]], axis=1).dropna()
    aligned.columns = ["asset", "benchmark"]

    if len(aligned) < max(120, int(roll_win)):
        st.warning("Not enough overlapping observations to compute robust relative risk.")
        return

    active = (aligned["asset"] - aligned["benchmark"]).dropna()
    if active.empty:
        st.warning("Active returns series is empty after alignment.")
        return

    # instantaneous (full-sample) relative VaR/ES
    rel_var = self.analytics.calculate_var(active, confidence_level=float(confidence), method="historical")
    te_ann = float(active.std(ddof=1) * np.sqrt(252) * 100)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(f"Relative VaR ({confidence*100:.0f}%)", f"{rel_var.get('var', float('nan')):.2f}%" if rel_var else "N/A")
    with k2:
        st.metric(f"Relative ES ({confidence*100:.0f}%)", f"{rel_var.get('cvar', float('nan')):.2f}%" if rel_var else "N/A")
    with k3:
        st.metric("Tracking Error (ann.)", f"{te_ann:.2f}%")

    st.markdown("#### Active Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active.index, y=active.values * 100, name="Active Returns (%)", mode="lines", fill="tozeroy"))
    if rel_var and "var" in rel_var and rel_var["var"] is not None:
        fig.add_hline(y=float(rel_var["var"]), line_dash="dash", line_color="red", annotation_text=f"VaR {confidence*100:.0f}%", annotation_position="bottom right")
    fig.update_layout(
        height=420,
        title=f"Active Returns: {asset} − {benchmark}",
        xaxis_title="Date",
        yaxis_title="Active Return (%)",
        template=getattr(self.visualizer, "template", "plotly_white"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="aa_active_returns_chart")

    st.divider()
    st.markdown("#### Rolling Relative VaR / ES (Historical) with Risk Bands")

    b1, b2, b3 = st.columns(3)
    with b1:
        green_thr = st.number_input("Green ≤ (|VaR|, %)", min_value=0.0, max_value=50.0, value=2.0, step=0.25, key="aa_rel_green")
    with b2:
        orange_thr = st.number_input("Orange ≤ (|VaR|, %)", min_value=0.0, max_value=50.0, value=4.0, step=0.25, key="aa_rel_orange")
    with b3:
        horizon = st.selectbox("Horizon (days)", options=[1, 5, 10, 21], index=0, key="aa_rel_horizon")

    # horizon scaling (sqrt-time) for VaR/ES magnitudes on active returns
    h = int(horizon)
    active_h = active * np.sqrt(h)

    q = (1.0 - float(confidence))
    # rolling historical VaR and ES
    var_roll = active_h.rolling(int(roll_win)).quantile(q) * 100
    def _es_func(x):
        if x is None or len(x) == 0:
            return np.nan
        x = np.asarray(x, dtype=float)
        v = np.nanpercentile(x, q * 100.0)
        tail = x[x <= v]
        return np.nanmean(tail) * 100.0 if tail.size else np.nan
    es_roll = active_h.rolling(int(roll_win)).apply(_es_func, raw=False)

    plot_df = pd.DataFrame({"VaR": var_roll, "ES": es_roll}).dropna()
    if plot_df.empty:
        st.info("Rolling series not yet available for the selected window/horizon.")
        return

    # bands over |VaR|
    y = plot_df["VaR"].values
    y_abs = np.abs(y)
    y_max = float(np.nanmax([y_abs.max(), orange_thr * 1.35, 1.0]))

    fig2 = go.Figure()
    x0, x1 = plot_df.index.min(), plot_df.index.max()
    fig2.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=0, y1=float(green_thr),
                   fillcolor="rgba(0,200,0,0.18)", line_width=0, layer="below")
    fig2.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=float(green_thr), y1=float(orange_thr),
                   fillcolor="rgba(255,165,0,0.18)", line_width=0, layer="below")
    fig2.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=float(orange_thr), y1=y_max,
                   fillcolor="rgba(255,0,0,0.16)", line_width=0, layer="below")

    fig2.add_trace(go.Scatter(x=plot_df.index, y=np.abs(plot_df["VaR"].values), mode="lines", name="|VaR| (%)"))
    fig2.add_trace(go.Scatter(x=plot_df.index, y=np.abs(plot_df["ES"].values), mode="lines", name="|ES| (%)"))
    fig2.update_layout(
        height=460,
        title=f"Rolling Relative Risk (h={h}d, window={int(roll_win)}d) — {asset} vs {benchmark}",
        xaxis_title="Date",
        yaxis_title="Magnitude (%)",
        template=getattr(self.visualizer, "template", "plotly_white"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig2.update_yaxes(range=[0, y_max])
    st.plotly_chart(fig2, use_container_width=True, key="aa_relrisk_roll_chart")


def _display_stress_testing(self, config: AnalysisConfiguration):
    """Display stress testing section."""
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    st.markdown('<div class="section-header"><h2>🧪 Stress Testing</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    st.subheader("Historical Stress Scenarios")

    stress_asset = st.selectbox("Select Asset", options=list(returns_df.columns), key="aa_stress_asset")
    if not stress_asset:
        return

    asset_returns = returns_df[stress_asset].dropna()
    if len(asset_returns) < 100:
        st.info("Need at least 100 observations for stress testing.")
        return

    scenarios = st.multiselect(
        "Shock Scenarios (additive returns)",
        options=[-0.01, -0.02, -0.05, -0.10, -0.15],
        default=[-0.01, -0.02, -0.05, -0.10],
        key="aa_stress_scenarios",
    )

    with st.spinner("Running stress tests..."):
        stress_results = self.analytics.stress_test(asset_returns, list(scenarios))

    if not stress_results:
        st.warning("Stress testing returned no results.")
        return

    # Table
    df = pd.DataFrame(stress_results).T.reset_index().rename(columns={"index": "Scenario"})
    # Format
    for col in ["shock", "shocked_return", "shocked_volatility", "max_drawdown", "var_95"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    if "loss" in df.columns:
        df["loss"] = df["loss"].map(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")

    st.dataframe(df, use_container_width=True)

    # Bar impact
    st.subheader("Scenario Impact Analysis (Loss)")
    fig = go.Figure()
    for scenario, data in stress_results.items():
        fig.add_trace(go.Bar(x=[scenario], y=[abs(float(data.get("loss", 0.0)))], name=scenario, text=f"${float(data.get('loss',0.0)):.1f}", textposition="auto"))
    fig.update_layout(
        title="Loss Impact by Stress Scenario",
        height=420,
        xaxis_title="Scenario",
        yaxis_title="Loss ($)",
        template=getattr(self.visualizer, "template", "plotly_white"),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="aa_stress_loss_bar")


def _display_reporting(self, config: AnalysisConfiguration):
    """Display institutional reporting section."""
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="section-header"><h2>📑 Institutional Reporting</h2></div>', unsafe_allow_html=True)

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    st.subheader("Performance Report Generator")

    c1, c2 = st.columns(2)
    with c1:
        report_assets = st.multiselect(
            "Assets to Include",
            options=list(returns_df.columns),
            default=list(returns_df.columns)[: min(5, len(returns_df.columns))],
            key="aa_report_assets",
        )
    with c2:
        report_metrics = st.multiselect(
            "Metrics to Include",
            options=["Returns", "Volatility", "Sharpe", "Sortino", "Max Drawdown", "VaR", "CVaR", "Beta", "Correlation"],
            default=["Returns", "Volatility", "Sharpe", "Max Drawdown"],
            key="aa_report_metrics",
        )

    if not report_assets:
        st.info("Select assets to generate a report.")
        return

    if st.button("Generate Report", key="aa_generate_report"):
        with st.spinner("Generating institutional report..."):
            rows = []
            for asset in report_assets:
                s = returns_df[asset].dropna()
                if len(s) < 60:
                    continue
                metrics = self.analytics.calculate_performance_metrics(s) or {}
                r = {"Asset": asset}

                if "Returns" in report_metrics:
                    r["Annual Return"] = float(metrics.get("annual_return", np.nan))
                if "Volatility" in report_metrics:
                    r["Annual Volatility"] = float(metrics.get("annual_volatility", np.nan))
                if "Sharpe" in report_metrics:
                    r["Sharpe"] = float(metrics.get("sharpe_ratio", np.nan))
                if "Sortino" in report_metrics:
                    r["Sortino"] = float(metrics.get("sortino_ratio", np.nan))
                if "Max Drawdown" in report_metrics:
                    r["Max Drawdown"] = float(metrics.get("max_drawdown", np.nan))
                if "VaR" in report_metrics:
                    r["VaR 95%"] = float(metrics.get("var_95", np.nan))
                if "CVaR" in report_metrics:
                    r["CVaR 95%"] = float(metrics.get("cvar_95", np.nan))
                rows.append(r)

            if not rows:
                st.warning("No metrics calculated. Check asset selection and data availability.")
                return

            metrics_df = pd.DataFrame(rows).set_index("Asset")

            # display with light formatting
            display_df = metrics_df.copy()
            for col in display_df.columns:
                if col in ["Annual Return", "Annual Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%"]:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%" if np.isfinite(x) else "N/A")
                else:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "N/A")

            st.dataframe(display_df, use_container_width=True)

            # correlation
            if "Correlation" in report_metrics and len(report_assets) > 1:
                st.subheader("Correlation Matrix (Aligned + PSD-safe)")
                aligned = returns_df[report_assets].dropna(how="any")
                if aligned.shape[0] < 30:
                    st.info("Not enough overlapping observations to compute a stable correlation matrix.")
                else:
                    corr = aligned.corr()
                    try:
                        # enforce correlation PSD using existing Higham method if present
                        corr_np = corr.values.astype(float)
                        corr_fixed = self.analytics._higham_nearest_correlation(corr_np, max_iter=100, tol=1e-7, epsilon=1e-12)
                        corr = pd.DataFrame(corr_fixed, index=corr.index, columns=corr.columns)
                    except Exception:
                        pass
                    try:
                        fig = self.visualizer.create_correlation_matrix(corr)
                        st.plotly_chart(fig, use_container_width=True, key="aa_report_corr")
                    except Exception as e:
                        st.info(f"Correlation heatmap unavailable: {e}")

            # Exports
            st.subheader("Export Report")
            csv = metrics_df.reset_index().to_csv(index=False)
            jsn = metrics_df.reset_index().to_json(orient="records", indent=2)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download CSV", data=csv, file_name="commodities_report.csv", mime="text/csv", key="aa_dl_csv")
            with d2:
                st.download_button("Download JSON", data=jsn, file_name="commodities_report.json", mime="application/json", key="aa_dl_json")


def _display_settings(self, config: AnalysisConfiguration):
    """Display settings section (cloud-safe, optional psutil)."""
    import sys
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="section-header"><h2>⚙️ Settings & Configuration</h2></div>', unsafe_allow_html=True)

    st.subheader("Analysis Configuration")

    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(getattr(config, "risk_free_rate", 0.02)) * 100.0,
        step=0.1,
        key="aa_rfr",
    ) / 100.0

    annual_trading_days = st.number_input(
        "Annual Trading Days",
        min_value=200,
        max_value=365,
        value=int(getattr(config, "annual_trading_days", 252)),
        step=1,
        key="aa_trading_days",
    )

    conf_levels = st.multiselect(
        "Confidence Levels for VaR",
        options=[0.90, 0.95, 0.99, 0.995],
        default=list(getattr(config, "confidence_levels", (0.90, 0.95, 0.99))),
        key="aa_conf_levels",
    )

    c1, c2 = st.columns(2)
    with c1:
        garch_p_min = st.number_input("GARCH p min", min_value=1, max_value=5, value=int(getattr(config, "garch_p_range", (1, 2))[0]), key="aa_garch_p_min")
        garch_p_max = st.number_input("GARCH p max", min_value=1, max_value=5, value=int(getattr(config, "garch_p_range", (1, 2))[1]), key="aa_garch_p_max")
    with c2:
        garch_q_min = st.number_input("GARCH q min", min_value=1, max_value=5, value=int(getattr(config, "garch_q_range", (1, 2))[0]), key="aa_garch_q_min")
        garch_q_max = st.number_input("GARCH q max", min_value=1, max_value=5, value=int(getattr(config, "garch_q_range", (1, 2))[1]), key="aa_garch_q_max")

    regime_states = st.slider(
        "Number of Regime States",
        min_value=2,
        max_value=5,
        value=int(getattr(config, "regime_states", 3)),
        key="aa_regime_states",
    )

    if st.button("Save Configuration", key="aa_save_cfg"):
        config.risk_free_rate = float(risk_free_rate)
        config.annual_trading_days = int(annual_trading_days)
        config.confidence_levels = tuple(conf_levels) if conf_levels else (0.90, 0.95, 0.99)
        config.garch_p_range = (int(garch_p_min), int(garch_p_max))
        config.garch_q_range = (int(garch_q_min), int(garch_q_max))
        config.regime_states = int(regime_states)

        # propagate to analytics engine
        self.analytics.risk_free_rate = float(risk_free_rate)
        self.analytics.annual_trading_days = int(annual_trading_days)

        st.session_state["analysis_config"] = config
        st.success("Configuration saved successfully!")

    st.divider()
    st.subheader("System Information")

    s1, s2 = st.columns(2)
    with s1:
        st.metric("Python Version", f"{sys.version.split()[0]}")
        st.metric("Pandas Version", pd.__version__)
        st.metric("NumPy Version", np.__version__)
    with s2:
        st.metric("Streamlit Version", st.__version__)
        try:
            import plotly
            st.metric("Plotly Version", plotly.__version__)
        except Exception:
            st.metric("Plotly Version", "N/A")

        # Memory usage (optional)
        try:
            import psutil
            p = psutil.Process()
            mem_mb = p.memory_info().rss / 1024 / 1024
            st.metric("Memory Usage", f"{mem_mb:.1f} MB")
        except Exception:
            st.metric("Memory Usage", "N/A (psutil missing)")

    st.divider()
    st.subheader("Dependency Status")

    try:
        deps_status = []
        for dep_name, dep_info in getattr(dep_manager, "dependencies", {}).items():
            status = "✅ Available" if dep_info.get("available", False) else "❌ Not Available"
            deps_status.append(f"{dep_name}: {status}")
        st.write("\n".join(deps_status) if deps_status else "Dependency registry not available.")
    except Exception:
        st.write("Dependency registry not available.")


def _display_portfolio_lab(self, config: AnalysisConfiguration):
    """Portfolio Lab with PyPortfolioOpt integration (optional dependency)."""
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="section-header"><h2>🧰 Portfolio Lab (PyPortfolioOpt Integration)</h2></div>', unsafe_allow_html=True)

    # optional dependency
    try:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        pypfopt_available = True
    except Exception:
        pypfopt_available = False

    if not pypfopt_available:
        st.warning("PyPortfolioOpt not available. Install with: pip install PyPortfolioOpt")
        return

    if not st.session_state.get("data_loaded", False):
        st.warning("Please load data first.")
        return

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.warning("No returns data available.")
        return

    st.subheader("PyPortfolioOpt Optimization")

    selected_assets = st.multiselect(
        "Select Assets for Optimization",
        options=list(returns_df.columns),
        default=list(returns_df.columns)[: min(6, len(returns_df.columns))],
        key="aa_pypfopt_assets",
    )
    if not selected_assets:
        return

    r = returns_df[selected_assets].dropna()
    if len(r) < 60:
        st.warning("Need at least 60 days of overlapping data for optimization.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        opt_method = st.selectbox(
            "Optimization Method",
            options=["max_sharpe", "min_volatility", "efficient_risk", "efficient_return"],
            key="aa_pypfopt_method",
        )
    with c2:
        returns_model = st.selectbox(
            "Returns Model",
            options=["mean_historical_return", "ema_historical_return", "capm_return"],
            key="aa_pypfopt_returns_model",
        )
    with c3:
        risk_model = st.selectbox(
            "Risk Model",
            options=["sample_cov", "semicovariance", "exp_cov", "ledoit_wolf", "oracle_approximating"],
            key="aa_pypfopt_risk_model",
        )

    # expected returns
    if returns_model == "mean_historical_return":
        mu = expected_returns.mean_historical_return(r)
    elif returns_model == "ema_historical_return":
        mu = expected_returns.ema_historical_return(r)
    else:
        bench_df = _icd__get_bench_df()
        if not bench_df.empty:
            market_returns = bench_df.iloc[:, 0].dropna()
            mu = expected_returns.capm_return(r, market_returns)
        else:
            mu = expected_returns.mean_historical_return(r)

    # risk model
    if risk_model == "sample_cov":
        S = risk_models.sample_cov(r)
    elif risk_model == "semicovariance":
        S = risk_models.semicovariance(r)
    elif risk_model == "exp_cov":
        S = risk_models.exp_cov(r)
    elif risk_model == "ledoit_wolf":
        S = risk_models.CovarianceShrinkage(r).ledoit_wolf()
    else:
        S = risk_models.CovarianceShrinkage(r).oracle_approximating()

    ef = EfficientFrontier(mu, S)

    # optimization
    try:
        if opt_method == "max_sharpe":
            ef.max_sharpe()
        elif opt_method == "min_volatility":
            ef.min_volatility()
        elif opt_method == "efficient_risk":
            target_vol = st.slider("Target Volatility", 0.05, 0.80, 0.20, 0.01, key="aa_target_vol")
            ef.efficient_risk(target_volatility=float(target_vol))
        else:
            target_ret = st.slider("Target Return", 0.01, 0.80, 0.10, 0.01, key="aa_target_ret")
            ef.efficient_return(target_return=float(target_ret))
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return

    w = ef.clean_weights()
    st.subheader("Optimal Weights")
    wdf = pd.DataFrame({"Asset": list(w.keys()), "Weight": list(w.values())})
    wdf["Weight %"] = (wdf["Weight"] * 100).map(lambda x: f"{x:.2f}%")
    st.dataframe(wdf[["Asset", "Weight %"]].set_index("Asset"), use_container_width=True)

    st.subheader("Portfolio Performance")
    perf = ef.portfolio_performance(verbose=False)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Expected Annual Return", f"{perf[0]*100:.2f}%")
    with k2:
        st.metric("Expected Annual Volatility", f"{perf[1]*100:.2f}%")
    with k3:
        st.metric("Sharpe Ratio", f"{perf[2]:.2f}")

    st.subheader("Discrete Allocation")
    port_value = st.number_input("Portfolio Value ($)", 10000, 10000000, 100000, 10000, key="aa_port_value")
    try:
        latest_prices = get_latest_prices(r)
        da = DiscreteAllocation(w, latest_prices, total_portfolio_value=float(port_value))
        allocation, leftover = da.lp_portfolio()

        if allocation:
            alloc_df = pd.DataFrame({
                "Asset": list(allocation.keys()),
                "Shares": list(allocation.values()),
                "Value": [float(allocation[a]) * float(latest_prices[a]) for a in allocation.keys()],
            })
            st.dataframe(alloc_df, use_container_width=True)
            st.metric("Leftover Cash", f"${float(leftover):.2f}")
        else:
            st.info("No discrete allocation found (constraints may be too tight).")
    except Exception as e:
        st.info(f"Discrete allocation not available: {e}")


# -----------------------------------------------------------------------------
# Bind missing methods to the class (safe: only if absent, or if a previous fallback binder was used)
# -----------------------------------------------------------------------------
def _icd__bind_method(name: str, fn):
    try:
        cur = getattr(InstitutionalCommoditiesDashboard, name, None)
        # If missing OR previously bound to a fallback helper, override with the merged implementation
        if cur is None or getattr(cur, "__name__", "").startswith("_icd_"):
            setattr(InstitutionalCommoditiesDashboard, name, fn)
    except Exception:
        try:
            setattr(InstitutionalCommoditiesDashboard, name, fn)
        except Exception:
            pass

_icd__bind_method("_display_advanced_analytics", _display_advanced_analytics)
_icd__bind_method("_display_risk_analytics", _display_risk_analytics)
_icd__bind_method("_display_ewma_ratio_signal", _display_ewma_ratio_signal)
_icd__bind_method("_display_portfolio", _display_portfolio)
_icd__bind_method("_display_rolling_beta", _display_rolling_beta)
_icd__bind_method("_display_relative_risk", _display_relative_risk)
_icd__bind_method("_display_stress_testing", _display_stress_testing)
_icd__bind_method("_display_reporting", _display_reporting)
_icd__bind_method("_display_settings", _display_settings)
_icd__bind_method("_display_portfolio_lab", _display_portfolio_lab)

# =============================================================================
# ✅ END MERGE PATCH
# =============================================================================


# Execute router (Streamlit entrypoint)
_run_app_router()
