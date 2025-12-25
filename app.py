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
    """Main dashboard class with superior architecture"""
    
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
        """Display professional institutional header (clean)."""

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

    def run(self):
        """Main app runner (Streamlit entry)."""
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
            if len(returns_df.columns) > 1:
                avg_corr = returns_df.corr().values[np.triu_indices(len(returns_df.columns), 1)].mean()
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card">
                    <div class="metric-label">🔗 Avg Correlation</div>
                    <div class="metric-value">{avg_corr:.3f}</div>
                </div>
                """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div class="metric-card">
                    <div class="metric-label">🔗 Avg Correlation</div>
                    <div class="metric-value">N/A</div>
                </div>
                """), unsafe_allow_html=True)
        
        with col4:
            total_days = len(returns_df) if not returns_df.empty else 0
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📅 Trading Days</div>
                <div class="metric-value">{total_days:,}</div>
            </div>
            """), unsafe_allow_html=True)
        
        # Asset performance table
        st.markdown("### 📈 Asset Performance Overview")
        
        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if 'Returns' in df.columns:
                returns = df['Returns'].dropna()
                if len(returns) > 0:
                    metadata = next(
                        (meta for category in COMMODITIES_UNIVERSE.values() 
                         for meta in category.values() if meta.symbol == symbol),
                        AssetMetadata(symbol, symbol, AssetCategory.BENCHMARK, "#666666")
                    )
                    
                    metrics = self.analytics.calculate_performance_metrics(returns)
                    
                    performance_data.append({
                        'Asset': symbol,
                        'Name': metadata.name,
                        'Category': metadata.category.value,
                        'Current Price': df['Adj_Close'].iloc[-1] if 'Adj_Close' in df.columns else df['Close'].iloc[-1],
                        '1D Return': df['Returns'].iloc[-1] * 100 if len(df) > 1 else 0,
                        'Annual Return': metrics.get('annual_return', 0),
                        'Annual Vol': metrics.get('annual_volatility', 0),
                        'Sharpe': metrics.get('sharpe_ratio', 0),
                        'Max DD': metrics.get('max_drawdown', 0),
                        'Color': metadata.color
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Style the dataframe
            styled_df = perf_df.style.format({
                'Current Price': '{:.2f}',
                '1D Return': '{:.2f}%',
                'Annual Return': '{:.2f}%',
                'Annual Vol': '{:.2f}%',
                'Sharpe': '{:.3f}',
                'Max DD': '{:.2f}%'
            }).background_gradient(
                subset=['Annual Return', 'Sharpe'],
                cmap='RdYlGn',
                vmin=-2, vmax=2
            ).background_gradient(
                subset=['Annual Vol', 'Max DD'],
                cmap='RdYlGn_r'
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400,
                column_config={
                    'Color': st.column_config.Column(disabled=True),
                    'Name': st.column_config.Column(width="medium"),
                    'Asset': st.column_config.Column(width="small")
                }
            )
        
        # Individual asset analysis
        st.markdown("### 📉 Detailed Asset Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed View",
            options=st.session_state.selected_assets,
            key="dashboard_asset_select",
            format_func=lambda x: f"{x} - {next((meta.name for category in COMMODITIES_UNIVERSE.values() for meta in category.values() if meta.symbol == x), x)}"
        )
        
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            
            # Create tabs for different visualizations
            chart_tabs = st.tabs(["Price Analysis", "Returns Analysis", "Technical Indicators"])
            
            with chart_tabs[0]:
                fig = self.visualizer.create_price_chart(
                    df,
                    f"{selected_asset} - Comprehensive Analysis",
                    show_indicators=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[1]:
                returns = df['Returns'].dropna()
                
                if len(returns) > 0:
                    # Get benchmark returns if available
                    benchmark_returns = None
                    if st.session_state.benchmark_data:
                        # Use first benchmark for comparison
                        first_benchmark = next(iter(st.session_state.benchmark_data.values()))
                        if 'Returns' in first_benchmark.columns:
                            benchmark_returns = first_benchmark['Returns'].dropna()
                    
                    fig = self.visualizer.create_performance_chart(
                        returns,
                        benchmark_returns,
                        f"{selected_asset} - Performance Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[2]:
                # Technical indicators summary
                tech_cols = st.columns(4)
                
                tech_indicators = {
                    'RSI': ('RSI', 30, 70),
                    'MACD': ('MACD', None, None),
                    'BB Position': ('BB_Position', 0.2, 0.8),
                    'Volume Ratio': ('Volume_Ratio', 0.8, 1.2)
                }
                
                for (name, (col, lower, upper)), tech_col in zip(tech_indicators.items(), tech_cols):
                    with tech_col:
                        if col in df.columns:
                            value = df[col].iloc[-1]
                            
                            # Determine status
                            if lower is not None and upper is not None:
                                if value < lower:
                                    status = "Oversold" if name == "RSI" else "Low"
                                    color = "var(--success)"
                                elif value > upper:
                                    status = "Overbought" if name == "RSI" else "High"
                                    color = "var(--danger)"
                                else:
                                    status = "Neutral"
                                    color = "var(--warning)"
                            else:
                                status = "Current"
                                color = "var(--primary)"
                            
                            st.markdown(textwrap.dedent(f"""
                            <div class="metric-card" style="border-left-color: {color};">
                                <div class="metric-label">{name}</div>
                                <div class="metric-value" style="color: {color};">{value:.2f}</div>
                                <div style="font-size: 0.85rem; color: {color}; margin-top: 5px;">
                                    {status}
                                </div>
                            </div>
                            """), unsafe_allow_html=True)

        # Correlation matrix (SMART + ROBUST)
        try:
            returns_src = st.session_state.get("returns_data", None)

            # Normalize to DataFrame
            if isinstance(returns_src, pd.DataFrame):
                returns_df = returns_src.copy()
            elif isinstance(returns_src, dict):
                returns_df = pd.DataFrame(returns_src)
            else:
                returns_df = pd.DataFrame()

            if isinstance(returns_df, pd.DataFrame) and returns_df.shape[1] > 1:
                st.markdown("### 🔗 Correlation Analysis (Smart)")

                with st.expander("⚙️ Correlation Settings", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        corr_return_type = st.selectbox(
                            "Return Series",
                            ["Simple Returns", "Log Returns"],
                            index=0,
                            key="corr_return_type"
                        )
                    with c2:
                        corr_method = st.selectbox(
                            "Method",
                            ["Pearson", "Spearman", "Kendall"],
                            index=0,
                            key="corr_method"
                        )
                    with c3:
                        corr_alignment = st.selectbox(
                            "Date Alignment",
                            ["Pairwise (max data)", "Intersection (same dates)"],
                            index=1,
                            key="corr_alignment"
                        )
                    with c4:
                        corr_lookback = st.slider(
                            "Lookback (trading days)",
                            min_value=60,
                            max_value=min(2520, max(60, int(returns_df.shape[0]))),
                            value=min(252, max(60, int(returns_df.shape[0]))),
                            step=10,
                            key="corr_lookback"
                        )

                    c5, c6, c7, c8 = st.columns(4)
                    with c5:
                        min_overlap = st.slider(
                            "Min overlap (days)",
                            min_value=20,
                            max_value=min(260, max(20, int(returns_df.shape[0]))),
                            value=min(60, max(20, int(returns_df.shape[0]//4) if returns_df.shape[0] >= 80 else 20)),
                            step=5,
                            key="corr_min_overlap"
                        )
                    with c6:
                        winsorize_on = st.checkbox("Winsorize (1%/99%)", value=True, key="corr_winsorize")
                    with c7:
                        cluster_on = st.checkbox("Cluster & reorder", value=True, key="corr_cluster")
                    with c8:
                        show_pairs = st.checkbox("Show top pairs", value=True, key="corr_show_pairs")

                

                # --- Optional: Ledoit-Wolf shrinkage + p-value/significance overlay
                c9, c10, c11, c12 = st.columns([1.4, 1.4, 1.2, 1.6])
                with c9:
                    use_lw = st.checkbox("Ledoit-Wolf shrinkage (sklearn)", value=True, key="corr_use_lw")
                with c10:
                    signif_on = st.checkbox("p-value / significance overlay", value=False, key="corr_signif_on")
                with c11:
                    signif_alpha = st.selectbox("α", options=[0.10, 0.05, 0.01], index=1, key="corr_signif_alpha")
                with c12:
                    signif_style = st.selectbox(
                        "Overlay output",
                        options=["Stars on cells", "Separate p-value table", "Both"],
                        index=2,
                        key="corr_signif_style"
                    )
# --- Build working frame
                df = returns_df.tail(int(corr_lookback)).copy()

                # Return type
                if isinstance(corr_return_type, str) and "log" in corr_return_type.lower():
                    df = np.log1p(df)

                # Winsorize (per-series) to reduce outlier-driven distortion
                if winsorize_on:
                    def _winsorize_col(s: pd.Series) -> pd.Series:
                        s = pd.to_numeric(s, errors="coerce")
                        if s.dropna().shape[0] < 20:
                            return s
                        lo = s.quantile(0.01)
                        hi = s.quantile(0.99)
                        return s.clip(lower=lo, upper=hi)
                    df = df.apply(_winsorize_col, axis=0)

                # Alignment mode
                if isinstance(corr_alignment, str) and "intersection" in corr_alignment.lower():
                    df_work = df.dropna(how="any")
                else:
                    df_work = df

                # Overlap counts (pairwise)
                mask = df_work.notna().astype(int)
                overlap = (mask.T @ mask).astype(int)

                # Correlation (pairwise by default in pandas; min_periods enforces data quality)
                corr = df_work.corr(method=str(corr_method).lower(), min_periods=int(min_overlap))


                # Optional shrinkage correlation (requires scikit-learn)
                lw_used = False
                if 'use_lw' in locals() and bool(use_lw):
                    try:
                        from sklearn.covariance import LedoitWolf  # type: ignore
                        # Ledoit-Wolf needs a complete (no-NaN) sample matrix
                        lw_df = df_work.dropna(how="any")
                        if lw_df.shape[0] < int(min_overlap) or lw_df.shape[0] < max(30, lw_df.shape[1] + 5):
                            st.warning("Ledoit-Wolf shrinkage needs sufficient fully-overlapping data. Using standard correlation instead.")
                        else:
                            lw = LedoitWolf().fit(lw_df.values)
                            cov = pd.DataFrame(lw.covariance_, index=lw_df.columns, columns=lw_df.columns)
                            d = np.sqrt(np.diag(cov))
                            corr = cov.div(d, axis=0).div(d, axis=1)
                            # For shrinkage, overlap is the fully-overlapping sample size (same for all pairs)
                            overlap = pd.DataFrame(int(lw_df.shape[0]), index=corr.index, columns=corr.columns)
                            lw_used = True
                    except Exception as _lw_e:
                        st.info("Ledoit-Wolf shrinkage unavailable (needs scikit-learn). Using standard correlation instead.")
                        try:
                            self._log_error(_lw_e, context="correlation_ledoitwolf")
                        except Exception:
                            pass

                # Optional clustering reorder (using distance = 1 - corr)
                if cluster_on:
                    try:
                        from scipy.cluster.hierarchy import linkage, leaves_list
                        from scipy.spatial.distance import squareform

                        cfill = corr.fillna(0.0).copy()
                        # Ensure diagonal = 1 for stable distance
                        np.fill_diagonal(cfill.values, 1.0)
                        dist = 1.0 - cfill.values
                        # Condense distance matrix for linkage
                        Z = linkage(squareform(dist, checks=False), method="average")
                        order = leaves_list(Z)
                        cols = corr.columns.to_numpy()[order].tolist()
                        corr = corr.loc[cols, cols]
                        overlap = overlap.loc[cols, cols]
                    except Exception:
                        pass

                

                # Optional: p-values + significance stars (computed pairwise with available overlap)
                pvals = None
                stars = None
                if 'signif_on' in locals() and bool(signif_on):
                    try:
                        _cols = corr.columns.tolist()
                        pvals = pd.DataFrame(np.nan, index=_cols, columns=_cols)

                        for _i, _a in enumerate(_cols):
                            for _j, _b in enumerate(_cols):
                                if _i == _j:
                                    pvals.iat[_i, _j] = 0.0
                                    continue
                                pair = df_work[[_a, _b]].dropna()
                                if pair.shape[0] < int(min_overlap):
                                    continue

                                _m = str(corr_method).lower()
                                if _m.startswith("spearman"):
                                    _, _p = stats.spearmanr(pair[_a].values, pair[_b].values)
                                elif _m.startswith("kendall"):
                                    _, _p = stats.kendalltau(pair[_a].values, pair[_b].values)
                                else:
                                    _, _p = stats.pearsonr(pair[_a].values, pair[_b].values)

                                pvals.iat[_i, _j] = float(_p) if _p is not None else np.nan

                        def _star(_p):
                            if pd.isna(_p):
                                return ""
                            if _p <= 0.01:
                                return "***"
                            if _p <= 0.05:
                                return "**"
                            if _p <= 0.10:
                                return "*"
                            return ""

                        stars = pvals.applymap(_star)
                    except Exception as _p_e:
                        pvals = None
                        stars = None
                        st.info("Could not compute p-values/significance overlay (continuing without).")
                        try:
                            self._log_error(_p_e, context="correlation_pvalues")
                        except Exception:
                            pass
# Build hovertext with overlap N
                hover = []
                corr_vals = corr.values
                cols = corr.columns.tolist()
                for i, rname in enumerate(cols):
                    row = []
                    for j, cname in enumerate(cols):
                        v = corr_vals[i, j]
                        n = int(overlap.iloc[i, j]) if (rname in overlap.index and cname in overlap.columns) else 0

                        # Optional p-value in hover (if enabled)
                        p_txt = ""
                        if pvals is not None:
                            try:
                                _p = pvals.loc[rname, cname]
                                if pd.isna(_p):
                                    p_txt = "<br>p: N/A"
                                else:
                                    p_txt = f"<br>p: {float(_p):.3g}"
                            except Exception:
                                p_txt = ""
                        if stars is not None and p_txt:
                            try:
                                _s = str(stars.loc[rname, cname])
                                if _s:
                                    p_txt = p_txt + f" {_s}"
                            except Exception:
                                pass

                        if pd.isna(v):
                            row.append(f"<b>{rname}</b> vs <b>{cname}</b><br>Corr: N/A<br>N: {n}{p_txt}")
                        else:
                            row.append(f"<b>{rname}</b> vs <b>{cname}</b><br>Corr: {v:.3f}<br>N: {n}{p_txt}")
                    hover.append(row)

                # Plot heatmap (directly, to include N in hover)


                # Cell text (optionally add significance stars)
                cell_text = corr.round(2).astype(str)
                if stars is not None and 'signif_style' in locals() and (("Stars" in str(signif_style)) or ("Both" in str(signif_style))):
                    try:
                        cell_text = cell_text + stars.fillna("")
                    except Exception:
                        pass
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        zmin=-1, zmax=1, zmid=0,
                        colorscale="RdBu",
                        text=cell_text.values,
                        texttemplate="%{text}",
                        hoverinfo="text",
                        hovertext=hover,
                        colorbar=dict(
                            title=dict(text="Correlation"),
                            tickformat=".2f"
                        ),
                    )
                )
                fig.update_layout(
                    title=dict(text="Asset Correlations (Robust)", x=0.5, font=dict(size=20)),
                    height=650,
                    template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
                    xaxis_tickangle=45,
                    xaxis=dict(side="bottom"),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=70, l=40, r=20, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)


                # Optional p-value table output
                if pvals is not None and 'signif_style' in locals() and (("Separate" in str(signif_style)) or ("Both" in str(signif_style))):
                    try:
                        st.markdown("#### p-values (pairwise)")
                        st.dataframe(
                            pvals.style.format("{:.3g}"),
                            use_container_width=True
                        )
                    except Exception:
                        pass

                # Pair summaries
                if show_pairs:
                    try:
                        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        s = upper.stack().dropna()
                        if not s.empty:
                            top_pos = s.sort_values(ascending=False).head(10).reset_index()
                            top_pos.columns = ["Asset A", "Asset B", "Correlation"]
                            top_neg = s.sort_values(ascending=True).head(10).reset_index()
                            top_neg.columns = ["Asset A", "Asset B", "Correlation"]

                            cpos, cneg = st.columns(2)
                            with cpos:
                                st.markdown("**Top Positive Pairs**")
                                st.dataframe(top_pos.style.format({"Correlation": "{:.3f}"}), use_container_width=True, hide_index=True)
                            with cneg:
                                st.markdown("**Top Negative Pairs**")
                                st.dataframe(top_neg.style.format({"Correlation": "{:.3f}"}), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
        except Exception as _corr_e:
            # Never hard-fail the page due to correlation viz
            try:
                self._log_error(_corr_e, context="correlation_analysis")
            except Exception:
                pass
    
    def _display_portfolio(self, config: AnalysisConfiguration):
        """Display portfolio analysis"""
        st.markdown('<div class="section-header"><h2>🧺 Portfolio Analysis</h2></div>', unsafe_allow_html=True)
        
        returns_data = st.session_state.get("returns_data", None)
        if returns_data is None or (isinstance(returns_data, pd.DataFrame) and returns_data.empty) or (isinstance(returns_data, dict) and len(returns_data) == 0):
            st.warning("⚠️ No return data available. Please load data first.")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty:
            st.warning("⚠️ Insufficient data for portfolio analysis")
            return
        
        # Portfolio configuration
        st.markdown("### ⚙️ Portfolio Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            weight_mode = st.radio(
                "Weighting Method",
                ["Equal Weight", "Optimized (Sharpe)", "Optimized (Min Variance)", "Custom Weights"],
                horizontal=True,
                key="portfolio_weight_mode"
            )
            
            if weight_mode == "Custom Weights":
                st.markdown("**Set Custom Weights:**")
                
                assets = returns_df.columns.tolist()
                n_cols = min(4, len(assets))
                cols = st.columns(n_cols)
                
                weight_inputs = []
                for i, asset in enumerate(assets):
                    with cols[i % n_cols]:
                        default_weight = 1.0 / len(assets)
                        current_weight = st.session_state.portfolio_weights.get(asset, default_weight)
                        
                        weight = st.slider(
                            asset,
                            min_value=0.0,
                            max_value=1.0,
                            value=current_weight,
                            step=0.01,
                            key=f"custom_weight_{asset}"
                        )
                        weight_inputs.append(weight)
                
                weights = np.array(weight_inputs)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            elif weight_mode.startswith("Optimized"):
                optimization_type = weight_mode.split("(")[1].rstrip(")")
                
                if st.button(f"🔄 Optimize Portfolio ({optimization_type})", type="primary", use_container_width=True, key=f"btn_optimize_portfolio_{optimization_type}"):
                    with st.spinner("Optimizing portfolio..."):
                        result = self.analytics.optimize_portfolio(
                            returns_df,
                            method=optimization_type.lower().replace(' ', '_'),
                            target_return=None
                        )
                        
                        if result['success']:
                            weights = np.array(list(result['weights'].values()))
                            st.session_state.portfolio_weights = result['weights']
                            st.session_state.portfolio_metrics = result['metrics']
                            st.success("✅ Portfolio optimized successfully!")
                        else:
                            st.warning(f"⚠️ Optimization failed: {result.get('message', 'Unknown error')}")
                            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                else:
                    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            
            else:  # Equal Weight
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                st.session_state.portfolio_weights = dict(zip(returns_df.columns, weights))
        
        with col2:
            # Display current weights
            st.markdown("**Current Weights:**")
            
            weight_data = []
            for asset, weight in st.session_state.portfolio_weights.items():
                metadata = next(
                    (meta for category in COMMODITIES_UNIVERSE.values() 
                     for meta in category.values() if meta.symbol == asset),
                    AssetMetadata(asset, asset, AssetCategory.BENCHMARK, "#666666")
                )
                
                weight_data.append({
                    'Asset': asset,
                    'Weight': weight,
                    'Color': metadata.color
                })
            
            for item in sorted(weight_data, key=lambda x: x['Weight'], reverse=True):
                st.markdown(textwrap.dedent(f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: {item['Color']}; font-weight: 600;">{item['Asset']}</span>
                        <span style="font-weight: 600;">{item['Weight']:.1%}</span>
                    </div>
                    <div style="background: var(--light); height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: {item['Color']}; width: {item['Weight']*100}%; height: 100%;"></div>
                    </div>
                </div>
                """), unsafe_allow_html=True)
        
        # Calculate portfolio metrics
        portfolio_returns = returns_df @ weights
        portfolio_metrics = self.analytics.calculate_performance_metrics(portfolio_returns)
        st.session_state.portfolio_metrics = portfolio_metrics
        
        # Performance metrics
        st.markdown("### 📊 Portfolio Performance")
        
        # Create metrics grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        metric_configs = [
            ("Annual Return", "annual_return", "{:.2f}%", 
             "positive" if portfolio_metrics.get('annual_return', 0) > 0 else "negative",
             "Total annualized return"),
            
            ("Annual Volatility", "annual_volatility", "{:.2f}%", "neutral",
             "Annualized standard deviation of returns"),
            
            ("Sharpe Ratio", "sharpe_ratio", "{:.3f}",
             "positive" if portfolio_metrics.get('sharpe_ratio', 0) > 0 else "negative",
             "Risk-adjusted return (Sharpe ratio)"),
            
            ("Max Drawdown", "max_drawdown", "{:.2f}%", "negative",
             "Maximum peak-to-trough decline"),
            
            ("Sortino Ratio", "sortino_ratio", "{:.3f}",
             "positive" if portfolio_metrics.get('sortino_ratio', 0) > 0 else "negative",
             "Downside risk-adjusted return"),
            
            ("Calmar Ratio", "calmar_ratio", "{:.3f}",
             "positive" if portfolio_metrics.get('calmar_ratio', 0) > 0 else "negative",
             "Return to max drawdown ratio"),
            
            ("Win Rate", "win_rate", "{:.1f}%", "positive",
             "Percentage of positive return periods"),
            
            ("Profit Factor", "profit_factor", "{:.2f}",
             "positive" if portfolio_metrics.get('profit_factor', 0) > 1 else "negative",
             "Gross profit to gross loss ratio")
        ]
        
        # Display metrics in a grid
        cols = st.columns(4)
        for i, (label, key, fmt, color_class, tooltip) in enumerate(metric_configs):
            with cols[i % 4]:
                value = portfolio_metrics.get(key, 0)
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card custom-tooltip" data-tooltip="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk metrics
        st.markdown("### ⚖️ Risk Metrics")
        
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        risk_configs = [
            ("VaR (95%)", "var_95", "{:.2f}%", "negative", "Value at Risk (95% confidence)"),
            ("CVaR (95%)", "cvar_95", "{:.2f}%", "negative", "Conditional VaR (95% confidence)"),
            ("VaR (99%)", "var_99", "{:.2f}%", "negative", "Value at Risk (99% confidence)"),
            ("CVaR (99%)", "cvar_99", "{:.2f}%", "negative", "Conditional VaR (99% confidence)")
        ]
        
        risk_cols = st.columns(4)
        for i, (label, key, fmt, color_class, tooltip) in enumerate(risk_configs):
            with risk_cols[i]:
                value = portfolio_metrics.get(key, 0)
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card custom-tooltip" data-tooltip="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Portfolio Visualization")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Risk decomposition
            risk_contributions = self.analytics._calculate_risk_contributions(returns_df, weights)
            
            fig = self.visualizer.create_risk_decomposition(
                risk_contributions,
                "Risk Contribution Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Performance radar
            radar_metrics = {
                'Return': portfolio_metrics.get('annual_return', 0) / 50,
                'Risk': 1 - min(portfolio_metrics.get('annual_volatility', 0) / 50, 1),
                'Sharpe': min(portfolio_metrics.get('sharpe_ratio', 0) / 3, 1),
                'Sortino': min(portfolio_metrics.get('sortino_ratio', 0) / 3, 1),
                'Win Rate': portfolio_metrics.get('win_rate', 0) / 100
            }
            
            categories = list(radar_metrics.keys())
            values = list(radar_metrics.values())
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
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
                title=dict(text="Performance Profile", x=0.5),
                height=400,
                template=self.visualizer.template
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative returns
        st.markdown("### 📊 Cumulative Returns")
        
        portfolio_returns_series = pd.Series(
            portfolio_returns,
            index=returns_df.index,
            name="Portfolio"
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_returns_series.index,
            y=(1 + portfolio_returns_series).cumprod(),
            name="Portfolio",
            line=dict(color=self.visualizer.colors['primary'], width=3),
            fill='tozeroy',
            fillcolor=f"rgba({int(self.visualizer.colors['primary'][1:3], 16)}, "
                     f"{int(self.visualizer.colors['primary'][3:5], 16)}, "
                     f"{int(self.visualizer.colors['primary'][5:7], 16)}, 0.1)"
        ))
        
        # Add benchmarks if available
        for benchmark_symbol, benchmark_data in st.session_state.benchmark_data.items():
            if 'Returns' in benchmark_data.columns:
                benchmark_returns = benchmark_data['Returns'].dropna()
                aligned_idx = portfolio_returns_series.index.intersection(benchmark_returns.index)
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
        
        # Export options
        st.markdown("### 💾 Export Data")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📥 Download Portfolio Metrics", use_container_width=True, key="btn_download_portfolio_metrics"):
                metrics_df = pd.DataFrame.from_dict(portfolio_metrics, orient='index', columns=['Value'])
                csv = metrics_df.to_csv()
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("📊 Download Returns Data", use_container_width=True, key="btn_download_returns_data"):
                returns_data = pd.DataFrame({
                    'Date': portfolio_returns_series.index,
                    'Portfolio_Return': portfolio_returns_series.values,
                    'Cumulative_Return': (1 + portfolio_returns_series).cumprod().values
                })
                csv = returns_data.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_col3:
            if st.button("📋 Download Weights", use_container_width=True, key="btn_download_weights"):
                weights_data = pd.DataFrame.from_dict(
                    st.session_state.portfolio_weights,
                    orient='index',
                    columns=['Weight']
                )
                csv = weights_data.to_csv()
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def _display_advanced_analytics(self, config: AnalysisConfiguration):
        """Display advanced analytics"""
        st.markdown('<div class="section-header"><h2>⚡ Advanced Analytics</h2></div>', unsafe_allow_html=True)
        
        returns_data = st.session_state.get("returns_data", None)
        if returns_data is None or (isinstance(returns_data, pd.DataFrame) and returns_data.empty) or (isinstance(returns_data, dict) and len(returns_data) == 0):
            st.warning("⚠️ Please load data first")
            return
        
        # Create tabs for different advanced analyses
        adv_tabs = st.tabs(["GARCH Modeling", "Regime Detection", "Risk Analysis", "Monte Carlo"])
        
        with adv_tabs[0]:
            self._display_garch_analysis(config)
        
        with adv_tabs[1]:
            self._display_regime_analysis(config)
        
        with adv_tabs[2]:
            self._display_risk_analysis(config, key_ns="advanced")
        
        with adv_tabs[3]:
            self._display_monte_carlo(config)
    
    def _display_garch_analysis(self, config: AnalysisConfiguration):
        """Display GARCH analysis"""
        st.markdown("### 📊 GARCH Volatility Modeling")
        
        selected_asset = st.selectbox(
            "Select Asset for GARCH Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="garch_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p_max = st.slider("ARCH Order (p max)", 1, 5, 2, 1)
            
            with col2:
                q_max = st.slider("GARCH Order (q max)", 1, 5, 2, 1)
            
            with col3:
                distributions = st.multiselect(
                    "Distributions",
                    ["normal", "t", "skewt"],
                    default=["normal", "t"]
                )
            
            if st.button("🔍 Run GARCH Analysis", type="primary", use_container_width=True, key="btn_run_garch"):
                with st.spinner("Running GARCH analysis..."):
                    result = self.analytics.garch_analysis(
                        returns,
                        p_range=(1, p_max),
                        q_range=(1, q_max),
                        distributions=distributions
                    )
                    
                    if result.get('available', False):
                        st.session_state.garch_results[selected_asset] = result
                        st.success("✅ GARCH analysis completed!")
                        
                        # Display results
                        best_model = result['best_model']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Model", f"GARCH({best_model['p']},{best_model['q']})")
                        with col2:
                            st.metric("Distribution", best_model['distribution'])
                        with col3:
                            st.metric("AIC", f"{best_model['aic']:.1f}")
                        
                        # Plot volatility
                        if 'conditional_volatility' in best_model:
                            fig = self.visualizer.create_garch_volatility(
                                returns,
                                best_model['conditional_volatility'],
                                None,
                                f"{selected_asset} - GARCH Volatility"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show model parameters
                        with st.expander("📋 Model Parameters"):
                            params_df = pd.DataFrame.from_dict(
                                best_model['params'],
                                orient='index',
                                columns=['Value']
                            )
                            st.dataframe(params_df.style.format({'Value': '{:.6f}'}))
                    
                    else:
                        st.warning(f"⚠️ {result.get('message', 'GARCH analysis failed')}")
    
    def _display_regime_analysis(self, config: AnalysisConfiguration):
        """Display regime detection analysis"""
        st.markdown("### 🔄 Market Regime Detection")
        
        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="regime_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_regimes = st.slider("Number of Regimes", 2, 5, 3, 1)
            
            with col2:
                features = st.multiselect(
                    "Features",
                    ["returns", "volatility", "volume"],
                    default=["returns", "volatility"]
                )
            
            if st.button("🔍 Detect Regimes", type="primary", use_container_width=True, key="btn_detect_regimes"):
                with st.spinner("Detecting market regimes..."):
                    result = self.analytics.detect_regimes(
                        returns,
                        n_regimes=n_regimes,
                        features=features
                    )
                    
                    if result.get('available', False):
                        st.session_state.regime_results[selected_asset] = result
                        st.success("✅ Regime detection completed!")
                        
                        # Display regime statistics
                        if result.get('regime_stats'):
                            stats_df = pd.DataFrame(result['regime_stats'])
                            st.dataframe(
                                stats_df.style.format({
                                    'frequency': '{:.2f}%',
                                    'mean_return': '{:.4f}%',
                                    'volatility': '{:.2f}%',
                                    'sharpe': '{:.3f}',
                                    'var_95': '{:.2f}%'
                                })
                            )
                        
                        # Plot regimes
                        if selected_asset in st.session_state.asset_data:
                            price_data = st.session_state.asset_data[selected_asset]
                            price_col = 'Adj_Close' if 'Adj_Close' in price_data.columns else 'Close'
                            price = price_data[price_col]
                            
                            fig = self.visualizer.create_regime_chart(
                                price,
                                result['regimes'],
                                result.get('regime_labels', {}),
                                f"{selected_asset} - Market Regimes"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning(f"⚠️ {result.get('message', 'Regime detection failed')}")
    
    
    def _display_risk_analytics(self, config: AnalysisConfiguration):
        """Backward-compatible alias for _display_risk_analysis (older call sites)."""
        return self._display_risk_analysis(config, key_ns="risk")


    def _display_ewma_ratio_signal(self, config: AnalysisConfiguration):
        """EWMA(22) / (EWMA(33) + EWMA(99)) volatility ratio signal with BB + alarm zones."""
        st.markdown("### 📉 EWMA Volatility Ratio Signal (Institutional Risk Indicator)")

        returns_data = st.session_state.get("returns_data", None)
        if returns_data is None:
            st.info("Load data first to compute the EWMA ratio signal.")
            return

        # Normalize to DataFrame
        if isinstance(returns_data, pd.DataFrame):
            returns_df = returns_data.copy()
        elif isinstance(returns_data, dict):
            returns_df = pd.DataFrame(returns_data)
        else:
            returns_df = pd.DataFrame()

        if returns_df.empty or returns_df.shape[1] == 0:
            st.info("No returns series available. Load assets first.")
            return

        # Controls
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.2, 1.2, 1.2])
        with c1:
            selected_asset = st.selectbox(
                "Select Asset",
                options=list(returns_df.columns),
                index=0,
                key="ewma_ratio_asset_select"
            )
        with c2:
            annualize = st.checkbox("Annualize vol", value=False, key="ewma_ratio_annualize")
        with c3:
            span_fast = st.number_input("EWMA Fast (days)", min_value=5, max_value=252, value=22, step=1, key="ewma_ratio_span_fast")
        with c4:
            span_mid = st.number_input("EWMA Mid (days)", min_value=5, max_value=252, value=33, step=1, key="ewma_ratio_span_mid")
        with c5:
            span_slow = st.number_input("EWMA Slow (days)", min_value=10, max_value=756, value=99, step=1, key="ewma_ratio_span_slow")

        st.markdown("#### 📌 Bands & Alarm Zones")
        b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 1.2])
        with b1:
            bb_window = st.slider("Bollinger window", min_value=10, max_value=120, value=20, step=5, key="ewma_ratio_bb_window")
        with b2:
            bb_k = st.slider("Bollinger k", min_value=1.0, max_value=3.5, value=2.0, step=0.1, key="ewma_ratio_bb_k")
        with b3:
            green_max = st.slider("Green max", min_value=0.05, max_value=0.95, value=0.35, step=0.01, key="ewma_ratio_green_max")
        with b4:
            red_min = st.slider("Red min", min_value=0.05, max_value=0.95, value=0.55, step=0.01, key="ewma_ratio_red_min")

        if float(red_min) <= float(green_max):
            st.warning("Red min must be greater than Green max. Auto-adjusting.")
            red_min = float(green_max) + 0.01

        # Compute signal
        try:
            series = returns_df[selected_asset]
        except Exception:
            st.error("Selected asset series could not be loaded.")
            return

        sig_df = self.analytics.compute_ewma_volatility_ratio(
            returns=series.dropna(),
            span_fast=int(span_fast),
            span_mid=int(span_mid),
            span_slow=int(span_slow),
            annualize=bool(annualize)
        )

        if sig_df.empty or "EWMA_RATIO" not in sig_df.columns:
            st.info("Not enough data to compute EWMA ratio signal for this asset.")
            return

        ratio = sig_df["EWMA_RATIO"].dropna()
        latest = float(ratio.iloc[-1]) if not ratio.empty else float("nan")

        # Status
        if np.isnan(latest):
            status = "N/A"
            status_color = self.visualizer.colors.get("gray", "#6b7280")
        elif latest <= float(green_max):
            status = "GREEN (Normal)"
            status_color = self.visualizer.colors.get("success", "#10b981")
        elif latest >= float(red_min):
            status = "RED (High Risk)"
            status_color = self.visualizer.colors.get("danger", "#ef4444")
        else:
            status = "ORANGE (Watch)"
            status_color = self.visualizer.colors.get("warning", "#f59e0b")

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Ratio</div>
                <div class="metric-value">{latest:.4f}</div>
            </div>
            """), unsafe_allow_html=True)
        with k2:
            v22 = float(sig_df.iloc[-1].get(f"EWMA_VOL_{int(span_fast)}", np.nan))
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">EWMA Vol ({int(span_fast)})</div>
                <div class="metric-value">{(v22*100):.2f}</div>
            </div>
            """), unsafe_allow_html=True)
        with k3:
            v33 = float(sig_df.iloc[-1].get(f"EWMA_VOL_{int(span_mid)}", np.nan))
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">EWMA Vol ({int(span_mid)})</div>
                <div class="metric-value">{(v33*100):.2f}</div>
            </div>
            """), unsafe_allow_html=True)
        with k4:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Alarm Zone</div>
                <div class="metric-value" style="color:{status_color}">{status}</div>
            </div>
            """), unsafe_allow_html=True)

        # Chart
        fig = self.visualizer.create_ewma_ratio_signal_chart(
            ewma_df=sig_df,
            title=f"{selected_asset} | EWMA({int(span_fast)}) / (EWMA({int(span_mid)}) + EWMA({int(span_slow)})) Ratio",
            bb_window=int(bb_window),
            bb_k=float(bb_k),
            green_max=float(green_max),
            red_min=float(red_min),
            show_bollinger=True,
            show_threshold_lines=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostics tables
        st.markdown("#### 🔍 Diagnostics & Recent Values")
        tail_n = min(30, int(sig_df.shape[0]))
        show_df = sig_df.tail(tail_n).copy()
        show_df = show_df.rename(columns={
            "EWMA_RATIO": "EWMA_RATIO_SIGNAL"
        })
        st.dataframe(
            show_df.style.format(precision=6),
            use_container_width=True
        )

        # Simple alert note
        st.info(
            "Interpretation: rising ratio typically indicates short-term volatility dominance "
            "vs medium/long EWMA vols. Use zones + Bollinger breaks as a risk signal overlay."
        )



    def _display_risk_analysis(self, config: AnalysisConfiguration, key_ns: str = "risk"):
        """Display risk analysis"""
        st.markdown("### ⚠️ Risk Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Risk Analysis",
            options=list(st.session_state.returns_data.keys()),
            key=f"risk_asset_select_{key_ns}"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            # VaR Calculation
            st.markdown("#### Value at Risk (VaR)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_method = st.selectbox(
                    "VaR Method",
                    ["historical", "parametric", "modified"],
                    key=f"var_method_{key_ns}"
                )
            
            with col2:
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.99],
                    value=0.95,
                    key=f"var_confidence_{key_ns}"
                )
            
            with col3:
                if st.button("📊 Calculate VaR", type="primary", use_container_width=True, key=f"{key_ns}__btn_calc_var"):
                    with st.spinner("Calculating VaR..."):
                        var_result = self.analytics.calculate_var(
                            returns,
                            confidence_level=confidence_level,
                            method=var_method
                        )
                        
                        if var_result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    f"VaR ({confidence_level:.0%})",
                                    f"{var_result['var']:.2f}%",
                                    help=f"Value at Risk using {var_method} method"
                                )
                            with col2:
                                st.metric(
                                    f"CVaR ({confidence_level:.0%})",
                                    f"{var_result['cvar']:.2f}%",
                                    help="Expected Shortfall (Conditional VaR)"
                                )
            
            # Stress Testing
            st.markdown("#### 🧪 Stress Testing")
            
            if st.button("⚡ Run Stress Test", type="primary", use_container_width=True, key=f"{key_ns}__btn_stress_test"):
                with st.spinner("Running stress tests..."):
                    stress_results = self.analytics.stress_test(returns)
                    
                    if stress_results:
                        stress_df = pd.DataFrame.from_dict(stress_results, orient='index')
                        st.dataframe(
                            stress_df.style.format({
                                'shock': '{:.1f}%',
                                'shocked_return': '{:.2f}%',
                                'shocked_volatility': '{:.2f}%',
                                'loss': '{:.2f}',
                                'max_drawdown': '{:.2f}%',
                                'var_95': '{:.2f}%'
                            })
                        )

    def _display_monte_carlo(self, config: AnalysisConfiguration):
        """Display Monte Carlo simulation"""
        st.markdown("### 🎲 Monte Carlo Simulation")
        
        selected_asset = st.selectbox(
            "Select Asset for Simulation",
            options=list(st.session_state.returns_data.keys()),
            key="monte_carlo_asset"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_simulations = st.slider(
                    "Number of Simulations",
                    1000, 50000, 10000, 1000
                )
            
            with col2:
                n_days = st.slider(
                    "Time Horizon (days)",
                    30, 1000, 252, 30
                )
            
            if st.button("🚀 Run Simulation", type="primary", use_container_width=True, key="btn_run_monte_carlo"):
                with st.spinner(f"Running {n_simulations:,} simulations..."):
                    mc_result = self.analytics.monte_carlo_simulation(
                        returns,
                        n_simulations=n_simulations,
                        n_days=n_days
                    )
                    
                    if mc_result:
                        st.session_state.monte_carlo_results[selected_asset] = mc_result
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Expected Final Value",
                                f"${mc_result['mean_final_value']:.2f}",
                                help="Mean of final portfolio values"
                            )
                        
                        with col2:
                            st.metric(
                                "VaR (95%)",
                                f"${mc_result['var_95_final']:.2f}",
                                help="5th percentile of final values"
                            )
                        
                        with col3:
                            st.metric(
                                "Probability of Loss",
                                f"{mc_result['probability_loss']:.1f}%",
                                help="Probability of ending below initial value"
                            )
                        
                        with col4:
                            st.metric(
                                "Expected Max",
                                f"${mc_result['expected_max']:.2f}",
                                help="Expected maximum value during simulation"
                            )
                        
                        # Plot some sample paths
                        if 'paths' in mc_result:
                            fig = go.Figure()
                            
                            # Plot a subset of paths for clarity
                            n_sample_paths = min(50, n_simulations)
                            sample_paths = mc_result['paths'][:n_sample_paths]
                            
                            for i in range(n_sample_paths):
                                fig.add_trace(go.Scatter(
                                    x=list(range(n_days)),
                                    y=sample_paths[i],
                                    mode='lines',
                                    line=dict(width=1, color='rgba(100, 100, 100, 0.1)'),
                                    showlegend=False
                                ))
                            
                            # Plot mean path
                            mean_path = mc_result['paths'].mean(axis=0)
                            fig.add_trace(go.Scatter(
                                x=list(range(n_days)),
                                y=mean_path,
                                mode='lines',
                                line=dict(width=3, color=self.visualizer.colors['primary']),
                                name='Mean Path'
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_asset} - Monte Carlo Simulation Paths",
                                height=500,
                                template=self.visualizer.template,
                                xaxis_title="Days",
                                yaxis_title="Portfolio Value ($)",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
    
    def _display_analytics(self, config: AnalysisConfiguration):
        """Display general analytics"""
        st.markdown('<div class="section-header"><h2>📈 Advanced Analytics</h2></div>', unsafe_allow_html=True)
        
        returns_data = st.session_state.get("returns_data", None)
        if returns_data is None or (isinstance(returns_data, pd.DataFrame) and returns_data.empty) or (isinstance(returns_data, dict) and len(returns_data) == 0):
            st.warning("⚠️ Please load data first")
            return
        
        # Rolling statistics
        st.markdown("### 📊 Rolling Statistics")
        
        selected_asset = st.selectbox(
            "Select Asset",
            options=list(st.session_state.returns_data.keys()),
            key="rolling_stats_asset"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2 = st.columns(2)
            
            with col1:
                window = st.slider(
                    "Rolling Window (days)",
                    20, 252, 60, 20,
                    key="rolling_window"
                )
            
            with col2:
                stat_type = st.selectbox(
                    "Statistic",
                    ["Mean", "Volatility", "Sharpe", "Skewness", "Kurtosis"],
                    key="rolling_stat"
                )
            
            # Calculate rolling statistic
            if stat_type == "Mean":
                rolling_stat = returns.rolling(window).mean() * 252 * 100
                y_title = "Annual Return (%)"
            elif stat_type == "Volatility":
                rolling_stat = returns.rolling(window).std() * np.sqrt(252) * 100
                y_title = "Annual Volatility (%)"
            elif stat_type == "Sharpe":
                rolling_mean = returns.rolling(window).mean() * 252
                rolling_std = returns.rolling(window).std() * np.sqrt(252)
                rolling_stat = (rolling_mean - config.risk_free_rate) / rolling_std
                rolling_stat = rolling_stat.replace([np.inf, -np.inf], np.nan)
                y_title = "Sharpe Ratio"
            elif stat_type == "Skewness":
                rolling_stat = returns.rolling(window).skew()
                y_title = "Skewness"
            else:  # Kurtosis
                rolling_stat = returns.rolling(window).kurt()
                y_title = "Kurtosis"
            
            # Plot rolling statistic
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_stat.index,
                y=rolling_stat.values,
                name=f"Rolling {stat_type}",
                line=dict(width=2, color=self.visualizer.colors['primary']),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.visualizer.colors['primary'][1:3], 16)}, "
                         f"{int(self.visualizer.colors['primary'][3:5], 16)}, "
                         f"{int(self.visualizer.colors['primary'][5:7], 16)}, 0.2)"
            ))
            
            fig.update_layout(
                title=f"{selected_asset} - Rolling {stat_type} ({window}-day window)",
                height=400,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title=y_title,
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(st.session_state.returns_data) > 1:
            st.markdown("### 🔗 Advanced Correlation Analysis")
            
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            
            if not returns_df.empty and len(returns_df.columns) > 1:
                # Calculate rolling correlations
                selected_pair = st.selectbox(
                    "Select Asset Pair",
                    options=[
                        f"{col1} vs {col2}" 
                        for i, col1 in enumerate(returns_df.columns) 
                        for j, col2 in enumerate(returns_df.columns) 
                        if i < j
                    ],
                    key="corr_pair"
                )
                
                if selected_pair:
                    col1, col2 = selected_pair.split(" vs ")
                    
                    col1_returns = returns_df[col1]
                    col2_returns = returns_df[col2]
                    
                    # Calculate rolling correlation
                    rolling_corr = col1_returns.rolling(window=60).corr(col2_returns)
                    
                    # Plot rolling correlation
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        name=f"Rolling Correlation ({col1} vs {col2})",
                        line=dict(width=2, color=self.visualizer.colors['primary'])
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title=f"Rolling Correlation: {col1} vs {col2} (60-day window)",
                        height=400,
                        template=self.visualizer.template,
                        hovermode='x unified',
                        yaxis_title="Correlation",
                        xaxis_title="Date",
                        yaxis_range=[-1, 1]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    

    def _display_stress_testing(self, config: AnalysisConfiguration):
        """Display stress testing tab (required by run())."""
        st.markdown('<div class="section-header"><h2>🧪 Stress Testing</h2></div>', unsafe_allow_html=True)

        returns_src = st.session_state.get("returns_data", None)
        if returns_src is None:
            st.warning("⚠️ No return data available. Please load data first.")
            return

        # Normalize to DataFrame
        if isinstance(returns_src, pd.DataFrame):
            returns_df = returns_src.copy()
        elif isinstance(returns_src, dict):
            returns_df = pd.DataFrame(returns_src)
        else:
            returns_df = pd.DataFrame()

        if returns_df.empty or returns_df.shape[1] == 0:
            st.warning("⚠️ Insufficient data for stress testing.")
            return

        assets = returns_df.columns.tolist()

        # Controls
        c0, c1, c2 = st.columns([2, 1, 1])
        with c0:
            scope = st.radio(
                "Scope",
                ["Portfolio", "Single Asset", "All Assets (Heatmap)"],
                horizontal=True,
                key="stress_scope"
            )
        with c1:
            notional = st.number_input(
                "Notional (base currency)",
                min_value=1000.0,
                value=1_000_000.0,
                step=50_000.0,
                format="%.0f",
                key="stress_notional"
            )
        with c2:
            conf = st.select_slider(
                "VaR Confidence",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                key="stress_var_conf"
            )

        shock_options = [-1, -2, -5, -10, -15, -20, -30]
        shocks_pct = st.multiselect(
            "Shock scenarios (%)",
            options=shock_options,
            default=[-1, -2, -5, -10],
            key="stress_shocks_pct"
        )
        if not shocks_pct:
            st.info("Select at least one shock scenario.")
            return

        shocks = [float(s) / 100.0 for s in shocks_pct]  # decimals (negative)

        # Helpers
        def _safe_series(x) -> pd.Series:
            s = pd.to_numeric(x, errors="coerce").dropna()
            return s

        def _var_cvar(series: pd.Series, cl: float) -> Tuple[float, float]:
            r = _safe_series(series)
            if r.empty:
                return (np.nan, np.nan)
            # Historical VaR/CVaR (%)
            q = np.quantile(r, 1 - cl)
            cvar = r[r <= q].mean() if (r <= q).any() else np.nan
            return (float(q) * 100.0, float(cvar) * 100.0)

        run_btn = st.button("⚡ Run Stress Testing", type="primary", use_container_width=True, key="btn_run_stress_tab")
        if not run_btn:
            st.caption("Configure scenarios above, then click **Run Stress Testing**.")
            return

        with st.spinner("Running stress testing..."):
            # --- Portfolio scope
            if scope == "Portfolio":
                # Try to use current weights; fallback equal weight
                w = st.session_state.get("portfolio_weights", {}) or {}
                weights = {}
                if isinstance(w, dict) and len(w) > 0:
                    for a in assets:
                        if a in w:
                            try:
                                weights[a] = float(w[a])
                            except Exception:
                                pass
                if len(weights) != len(assets):
                    weights = {a: 1.0 / len(assets) for a in assets}

                # Normalize weights
                ssum = sum(weights.values())
                if ssum <= 0:
                    weights = {a: 1.0 / len(assets) for a in assets}
                    ssum = 1.0
                weights = {a: v / ssum for a, v in weights.items()}

                port_rets = (returns_df[assets].apply(pd.to_numeric, errors="coerce") * pd.Series(weights)).sum(axis=1)
                port_rets = _safe_series(port_rets)

                if port_rets.empty:
                    st.warning("⚠️ Portfolio returns are empty after cleaning.")
                    return

                var_val, cvar_val = _var_cvar(port_rets, float(conf))
                ann_vol = float(port_rets.std() * np.sqrt(config.annual_trading_days) * 100.0) if port_rets.std() == port_rets.std() else np.nan
                ann_ret = float(port_rets.mean() * config.annual_trading_days * 100.0) if port_rets.mean() == port_rets.mean() else np.nan

                st.markdown("#### Portfolio baseline (historical)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Annualized Return", f"{ann_ret:.2f}%")
                m2.metric("Annualized Volatility", f"{ann_vol:.2f}%")
                m3.metric(f"VaR / CVaR ({conf:.0%})", f"{var_val:.2f}% / {cvar_val:.2f}%")

                # Shock P&L table
                rows = []
                for sh in shocks:
                    pnl = -sh * float(notional)  # positive loss for negative shock
                    rows.append({
                        "Shock (%)": sh * 100.0,
                        "Notional Loss": pnl,
                        "Post-shock Notional": float(notional) * (1.0 + sh)
                    })
                df_out = pd.DataFrame(rows)
                st.markdown("#### Portfolio scenario losses (instantaneous shock)")
                st.dataframe(
                    df_out.style.format({"Shock (%)": "{:.1f}", "Notional Loss": "{:,.0f}", "Post-shock Notional": "{:,.0f}"}),
                    use_container_width=True,
                    hide_index=True
                )

            # --- Single asset scope
            elif scope == "Single Asset":
                sel = st.selectbox("Select Asset", options=assets, key="stress_single_asset")
                s = _safe_series(returns_df[sel])

                if s.empty:
                    st.warning("⚠️ Selected asset has no usable return data.")
                    return

                var_val, cvar_val = _var_cvar(s, float(conf))
                ann_vol = float(s.std() * np.sqrt(config.annual_trading_days) * 100.0) if s.std() == s.std() else np.nan
                ann_ret = float(s.mean() * config.annual_trading_days * 100.0) if s.mean() == s.mean() else np.nan

                st.markdown("#### Baseline (historical)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Annualized Return", f"{ann_ret:.2f}%")
                m2.metric("Annualized Volatility", f"{ann_vol:.2f}%")
                m3.metric(f"VaR / CVaR ({conf:.0%})", f"{var_val:.2f}% / {cvar_val:.2f}%")

                rows = []
                for sh in shocks:
                    pnl = -sh * float(notional)
                    rows.append({
                        "Shock (%)": sh * 100.0,
                        "Notional Loss": pnl,
                        "Post-shock Notional": float(notional) * (1.0 + sh)
                    })
                df_out = pd.DataFrame(rows)

                st.markdown("#### Scenario losses (instantaneous shock)")
                st.dataframe(
                    df_out.style.format({"Shock (%)": "{:.1f}", "Notional Loss": "{:,.0f}", "Post-shock Notional": "{:,.0f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                # Optional: show analytics-based stress table (legacy)
                with st.expander("Legacy stress metrics (distribution shocked by adding shock to all days)", expanded=False):
                    legacy = {}
                    for sh in shocks:
                        legacy.update(self.analytics.stress_test(s, scenarios=[sh]))
                    if legacy:
                        ldf = pd.DataFrame.from_dict(legacy, orient="index").reset_index().rename(columns={"index": "Scenario"})
                        st.dataframe(ldf, use_container_width=True)

            # --- All assets heatmap
            else:
                # Heatmap of notional losses by asset & shock
                loss_mat = pd.DataFrame(
                    index=assets,
                    columns=[f"{sh*100:.0f}%" for sh in shocks],
                    dtype=float
                )
                for a in assets:
                    for sh in shocks:
                        loss_mat.loc[a, f"{sh*100:.0f}%"] = -sh * float(notional)

                fig = go.Figure(
                    data=go.Heatmap(
                        z=loss_mat.values,
                        x=loss_mat.columns,
                        y=loss_mat.index,
                        hovertemplate="<b>%{y}</b><br>Shock: %{x}<br>Loss: %{z:,.0f}<extra></extra>",
                        colorbar=dict(title=dict(text="Loss"))
                    )
                )
                fig.update_layout(
                    title=dict(text="Stress Loss Heatmap (Instantaneous Shock)", x=0.5),
                    height=max(450, 28 * len(assets)),
                    template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
                    margin=dict(t=70, l=40, r=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption("Losses are computed as: **Loss = -Shock × Notional** (shock is negative).")

    def _display_reporting(self, config: AnalysisConfiguration):
        """Compatibility wrapper: run() calls _display_reporting; older code uses _display_reports."""
        return self._display_reports(config)


    def _display_reports(self, config: AnalysisConfiguration):
        """Display reporting interface"""
        st.markdown('<div class="section-header"><h2>📋 Professional Reports</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first to generate reports")
            return
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                [
                    "Portfolio Summary",
                    "Risk Analysis", 
                    "Performance Attribution",
                    "Comprehensive Analysis",
                    "Executive Summary"
                ],
                key="report_type"
            )
        
        with col2:
            report_format = st.selectbox(
                "Format",
                ["HTML", "PDF", "Markdown", "JSON"],
                key="report_format"
            )
        
        # Report options
        st.markdown("### 📊 Report Options")
        
        options_cols = st.columns(4)
        
        with options_cols[0]:
            include_charts = st.checkbox("Include Charts", value=True, key="include_charts")
        
        with options_cols[1]:
            include_tables = st.checkbox("Include Tables", value=True, key="include_tables")
        
        with options_cols[2]:
            include_metrics = st.checkbox("Include Metrics", value=True, key="include_metrics")
        
        with options_cols[3]:
            include_details = st.checkbox("Include Details", value=True, key="include_details")
        
        # Generate report
        if st.button("📄 Generate Report", type="primary", use_container_width=True, key="btn_generate_report"):
            with st.spinner("Generating professional report..."):
                try:
                    # Prepare report data
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'report_type': report_type,
                        'assets': st.session_state.selected_assets,
                        'benchmarks': st.session_state.selected_benchmarks,
                        'portfolio_weights': st.session_state.portfolio_weights,
                        'portfolio_metrics': st.session_state.portfolio_metrics,
                        'config': {
                            'include_charts': include_charts,
                            'include_tables': include_tables,
                            'include_metrics': include_metrics,
                            'include_details': include_details
                        }
                    }
                    
                    # Generate report based on format
                    if report_format == "HTML":
                        report_content = self._generate_html_report(report_data)
                        file_name = f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        mime_type = "text/html"
                    
                    elif report_format == "Markdown":
                        report_content = self._generate_markdown_report(report_data)
                        file_name = f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        mime_type = "text/markdown"
                    
                    elif report_format == "JSON":
                        report_content = json.dumps(report_data, indent=2)
                        file_name = f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        mime_type = "application/json"
                    
                    else:  # PDF placeholder
                        report_content = self._generate_html_report(report_data)
                        file_name = f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        mime_type = "text/html"
                        st.info("📝 PDF generation requires additional libraries. Downloading HTML version instead.")
                    
                    # Display preview for HTML
                    if report_format == "HTML":
                        st.markdown("### 📊 Report Preview")
                        st.components.v1.html(report_content, height=600, scrolling=True)
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download {report_format.upper()} Report",
                        data=report_content,
                        file_name=file_name,
                        mime=mime_type
                    )
                    
                except Exception as e:
                    st.error(f"❌ Failed to generate report: {str(e)}")
                    self._log_error(e, "Report generation")
        
        # Quick snapshot
        st.markdown("### 📸 Quick Snapshot")
        
        if st.button("Take Snapshot", use_container_width=True, key="take_snapshot"):
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'platform_version': 'v6.0',
                'assets_loaded': st.session_state.selected_assets,
                'benchmarks_loaded': st.session_state.selected_benchmarks,
                'data_points': self._safe_data_points(st.session_state.get("returns_data", None)),
                'portfolio_weights': st.session_state.portfolio_weights,
                'portfolio_metrics_summary': {
                    k: v for k, v in st.session_state.portfolio_metrics.items()
                    if k in ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
                },
                'system_info': {
                    'dependencies': {dep: info.get('available', False) 
                                   for dep, info in dep_manager.dependencies.items()},
                    'python_version': os.sys.version,
                    'streamlit_version': st.__version__
                }
            }
            
            st.json(snapshot, expanded=False)
            
            st.download_button(
                label="📥 Download JSON Snapshot",
                data=json.dumps(snapshot, indent=2),
                file_name=f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Institutional Commodities Analytics Report</title>
            <style>
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    line-height: 1.6;
                    color: #1f2937;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px;
                    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
                }}
                
                .header {{
                    background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
                    color: white;
                    padding: 50px;
                    border-radius: 20px;
                    margin-bottom: 40px;
                    text-align: center;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                    background-size: 30px 30px;
                    opacity: 0.3;
                    animation: float 20s linear infinite;
                }}
                
                @keyframes float {{
                    0% {{ transform: translate(0, 0) rotate(0deg); }}
                    100% {{ transform: translate(-30px, -30px) rotate(360deg); }}
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 3em;
                    font-weight: 800;
                    position: relative;
                    z-index: 1;
                }}
                
                .header p {{
                    margin: 15px 0 0;
                    opacity: 0.95;
                    font-size: 1.3em;
                    position: relative;
                    z-index: 1;
                }}
                
                .section {{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
                    border-left: 5px solid #1a2980;
                }}
                
                .section-title {{
                    color: #1a2980;
                    border-bottom: 3px solid #26d0ce;
                    padding-bottom: 15px;
                    margin-bottom: 25px;
                    font-size: 1.8em;
                    font-weight: 700;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 25px;
                    margin: 30px 0;
                }}
                
                .metric-card {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    border-left: 4px solid #1a2980;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 30px rgba(0,0,0,0.12);
                }}
                
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: 800;
                    color: #1a2980;
                    margin: 15px 0;
                    background: linear-gradient(135deg, #1a2980, #26d0ce);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: #6c757d;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                    font-weight: 600;
                }}
                
                .positive {{ color: #10b981; }}
                .negative {{ color: #ef4444; }}
                .neutral {{ color: #f59e0b; }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                }}
                
                th {{
                    background: linear-gradient(135deg, #1a2980, #26d0ce);
                    color: white;
                    padding: 18px 20px;
                    text-align: left;
                    font-weight: 600;
                    font-size: 0.95em;
                    text-transform: uppercase;
                    letter-spacing: 0.8px;
                }}
                
                td {{
                    padding: 16px 20px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                
                tr:hover {{
                    background-color: #f9fafb;
                }}
                
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                }}
                
                .disclaimer {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 25px;
                    border-radius: 8px;
                    margin-top: 40px;
                    font-size: 0.9em;
                    color: #856404;
                    line-height: 1.8;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 30px;
                    border-top: 2px solid #e5e7eb;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
                
                @media print {{
                    body {{
                        background: white;
                        padding: 20px;
                    }}
                    
                    .header {{
                        background: #1a2980;
                        box-shadow: none;
                    }}
                    
                    .metric-card {{
                        break-inside: avoid;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏛️ Institutional Commodities Analytics Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report Type: {report_data['report_type']}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Assets</div>
                        <div class="metric-value">{len(report_data['assets'])}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Annual Return</div>
                        <div class="metric-value {'positive' if report_data['portfolio_metrics'].get('annual_return', 0) > 0 else 'negative'}">
                            {report_data['portfolio_metrics'].get('annual_return', 0):.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{report_data['portfolio_metrics'].get('sharpe_ratio', 0):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{report_data['portfolio_metrics'].get('max_drawdown', 0):.2f}%</div>
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
                            <th>Sharpe Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
                            f'''<tr>
                                <td><strong>{asset}</strong></td>
                                <td>{weight:.1%}</td>
                                <td class="{'positive' if report_data['portfolio_metrics'].get('annual_return', 0) > 0 else 'negative'}">
                                    {report_data['portfolio_metrics'].get('annual_return', 0):.2f}%
                                </td>
                                <td>{report_data['portfolio_metrics'].get('annual_volatility', 0):.2f}%</td>
                                <td>{report_data['portfolio_metrics'].get('sharpe_ratio', 0):.3f}</td>
                            </tr>'''
                            for asset, weight in report_data['portfolio_weights'].items()
                        ])}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">⚖️ Risk Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Annual Volatility</div>
                        <div class="metric-value">{report_data['portfolio_metrics'].get('annual_volatility', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">VaR (95%)</div>
                        <div class="metric-value negative">{report_data['portfolio_metrics'].get('var_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">CVaR (95%)</div>
                        <div class="metric-value negative">{report_data['portfolio_metrics'].get('cvar_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{report_data['portfolio_metrics'].get('sortino_ratio', 0):.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This report is generated for informational purposes only by the 
                Institutional Commodities Analytics Platform v6.0. Past performance is not indicative of future results. 
                The information provided does not constitute investment advice and should not be relied upon for making 
                investment decisions. Consult with a qualified financial advisor before making investment decisions. 
                Data source: Yahoo Finance. Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
            </div>
            
            <div class="footer">
                <p>© {datetime.now().year} Institutional Commodities Analytics Platform v6.0</p>
                <p>Confidential - For Institutional Use Only</p>
                <p>Report ID: {hashlib.md5(str(report_data).encode()).hexdigest()[:16]}</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        markdown = f"""
# 🏛️ Institutional Commodities Analytics Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_data['report_type']}
**Report ID:** {hashlib.md5(str(report_data).encode()).hexdigest()[:16]}

## 📊 Executive Summary

- **Total Assets:** {len(report_data['assets'])}
- **Annual Return:** {report_data['portfolio_metrics'].get('annual_return', 0):.2f}%
- **Sharpe Ratio:** {report_data['portfolio_metrics'].get('sharpe_ratio', 0):.3f}
- **Max Drawdown:** {report_data['portfolio_metrics'].get('max_drawdown', 0):.2f}%
- **Annual Volatility:** {report_data['portfolio_metrics'].get('annual_volatility', 0):.2f}%

## 📈 Portfolio Composition

| Asset | Weight | Annual Return | Volatility | Sharpe Ratio |
|-------|--------|---------------|------------|--------------|
"""
        
        for asset, weight in report_data['portfolio_weights'].items():
            markdown += f"| {asset} | {weight:.1%} | {report_data['portfolio_metrics'].get('annual_return', 0):.2f}% | {report_data['portfolio_metrics'].get('annual_volatility', 0):.2f}% | {report_data['portfolio_metrics'].get('sharpe_ratio', 0):.3f} |\n"
        
        markdown += """
## ⚖️ Risk Metrics

- **VaR (95%):** {:.2f}%
- **CVaR (95%):** {:.2f}%
- **VaR (99%):** {:.2f}%
- **CVaR (99%):** {:.2f}%
- **Sortino Ratio:** {:.3f}
- **Calmar Ratio:** {:.3f}
- **Win Rate:** {:.1f}%
- **Profit Factor:** {:.2f}

## 📋 System Information

- **Platform Version:** v6.0
- **Assets Loaded:** {}
- **Benchmarks Loaded:** {}
- **Data Points:** {:,}

## ⚠️ Disclaimer

This report is generated for informational purposes only by the Institutional Commodities Analytics Platform v6.0. 
Past performance is not indicative of future results. The information provided does not constitute investment advice 
and should not be relied upon for making investment decisions. Consult with a qualified financial advisor before 
making investment decisions. Data source: Yahoo Finance.

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Confidential - For Institutional Use Only*
""".format(
            report_data['portfolio_metrics'].get('var_95', 0),
            report_data['portfolio_metrics'].get('cvar_95', 0),
            report_data['portfolio_metrics'].get('var_99', 0),
            report_data['portfolio_metrics'].get('cvar_99', 0),
            report_data['portfolio_metrics'].get('sortino_ratio', 0),
            report_data['portfolio_metrics'].get('calmar_ratio', 0),
            report_data['portfolio_metrics'].get('win_rate', 0),
            report_data['portfolio_metrics'].get('profit_factor', 0),
            len(report_data['assets']),
            len(report_data['benchmarks']),
            self._safe_data_points(st.session_state.get("returns_data", None))
        )
        
        return markdown
    
def _display_portfolio_lab(self, config: AnalysisConfiguration):
    """Portfolio Lab (PyPortfolioOpt + manual portfolio builder + risk decomposition).

    This tab is **additive** (does not replace existing Portfolio tab). It provides:
    - PyPortfolioOpt strategies (EF: max Sharpe, min vol, efficient return/risk, L2 reg, etc.)
    - Manual portfolio builder via sliders (user-defined weights)
    - Comparative portfolio analysis (saved portfolios)
    - Risk contribution decomposition (covariance + PCA factors)
    - VaR / CVaR(ES) + Relative VaR vs benchmark (Historical, Parametric, Monte Carlo)
    - PCA drivers for EWMA & GARCH volatility proxies
    """
    st.markdown('<div class="section-header"><h2>🧰 Portfolio Lab (PyPortfolioOpt)</h2></div>', unsafe_allow_html=True)

    # Pull aligned returns from session
    returns_df = st.session_state.get("returns_data", pd.DataFrame())
    if returns_df is None or not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        st.info("No returns matrix found. Please load data from the sidebar.")
        return

    # Build prices DataFrame from asset_data (Adj_Close preferred)
    asset_data = st.session_state.get("asset_data", {})
    prices_df = pd.DataFrame()
    if isinstance(asset_data, dict) and asset_data:
        series_map = {}
        for sym, df in asset_data.items():
            try:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    col = "Adj_Close" if "Adj_Close" in df.columns else ("Close" if "Close" in df.columns else None)
                    if col:
                        s = pd.to_numeric(df[col], errors="coerce")
                        s.name = sym
                        series_map[sym] = s
            except Exception:
                continue
        if series_map:
            prices_df = pd.concat(series_map.values(), axis=1).sort_index()

    # Benchmarks (optional)
    bench_df = st.session_state.get("benchmark_returns_data", pd.DataFrame())
    bench_dict = {}
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        for c in bench_df.columns:
            s = pd.to_numeric(bench_df[c], errors="coerce").dropna()
            if not s.empty:
                s.name = c
                bench_dict[c] = s

    # Delegate the whole suite to the patch module (merged below)
    try:
        render_portfolio_lab_suite(
            prices_df=prices_df,
            returns_df=returns_df,
            benchmark_returns=bench_dict,
            key_ns="portfolio_lab",
        )
    except NameError:
        st.error("Portfolio Lab patch is not available (render_portfolio_lab_suite missing). Please ensure the merged file is used.")
    except Exception as e:
        st.error(f"Portfolio Lab failed: {e}")
        st.code(traceback.format_exc())

    def _display_settings(self, config: AnalysisConfiguration):
        """Display settings and system information"""
        st.markdown('<div class="section-header"><h2>⚙️ Settings & System Info</h2></div>', unsafe_allow_html=True)
        
        # Platform information
        st.markdown("### 🏛️ Platform Information")
        
        info_cols = st.columns(3)
        
        with info_cols[0]:
            st.metric("Platform Version", "v6.0")
        
        with info_cols[1]:
            st.metric("Python Version", os.sys.version.split()[0])
        
        with info_cols[2]:
            st.metric("Streamlit Version", st.__version__)
        
        # Dependencies status
        st.markdown("### 📦 Dependencies Status")
        
        deps_cols = st.columns(3)
        deps_list = list(dep_manager.dependencies.items())
        
        for i, (dep_name, dep_info) in enumerate(deps_list):
            with deps_cols[i % 3]:
                status = "🟢 Available" if dep_info.get('available', False) else "🔴 Not Available"
                st.markdown(f"**{dep_name}:** {status}")
        
        # System configuration
        st.markdown("### ⚙️ System Configuration")
        
        with st.expander("View Configuration Details"):
            config_dict = {
                'risk_free_rate': f"{config.risk_free_rate:.2%}",
                'annual_trading_days': config.annual_trading_days,
                'confidence_levels': [f"{cl:.0%}" for cl in config.confidence_levels],
                'backtest_window': config.backtest_window,
                'optimization_method': config.optimization_method
            }
            
            st.json(config_dict)
        
        # Performance monitoring
        st.markdown("### 📊 Performance Monitoring")
        
        runtime = datetime.now() - self.start_time
        assets_loaded = len(st.session_state.asset_data)
        data_points = self._safe_data_points(st.session_state.get("returns_data", None))
        
        perf_cols = st.columns(4)
        
        with perf_cols[0]:
            st.metric("Runtime", f"{runtime.total_seconds():.0f}s")
        
        with perf_cols[1]:
            st.metric("Assets Loaded", assets_loaded)
        
        with perf_cols[2]:
            st.metric("Data Points", f"{data_points:,}")
        
        with perf_cols[3]:
            st.metric("Memory Usage", "Optimized")
        
        # Error log
        if st.session_state.error_log:
            st.markdown("### ⚠️ Error Log")
            
            with st.expander("View Error Log"):
                for error in st.session_state.error_log[-5:]:  # Show last 5 errors
                    st.markdown(f"""
                    **{error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}** - {error['context']}
                    ```
                    {error['error'][:200]}
                    ```
                    """)
        
        # Reset options
        st.markdown("### 🔄 Reset Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear Cache", use_container_width=True, key="btn_clear_cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache cleared successfully!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Application", use_container_width=True, type="secondary", key="btn_reset_application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Export configuration
        st.markdown("### 💾 Export Configuration")
        
        if st.button("📥 Export Configuration", use_container_width=True, key="btn_export_configuration"):
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'selected_assets': st.session_state.selected_assets,
                'selected_benchmarks': st.session_state.selected_benchmarks,
                'portfolio_weights': st.session_state.portfolio_weights,
                'analysis_config': {
                    'start_date': config.start_date.isoformat(),
                    'end_date': config.end_date.isoformat(),
                    'risk_free_rate': config.risk_free_rate,
                    'confidence_levels': config.confidence_levels,
                    'optimization_method': config.optimization_method
                }
            }
            
            config_json = json.dumps(config_data, indent=2)
            
            st.download_button(
                label="Download JSON Configuration",
                data=config_json,
                file_name=f"ica_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # =============================================================================
    # INSTITUTIONAL RELATIVE METRICS TABS (Tracking Error / Rolling Beta / Relative Risk)
    # =============================================================================

    @staticmethod
    def _to_returns_df(returns_data: Any) -> pd.DataFrame:
        """Normalize returns input (DataFrame or dict) into a DataFrame."""
        if returns_data is None:
            return pd.DataFrame()
        if isinstance(returns_data, pd.DataFrame):
            return returns_data.copy()
        if isinstance(returns_data, dict):
            try:
                return pd.DataFrame(returns_data).copy()
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def _safe_align_pair(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two series on the same index and drop missing."""
        if a is None or b is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        df = pd.concat([a, b], axis=1).dropna(how="any")
        if df.shape[1] != 2 or df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return df.iloc[:, 0], df.iloc[:, 1]

    def _get_scope_returns(
        self,
        returns_df: pd.DataFrame,
        scope: str,
        asset: Optional[str] = None,
    ) -> Tuple[pd.Series, str]:
        """Return series and label for 'Asset' or 'Portfolio' scope."""
        if returns_df is None or returns_df.empty:
            return pd.Series(dtype=float), "N/A"

        cols = list(returns_df.columns)
        scope = str(scope).strip()

        if scope == "Portfolio":
            # Use current weights if present; otherwise equal-weight across available assets
            weights = st.session_state.get("portfolio_weights", None)
            if isinstance(weights, dict) and len(weights) > 0:
                w = pd.Series(weights, dtype=float)
                w = w.reindex(cols).fillna(0.0)
                if float(w.abs().sum()) <= 0:
                    w = pd.Series(1.0 / len(cols), index=cols)
                else:
                    if float(w.sum()) != 0:
                        w = w / float(w.sum())
            else:
                w = pd.Series(1.0 / len(cols), index=cols)

            port = (returns_df[cols].mul(w, axis=1)).sum(axis=1)
            return port.dropna(), "Portfolio"

        # Asset scope
        if asset is None:
            asset = cols[0]
        if asset not in returns_df.columns:
            asset = cols[0]
        return returns_df[asset].dropna(), str(asset)

    def _select_benchmark_series(self, bench_returns_df: pd.DataFrame, key_ns: str = "") -> Tuple[pd.Series, str]:
        """Select benchmark series from loaded benchmark returns."""
        if bench_returns_df is None or bench_returns_df.empty:
            return pd.Series(dtype=float), "N/A"

        bcols = list(bench_returns_df.columns)
        default_idx = 0
        selected = st.selectbox(
            "Benchmark",
            bcols,
            index=default_idx,
            key=f"{key_ns}rel_benchmark_selectbox",
            help="Benchmark used for relative metrics (Tracking Error / Beta / Relative VaR).",
        )
        return bench_returns_df[selected].dropna(), str(selected)

    @staticmethod
    def _add_zone_bands(fig: go.Figure, x0, x1, bands: List[Tuple[float, float, str, str]]) -> None:
        """Add colored band rectangles to a Plotly figure.

        bands: list of (ymin, ymax, fillcolor, name)
        """
        for ymin, ymax, fillcolor, name in bands:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=ymin,
                y1=ymax,
                fillcolor=fillcolor,
                opacity=0.18,
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=x0,
                y=(ymin + ymax) / 2.0,
                xref="x",
                yref="y",
                text=name,
                showarrow=False,
                xanchor="left",
                font=dict(size=10),
                opacity=0.7,
            )

    def _display_tracking_error(self, config: AnalysisConfiguration):
        """Interactive Tracking Error analytics with institutional band zones."""
        st.markdown("### 🎯 Tracking Error (Institutional Band Monitoring)")

        returns_df = self._to_returns_df(st.session_state.get("returns_data", None))
        bench_returns_df = self._to_returns_df(st.session_state.get("benchmark_returns_data", None))

        if returns_df.empty:
            st.info("Load data first to compute Tracking Error.")
            return
        if bench_returns_df.empty:
            st.warning("No benchmark returns available. Please select at least one benchmark in the sidebar and reload data.")
            return

        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.1])
        with c1:
            scope = st.selectbox("Scope", ["Asset", "Portfolio"], index=0, key="te_scope")
        with c2:
            selected_asset = None
            if scope == "Asset":
                selected_asset = st.selectbox("Asset", list(returns_df.columns), index=0, key="te_asset")
        with c3:
            window = st.number_input("Rolling window (days)", min_value=20, max_value=252, value=int(getattr(config, "rolling_window", 60)), step=5, key="te_window")
        with c4:
            annual_days = st.number_input("Annualization days", min_value=200, max_value=365, value=int(getattr(config, "annual_trading_days", 252)), step=1, key="te_annual_days")

        st.markdown("#### 📌 Policy Bands")
        b1, b2, b3 = st.columns([1.2, 1.2, 1.2])
        with b1:
            te_green = st.number_input("Green max (annual TE)", min_value=0.0, max_value=1.0, value=0.04, step=0.005, format="%.3f", key="te_green")
        with b2:
            te_orange = st.number_input("Orange max (annual TE)", min_value=0.0, max_value=2.0, value=0.08, step=0.005, format="%.3f", key="te_orange")
        with b3:
            show_week_table = st.checkbox("Show weekly monitoring table", value=True, key="te_show_week_table")

        # Select returns
        scope_returns, scope_label = self._get_scope_returns(returns_df, scope=scope, asset=selected_asset)
        bench_series, bench_label = self._select_benchmark_series(bench_returns_df, key_ns="te_")

        a, b = self._safe_align_pair(scope_returns, bench_series)
        if a.empty or b.empty:
            st.warning("Insufficient overlap between scope returns and benchmark returns.")
            return

        active = (a - b).dropna()
        if active.empty or len(active) < int(window):
            st.warning("Not enough data points to compute rolling Tracking Error.")
            return

        te_annual = float(active.std(ddof=1) * math.sqrt(float(annual_days)))
        te_roll = active.rolling(int(window)).std(ddof=1) * math.sqrt(float(annual_days))

        # Zone label
        if te_annual <= float(te_green):
            zone = "GREEN"
        elif te_annual <= float(te_orange):
            zone = "ORANGE"
        else:
            zone = "RED"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scope", scope_label)
        k2.metric("Benchmark", bench_label)
        k3.metric("Current TE (annualized)", f"{100*te_annual:.2f}%")
        k4.metric("Policy Zone", zone)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=te_roll.index,
            y=te_roll.values,
            mode="lines",
            name=f"Rolling TE ({int(window)}d, annualized)",
        ))

        # Add policy band overlays
        x0 = te_roll.index.min()
        x1 = te_roll.index.max()
        bands = [
            (0.0, float(te_green), "rgba(0, 200, 0, 0.35)", "GREEN"),
            (float(te_green), float(te_orange), "rgba(255, 165, 0, 0.35)", "ORANGE"),
            (float(te_orange), float(max(float(te_orange)*1.5, float(te_roll.max())*1.05)), "rgba(220, 20, 60, 0.35)", "RED"),
        ]
        try:
            self._add_zone_bands(fig, x0, x1, bands)
        except Exception:
            pass

        # Policy lines
        fig.add_hline(y=float(te_green), line_dash="dash", annotation_text="Green max", opacity=0.6)
        fig.add_hline(y=float(te_orange), line_dash="dash", annotation_text="Orange max", opacity=0.6)

        fig.update_layout(
            title=dict(text=f"Tracking Error vs {bench_label} — {scope_label}", x=0.5),
            xaxis_title="Date",
            yaxis_title="Tracking Error (annualized)",
            height=520,
            template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
            margin=dict(t=60, l=40, r=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Weekly monitoring table
        if show_week_table:
            st.markdown("#### 📅 Weekly Monitoring (Rolling TE snapshots)")
            try:
                # Week-ending series
                te_week = te_roll.resample("W-FRI").last().dropna()
                last_n = st.slider("Weeks to display", min_value=8, max_value=104, value=26, step=2, key="te_weeks_n")
                te_week = te_week.tail(int(last_n))
                if te_week.empty:
                    st.info("No weekly points available yet.")
                else:
                    def _zone(v: float) -> str:
                        if v <= float(te_green):
                            return "GREEN"
                        if v <= float(te_orange):
                            return "ORANGE"
                        return "RED"

                    out = pd.DataFrame({
                        "Week End": te_week.index.date,
                        "Tracking Error (annualized)": te_week.values,
                    })
                    out["Tracking Error %"] = out["Tracking Error (annualized)"].astype(float) * 100.0
                    out["Zone"] = out["Tracking Error (annualized)"].astype(float).apply(_zone)
                    out = out[["Week End", "Tracking Error %", "Zone"]]

                    # Styler (works in Streamlit)
                    def _style_zone(s):
                        z = str(s)
                        if z == "GREEN":
                            return "background-color: rgba(0, 200, 0, 0.20)"
                        if z == "ORANGE":
                            return "background-color: rgba(255, 165, 0, 0.25)"
                        if z == "RED":
                            return "background-color: rgba(220, 20, 60, 0.25)"
                        return ""

                    sty = out.style.format({"Tracking Error %": "{:.2f}"}).applymap(_style_zone, subset=["Zone"])
                    st.dataframe(sty, use_container_width=True, hide_index=True)
            except Exception as e:
                self._log_error(e, context="tracking_error_weekly_table")
                st.warning("Weekly monitoring table could not be computed.")

    def _display_rolling_beta(self, config: AnalysisConfiguration):
        """Rolling beta vs benchmark (asset or portfolio) with interactive chart."""
        st.markdown("### β Rolling Beta vs Benchmark")

        returns_df = self._to_returns_df(st.session_state.get("returns_data", None))
        bench_returns_df = self._to_returns_df(st.session_state.get("benchmark_returns_data", None))

        if returns_df.empty:
            st.info("Load data first to compute rolling beta.")
            return
        if bench_returns_df.empty:
            st.warning("No benchmark returns available. Please select at least one benchmark in the sidebar and reload data.")
            return

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
        with c1:
            scope = st.selectbox("Scope", ["Asset", "Portfolio"], index=0, key="beta_scope")
        with c2:
            selected_asset = None
            if scope == "Asset":
                selected_asset = st.selectbox("Asset", list(returns_df.columns), index=0, key="beta_asset")
        with c3:
            window = st.number_input("Rolling window (days)", min_value=20, max_value=252, value=int(getattr(config, "rolling_window", 60)), step=5, key="beta_window")

        scope_returns, scope_label = self._get_scope_returns(returns_df, scope=scope, asset=selected_asset)
        bench_series, bench_label = self._select_benchmark_series(bench_returns_df, key_ns="beta_")

        a, b = self._safe_align_pair(scope_returns, bench_series)
        if a.empty or b.empty or len(a) < int(window):
            st.warning("Insufficient overlap to compute rolling beta.")
            return

        # Rolling beta = cov(a,b)/var(b)
        df = pd.concat([a, b], axis=1).dropna()
        df.columns = ["asset", "bench"]

        cov = df["asset"].rolling(int(window)).cov(df["bench"])
        var = df["bench"].rolling(int(window)).var()
        beta = (cov / var).replace([np.inf, -np.inf], np.nan).dropna()

        if beta.empty:
            st.warning("Rolling beta series is empty after processing.")
            return

        k1, k2, k3 = st.columns(3)
        k1.metric("Scope", scope_label)
        k2.metric("Benchmark", bench_label)
        k3.metric("Latest Rolling Beta", f"{float(beta.iloc[-1]):.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=beta.index,
            y=beta.values,
            mode="lines",
            name=f"Rolling Beta ({int(window)}d)",
        ))

        fig.add_hline(y=1.0, line_dash="dash", opacity=0.6, annotation_text="β=1")
        fig.add_hline(y=0.0, line_dash="dot", opacity=0.5, annotation_text="β=0")

        fig.update_layout(
            title=dict(text=f"Rolling Beta vs {bench_label} — {scope_label}", x=0.5),
            xaxis_title="Date",
            yaxis_title="Beta",
            height=520,
            template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
            margin=dict(t=60, l=40, r=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostics
        with st.expander("Beta diagnostics", expanded=False):
            st.write("Beta is computed as rolling Cov(asset, benchmark) / Var(benchmark) on daily returns.")
            st.write(f"Window: {int(window)} trading days; observations: {int(beta.shape[0])}.")

    @staticmethod
    def _rolling_var_cvar_es(
        r: pd.Series,
        window: int,
        alpha: float = 0.95,
    ) -> pd.DataFrame:
        """Compute rolling Historical VaR/CVaR/ES (loss as positive numbers)."""
        rr = r.dropna()
        if rr.empty or len(rr) < int(window):
            return pd.DataFrame()

        q = 1.0 - float(alpha)

        def _var(x: np.ndarray) -> float:
            v = np.quantile(x, q)
            return float(-v)

        def _cvar(x: np.ndarray) -> float:
            v = np.quantile(x, q)
            tail = x[x <= v]
            if tail.size == 0:
                return float(-v)
            return float(-np.mean(tail))

        # ES in this context == CVaR for historical
        roll = rr.rolling(int(window))

        var_s = roll.apply(lambda x: _var(np.asarray(x, dtype=float)), raw=False)
        cvar_s = roll.apply(lambda x: _cvar(np.asarray(x, dtype=float)), raw=False)
        es_s = cvar_s.copy()

        out = pd.DataFrame({"VaR": var_s, "CVaR": cvar_s, "ES": es_s}).dropna()
        return out

    def _display_relative_risk(self, config: AnalysisConfiguration):
        """Relative VaR/CVaR/ES vs benchmark with band zones."""
        st.markdown("### 📉 Relative VaR / CVaR / ES vs Benchmark (Band Zones)")

        returns_df = self._to_returns_df(st.session_state.get("returns_data", None))
        bench_returns_df = self._to_returns_df(st.session_state.get("benchmark_returns_data", None))

        if returns_df.empty:
            st.info("Load data first to compute relative risk metrics.")
            return
        if bench_returns_df.empty:
            st.warning("No benchmark returns available. Please select at least one benchmark in the sidebar and reload data.")
            return

        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
        with c1:
            scope = st.selectbox("Scope", ["Asset", "Portfolio"], index=0, key="relrisk_scope")
        with c2:
            selected_asset = None
            if scope == "Asset":
                selected_asset = st.selectbox("Asset", list(returns_df.columns), index=0, key="relrisk_asset")
        with c3:
            window = st.number_input("Rolling window (days)", min_value=60, max_value=756, value=250, step=25, key="relrisk_window")
        with c4:
            alpha = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="relrisk_alpha")

        scope_returns, scope_label = self._get_scope_returns(returns_df, scope=scope, asset=selected_asset)
        bench_series, bench_label = self._select_benchmark_series(bench_returns_df, key_ns="relrisk_")

        a, b = self._safe_align_pair(scope_returns, bench_series)
        if a.empty or b.empty or len(a) < int(window):
            st.warning("Insufficient overlap to compute rolling relative risk.")
            return

        # Rolling risk
        risk_a = self._rolling_var_cvar_es(a, window=int(window), alpha=float(alpha))
        risk_b = self._rolling_var_cvar_es(b, window=int(window), alpha=float(alpha))

        if risk_a.empty or risk_b.empty:
            st.warning("Risk series could not be computed (empty).")
            return

        # Align
        common = risk_a.join(risk_b, how="inner", lsuffix="_asset", rsuffix="_bench").dropna()
        if common.empty:
            st.warning("No overlapping rolling window risk points after alignment.")
            return

        metric = st.selectbox("Metric", ["VaR", "CVaR", "ES"], index=0, key="relrisk_metric")
        asset_col = f"{metric}_asset"
        bench_col = f"{metric}_bench"

        # Relative (asset - benchmark) in loss terms (positive numbers)
        rel = (common[asset_col] - common[bench_col]).astype(float)

        st.markdown("#### 📌 Relative Risk Bands")
        b1, b2, b3 = st.columns([1.1, 1.1, 1.1])
        with b1:
            green_max = st.number_input("Green max (relative loss)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="relrisk_green")
        with b2:
            orange_max = st.number_input("Orange max (relative loss)", min_value=-1.0, max_value=2.0, value=0.01, step=0.001, format="%.3f", key="relrisk_orange")
        with b3:
            show_components = st.checkbox("Show asset/benchmark components", value=True, key="relrisk_show_components")

        # KPIs
        latest_rel = float(rel.iloc[-1])
        if latest_rel <= float(green_max):
            zone = "GREEN"
        elif latest_rel <= float(orange_max):
            zone = "ORANGE"
        else:
            zone = "RED"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scope", scope_label)
        k2.metric("Benchmark", bench_label)
        k3.metric(f"Latest Relative {metric}", f"{100*latest_rel:.2f}%")
        k4.metric("Policy Zone", zone)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rel.index,
            y=rel.values,
            mode="lines",
            name=f"Relative {metric} (Asset - Benchmark)",
        ))

        # Bands
        x0 = rel.index.min()
        x1 = rel.index.max()
        # Expand y-range
        ymax = float(max(rel.max(), float(orange_max)) * 1.25) if np.isfinite(rel.max()) else float(orange_max) * 1.25
        ymin = float(min(rel.min(), float(green_max)) * 1.25) if np.isfinite(rel.min()) else float(green_max) * 1.25

        bands = [
            (ymin, float(green_max), "rgba(0, 200, 0, 0.35)", "GREEN"),
            (float(green_max), float(orange_max), "rgba(255, 165, 0, 0.35)", "ORANGE"),
            (float(orange_max), ymax, "rgba(220, 20, 60, 0.35)", "RED"),
        ]
        try:
            self._add_zone_bands(fig, x0, x1, bands)
        except Exception:
            pass

        fig.add_hline(y=float(green_max), line_dash="dash", opacity=0.6, annotation_text="Green max")
        fig.add_hline(y=float(orange_max), line_dash="dash", opacity=0.6, annotation_text="Orange max")
        fig.add_hline(y=0.0, line_dash="dot", opacity=0.5, annotation_text="0")

        fig.update_layout(
            title=dict(text=f"Relative {metric} vs {bench_label} — {scope_label} (α={float(alpha):.2f})", x=0.5),
            xaxis_title="Date",
            yaxis_title=f"Relative {metric} (loss, positive)",
            height=540,
            template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
            margin=dict(t=60, l=40, r=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_components:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=common.index,
                y=common[asset_col].values,
                mode="lines",
                name=f"{scope_label} {metric}",
            ))
            fig2.add_trace(go.Scatter(
                x=common.index,
                y=common[bench_col].values,
                mode="lines",
                name=f"{bench_label} {metric}",
            ))
            fig2.update_layout(
                title=dict(text=f"{metric} Components (Historical rolling window)", x=0.5),
                xaxis_title="Date",
                yaxis_title=f"{metric} (loss, positive)",
                height=520,
                template=self.visualizer.template if hasattr(self, "visualizer") else "plotly_white",
                margin=dict(t=60, l=40, r=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            )
            st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main application entry point"""
    try:
        # Hide streamlit default elements
        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        # Add custom CSS for additional styling
        st.markdown(textwrap.dedent("""
        <style>
            .stAlert { border-radius: 10px; }
            .stButton > button { border-radius: 8px; }
            .stSelectbox, .stMultiselect { border-radius: 8px; }
            .stSlider { border-radius: 8px; }
        </style>
        """), unsafe_allow_html=True)
        
        # Initialize and run dashboard
        dashboard = InstitutionalCommoditiesDashboard()
        dashboard.run()
        
    except Exception as e:
        # Comprehensive error handling
        st.error(f"""
        ## 🚨 Application Error
        
        An unexpected error occurred in the Institutional Commodities Analytics Platform.
        
        **Error Details:** {str(e)}
        
        ### 🔧 Troubleshooting Steps:
        1. Refresh the page
        2. Clear your browser cache
        3. Check your internet connection
        4. Try selecting different assets or date ranges
        
        If the problem persists, please contact support with the error details above.
        """)
        
        # Log error for debugging
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'streamlit_version': st.__version__,
            'python_version': os.sys.version
        }
        
        st.code(json.dumps(error_log, indent=2), language='json')


# =============================================================================
# BEGIN PORTFOLIO LAB PATCH (Merged)
# =============================================================================

# =============================================================================
# 🧩 PATCH PACK v1.0 — Portfolio Lab + Risk Decomposition + VaR (3 Methods) + PCA Vol Drivers
# Target: "🏛️ Institutional Commodities Analytics Platform v7.0" (Streamlit)
#
# ✅ Paste this entire file CONTENT at the *BOTTOM* of your existing app.py
# ✅ Then add the tiny hook snippets shown at the end ("INTEGRATION HOOKS")
#
# This patch is strictly additive:
# - Does NOT remove/replace your existing features
# - Adds Copper + Aluminum futures tickers
# - Adds a full "Portfolio Lab" suite using PyPortfolioOpt (optional, graceful fallback)
# - Adds manual-weight portfolio builder + multi-portfolio comparative analysis
# - Adds portfolio volatility risk contributions (assets + PCA factors)
# - Adds VaR + CVaR/ES and Relative VaR (active) with 3 methods (Historical / Parametric / Monte Carlo)
# - Adds PCA-driven volatility factor analysis for GARCH vol + Portfolio EWMA vol
#
# NOTE:
# - PyPortfolioOpt requires: PyPortfolioOpt + cvxpy + scikit-learn
# - This patch handles missing packages gracefully and shows install hints inside the app.
# =============================================================================
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

# Optional (used if available)
try:
    from sklearn.decomposition import PCA
    _SKLEARN_OK = True
except Exception:
    PCA = None
    _SKLEARN_OK = False

# Optional (used if available in your app)
try:
    from arch import arch_model
    _ARCH_OK = True
except Exception:
    arch_model = None
    _ARCH_OK = False

# Optional: PyPortfolioOpt (graceful fallback)
_PYPFOPT_OK = False
try:
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.cla import CLA

    # Optional advanced optimizers (availability varies by version)
    try:
        from pypfopt.efficient_cvar import EfficientCVaR
    except Exception:
        EfficientCVaR = None
    try:
        from pypfopt.efficient_semivariance import EfficientSemivariance
    except Exception:
        EfficientSemivariance = None
    try:
        from pypfopt.efficient_cdar import EfficientCDaR
    except Exception:
        EfficientCDaR = None
    try:
        from pypfopt.black_litterman import BlackLittermanModel
    except Exception:
        BlackLittermanModel = None

    _PYPFOPT_OK = True
except Exception:
    _PYPFOPT_OK = False


# =============================================================================
# 1) FUTURES UNIVERSE PATCH: Add Copper + Aluminum
# =============================================================================

def patch_futures_universe(existing_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Add Copper + Aluminum tickers to your futures universe.
    Yahoo Finance tickers:
      - Copper (COMEX): HG=F
      - Aluminum (COMEX): ALI=F
    """
    base = dict(existing_map) if isinstance(existing_map, dict) else {}

    # Keep existing keys untouched; only add missing
    additions = {
        "Copper (COMEX) - HG=F": "HG=F",
        "Aluminum (COMEX) - ALI=F": "ALI=F",
    }
    for k, v in additions.items():
        if k not in base:
            base[k] = v
    return base


# =============================================================================
# 2) UTILITIES — Returns, EWMA Vol, Covariance, Risk Contributions
# =============================================================================

def _to_prices_df(prices: Any) -> pd.DataFrame:
    if prices is None:
        return pd.DataFrame()
    if isinstance(prices, pd.Series):
        return prices.to_frame()
    if isinstance(prices, pd.DataFrame):
        return prices.copy()
    return pd.DataFrame(prices)

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    method:
      - 'log': log returns
      - 'simple': pct returns
    """
    prices = _to_prices_df(prices)
    prices = prices.sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.ffill().bfill()

    if prices.empty or prices.shape[0] < 3:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    if method.lower() == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices).diff()

    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    return rets

def annualize_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    # geometric
    return float((1.0 + r).prod() ** (periods_per_year / max(1, r.shape[0])) - 1.0)

def annualize_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods_per_year))

def max_drawdown(cum: pd.Series) -> float:
    s = cum.dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())

def ewma_volatility(returns: pd.Series, span: int = 33, periods_per_year: int = 252) -> pd.Series:
    """
    EWMA volatility (annualized). Uses span for EWM std.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    vol = r.ewm(span=span, adjust=False).std(bias=False) * np.sqrt(periods_per_year)
    vol.name = f"EWMA_Vol_{span}"
    return vol

def _align_for_cov(returns_df: pd.DataFrame) -> pd.DataFrame:
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    # drop all-NA columns
    x = x.dropna(axis=1, how="all")
    # keep common rows
    x = x.dropna(axis=0, how="any")
    return x

def sample_covariance(returns_df: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 1:
        return pd.DataFrame()
    cov = x.cov()
    if annualize:
        cov *= periods_per_year
    return cov

def ledoit_wolf_covariance(returns_df: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance (requires sklearn). If sklearn missing, fallback to sample covariance.
    """
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 1:
        return pd.DataFrame()
    if not _SKLEARN_OK:
        return sample_covariance(x, annualize=annualize, periods_per_year=periods_per_year)

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(x.values)
        cov = pd.DataFrame(lw.covariance_, index=x.columns, columns=x.columns)
        if annualize:
            cov *= periods_per_year
        return cov
    except Exception:
        return sample_covariance(x, annualize=annualize, periods_per_year=periods_per_year)

def risk_contribution(weights: np.ndarray, cov_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Risk contribution decomposition (assets):
      RC_i = w_i * (Σ w)_i / σ_p
    """
    if cov_annual is None or cov_annual.empty:
        return pd.DataFrame()
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    cols = list(cov_annual.columns)
    if w.shape[0] != len(cols):
        return pd.DataFrame()
    Sigma = cov_annual.values
    port_var = float(w.T @ Sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    mrc = (Sigma @ w).flatten()  # marginal risk contributions (unnormalized)
    rc = (w.flatten() * mrc) / (port_vol if port_vol > 0 else np.nan)

    out = pd.DataFrame({
        "weight": w.flatten(),
        "marginal_contrib": mrc,
        "risk_contrib": rc,
        "risk_contrib_pct": rc / (port_vol if port_vol > 0 else np.nan)
    }, index=cols)
    out = out.sort_values("risk_contrib_pct", ascending=False)
    return out

def pca_factor_risk_contribution(weights: np.ndarray, cov_annual: pd.DataFrame, top_k: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PCA factor risk decomposition:
      - eigen-decompose Σ = V Λ V'
      - portfolio variance = Σ_j ( (w' v_j)^2 * λ_j )
    Returns:
      factor_var_contrib: variance contribution by PCA factor
      factor_loading: loadings matrix (assets x factors)
    """
    if cov_annual is None or cov_annual.empty:
        return pd.DataFrame(), pd.DataFrame()

    Sigma = cov_annual.values
    cols = list(cov_annual.columns)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    if w.shape[0] != len(cols):
        return pd.DataFrame(), pd.DataFrame()

    # Symmetrize for numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    k = int(min(max(1, top_k), len(evals)))
    evals_k = evals[:k]
    evecs_k = evecs[:, :k]

    # Factor exposures: b_j = w' v_j
    b = (w.T @ evecs_k).flatten()
    var_contrib = (b ** 2) * evals_k
    total_var = float((w.T @ Sigma @ w).item())
    total_var = max(total_var, 0.0)

    df_var = pd.DataFrame({
        "eigenvalue": evals_k,
        "exposure_wTv": b,
        "variance_contrib": var_contrib,
        "variance_contrib_pct": var_contrib / (total_var if total_var > 0 else np.nan),
    }, index=[f"PC{i+1}" for i in range(k)]).sort_values("variance_contrib_pct", ascending=False)

    df_load = pd.DataFrame(evecs_k, index=cols, columns=[f"PC{i+1}" for i in range(k)])
    return df_var, df_load


# =============================================================================
# 3) VaR / CVaR(ES) + Relative VaR (Active Returns) — 3 Methods
# =============================================================================

@dataclass
class VaRResult:
    var: float
    cvar_es: float
    method: str
    alpha: float
    horizon_days: int

def _scale_var(value_1d: float, horizon_days: int, method: str) -> float:
    """
    For parametric / Monte Carlo under iid normal-ish assumption: sqrt(h)
    For historical: use empirical scaling by resampling if we want, but for simplicity:
      - keep sqrt scaling as default (documented)
    """
    if horizon_days <= 1 or value_1d is None or np.isnan(value_1d):
        return value_1d
    return float(value_1d * np.sqrt(horizon_days))

def historical_var(returns: pd.Series, alpha: float = 0.05, horizon_days: int = 1) -> VaRResult:
    r = returns.dropna()
    if r.empty:
        return VaRResult(np.nan, np.nan, "Historical", alpha, horizon_days)

    losses = -r
    var_1d = float(np.quantile(losses, 1.0 - (1.0 - alpha)))  # alpha quantile of losses
    # ES = mean loss beyond VaR threshold
    tail = losses[losses >= var_1d]
    es_1d = float(tail.mean()) if len(tail) > 0 else float(losses.max())
    return VaRResult(_scale_var(var_1d, horizon_days, "Historical"), _scale_var(es_1d, horizon_days, "Historical"), "Historical", alpha, horizon_days)

def parametric_var(returns: pd.Series, alpha: float = 0.05, horizon_days: int = 1) -> VaRResult:
    r = returns.dropna()
    if r.empty:
        return VaRResult(np.nan, np.nan, "Parametric (Gaussian)", alpha, horizon_days)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0 or np.isnan(sigma):
        return VaRResult(np.nan, np.nan, "Parametric (Gaussian)", alpha, horizon_days)

    from scipy.stats import norm
    z = float(norm.ppf(alpha))
    # VaR on losses: -(mu + z*sigma)
    var_1d = float(-(mu + z * sigma))
    # ES (Gaussian) on losses
    es_1d = float(-(mu) + sigma * (norm.pdf(z) / alpha))
    return VaRResult(_scale_var(var_1d, horizon_days, "Parametric"), _scale_var(es_1d, horizon_days, "Parametric"), "Parametric (Gaussian)", alpha, horizon_days)

def monte_carlo_var(returns_df: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05, horizon_days: int = 1,
                    n_sims: int = 20000, seed: int = 42, use_shrinkage_cov: bool = True) -> VaRResult:
    """
    Monte Carlo VaR for portfolio using multivariate normal simulation:
      r_sim ~ N(mu, cov)
      portfolio_sim = r_sim @ w
    """
    x = _align_for_cov(returns_df)
    if x.empty:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    mu = x.mean().values
    cov = ledoit_wolf_covariance(x, annualize=False) if use_shrinkage_cov else sample_covariance(x, annualize=False)
    if cov.empty:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    w = np.asarray(weights, dtype=float).flatten()
    if len(w) != x.shape[1]:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    rng = np.random.default_rng(int(seed))
    sims = rng.multivariate_normal(mean=mu, cov=cov.values, size=int(n_sims))
    port = sims @ w
    losses = -port
    var_1d = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_1d]
    es_1d = float(tail.mean()) if len(tail) > 0 else float(losses.max())
    return VaRResult(_scale_var(var_1d, horizon_days, "MC"), _scale_var(es_1d, horizon_days, "MC"), "Monte Carlo (MVN)", alpha, horizon_days)

def compute_var_suite(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float,
    horizon_days: int,
    n_sims: int,
    seed: int,
    use_shrinkage_cov: bool
) -> pd.DataFrame:
    """
    Compute VaR + ES for 3 methods, return a tidy DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    h = historical_var(portfolio_returns, alpha=alpha, horizon_days=horizon_days)
    p = parametric_var(portfolio_returns, alpha=alpha, horizon_days=horizon_days)
    m = monte_carlo_var(asset_returns, weights=weights, alpha=alpha, horizon_days=horizon_days,
                        n_sims=n_sims, seed=seed, use_shrinkage_cov=use_shrinkage_cov)

    for res in [h, p, m]:
        rows.append({
            "method": res.method,
            "alpha": res.alpha,
            "horizon_days": res.horizon_days,
            "VaR": res.var,
            "ES(CVaR)": res.cvar_es
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4) GARCH VOL + PCA VOL DRIVERS
# =============================================================================

def _garch_vol_series(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal",
                      annualize: bool = True, periods_per_year: int = 252) -> pd.Series:
    """
    GARCH(p,q) conditional volatility series. Requires 'arch'.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    if not _ARCH_OK:
        return pd.Series(dtype=float)

    # Scale returns for stability (arch typically uses %)
    r_pct = r * 100.0

    try:
        am = arch_model(r_pct, vol="Garch", p=int(p), q=int(q), dist=str(dist), rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility / 100.0  # back to decimal
        if annualize:
            cond_vol = cond_vol * np.sqrt(periods_per_year)
        cond_vol.name = f"GARCH({p},{q})_vol"
        return cond_vol
    except Exception:
        return pd.Series(dtype=float)

def compute_garch_vol_matrix(returns_df: pd.DataFrame, p: int = 1, q: int = 1, dist: str = "normal",
                             annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute GARCH vol for each asset (may be expensive). Use on a selected subset.
    """
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all")
    if x.empty:
        return pd.DataFrame()

    out = {}
    for col in x.columns:
        v = _garch_vol_series(x[col], p=p, q=q, dist=dist, annualize=annualize, periods_per_year=periods_per_year)
        if not v.empty:
            out[col] = v

    if not out:
        return pd.DataFrame()
    vol_df = pd.concat(out, axis=1).dropna(how="all")
    return vol_df

def pca_on_matrix(df: pd.DataFrame, n_components: int = 4, standardize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA on time x assets matrix.
    Returns:
      scores_df: time x k
      loadings_df: assets x k
      evr_df: explained variance ratios
    """
    if df is None or df.empty or df.shape[1] < 2 or not _SKLEARN_OK:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.empty or x.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = x.values
    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        X = (X - mu) / sd

    k = int(min(max(2, n_components), x.shape[1]))
    pca = PCA(n_components=k)
    scores = pca.fit_transform(X)  # time x k
    loadings = pca.components_.T     # assets x k

    scores_df = pd.DataFrame(scores, index=x.index, columns=[f"PC{i+1}" for i in range(k)])
    loadings_df = pd.DataFrame(loadings, index=x.columns, columns=[f"PC{i+1}" for i in range(k)])
    evr_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(k)],
        "ExplainedVarRatio": pca.explained_variance_ratio_,
        "CumExplained": np.cumsum(pca.explained_variance_ratio_)
    })
    return scores_df, loadings_df, evr_df


# =============================================================================
# 5) PORTFOLIO LAB — Manual Builder + PyPortfolioOpt Strategies + Comparative Analysis
# =============================================================================

@dataclass
class PortfolioSpec:
    name: str
    weights: Dict[str, float]
    benchmark: Optional[str] = None

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in w.items() if v is not None and np.isfinite(v) and float(v) >= 0.0}
    s = float(sum(w.values()))
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}

def _weights_to_array(weights: Dict[str, float], cols: List[str]) -> np.ndarray:
    return np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)

def compute_portfolio_returns(returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.empty:
        return pd.Series(dtype=float)

    cols = list(x.columns)
    w = _weights_to_array(weights, cols)
    if np.allclose(w.sum(), 0.0):
        return pd.Series(index=x.index, dtype=float)

    pr = x.values @ w
    s = pd.Series(pr, index=x.index, name="Portfolio")
    return s

def compute_beta_alpha(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> Tuple[float, float]:
    """
    CAPM beta + Jensen alpha (annualized alpha approximation).
    """
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan, np.nan
    p = p.loc[idx]
    b = b.loc[idx]

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    p_ex = p - rf_daily
    b_ex = b - rf_daily

    var_b = float(np.var(b_ex, ddof=1))
    if var_b <= 0 or np.isnan(var_b):
        return np.nan, np.nan
    cov_pb = float(np.cov(p_ex, b_ex, ddof=1)[0, 1])
    beta = cov_pb / var_b

    # alpha daily = mean(p_ex) - beta * mean(b_ex)
    alpha_daily = float(p_ex.mean() - beta * b_ex.mean())
    alpha_annual = float((1.0 + alpha_daily) ** periods_per_year - 1.0)
    return float(beta), float(alpha_annual)

def tracking_error(portfolio: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan
    active = (p.loc[idx] - b.loc[idx]).dropna()
    if active.empty:
        return np.nan
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))

def information_ratio(portfolio: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan
    active = (p.loc[idx] - b.loc[idx]).dropna()
    if active.empty:
        return np.nan
    te = active.std(ddof=1)
    if te <= 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))

def portfolio_metrics_table(
    portfolio_ret: pd.Series,
    bench_ret: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    r = portfolio_ret.dropna()
    if r.empty:
        return {
            "AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan,
            "MaxDrawdown": np.nan, "Beta": np.nan, "Alpha": np.nan,
            "TrackingError": np.nan, "InfoRatio": np.nan
        }

    cum = (1.0 + r).cumprod()
    out = {
        "AnnReturn": annualize_return(r, periods_per_year),
        "AnnVol": annualize_vol(r, periods_per_year),
        "Sharpe": sharpe_ratio(r, rf_annual, periods_per_year),
        "MaxDrawdown": max_drawdown(cum),
        "Beta": np.nan,
        "Alpha": np.nan,
        "TrackingError": np.nan,
        "InfoRatio": np.nan
    }
    if bench_ret is not None and isinstance(bench_ret, pd.Series):
        beta, alpha = compute_beta_alpha(r, bench_ret, rf_annual, periods_per_year)
        out["Beta"] = beta
        out["Alpha"] = alpha
        out["TrackingError"] = tracking_error(r, bench_ret, periods_per_year)
        out["InfoRatio"] = information_ratio(r, bench_ret, periods_per_year)
    return out

def _plot_cum_performance(portfolios: Dict[str, pd.Series], title: str, key: str) -> None:
    fig = go.Figure()
    for name, r in portfolios.items():
        rr = r.dropna()
        if rr.empty:
            continue
        cum = (1.0 + rr).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, mode="lines"))
    fig.update_layout(
        title=title,
        height=420,
        xaxis_title="Date",
        yaxis_title="Cumulative Growth",
        legend_title="Series",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def _plot_risk_contrib_bar(rc_df: pd.DataFrame, title: str, key: str) -> None:
    if rc_df is None or rc_df.empty:
        st.info("Risk contribution cannot be computed (insufficient data).")
        return
    df = rc_df.reset_index().rename(columns={"index": "Asset"})
    fig = px.bar(df, x="Asset", y="risk_contrib_pct", title=title)
    fig.update_layout(height=420, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)

def _plot_pca_loadings_heatmap(loadings: pd.DataFrame, title: str, key: str) -> None:
    if loadings is None or loadings.empty:
        st.info("PCA loadings are not available.")
        return
    fig = px.imshow(loadings.values, x=loadings.columns, y=loadings.index, aspect="auto", title=title)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)

def _ensure_session_portfolios(key_ns: str) -> None:
    k = f"{key_ns}__saved_portfolios"
    if k not in st.session_state or not isinstance(st.session_state.get(k), list):
        st.session_state[k] = []

def _save_portfolio_spec(spec: PortfolioSpec, key_ns: str) -> None:
    _ensure_session_portfolios(key_ns)
    k = f"{key_ns}__saved_portfolios"
    st.session_state[k].append(spec)

def _get_saved_portfolios(key_ns: str) -> List[PortfolioSpec]:
    _ensure_session_portfolios(key_ns)
    return list(st.session_state.get(f"{key_ns}__saved_portfolios", []))

def _clear_saved_portfolios(key_ns: str) -> None:
    st.session_state[f"{key_ns}__saved_portfolios"] = []

def _render_install_hints() -> None:
    st.warning("PyPortfolioOpt / scikit-learn / cvxpy may not be installed in your environment.")
    st.code(
        "requirements.txt (Streamlit Cloud) suggested entries:\n"
        "PyPortfolioOpt==1.5.5\n"
        "cvxpy>=1.3\n"
        "scikit-learn>=1.2\n"
        "arch>=6.3\n"
        "scipy>=1.10\n"
    )

def _pypfopt_expected_returns_and_cov(returns_df: pd.DataFrame, cov_method: str, key_ns: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 2:
        return None, None

    # Expected returns
    # Use annualized mean historical returns
    mu = x.mean() * 252.0

    # Covariance
    if cov_method == "Ledoit-Wolf":
        cov = ledoit_wolf_covariance(x, annualize=True, periods_per_year=252)
    else:
        cov = sample_covariance(x, annualize=True, periods_per_year=252)

    # Ensure order consistency
    mu = mu.loc[cov.columns]
    return mu, cov

def _run_pypfopt_strategies(
    returns_df: pd.DataFrame,
    cov_method: str,
    rf_annual: float,
    key_ns: str
) -> Dict[str, Dict[str, float]]:
    """
    Runs multiple PyPortfolioOpt strategies and returns dict {strategy_name: weights_dict}
    """
    if not _PYPFOPT_OK:
        return {}

    mu, cov = _pypfopt_expected_returns_and_cov(returns_df, cov_method, key_ns)
    if mu is None or cov is None or cov.empty:
        return {}

    assets = list(cov.columns)

    out: Dict[str, Dict[str, float]] = {}

    # --- Efficient Frontier base
    try:
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        w = ef.max_sharpe(risk_free_rate=float(rf_annual))
        out["Max Sharpe (EF)"] = dict(ef.clean_weights())
    except Exception:
        pass

    try:
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        w = ef.min_volatility()
        out["Min Volatility (EF)"] = dict(ef.clean_weights())
    except Exception:
        pass

    # Efficient risk/return targets
    try:
        target_vol = float(st.session_state.get(f"{key_ns}__target_vol", 0.15))
        ef = EfficientFrontier(mu, cov)
        w = ef.efficient_risk(target_volatility=target_vol)
        out[f"Efficient Risk (Target Vol={target_vol:.2%})"] = dict(ef.clean_weights())
    except Exception:
        pass

    try:
        target_ret = float(st.session_state.get(f"{key_ns}__target_ret", 0.10))
        ef = EfficientFrontier(mu, cov)
        w = ef.efficient_return(target_return=target_ret)
        out[f"Efficient Return (Target Ret={target_ret:.2%})"] = dict(ef.clean_weights())
    except Exception:
        pass

    # Quadratic utility
    try:
        gamma = float(st.session_state.get(f"{key_ns}__gamma", 1.0))
        ef = EfficientFrontier(mu, cov)
        w = ef.max_quadratic_utility(risk_aversion=gamma)
        out[f"Max Quadratic Utility (γ={gamma:.2f})"] = dict(ef.clean_weights())
    except Exception:
        pass

    # HRP
    try:
        hrp = HRPOpt(returns_df.dropna(how="any"))
        hrp_w = hrp.optimize()
        out["Hierarchical Risk Parity (HRP)"] = {k: float(v) for k, v in hrp_w.items()}
    except Exception:
        pass

    # CLA
    try:
        cla = CLA(mu, cov)
        cla_w = cla.max_sharpe()
        out["Critical Line Algorithm (CLA)"] = {k: float(v) for k, v in cla_w.items()}
    except Exception:
        pass

    # Efficient CVaR
    if EfficientCVaR is not None:
        try:
            ec = EfficientCVaR(mu, cov, returns_df.dropna(how="any"))
            cvar_w = ec.min_cvar()
            out["Min CVaR (EfficientCVaR)"] = dict(ec.clean_weights())
        except Exception:
            pass

    # Efficient Semivariance
    if EfficientSemivariance is not None:
        try:
            es = EfficientSemivariance(mu, cov, returns_df.dropna(how="any"))
            semi_w = es.min_semivariance()
            out["Min Semivariance (EfficientSemivariance)"] = dict(es.clean_weights())
        except Exception:
            pass

    # Efficient CDaR
    if EfficientCDaR is not None:
        try:
            cd = EfficientCDaR(mu, cov, returns_df.dropna(how="any"))
            cd_w = cd.min_cdar()
            out["Min CDaR (EfficientCDaR)"] = dict(cd.clean_weights())
        except Exception:
            pass

    return out

def _render_manual_portfolio_builder(assets: List[str], key_ns: str) -> PortfolioSpec:
    st.subheader("Manual Portfolio Builder (User-Defined Weights)")
    sel = st.multiselect("Select assets", options=assets, default=assets[:min(6, len(assets))], key=f"{key_ns}__asset_select")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        auto_normalize = st.checkbox("Auto-normalize weights to 100%", value=True, key=f"{key_ns}__auto_norm")
    with colB:
        name = st.text_input("Portfolio name", value="Custom_1", key=f"{key_ns}__pname")
    with colC:
        bench = st.text_input("Benchmark ticker/name (optional, must exist in your benchmark returns dict)", value="", key=f"{key_ns}__bench")

    weights: Dict[str, float] = {}
    if not sel:
        st.info("Select at least 1 asset.")
        return PortfolioSpec(name=name, weights={}, benchmark=bench if bench else None)

    st.caption("Set weights using sliders (0–100). Tip: turn on Auto-normalize.")
    cols = st.columns(2)
    for i, a in enumerate(sel):
        with cols[i % 2]:
            weights[a] = st.slider(f"Weight: {a}", min_value=0.0, max_value=100.0, value=float(100.0/len(sel)),
                                   step=0.5, key=f"{key_ns}__w__{a}")

    weights = {k: v/100.0 for k, v in weights.items()}
    if auto_normalize:
        weights = _normalize_weights(weights)

    s = sum(weights.values())
    st.write(f"Weight sum: **{s:.4f}**")
    if abs(s - 1.0) > 1e-3:
        st.warning("Weights do not sum to 1.0. Enable Auto-normalize or adjust sliders.")

    return PortfolioSpec(name=name, weights=weights, benchmark=bench if bench else None)

def render_portfolio_lab_suite(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
    key_ns: str = "portfolio_lab"
) -> None:
    """
    Main entrypoint. Call this inside a new Streamlit tab without removing anything else.
    Inputs expected:
      - prices_df: wide dataframe of prices for assets (columns=assets)
      - returns_df: wide dataframe of returns aligned to prices_df
      - benchmark_returns: dict[str, Series] of benchmark returns (e.g., {'SPY': spy_rets, 'XU100': xu100_rets})
    """
    st.header("🏦 Portfolio Lab — PyPortfolioOpt + Manual Portfolios + Risk Decomposition")
    if prices_df is None or returns_df is None or returns_df.empty:
        st.info("No returns data provided to Portfolio Lab. Provide prices_df/returns_df from your existing pipeline.")
        return

    benchmark_returns = benchmark_returns or {}

    # Align and clean
    R = returns_df.copy()
    R = R.replace([np.inf, -np.inf], np.nan)
    R = R.dropna(axis=1, how="all")
    if R.shape[1] < 1:
        st.info("Not enough assets for portfolio analysis.")
        return

    # ---------------- Sidebar controls
    with st.sidebar.expander("📌 Portfolio Lab Controls", expanded=False):
        st.session_state[f"{key_ns}__rf_annual"] = st.number_input("Risk-free rate (annual, decimal)", value=float(st.session_state.get(f"{key_ns}__rf_annual", 0.03)),
                                                                  step=0.005, format="%.4f", key=f"{key_ns}__rf_annual_in")
        st.session_state[f"{key_ns}__target_vol"] = st.slider("Target Volatility (Efficient Risk)", min_value=0.05, max_value=0.50,
                                                              value=float(st.session_state.get(f"{key_ns}__target_vol", 0.15)), step=0.01, key=f"{key_ns}__target_vol_in")
        st.session_state[f"{key_ns}__target_ret"] = st.slider("Target Return (Efficient Return)", min_value=-0.10, max_value=0.60,
                                                              value=float(st.session_state.get(f"{key_ns}__target_ret", 0.10)), step=0.01, key=f"{key_ns}__target_ret_in")
        st.session_state[f"{key_ns}__gamma"] = st.slider("Risk Aversion γ (Quadratic Utility)", min_value=0.1, max_value=10.0,
                                                         value=float(st.session_state.get(f"{key_ns}__gamma", 1.0)), step=0.1, key=f"{key_ns}__gamma_in")

    rf_annual = float(st.session_state.get(f"{key_ns}__rf_annual", 0.03))

    tabs = st.tabs([
        "① Manual Portfolios",
        "② PyPortfolioOpt Strategies",
        "③ Comparative Analysis",
        "④ Volatility Contributions",
        "⑤ VaR + Relative VaR (3 Methods)",
        "⑥ PCA Vol Drivers (GARCH + EWMA)"
    ])

    # =============================================================================
    # TAB 1 — Manual Portfolios
    # =============================================================================
    with tabs[0]:
        assets = list(R.columns)
        spec = _render_manual_portfolio_builder(assets, key_ns=key_ns)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("💾 Save Portfolio", key=f"{key_ns}__save_btn"):
                if spec.weights:
                    _save_portfolio_spec(spec, key_ns=key_ns)
                    st.success(f"Saved portfolio: {spec.name}")
                else:
                    st.warning("No weights to save.")
        with c2:
            if st.button("🧹 Clear Saved Portfolios", key=f"{key_ns}__clear_btn"):
                _clear_saved_portfolios(key_ns=key_ns)
                st.info("Saved portfolios cleared.")
        with c3:
            st.write(f"Saved count: **{len(_get_saved_portfolios(key_ns))}**")

        if spec.weights:
            pr = compute_portfolio_returns(R[list(spec.weights.keys())], spec.weights)
            st.subheader("Preview: Custom Portfolio Performance")
            _plot_cum_performance({spec.name: pr}, "Cumulative Growth (Custom Portfolio)", key=f"{key_ns}__cum_custom")
            mt = portfolio_metrics_table(pr, bench_ret=benchmark_returns.get(spec.benchmark) if spec.benchmark else None, rf_annual=rf_annual)
            st.dataframe(pd.DataFrame([mt], index=[spec.name]).T, use_container_width=True)

    # =============================================================================
    # TAB 2 — PyPortfolioOpt Strategies
    # =============================================================================
    with tabs[1]:
        st.subheader("PyPortfolioOpt Strategies (Institutional Suite)")
        if not _PYPFOPT_OK:
            _render_install_hints()
            st.info("Manual portfolios remain fully available even without PyPortfolioOpt.")
        else:
            cov_method = st.selectbox("Covariance Method", ["Ledoit-Wolf", "Sample"], index=0, key=f"{key_ns}__cov_method")
            st.caption("Strategies run on the aligned returns window (rows with missing values dropped).")

            strategies = _run_pypfopt_strategies(R, cov_method=cov_method, rf_annual=rf_annual, key_ns=key_ns)
            if not strategies:
                st.warning("No strategies produced weights (insufficient data or solver constraints). Try fewer assets or longer history.")
            else:
                st.success(f"Strategies computed: {len(strategies)}")

                # Show weights
                strat_names = list(strategies.keys())
                chosen = st.multiselect("Choose strategies to save/compare", strat_names, default=strat_names[:min(4, len(strat_names))], key=f"{key_ns}__choose_strats")
                if chosen:
                    for sname in chosen:
                        w = strategies[sname]
                        w_norm = _normalize_weights(w)
                        st.markdown(f"**{sname}**")
                        w_df = pd.DataFrame({"weight": pd.Series(w_norm)}).sort_values("weight", ascending=False)
                        st.dataframe(w_df, use_container_width=True)

                        if st.button(f"Save → {sname}", key=f"{key_ns}__save_strat__{sname}"):
                            _save_portfolio_spec(PortfolioSpec(name=sname, weights=w_norm, benchmark=None), key_ns=key_ns)
                            st.success(f"Saved: {sname}")

    # =============================================================================
    # TAB 3 — Comparative Analysis
    # =============================================================================
    with tabs[2]:
        st.subheader("Comparative Analysis — Saved Portfolios")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("No saved portfolios yet. Save from Manual Portfolios or PyPortfolioOpt Strategies.")
        else:
            # Build returns series for each saved portfolio
            series_map: Dict[str, pd.Series] = {}
            rows = []
            for spec in saved:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                if not cols:
                    continue
                w = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                pr = compute_portfolio_returns(R[cols], w)
                series_map[spec.name] = pr

                bench = benchmark_returns.get(spec.benchmark) if spec.benchmark else None
                mt = portfolio_metrics_table(pr, bench_ret=bench, rf_annual=rf_annual)
                rows.append({"Portfolio": spec.name, "Benchmark": spec.benchmark or "", **mt})

            if not series_map:
                st.warning("Saved portfolios could not be computed with current returns universe.")
            else:
                _plot_cum_performance(series_map, "Cumulative Growth — Saved Portfolios", key=f"{key_ns}__cum_saved")
                mt_df = pd.DataFrame(rows).set_index("Portfolio")
                st.dataframe(mt_df, use_container_width=True)

    # =============================================================================
    # TAB 4 — Volatility Contributions (Assets + PCA Factors)
    # =============================================================================
    with tabs[3]:
        st.subheader("Portfolio Volatility Contributions")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("Save at least one portfolio to compute volatility contributions.")
        else:
            names = [s.name for s in saved]
            pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_vol_contrib")
            spec = next((s for s in saved if s.name == pick), None)
            if spec is None:
                st.info("Portfolio not found.")
            else:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                subR = _align_for_cov(R[cols])
                if subR.empty or subR.shape[1] < 2:
                    st.warning("Need at least 2 assets with sufficient overlap.")
                else:
                    cov_method = st.selectbox("Covariance for decomposition", ["Ledoit-Wolf", "Sample"], index=0, key=f"{key_ns}__cov_dec")
                    covA = ledoit_wolf_covariance(subR, annualize=True) if cov_method == "Ledoit-Wolf" else sample_covariance(subR, annualize=True)
                    w_arr = _weights_to_array(w_norm, list(subR.columns))

                    rc = risk_contribution(w_arr, covA)
                    _plot_risk_contrib_bar(rc, f"Asset Risk Contribution (% of Portfolio Vol) — {pick}", key=f"{key_ns}__rc_bar")

                    with st.expander("Risk Contribution Table"):
                        st.dataframe(rc, use_container_width=True)

                    st.markdown("---")
                    st.subheader("PCA Factor Risk Decomposition (Covariance Eigen-Factors)")
                    top_k = st.slider("Number of PCA factors", min_value=2, max_value=min(12, subR.shape[1]),
                                      value=min(6, subR.shape[1]), step=1, key=f"{key_ns}__pca_k_risk")
                    df_var, df_load = pca_factor_risk_contribution(w_arr, covA, top_k=top_k)
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.dataframe(df_var, use_container_width=True)
                    with c2:
                        _plot_pca_loadings_heatmap(df_load, "PCA Factor Loadings (Eigenvectors)", key=f"{key_ns}__pca_load_risk")

    # =============================================================================
    # TAB 5 — VaR + Relative VaR (Active) — 3 Methods
    # =============================================================================
    with tabs[4]:
        st.subheader("VaR + CVaR(ES) + Relative VaR (Active vs Benchmark) — 3 Methods")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("Save at least one portfolio first.")
        else:
            names = [s.name for s in saved]
            pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_var")
            spec = next((s for s in saved if s.name == pick), None)
            if spec is None:
                st.info("Portfolio not found.")
            else:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                subR = _align_for_cov(R[cols])

                if subR.empty:
                    st.warning("Insufficient overlapping returns.")
                else:
                    w_arr = _weights_to_array(w_norm, list(subR.columns))
                    pr = compute_portfolio_returns(subR, w_norm)

                    bmk_key = st.selectbox("Benchmark (for Relative VaR)", options=["(None)"] + list(benchmark_returns.keys()),
                                           index=0, key=f"{key_ns}__bmk_var")
                    bench = benchmark_returns.get(bmk_key) if bmk_key != "(None)" else None

                    alpha = st.select_slider("Confidence level", options=[0.10, 0.05, 0.025, 0.01], value=0.05, key=f"{key_ns}__alpha")
                    horizon = st.selectbox("Horizon (days)", options=[1, 5, 10, 21], index=0, key=f"{key_ns}__horizon")
                    n_sims = st.slider("Monte Carlo simulations", min_value=5000, max_value=80000, value=20000, step=5000, key=f"{key_ns}__n_sims")
                    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1, key=f"{key_ns}__seed")
                    use_shrink = st.checkbox("Use shrinkage covariance (Ledoit-Wolf) in Monte Carlo", value=True, key=f"{key_ns}__mc_shrink")

                    st.markdown("### Portfolio VaR/ES")
                    var_df = compute_var_suite(pr, subR, w_arr, alpha=alpha, horizon_days=horizon, n_sims=int(n_sims), seed=int(seed), use_shrinkage_cov=use_shrink)
                    st.dataframe(var_df, use_container_width=True)

                    if bench is not None and isinstance(bench, pd.Series):
                        # Relative/Active VaR
                        idx = pr.dropna().index.intersection(bench.dropna().index)
                        if len(idx) >= 60:
                            active = (pr.loc[idx] - bench.loc[idx]).dropna()
                            st.markdown("### Relative (Active) VaR/ES — (Portfolio - Benchmark)")
                            # For Monte Carlo on active returns, we simulate portfolio and benchmark separately is complex;
                            # practical approach: compute active series and use historical/parametric on it,
                            # and for MC approximate using active mean/var derived from assets vs benchmark covariance if available.
                            rel_hist = historical_var(active, alpha=alpha, horizon_days=horizon)
                            rel_para = parametric_var(active, alpha=alpha, horizon_days=horizon)

                            rel_rows = [
                                {"method": rel_hist.method, "alpha": rel_hist.alpha, "horizon_days": rel_hist.horizon_days, "VaR": rel_hist.var, "ES(CVaR)": rel_hist.cvar_es},
                                {"method": rel_para.method, "alpha": rel_para.alpha, "horizon_days": rel_para.horizon_days, "VaR": rel_para.var, "ES(CVaR)": rel_para.cvar_es},
                            ]
                            rel_df = pd.DataFrame(rel_rows)
                            st.dataframe(rel_df, use_container_width=True)

                            # Interactive chart of rolling active VaR (historical)
                            window = st.slider("Rolling window (days) for active VaR chart", min_value=60, max_value=504, value=252, step=21, key=f"{key_ns}__roll_win")
                            alpha_roll = alpha

                            active_losses = -active
                            roll_var = active_losses.rolling(window).quantile(alpha_roll)
                            roll_es = active_losses.rolling(window).apply(lambda s: float(pd.Series(s)[pd.Series(s) >= np.quantile(pd.Series(s), alpha_roll)].mean()) if len(s.dropna()) > 0 else np.nan, raw=False)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=active.index, y=active.values, name="Active Return", mode="lines"))
                            fig.add_trace(go.Scatter(x=roll_var.index, y=-roll_var.values, name=f"Rolling VaR (α={alpha_roll})", mode="lines"))
                            fig.add_trace(go.Scatter(x=roll_es.index, y=-roll_es.values, name=f"Rolling ES (α={alpha_roll})", mode="lines"))
                            fig.update_layout(
                                title="Active Return vs Rolling VaR/ES (Historical)",
                                height=460, xaxis_title="Date", yaxis_title="Return",
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_ns}__active_var_chart")
                        else:
                            st.info("Benchmark overlap is not sufficient for Relative VaR (need ~60+ points).")

    # =============================================================================
    # TAB 6 — PCA Vol Drivers (GARCH vol matrix + Portfolio EWMA vol)
    # =============================================================================
    with tabs[5]:
        st.subheader("PCA Volatility Drivers — GARCH Vol & Portfolio EWMA Vol")
        if not _SKLEARN_OK:
            st.warning("scikit-learn is required for PCA. Install scikit-learn to enable this tab.")
        else:
            saved = _get_saved_portfolios(key_ns=key_ns)
            if not saved:
                st.info("Save at least one portfolio.")
            else:
                names = [s.name for s in saved]
                pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_pca_vol")
                spec = next((s for s in saved if s.name == pick), None)
                if spec is None:
                    st.info("Portfolio not found.")
                else:
                    cols = [c for c in spec.weights.keys() if c in R.columns]
                    w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                    subR = _align_for_cov(R[cols])

                    if subR.empty or subR.shape[1] < 2:
                        st.warning("Need at least 2 assets with sufficient overlap.")
                    else:
                        # Portfolio returns + vols
                        pr = compute_portfolio_returns(subR, w_norm)
                        ewma_span = st.selectbox("EWMA span", options=[22, 33, 55, 99], index=1, key=f"{key_ns}__ewma_span")
                        port_ewma = ewma_volatility(pr, span=int(ewma_span))

                        # GARCH vol matrix for assets
                        st.caption("GARCH vol computation can be heavy. Use a smaller asset set if needed.")
                        garch_p = st.selectbox("GARCH p", options=[1, 2], index=0, key=f"{key_ns}__garch_p")
                        garch_q = st.selectbox("GARCH q", options=[1, 2], index=0, key=f"{key_ns}__garch_q")
                        dist = st.selectbox("GARCH distribution", options=["normal", "t"], index=0, key=f"{key_ns}__garch_dist")

                        if not _ARCH_OK:
                            st.warning("arch package not available; cannot compute GARCH vol. Install 'arch' to enable.")
                        else:
                            with st.spinner("Computing asset GARCH vol matrix..."):
                                garch_vol = compute_garch_vol_matrix(subR, p=int(garch_p), q=int(garch_q), dist=str(dist), annualize=True)

                            if garch_vol.empty:
                                st.warning("GARCH vol matrix is empty (model failed or insufficient data).")
                            else:
                                # PCA on GARCH vol matrix
                                n_comp = st.slider("PCA components", min_value=2, max_value=min(8, garch_vol.shape[1]), value=min(4, garch_vol.shape[1]), step=1, key=f"{key_ns}__pca_k_vol")
                                scores, loadings, evr = pca_on_matrix(garch_vol, n_components=int(n_comp), standardize=True)

                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**Explained variance (GARCH vol factors)**")
                                    st.dataframe(evr, use_container_width=True)
                                with c2:
                                    _plot_pca_loadings_heatmap(loadings, "PCA Loadings on Asset GARCH Vol", key=f"{key_ns}__pca_load_garch")

                                # Correlation of PCA factors with portfolio EWMA vol
                                if not scores.empty and not port_ewma.empty:
                                    # Align
                                    idx = scores.index.intersection(port_ewma.index)
                                    if len(idx) > 50:
                                        corr = scores.loc[idx].corrwith(port_ewma.loc[idx])
                                        corr_df = pd.DataFrame({"CorrWith_Port_EWMA": corr}).sort_values("CorrWith_Port_EWMA", ascending=False)
                                        st.markdown("**Correlation: PCA Vol Factors vs Portfolio EWMA Vol**")
                                        st.dataframe(corr_df, use_container_width=True)

                                        fig = go.Figure()
                                        for pc in scores.columns:
                                            fig.add_trace(go.Scatter(x=scores.index, y=scores[pc], name=pc, mode="lines"))
                                        fig.update_layout(title="PCA Factor Scores (GARCH Vol Factors)", height=420,
                                                          xaxis_title="Date", yaxis_title="Score",
                                                          margin=dict(l=10, r=10, t=50, b=10))
                                        st.plotly_chart(fig, use_container_width=True, key=f"{key_ns}__pca_scores_chart")

                                        fig2 = go.Figure()
                                        fig2.add_trace(go.Scatter(x=port_ewma.index, y=port_ewma.values, name=f"Portfolio EWMA Vol (span={ewma_span})", mode="lines"))
                                        fig2.update_layout(title="Portfolio EWMA Volatility", height=360,
                                                           xaxis_title="Date", yaxis_title="Annualized Vol",
                                                           margin=dict(l=10, r=10, t=50, b=10))
                                        st.plotly_chart(fig2, use_container_width=True, key=f"{key_ns}__port_ewma_chart")


# =============================================================================
# 6) INTEGRATION HOOKS (tiny snippets)
# =============================================================================
#
# A) Add Copper + Aluminum to your futures universe:
#    If you have a dict like FUTURES_TICKERS or FUTURES_MAP, patch it:
#
#        FUTURES_TICKERS = patch_futures_universe(FUTURES_TICKERS)
#
#    (Do this right after your futures dict is defined.)
#
# B) Add a NEW TAB in your existing tab router (without removing anything):
#
#    If you have something like:
#        tabs = st.tabs([...existing...])
#    add another label at the end, e.g. "Portfolio Lab"
#
#    Then inside that tab call:
#
#        render_portfolio_lab_suite(
#            prices_df=prices_df,                 # your assets prices wide df
#            returns_df=returns_df,               # your assets returns wide df
#            benchmark_returns=bench_rets_dict,   # dict { 'SPY': spy_ret, 'XU100': xu100_ret, ... }
#            key_ns="portfolio_lab"
#        )
#
#    Note: benchmark_returns is optional. If you already compute benchmarks, pass them.
# =============================================================================

# =============================================================================
# END PORTFOLIO LAB PATCH (Merged)
# =============================================================================


if __name__ == "__main__":
    main()
