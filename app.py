"""
🏛️ Institutional Commodities Analytics Platform v6.1
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • Machine Learning • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance
By Murat KONUKLAR
"""

# =============================================================================
# BUILD / VERSION
# =============================================================================
__ICD_BUILD__ = "v7.3.8_DEPLOY_VERIFY"

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
# EXCEL EXPORT (CLOUD-SAFE ENGINE FALLBACK)
# =============================================================================
def icd_safe_excel_writer(buffer_obj):
    """
    Create a pandas ExcelWriter with a robust engine fallback.

    Streamlit Cloud deployments sometimes omit optional Excel dependencies.
    This helper tries `openpyxl` first (common default), then `xlsxwriter`.
    If neither engine is available, returns (None, None) and the caller can
    disable Excel export gracefully instead of crashing the app.
    """
    # Try openpyxl (preferred for .xlsx read/write compatibility)
    try:
        import openpyxl  # noqa: F401
        return pd.ExcelWriter(buffer_obj, engine="openpyxl"), "openpyxl"
    except Exception:
        pass

    # Try xlsxwriter (fast writer-only engine; great fallback on Cloud)
    try:
        import xlsxwriter  # noqa: F401
        return pd.ExcelWriter(buffer_obj, engine="xlsxwriter"), "xlsxwriter"
    except Exception:
        pass

    return None, None

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
    page_title="Institutional Commodities Platform v6.0 by Murat KONUKLAR",
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

# Build identifier (helps verify the deployed code on Streamlit Cloud)
try:
    st.sidebar.caption(f"Build: {__ICD_BUILD__}")
except Exception:
    pass

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
    start_date: datetime = field(default_factory=lambda: (datetime.now() - timedelta(days=1095)))
    end_date: datetime = field(default_factory=lambda: datetime.now())
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
    "Precious Metals": {
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
    "Industrial Metals": {
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
    "Energy": {
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
    "Agriculture": {
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
                border: 1px solid rgba(255, 255,255, 0.1);
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
        """Advanced portfolio optimization with robust covariance handling"""
        
        if returns_df.empty or len(returns_df) < 60:
            return {'success': False, 'message': 'Insufficient data'}
        
        n_assets = returns_df.shape[1]
        
        # Check for sufficient data
        if len(returns_df) < 2 * n_assets:
            return {'success': False, 'message': f'Insufficient data points ({len(returns_df)}) for {n_assets} assets. Need at least {2 * n_assets} observations.'}
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'sum_to_one': True
            }
        
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n_assets))
        
        # Initial weights (equal weight)
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
        
        # Calculate covariance matrix with enhanced numerical stability
        try:
            # Drop any assets with zero variance or NaN returns
            valid_assets = []
            for col in returns_df.columns:
                if returns_df[col].std() > 1e-8 and not returns_df[col].isna().all():
                    valid_assets.append(col)
            
            if len(valid_assets) < 2:
                return {'success': False, 'message': 'Insufficient valid assets for optimization'}
            
            # Use only valid assets
            returns_df = returns_df[valid_assets]
            n_assets = len(valid_assets)
            
            # Recalculate bounds and initial weights
            bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                          for _ in range(n_assets))
            init_weights = np.ones(n_assets) / n_assets
            
            # Calculate mean returns and covariance
            mean_returns = returns_df.mean() * self.annual_trading_days
            cov_matrix = returns_df.cov() * self.annual_trading_days
            
            # Enhanced regularization
            cov_matrix = self._ensure_psd_covariance(
                cov_matrix,
                method="higham",
                epsilon=1e-12,
                max_iter=100,
                tol=1e-7,
            )
            
            # Add small ridge regularization to ensure numerical stability
            ridge_lambda = 1e-6
            identity_matrix = np.eye(n_assets)
            cov_matrix = (1 - ridge_lambda) * cov_matrix + ridge_lambda * np.mean(np.diag(cov_matrix)) * identity_matrix
            
            # Check condition number
            cond_number = np.linalg.cond(cov_matrix)
            if cond_number > 1e10:
                # Add more regularization for ill-conditioned matrices
                ridge_lambda = 1e-4
                cov_matrix = (1 - ridge_lambda) * cov_matrix + ridge_lambda * np.mean(np.diag(cov_matrix)) * identity_matrix
                
        except Exception as e:
            return {'success': False, 'message': f'Covariance matrix calculation failed: {str(e)}'}
        
        # Define objective functions
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        def portfolio_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_vol = np.sqrt(max(weights.T @ cov_matrix @ weights, 1e-12))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 1e-12 else 1e6
        
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
        
        # Perform optimization with fallback methods
        try:
            result = optimize.minimize(
                objective,
                x0=init_weights,
                bounds=bounds,
                constraints=opt_constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-8}
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
                    'n_iterations': result.nit,
                    'condition_number': float(np.linalg.cond(cov_matrix))
                }
            else:
                # Try alternative method if SLSQP fails
                try:
                    result = optimize.minimize(
                        objective,
                        x0=init_weights,
                        bounds=bounds,
                        constraints=opt_constraints,
                        method='trust-constr',
                        options={'maxiter': 500, 'verbose': 0}
                    )
                    
                    if result.success:
                        optimized_weights = result.x
                        optimized_weights = optimized_weights / np.sum(optimized_weights)
                        
                        portfolio_returns = returns_df @ optimized_weights
                        metrics = self.calculate_performance_metrics(portfolio_returns)
                        
                        return {
                            'success': True,
                            'weights': dict(zip(returns_df.columns, optimized_weights)),
                            'metrics': metrics,
                            'risk_contributions': {},
                            'diversification_ratio': 1.0,
                            'objective_value': -result.fun if method == 'sharpe' else result.fun,
                            'n_iterations': result.nit,
                            'method_used': 'trust-constr'
                        }
                    else:
                        return {'success': False, 'message': f'Optimization failed: {result.message}'}
                        
                except Exception as e2:
                    return {'success': False, 'message': f'Both SLSQP and trust-constr failed: {str(e2)}'}
                    
        except Exception as e:
            return {'success': False, 'message': f'Optimization error: {str(e)}'}
    
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
    
    def _validate_returns_data(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean returns data for optimization"""
        if returns_df.empty:
            return returns_df
        
        # Remove assets with insufficient data
        min_obs = max(60, returns_df.shape[1] * 2)
        returns_df = returns_df.dropna(thresh=min_obs, axis=1)
        
        # Remove assets with zero or near-zero variance
        valid_cols = []
        for col in returns_df.columns:
            if returns_df[col].std() > 1e-8:
                valid_cols.append(col)
        
        returns_df = returns_df[valid_cols]
        
        # Remove any remaining NaNs by forward filling then backward filling
        returns_df = returns_df.ffill().bfill()
        
        # If still NaN, drop those rows
        returns_df = returns_df.dropna()
        
        return returns_df
    
    # =========================================================================
    # GARCH MODELING
    # =========================================================================
    
    def garch_analysis(
        self,
        returns: pd.Series,
        p: Optional[int] = None,
        q: Optional[int] = None,
        p_range: Tuple[int, int] = (1, 2),
        q_range: Tuple[int, int] = (1, 2),
        distributions: List[str] = None,
        dist: Optional[str] = None,
        annualize: bool = True
    ) -> Dict[str, Any]:
        """Perform GARCH analysis with Cloud-safe behavior and UI-compatible output.

        Fixes:
        - Streamlit UI calls this method with `p=` and `q=`; we now accept those keywords.
        - Returns `success` (alias of `available`) and exposes `conditional_volatility` for plotting.
        - Annualizes conditional volatility to match the visualization (Realized Vol is annualized).
        """
        # Dependency gate
        if not dep_manager.is_available("arch"):
            return {
                "available": False,
                "success": False,
                "message": "ARCH package not available. Add `arch` to requirements.txt to enable GARCH.",
            }

        if distributions is None:
            distributions = ["normal", "t", "skewt"]
        if dist is not None:
            distributions = [str(dist)]

        # Robust return cleaning
        try:
            r = pd.to_numeric(returns, errors="coerce")
        except Exception:
            r = returns.copy()
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        try:
            r = r[~r.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass

        if r is None or r.empty or len(r) < 60:
            return {
                "available": False,
                "success": False,
                "message": "Insufficient data for GARCH (need at least ~60 observations).",
                "n_obs": 0 if r is None else int(len(r)),
            }

        # Allow UI-style single model selection
        if p is not None:
            p_range = (int(p), int(p))
        if q is not None:
            q_range = (int(q), int(q))

        # Scale to percent for arch_model stability (common convention)
        returns_scaled = r.values.astype(float) * 100.0

        arch_model = dep_manager.dependencies["arch"]["arch_model"]

        # FIX: Use self.annual_trading_days instead of self.cfg
        ann_scale = math.sqrt(float(self.annual_trading_days)) if annualize else 1.0

        results: List[Dict[str, Any]] = []
        best = None  # track best by BIC
        best_bic = None

        for pp in range(int(p_range[0]), int(p_range[1]) + 1):
            for qq in range(int(q_range[0]), int(q_range[1]) + 1):
                for d in distributions:
                    try:
                        model = arch_model(
                            returns_scaled,
                            mean="Constant",
                            vol="GARCH",
                            p=int(pp),
                            q=int(qq),
                            dist=str(d),
                            rescale=False
                        )
                        fit = model.fit(disp="off", show_warning=False, update_freq=0)

                        # Conditional vol from arch is in percent (because input is percent).
                        # Convert to annualized decimal to match plotting.
                        cond_vol = np.asarray(fit.conditional_volatility, dtype=float)  # percent (daily)
                        cond_vol_dec = (cond_vol / 100.0) * ann_scale  # annualized decimal (if annualize)

                        cond_series = pd.Series(cond_vol_dec, index=r.index[:len(cond_vol_dec)])

                        row = {
                            "p": int(pp),
                            "q": int(qq),
                            "distribution": str(d),
                            "aic": float(getattr(fit, "aic", np.nan)),
                            "bic": float(getattr(fit, "bic", np.nan)),
                            "log_likelihood": float(getattr(fit, "loglikelihood", np.nan)),
                            "converged": bool(getattr(fit, "convergence_flag", 1) == 0),
                            "params": dict(getattr(fit, "params", {})),
                            "conditional_volatility": cond_series,
                        }
                        results.append(row)

                        bic_val = row["bic"]
                        if np.isfinite(bic_val):
                            if best_bic is None or bic_val < best_bic:
                                best_bic = bic_val
                                best = row

                    except Exception:
                        continue

        if not results or best is None:
            return {
                "available": False,
                "success": False,
                "message": "No GARCH models converged.",
                "n_models_tested": int(len(results)),
            }

        # Prepare a lightweight best_model dict for JSON (exclude the heavy series)
        best_model_json = {k: v for k, v in best.items() if k != "conditional_volatility"}

        return {
            "available": True,
            "success": True,
            "message": "GARCH model fit successful.",
            "best_model": best_model_json,
            "all_models": [
                {k: v for k, v in row.items() if k != "conditional_volatility"} for row in results
            ],
            "n_models_tested": int(len(results)),
            "conditional_volatility": best.get("conditional_volatility"),
            "returns": r,
            "annualized": bool(annualize),
        }
    
    def garch_forecast(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
        forecast_horizon: int = 30,
        dist: str = "normal"
    ) -> Dict[str, Any]:
        """Generate GARCH volatility forecasts"""
        if not dep_manager.is_available("arch"):
            return {"success": False, "message": "ARCH package not available"}
        
        try:
            # Clean returns
            r = returns.dropna()
            if len(r) < 100:
                return {"success": False, "message": "Insufficient data for forecasting"}
            
            # Scale returns
            returns_scaled = r.values.astype(float) * 100.0
            # =============================================================================
# COMPLETION OF THE CODE WITH ALL NECESSARY METHODS
# =============================================================================

    def regime_detection(
        self,
        prices: pd.Series,
        returns: pd.Series,
        n_states: int = 3,
        n_features: int = 5
    ) -> Dict[str, Any]:
        """Detect market regimes using Hidden Markov Models"""
        if not dep_manager.is_available("hmmlearn"):
            return {"success": False, "message": "hmmlearn not available"}
        
        try:
            from sklearn.preprocessing import StandardScaler
            from hmmlearn.hmm import GaussianHMM
            
            # Prepare features
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = returns.values
            features['volatility'] = returns.rolling(20).std().values
            features['volume'] = prices if 'Volume' in prices.name else 0
            
            # Technical indicators as features
            features['sma_ratio'] = (prices / prices.rolling(20).mean()).values
            features['rsi'] = self._calculate_rsi(prices).values
            features['atr'] = self._calculate_atr(prices).values
            
            # Drop NaN
            features = features.dropna()
            
            if len(features) < 100:
                return {"success": False, "message": "Insufficient data for regime detection"}
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit HMM
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=1000,
                random_state=42
            )
            
            hmm.fit(scaled_features)
            
            # Predict states
            states = hmm.predict(scaled_features)
            state_probs = hmm.predict_proba(scaled_features)
            
            # Calculate state statistics
            state_stats = {}
            for state in range(n_states):
                mask = states == state
                if mask.any():
                    state_returns = returns.iloc[mask]
                    state_stats[state] = {
                        'count': int(mask.sum()),
                        'mean_return': float(state_returns.mean()),
                        'volatility': float(state_returns.std()),
                        'sharpe': float((state_returns.mean() - self.risk_free_rate/252) / state_returns.std() if state_returns.std() > 0 else 0),
                        'probability': float(state_probs[:, state].mean())
                    }
            
            # Transition matrix
            transition_matrix = hmm.transmat_
            
            return {
                "success": True,
                "states": states,
                "state_probabilities": state_probs,
                "state_statistics": state_stats,
                "transition_matrix": transition_matrix,
                "model_score": float(hmm.score(scaled_features)),
                "features_used": list(features.columns),
                "n_observations": len(features)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Regime detection failed: {str(e)}"}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        atr = (high - low).rolling(period).mean()
        return atr
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        initial_price: float,
        n_simulations: int = 10000,
        n_days: int = 252,
        confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulation for price forecasting"""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 60:
                return {"success": False, "message": "Insufficient data for simulation"}
            
            # Calculate parameters
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            
            # Generate simulations
            np.random.seed(42)
            simulations = np.zeros((n_simulations, n_days))
            
            for i in range(n_simulations):
                # Random returns from normal distribution
                daily_returns = np.random.normal(mu, sigma, n_days)
                # Calculate price path
                price_path = initial_price * np.cumprod(1 + daily_returns)
                simulations[i] = price_path
            
            # Calculate statistics
            final_prices = simulations[:, -1]
            
            # Calculate Value at Risk (VaR)
            var_results = {}
            for cl in confidence_levels:
                var = np.percentile(final_prices, (1 - cl) * 100)
                var_results[f"var_{int(cl*100)}"] = float(var)
            
            # Calculate Expected Shortfall (CVaR)
            cvar_results = {}
            for cl in confidence_levels:
                threshold = np.percentile(final_prices, (1 - cl) * 100)
                cvar = final_prices[final_prices <= threshold].mean()
                cvar_results[f"cvar_{int(cl*100)}"] = float(cvar)
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for cl in confidence_levels:
                lower = np.percentile(final_prices, (1 - cl) * 100 / 2)
                upper = np.percentile(final_prices, 100 - (1 - cl) * 100 / 2)
                confidence_intervals[f"ci_{int(cl*100)}"] = (float(lower), float(upper))
            
            # Calculate probability of profit/loss
            prob_profit = np.mean(final_prices > initial_price) * 100
            prob_loss = 100 - prob_profit
            
            # Expected return and volatility
            expected_return = np.mean(final_prices) / initial_price - 1
            expected_volatility = np.std(final_prices) / initial_price
            
            return {
                "success": True,
                "simulations": simulations,
                "final_prices": final_prices,
                "initial_price": float(initial_price),
                "expected_price": float(np.mean(final_prices)),
                "median_price": float(np.median(final_prices)),
                "var_results": var_results,
                "cvar_results": cvar_results,
                "confidence_intervals": confidence_intervals,
                "prob_profit": float(prob_profit),
                "prob_loss": float(prob_loss),
                "expected_return": float(expected_return * 100),
                "expected_volatility": float(expected_volatility * 100),
                "mu": float(mu),
                "sigma": float(sigma),
                "n_simulations": n_simulations,
                "n_days": n_days
            }
            
        except Exception as e:
            return {"success": False, "message": f"Monte Carlo simulation failed: {str(e)}"}
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    def correlation_analysis(
        self,
        returns_df: pd.DataFrame,
        method: str = 'pearson',
        rolling_window: int = 60
    ) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        try:
            # Clean data
            returns_clean = returns_df.dropna()
            
            if returns_clean.empty or len(returns_clean) < rolling_window:
                return {"success": False, "message": "Insufficient data for correlation analysis"}
            
            # Static correlation matrix
            if method == 'pearson':
                corr_matrix = returns_clean.corr()
            elif method == 'spearman':
                corr_matrix = returns_clean.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = returns_clean.corr(method='kendall')
            else:
                corr_matrix = returns_clean.corr()
            
            # Calculate rolling correlations
            rolling_corrs = {}
            assets = returns_clean.columns.tolist()
            
            if len(assets) >= 2:
                for i in range(len(assets)):
                    for j in range(i + 1, len(assets)):
                        pair = f"{assets[i]}-{assets[j]}"
                        rolling_corr = returns_clean[assets[i]].rolling(window=rolling_window).corr(returns_clean[assets[j]])
                        rolling_corrs[pair] = rolling_corr
            
            # Calculate correlation statistics
            corr_stats = {
                'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
                'min_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
                'correlation_volatility': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std())
            }
            
            # Eigenvalue decomposition for correlation structure
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix.values)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate explained variance
            explained_variance = eigenvalues / np.sum(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance)
            
            return {
                "success": True,
                "correlation_matrix": corr_matrix,
                "rolling_correlations": rolling_corrs,
                "correlation_statistics": corr_stats,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "explained_variance": explained_variance,
                "cumulative_variance": cumulative_variance,
                "n_assets": len(assets),
                "method": method,
                "rolling_window": rolling_window
            }
            
        except Exception as e:
            return {"success": False, "message": f"Correlation analysis failed: {str(e)}"}

# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate professional reports and exports"""
    
    def __init__(self):
        self.data_manager = EnhancedDataManager()
    
    def generate_performance_report(
        self,
        metrics: Dict[str, Any],
        asset_name: str,
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            "header": {
                "asset": asset_name,
                "analysis_date": analysis_date.isoformat(),
                "report_type": "Performance Analysis",
                "version": "1.0"
            },
            "summary": {
                "return_metrics": {
                    "total_return": metrics.get('total_return', 0),
                    "annual_return": metrics.get('annual_return', 0),
                    "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                    "sortino_ratio": metrics.get('sortino_ratio', 0)
                },
                "risk_metrics": {
                    "annual_volatility": metrics.get('annual_volatility', 0),
                    "max_drawdown": metrics.get('max_drawdown', 0),
                    "var_95": metrics.get('var_95', 0),
                    "cvar_95": metrics.get('cvar_95', 0)
                },
                "trading_metrics": {
                    "win_rate": metrics.get('win_rate', 0),
                    "profit_factor": metrics.get('profit_factor', 0),
                    "avg_gain": metrics.get('avg_gain', 0),
                    "avg_loss": metrics.get('avg_loss', 0)
                }
            },
            "detailed_analysis": metrics
        }
        
        return report
    
    def generate_portfolio_report(
        self,
        optimization_results: Dict[str, Any],
        allocation: Dict[str, float],
        analysis_date: datetime
    ) -> Dict[str, Any]:
        """Generate portfolio optimization report"""
        
        report = {
            "header": {
                "report_type": "Portfolio Optimization",
                "analysis_date": analysis_date.isoformat(),
                "optimization_method": optimization_results.get('method_used', 'SLSQP'),
                "status": "Success" if optimization_results.get('success') else "Failed"
            },
            "optimization_results": {
                "success": optimization_results.get('success', False),
                "objective_value": optimization_results.get('objective_value', 0),
                "iterations": optimization_results.get('n_iterations', 0),
                "condition_number": optimization_results.get('condition_number', 0)
            },
            "portfolio_allocation": allocation,
            "portfolio_metrics": optimization_results.get('metrics', {}),
            "risk_analysis": {
                "risk_contributions": optimization_results.get('risk_contributions', {}),
                "diversification_ratio": optimization_results.get('diversification_ratio', 1.0)
            }
        }
        
        return report
    
    def create_excel_export(
        self,
        data_dict: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any],
        filename: str = "commodities_analysis"
    ) -> BytesIO:
        """Create comprehensive Excel export"""
        
        buffer = BytesIO()
        writer, engine = icd_safe_excel_writer(buffer)
        
        if writer is None:
            st.warning("Excel export not available - no engine found")
            return buffer
        
        try:
            # Write data sheets
            for sheet_name, df in data_dict.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31])
            
            # Write analysis results
            if analysis_results:
                analysis_df = pd.DataFrame([analysis_results])
                analysis_df.to_excel(writer, sheet_name="Analysis_Results")
            
            # Create summary sheet
            summary_data = {
                "Metric": ["Analysis Date", "Assets Analyzed", "Status"],
                "Value": [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    len(data_dict),
                    "Completed"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            writer.close()
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            return buffer
    
    def create_pdf_report(self, report_data: Dict[str, Any]) -> BytesIO:
        """Create PDF report (simplified version for Streamlit Cloud)"""
        buffer = BytesIO()
        
        # Create HTML report
        html_content = self._create_html_report(report_data)
        
        # For Streamlit Cloud, we'll return HTML that can be displayed
        # In production, you would use weasyprint or reportlab to create PDF
        buffer.write(html_content.encode())
        buffer.seek(0)
        
        return buffer
    
    def _create_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #1a2980, #26d0ce); 
                          color: white; padding: 30px; border-radius: 10px; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; margin: 10px 0; 
                               border-radius: 8px; border-left: 5px solid #1a2980; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #1a2980; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background-color: #1a2980; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏛️ Institutional Commodities Analysis</h1>
                <p>Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Performance Summary</h2>
            <div class="metric-card">
                <h3>Return Metrics</h3>
                <p>Annual Return: <span class="metric-value">{report_data.get('summary', {}).get('return_metrics', {}).get('annual_return', 0):.2f}%</span></p>
                <p>Sharpe Ratio: <span class="metric-value">{report_data.get('summary', {}).get('return_metrics', {}).get('sharpe_ratio', 0):.2f}</span></p>
            </div>
            
            <div class="metric-card">
                <h3>Risk Metrics</h3>
                <p>Annual Volatility: <span class="metric-value">{report_data.get('summary', {}).get('risk_metrics', {}).get('annual_volatility', 0):.2f}%</span></p>
                <p>Max Drawdown: <span class="metric-value">{report_data.get('summary', {}).get('risk_metrics', {}).get('max_drawdown', 0):.2f}%</span></p>
            </div>
            
            <h2>Detailed Analysis</h2>
            <table class="table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add detailed metrics
        for key, value in report_data.get('detailed_analysis', {}).items():
            if isinstance(value, (int, float)):
                html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html

# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

class StreamlitUI:
    """Streamlit UI components for the platform"""
    
    @staticmethod
    def render_header():
        """Render main application header"""
        st.markdown("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.8rem;">🏛️ Institutional Commodities Analytics</h1>
                <p style="font-size: 1.2rem; opacity: 0.9; margin: 0.5rem 0 0 0;">
                    Advanced Portfolio Analytics • GARCH Volatility Modeling • Machine Learning Regime Detection
                </p>
                <div style="margin-top: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span class="status-badge status-success">Live Market Data</span>
                    <span class="status-badge status-info">Real-time Analytics</span>
                    <span class="status-badge status-warning">Institutional Grade</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_card(title: str, value: Any, change: Optional[float] = None, 
                          format_str: str = "{:,.2f}", icon: str = "📊"):
        """Render a metric card"""
        
        if isinstance(value, (int, float)):
            display_value = format_str.format(value)
        else:
            display_value = str(value)
        
        change_html = ""
        if change is not None:
            change_color = "var(--success)" if change >= 0 else "var(--danger)"
            change_icon = "📈" if change >= 0 else "📉"
            change_html = f"""
                <div style="font-size: 0.9rem; color: {change_color}; margin-top: 0.5rem;">
                    {change_icon} {abs(change):.2f}%
                </div>
            """
        
        html = f"""
            <div class="metric-card">
                <div class="metric-label">
                    {icon} {title}
                </div>
                <div class="metric-value">
                    {display_value}
                </div>
                {change_html}
            </div>
        """
        
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_config():
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=1095),
                    max_value=datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            
            # Asset selection
            st.markdown("### 📊 Asset Selection")
            
            selected_assets = []
            for category, assets in COMMODITIES_UNIVERSE.items():
                with st.expander(f"{category}", expanded=True):
                    for symbol, metadata in assets.items():
                        if st.checkbox(
                            f"{metadata.name} ({symbol})",
                            value=True,
                            key=f"asset_{symbol}"
                        ):
                            selected_assets.append(symbol)
            
            # Benchmark selection
            st.markdown("### 📈 Benchmarks")
            selected_benchmarks = st.multiselect(
                "Select Benchmarks",
                options=list(BENCHMARKS.keys()),
                default=["^GSPC", "GLD"],
                format_func=lambda x: BENCHMARKS[x]["name"]
            )
            
            # Analysis parameters
            st.markdown("### 🔧 Analysis Parameters")
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1
            ) / 100
            
            garch_p = st.slider("GARCH p parameter", 1, 3, 1)
            garch_q = st.slider("GARCH q parameter", 1, 3, 1)
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "risk_free_rate": risk_free_rate,
                "garch_p": garch_p,
                "garch_q": garch_q
            }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Initialize components
    ui = StreamlitUI()
    data_manager = EnhancedDataManager()
    analytics = InstitutionalAnalytics()
    report_generator = ReportGenerator()
    
    # Render header
    ui.render_header()
    
    # Get configuration from sidebar
    config = ui.render_sidebar_config()
    
    # Check if assets are selected
    if not config["selected_assets"]:
        st.warning("⚠️ Please select at least one asset to analyze")
        return
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "📊 Portfolio Analysis",
        "📈 GARCH Volatility",
        "🔍 Regime Detection",
        "🎯 Monte Carlo",
        "📋 Reports"
    ])
    
    # Tab 1: Portfolio Analysis
    with tabs[0]:
        st.markdown("## 📊 Portfolio Analysis")
        
        with st.spinner("Fetching market data..."):
            # Fetch data
            all_symbols = config["selected_assets"] + config["selected_benchmarks"]
            data = data_manager.fetch_multiple_assets(
                all_symbols,
                config["start_date"],
                config["end_date"]
            )
            
            if not data:
                st.error("Failed to fetch data. Please try again.")
                return
            
            # Calculate returns
            returns_data = {}
            for symbol, df in data.items():
                if not df.empty and 'Adj_Close' in df.columns:
                    returns = df['Adj_Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                st.warning("Insufficient data for portfolio analysis")
                return
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # Display correlation heatmap
            st.markdown("### Correlation Matrix")
            corr_matrix = returns_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Asset Correlations",
                height=500,
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio optimization
            st.markdown("### Portfolio Optimization")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["sharpe", "min_variance", "max_return"],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            with col2:
                min_weight = st.slider("Min Weight %", 0, 20, 0) / 100
            with col3:
                max_weight = st.slider("Max Weight %", 50, 100, 100) / 100
            
            if st.button("Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    constraints = {
                        'min_weight': min_weight,
                        'max_weight': max_weight,
                        'sum_to_one': True
                    }
                    
                    result = analytics.optimize_portfolio(
                        returns_df[config["selected_assets"]],
                        method=optimization_method,
                        constraints=constraints
                    )
                    
                    if result["success"]:
                        # Display optimized weights
                        weights_df = pd.DataFrame(
                            list(result["weights"].items()),
                            columns=["Asset", "Weight"]
                        )
                        weights_df["Weight"] = weights_df["Weight"] * 100
                        
                        # Create pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=weights_df["Asset"],
                            values=weights_df["Weight"],
                            hole=.3,
                            textinfo='label+percent'
                        )])
                        
                        fig.update_layout(
                            title="Optimized Portfolio Allocation",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics
                        st.markdown("### Portfolio Performance Metrics")
                        
                        metrics = result["metrics"]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            ui.render_metric_card(
                                "Annual Return",
                                metrics.get('annual_return', 0),
                                format_str="{:.2f}%"
                            )
                        
                        with col2:
                            ui.render_metric_card(
                                "Sharpe Ratio",
                                metrics.get('sharpe_ratio', 0),
                                format_str="{:.2f}"
                            )
                        
                        with col3:
                            ui.render_metric_card(
                                "Max Drawdown",
                                metrics.get('max_drawdown', 0),
                                format_str="{:.2f}%"
                            )
                        
                        with col4:
                            ui.render_metric_card(
                                "Win Rate",
                                metrics.get('win_rate', 0),
                                format_str="{:.2f}%"
                            )
                    else:
                        st.error(f"Optimization failed: {result['message']}")
    
    # Tab 2: GARCH Volatility
    with tabs[1]:
        st.markdown("## 📈 GARCH Volatility Analysis")
        
        if config["selected_assets"]:
            selected_asset = st.selectbox(
                "Select Asset for GARCH Analysis",
                config["selected_assets"],
                format_func=lambda x: COMMODITIES_UNIVERSE[
                    [cat for cat, assets in COMMODITIES_UNIVERSE.items() if x in assets][0]
                ][x].name
            )
            
            if selected_asset in data and not data[selected_asset].empty:
                df = data[selected_asset]
                
                if 'Adj_Close' in df.columns:
                    returns = df['Adj_Close'].pct_change().dropna()
                    
                    # Perform GARCH analysis
                    garch_result = analytics.garch_analysis(
                        returns,
                        p=config["garch_p"],
                        q=config["garch_q"]
                    )
                    
                    if garch_result.get("success", False):
                        # Display conditional volatility
                        st.markdown("### Conditional Volatility")
                        
                        if "conditional_volatility" in garch_result:
                            vol_series = garch_result["conditional_volatility"]
                            
                            # Calculate realized volatility for comparison
                            realized_vol = returns.rolling(20).std() * np.sqrt(252)
                            
                            # Create comparison plot
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=vol_series.index,
                                y=vol_series.values * 100,
                                mode='lines',
                                name='GARCH Conditional Vol',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=realized_vol.index,
                                y=realized_vol.values * 100,
                                mode='lines',
                                name='Realized Vol (20D)',
                                line=dict(color='blue', width=1, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="Volatility Comparison: GARCH vs Realized",
                                xaxis_title="Date",
                                yaxis_title="Volatility (%)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display GARCH parameters
                        st.markdown("### GARCH Model Parameters")
                        
                        if "best_model" in garch_result:
                            best_model = garch_result["best_model"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("p parameter", best_model.get("p", "N/A"))
                            
                            with col2:
                                st.metric("q parameter", best_model.get("q", "N/A"))
                            
                            with col3:
                                st.metric("Distribution", best_model.get("distribution", "N/A"))
                            
                            with col4:
                                st.metric("BIC", f"{best_model.get('bic', 0):.2f}")
                        
                        # Display model comparison
                        if "all_models" in garch_result:
                            st.markdown("### Model Comparison")
                            
                            models_df = pd.DataFrame(garch_result["all_models"])
                            if not models_df.empty:
                                st.dataframe(
                                    models_df.sort_values("bic").style.highlight_min(
                                        subset=["bic", "aic"], color="lightgreen"
                                    ),
                                    use_container_width=True
                                )
                    else:
                        st.warning(garch_result.get("message", "GARCH analysis failed"))
    
    # Tab 3: Regime Detection
    with tabs[2]:
        st.markdown("## 🔍 Market Regime Detection")
        
        if config["selected_assets"] and dep_manager.is_available("hmmlearn"):
            selected_asset = st.selectbox(
                "Select Asset for Regime Analysis",
                config["selected_assets"],
                key="regime_asset",
                format_func=lambda x: COMMODITIES_UNIVERSE[
                    [cat for cat, assets in COMMODITIES_UNIVERSE.items() if x in assets][0]
                ][x].name
            )
            
            if selected_asset in data and not data[selected_asset].empty:
                df = data[selected_asset]
                
                if 'Adj_Close' in df.columns:
                    prices = df['Adj_Close']
                    returns = prices.pct_change().dropna()
                    
                    # Perform regime detection
                    regime_result = analytics.regime_detection(
                        prices,
                        returns,
                        n_states=3
                    )
                    
                    if regime_result.get("success", False):
                        # Plot regime states
                        st.markdown("### Market Regimes")
                        
                        states = regime_result["states"]
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Price with Regimes", "Regime States"),
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Price trace
                        fig.add_trace(
                            go.Scatter(
                                x=prices.index,
                                y=prices.values,
                                mode='lines',
                                name='Price',
                                line=dict(color='blue', width=1)
                            ),
                            row=1, col=1
                        )
                        
                        # Regime states as background
                        unique_states = np.unique(states)
                        colors = ['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.1)', 'rgba(0,0,255,0.1)']
                        
                        for i, state in enumerate(unique_states):
                            mask = states == state
                            if mask.any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=prices.index[mask],
                                        y=prices.values[mask],
                                        mode='markers',
                                        name=f'Regime {state}',
                                        marker=dict(color=colors[i % len(colors)], size=6),
                                        showlegend=True
                                    ),
                                    row=1, col=1
                                )
                        
                        # Regime state trace
                        fig.add_trace(
                            go.Scatter(
                                x=prices.index[:len(states)],
                                y=states,
                                mode='lines',
                                name='Regime',
                                line=dict(color='purple', width=2)
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=600,
                            showlegend=True,
                            hovermode='x unified'
                        )
                        
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="Regime", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display regime statistics
                        st.markdown("### Regime Statistics")
                        
                        if "state_statistics" in regime_result:
                            stats_df = pd.DataFrame(
                                regime_result["state_statistics"]
                            ).T
                            
                            if not stats_df.empty:
                                st.dataframe(
                                    stats_df.style.format({
                                        'mean_return': '{:.4f}',
                                        'volatility': '{:.4f}',
                                        'sharpe': '{:.2f}',
                                        'probability': '{:.2%}'
                                    }),
                                    use_container_width=True
                                )
                    else:
                        st.warning(regime_result.get("message", "Regime detection failed"))
        else:
            st.info("Regime detection requires hmmlearn. Install with: pip install hmmlearn scikit-learn")
    
    # Tab 4: Monte Carlo Simulation
    with tabs[3]:
        st.markdown("## 🎯 Monte Carlo Simulation")
        
        if config["selected_assets"]:
            selected_asset = st.selectbox(
                "Select Asset for Simulation",
                config["selected_assets"],
                key="mc_asset",
                format_func=lambda x: COMMODITIES_UNIVERSE[
                    [cat for cat, assets in COMMODITIES_UNIVERSE.items() if x in assets][0]
                ][x].name
            )
            
            if selected_asset in data and not data[selected_asset].empty:
                df = data[selected_asset]
                
                if 'Adj_Close' in df.columns:
                    current_price = df['Adj_Close'].iloc[-1]
                    returns = df['Adj_Close'].pct_change().dropna()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_simulations = st.slider(
                            "Number of Simulations",
                            min_value=1000,
                            max_value=20000,
                            value=10000,
                            step=1000
                        )
                    
                    with col2:
                        forecast_days = st.slider(
                            "Forecast Horizon (Days)",
                            min_value=30,
                            max_value=500,
                            value=252,
                            step=30
                        )
                    
                    with col3:
                        initial_price = st.number_input(
                            "Initial Price",
                            value=float(current_price),
                            min_value=0.1,
                            step=1.0
                        )
                    
                    if st.button("Run Monte Carlo Simulation", type="primary"):
                        with st.spinner(f"Running {n_simulations:,} simulations..."):
                            mc_result = analytics.monte_carlo_simulation(
                                returns,
                                initial_price,
                                n_simulations=n_simulations,
                                n_days=forecast_days
                            )
                            
                            if mc_result.get("success", False):
                                # Display simulation results
                                st.markdown("### Simulation Results")
                                
                                # Show key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    ui.render_metric_card(
                                        "Expected Price",
                                        mc_result.get("expected_price", 0),
                                        format_str="${:,.2f}"
                                    )
                                
                                with col2:
                                    ui.render_metric_card(
                                        "Probability of Profit",
                                        mc_result.get("prob_profit", 0),
                                        format_str="{:.1f}%"
                                    )
                                
                                with col3:
                                    var_95 = mc_result.get("var_results", {}).get("var_95", 0)
                                    ui.render_metric_card(
                                        "95% VaR",
                                        var_95,
                                        format_str="${:,.2f}"
                                    )
                                
                                with col4:
                                    cvar_95 = mc_result.get("cvar_results", {}).get("cvar_95", 0)
                                    ui.render_metric_card(
                                        "95% CVaR",
                                        cvar_95,
                                        format_str="${:,.2f}"
                                    )
                                
                                # Plot simulation paths
                                st.markdown("### Simulation Paths")
                                
                                simulations = mc_result.get("simulations", np.array([]))
                                if simulations.size > 0:
                                    # Plot sample of paths
                                    n_sample_paths = min(100, simulations.shape[0])
                                    sample_indices = np.random.choice(
                                        simulations.shape[0],
                                        n_sample_paths,
                                        replace=False
                                    )
                                    
                                    fig = go.Figure()
                                    
                                    for idx in sample_indices:
                                        fig.add_trace(go.Scatter(
                                            x=list(range(forecast_days)),
                                            y=simulations[idx],
                                            mode='lines',
                                            line=dict(width=1, color='rgba(0,100,255,0.1)'),
                                            showlegend=False
                                        ))
                                    
                                    # Add mean and confidence intervals
                                    mean_path = simulations.mean(axis=0)
                                    std_path = simulations.std(axis=0)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=list(range(forecast_days)),
                                        y=mean_path,
                                        mode='lines',
                                        name='Mean Path',
                                        line=dict(color='red', width=3)
                                    ))
                                    
                                    # Add confidence intervals
                                    for ci_level in [0.90, 0.95]:
                                        lower = np.percentile(simulations, (1 - ci_level) * 50, axis=0)
                                        upper = np.percentile(simulations, 100 - (1 - ci_level) * 50, axis=0)
                                        
                                        fig.add_trace(go.Scatter(
                                            x=list(range(forecast_days)),
                                            y=upper,
                                            mode='lines',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=list(range(forecast_days)),
                                            y=lower,
                                            mode='lines',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            name=f'{int(ci_level*100)}% CI',
                                            fillcolor=f'rgba(255,0,0,{0.3/ci_level})'
                                        ))
                                    
                                    fig.update_layout(
                                        title=f"Monte Carlo Simulation Paths (Sample of {n_sample_paths})",
                                        xaxis_title="Days Ahead",
                                        yaxis_title="Price",
                                        height=500,
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display final price distribution
                                    st.markdown("### Final Price Distribution")
                                    
                                    final_prices = mc_result.get("final_prices", np.array([]))
                                    
                                    if final_prices.size > 0:
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Histogram(
                                            x=final_prices,
                                            nbinsx=50,
                                            name='Price Distribution',
                                            marker_color='blue',
                                            opacity=0.7
                                        ))
                                        
                                        # Add vertical lines for key statistics
                                        fig.add_vline(
                                            x=initial_price,
                                            line_dash="dash",
                                            line_color="green",
                                            annotation_text="Initial Price"
                                        )
                                        
                                        fig.add_vline(
                                            x=np.mean(final_prices),
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text="Mean"
                                        )
                                        
                                        fig.add_vline(
                                            x=np.median(final_prices),
                                            line_dash="dash",
                                            line_color="orange",
                                            annotation_text="Median"
                                        )
                                        
                                        fig.update_layout(
                                            title="Distribution of Final Prices",
                                            xaxis_title="Final Price",
                                            yaxis_title="Frequency",
                                            height=400,
                                            bargap=0.1
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(mc_result.get("message", "Monte Carlo simulation failed"))
    
    # Tab 5: Reports
    with tabs[4]:
        st.markdown("## 📋 Reports & Export")
        
        # Generate report options
        report_type = st.selectbox(
            "Select Report Type",
            ["Performance Report", "Portfolio Report", "Full Analysis Export"]
        )
        
        if report_type == "Performance Report":
            if config["selected_assets"]:
                selected_asset = st.selectbox(
                    "Select Asset",
                    config["selected_assets"],
                    key="report_asset",
                    format_func=lambda x: COMMODITIES_UNIVERSE[
                        [cat for cat, assets in COMMODITIES_UNIVERSE.items() if x in assets][0]
                    ][x].name
                )
                
                if selected_asset in data and not data[selected_asset].empty:
                    df = data[selected_asset]
                    
                    if 'Adj_Close' in df.columns:
                        returns = df['Adj_Close'].pct_change().dropna()
                        
                        # Calculate metrics
                        analytics.risk_free_rate = config["risk_free_rate"]
                        metrics = analytics.calculate_performance_metrics(returns)
                        
                        if metrics:
                            # Generate report
                            report = report_generator.generate_performance_report(
                                metrics,
                                selected_asset,
                                datetime.now()
                            )
                            
                            # Display report
                            st.markdown("### Performance Report Preview")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.json(report["summary"], expanded=False)
                            
                            with col2:
                                # Create download button for JSON
                                report_json = json.dumps(report, indent=2)
                                st.download_button(
                                    label="📥 Download JSON Report",
                                    data=report_json,
                                    file_name=f"performance_report_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.json",
                                    mime="application/json"
                                )
                                
                                # Create Excel export
                                excel_data = {
                                    "Price_Data": df,
                                    "Returns": pd.DataFrame(returns, columns=['Returns'])
                                }
                                
                                excel_buffer = report_generator.create_excel_export(
                                    excel_data,
                                    metrics,
                                    filename=f"performance_{selected_asset}"
                                )
                                
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=excel_buffer,
                                    file_name=f"performance_analysis_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
        
        elif report_type == "Full Analysis Export":
            st.markdown("### Export Complete Analysis")
            
            if st.button("Generate Full Export", type="primary"):
                with st.spinner("Compiling analysis data..."):
                    # Collect all data
                    export_data = {}
                    
                    for symbol in config["selected_assets"]:
                        if symbol in data:
                            export_data[f"{symbol}_data"] = data[symbol]
                    
                    # Create comprehensive export
                    excel_buffer = report_generator.create_excel_export(
                        export_data,
                        {"analysis_date": datetime.now().isoformat()},
                        filename="full_commodities_analysis"
                    )
                    
                    st.success("Analysis compiled successfully!")
                    
                    st.download_button(
                        label="📥 Download Full Analysis (Excel)",
                        data=excel_buffer,
                        file_name=f"commodities_full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
        
        # System diagnostics
        with st.expander("System Diagnostics"):
            st.markdown("### Dependencies Status")
            
            for dep_name, dep_info in dep_manager.dependencies.items():
                status = "✅ Available" if dep_info.get('available') else "❌ Not Available"
                st.text(f"{dep_name}: {status}")
            
            st.markdown("### Cache Information")
            st.text(f"Cache entries: {len(st.session_state) if 'session_state' in dir(st) else 'N/A'}")
            
            # Memory usage (approximate)
            import sys
            memory_mb = sys.getsizeof(st.session_state) / (1024 * 1024) if 'session_state' in dir(st) else 0
            st.text(f"Session memory: {memory_mb:.2f} MB")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data_cache = {}
            st.session_state.analysis_cache = {}
        
        # Run main application
        main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
        
        # Provide recovery options
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
