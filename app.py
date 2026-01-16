"""
🏛️ Institutional Commodities Analytics Platform v6.1
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • Machine Learning • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance
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
        method: str = "historical",
        horizon: int = 1,
        use_log_aggregation: bool = True
    ) -> Dict[str, Any]:
        """Robust VaR / CVaR(ES) / ES engine (NaN-proof, horizon-aware).

        Fixes common production issues:
        - NaNs in VaR/CVaR/ES from residual NaNs/Infs or tiny effective samples.
        - Incorrect multi-day scaling (sqrt approximation) by computing horizon returns directly.
        - Key mismatches between analytics output and Streamlit UI expectations.

        Returns POSITIVE loss measures in decimal units:
        - VaR: 0.02 means 2% loss
        - CVaR/ES: expected shortfall (positive)
        """
        # Defensive cleaning: numeric, drop inf/nan, stable order, unique index
        try:
            rr = pd.to_numeric(returns, errors="coerce")
        except Exception:
            rr = returns.copy()

        try:
            rr = rr.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            rr = rr.dropna() if hasattr(rr, "dropna") else rr

        try:
            rr = rr[~rr.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass

        if rr is None or getattr(rr, "empty", False):
            return {"success": False, "message": "No valid returns available for VaR.", "n_obs": 0, "horizon": int(horizon)}

        # Horizon aggregation (compute H-day returns explicitly)
        try:
            h = int(horizon)
        except Exception:
            h = 1
        h = max(1, h)

        if h > 1:
            try:
                if use_log_aggregation:
                    # log aggregation is numerically stable: exp(sum(log(1+r))) - 1
                    lr = np.log1p(rr.astype(float))
                    agg = lr.rolling(h).sum()
                    rr_h = np.expm1(agg).dropna()
                else:
                    rr_h = rr.astype(float).rolling(h).sum().dropna()
            except Exception:
                rr_h = rr.copy()
        else:
            rr_h = rr.copy()

        if rr_h is None or getattr(rr_h, "empty", False):
            return {"success": False, "message": "No valid horizon-aggregated returns for VaR.", "n_obs": 0, "horizon": int(h)}

        # Final sanitize (nanquantile safety)
        try:
            rr_h = pd.to_numeric(rr_h, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            pass

        n = int(len(rr_h))
        if n <= 0:
            return {"success": False, "message": "No valid returns available after cleaning.", "n_obs": 0, "horizon": int(h)}

        method = (method or "historical").lower().strip()
        cl = float(confidence_level) if confidence_level is not None else 0.95
        cl = min(max(cl, 0.50), 0.999)  # clamp to safe range
        alpha = 1.0 - cl  # tail probability (e.g., 0.05 for 95% VaR)

        # Moments (guard against NaN std when n < 2)
        mu = float(rr_h.mean()) if n > 0 else 0.0
        if n >= 2:
            sigma = float(rr_h.std(ddof=1))
            if not np.isfinite(sigma):
                sigma = float(rr_h.std(ddof=0))
        else:
            sigma = 0.0

        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sigma):
            sigma = 0.0

        warning = ""
        if n < 60:
            warning = f"Small effective sample (n={n}). Results may be unstable."

        var = 0.0
        cvar = 0.0

        try:
            if method == "historical":
                q = float(np.nanquantile(rr_h.values, alpha))
                var = -q
                tail = rr_h[rr_h <= q]
                cvar = -float(np.nanmean(tail.values)) if len(tail) > 0 else float(var)

            elif method == "parametric":
                if sigma < 1e-12:
                    var, cvar = 0.0, 0.0
                else:
                    z = float(stats.norm.ppf(alpha))
                    q = mu + sigma * z
                    var = -(q)
                    pdf = float(stats.norm.pdf(z))
                    cvar = -mu + sigma * (pdf / max(alpha, 1e-12))

            elif method == "modified":
                # Cornish-Fisher adjusted quantile (using empirical skew/excess kurtosis)
                if sigma < 1e-12:
                    var, cvar = 0.0, 0.0
                else:
                    z = float(stats.norm.ppf(alpha))
                    try:
                        s = float(rr_h.skew())
                    except Exception:
                        s = 0.0
                    try:
                        k_ex = float(rr_h.kurtosis())  # pandas: excess kurtosis by default
                    except Exception:
                        k_ex = 0.0
                    if not np.isfinite(s):
                        s = 0.0
                    if not np.isfinite(k_ex):
                        k_ex = 0.0

                    z_cf = (
                        z
                        + (1.0 / 6.0) * (z**2 - 1.0) * s
                        + (1.0 / 24.0) * (z**3 - 3.0 * z) * k_ex
                        - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
                    )
                    q = mu + sigma * z_cf
                    var = -(q)

                    # Robust ES estimate from empirical tail below the adjusted quantile
                    tail = rr_h[rr_h <= q]
                    cvar = -float(np.nanmean(tail.values)) if len(tail) > 0 else float(var)

            else:
                # Unknown method -> default to historical
                q = float(np.nanquantile(rr_h.values, alpha))
                var = -q
                tail = rr_h[rr_h <= q]
                cvar = -float(np.nanmean(tail.values)) if len(tail) > 0 else float(var)

        except Exception as e:
            return {
                "success": False,
                "message": f"VaR computation failed: {e}",
                "method": method,
                "confidence_level": float(cl),
                "n_obs": int(n),
                "horizon": int(h),
                "warning": warning,
            }

        # Final output sanitation
        if not np.isfinite(var) or not np.isfinite(cvar):
            return {
                "success": False,
                "message": "VaR computation produced non-finite output (NaN/Inf). Check return series cleaning/overlap.",
                "method": method,
                "confidence_level": float(cl),
                "n_obs": int(n),
                "horizon": int(h),
                "warning": warning,
            }

        # Ensure non-negative loss magnitudes (can happen if returns are strongly positive)
        var = float(max(var, 0.0))
        cvar = float(max(cvar, 0.0))

        return {
            "success": True,
            "VaR": var,
            "CVaR": cvar,
            "ES": cvar,
            "confidence_level": float(cl),
            "method": method,
            "n_obs": int(n),
            "horizon": int(h),
            "warning": warning,
            "mu": float(mu),
            "sigma": float(sigma),
        }

    def stress_test(
        self,
        returns: pd.Series,
        scenarios: List[float] = None,
        shock: Optional[float] = None,
        duration: int = 1
    ) -> Dict[str, Any]:
        """Perform stress testing.

        Supports two modes (backward compatible):
        1) Scenario grid: pass `scenarios=[...]` (default) to apply additive return shocks and report metrics.
        2) Single shock path: pass `shock=<total shock>` and `duration=<days>` to distribute the total shock
           over the first `duration` observations (compounded) and return a simulated path.

        Notes:
        - If your UI passes `shock=` and `duration=`, this method will not raise an error.
        - min length / data quality checks are handled upstream in UI; this method is defensive anyway.
        """
        # Defensive clean-up
        try:
            returns_clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            returns_clean = returns.dropna()

        if returns_clean is None or len(returns_clean) == 0:
            return {"success": False, "message": "No valid returns provided for stress test."}

        # Base (unshocked) path for reference
        base_path = (1.0 + returns_clean).cumprod()

        # Mode 2: single shock path (used by your Streamlit UI)
        if shock is not None:
            try:
                shock_total = float(shock)
            except Exception:
                return {"success": False, "message": f"Invalid shock value: {shock}"}

            try:
                dur = max(1, int(duration))
            except Exception:
                dur = 1

            # Convert total shock into an equivalent per-day compounded shock
            try:
                daily_shock = (1.0 + shock_total) ** (1.0 / float(dur)) - 1.0
            except Exception:
                daily_shock = shock_total / float(dur)

            shocked = returns_clean.copy()
            k = min(dur, len(shocked))
            if k > 0:
                shocked.iloc[:k] = shocked.iloc[:k] + daily_shock

            path = (1.0 + shocked).cumprod()
            try:
                metrics = self.calculate_performance_metrics(shocked)
            except Exception as e:
                metrics = {"error": str(e)}

            return {
                "success": True,
                "mode": "single_shock",
                "shock_total": shock_total,
                "duration_days": dur,
                "daily_shock": float(daily_shock),
                "metrics": metrics,
                "path": path,
                "base_path": base_path
            }

        # Mode 1: scenario grid (legacy/default)
        if scenarios is None:
            scenarios = [-0.01, -0.02, -0.05, -0.10]

        results = {}
        for sc in scenarios:
            try:
                sc = float(sc)
            except Exception:
                continue
            shocked_returns = returns_clean + sc
            try:
                results[str(sc)] = self.calculate_performance_metrics(shocked_returns)
            except Exception as e:
                results[str(sc)] = {"error": str(e)}

        return {
            "success": True,
            "mode": "scenario_grid",
            "scenarios": list(scenarios),
            "results": results,
            "base_path": base_path
        }


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
        returns: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Performance Analysis"
    ) -> go.Figure:
        """Create performance visualization with multiple metrics.

        Robustly supports both pd.Series (single strategy/portfolio) and pd.DataFrame
        (multi-asset or multi-strategy) inputs.
        """

        # -----------------------------
        # Normalize input -> DataFrame
        # -----------------------------
        if returns is None:
            returns_df = pd.DataFrame()
        elif isinstance(returns, pd.DataFrame):
            returns_df = returns.copy()
        else:
            name = getattr(returns, "name", None) or "Portfolio"
            returns_df = pd.DataFrame({name: returns})

        # Coerce to numeric and drop empty rows/cols safely
        if not returns_df.empty:
            returns_df = returns_df.apply(pd.to_numeric, errors="coerce")
            returns_df = returns_df.dropna(how="all")
            returns_df = returns_df.dropna(axis=1, how="all")

        # Align benchmark to returns index (if present)
        bmk = None
        if benchmark_returns is not None:
            try:
                bmk = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
                if (bmk is not None) and (not returns_df.empty):
                    common_idx = returns_df.index.intersection(bmk.index)
                    returns_df = returns_df.loc[common_idx]
                    bmk = bmk.loc[common_idx]
            except Exception:
                bmk = None

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
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )

        cols = list(returns_df.columns) if not returns_df.empty else []
        palette = [
            self.colors.get("primary", "#1f77b4"),
            self.colors.get("secondary", "#ff7f0e"),
            self.colors.get("success", "#2ca02c"),
            self.colors.get("warning", "#d62728"),
            self.colors.get("danger", "#9467bd"),
            self.colors.get("gray", "#7f7f7f"),
        ]

        # -----------------------------
        # Cumulative returns (row 1, col 1)
        # -----------------------------
        for i, col in enumerate(cols):
            s = returns_df[col].dropna()
            if s.empty:
                continue
            cumulative = (1 + s).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    name=str(col),
                    line=dict(color=palette[i % len(palette)], width=3 if len(cols) == 1 else 2),
                    fill='tozeroy' if len(cols) == 1 else None,
                ),
                row=1, col=1
            )

        if bmk is not None and len(bmk) > 0:
            benchmark_cumulative = (1 + bmk).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name="Benchmark",
                    line=dict(color=self.colors.get("gray", "#888888"), width=2, dash='dash')
                ),
                row=1, col=1
            )

        # -----------------------------
        # Drawdown (row 1, col 2)
        # -----------------------------
        for i, col in enumerate(cols):
            s = returns_df[col].dropna()
            if s.empty:
                continue
            cumulative = (1 + s).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name=f"{col} Drawdown" if len(cols) > 1 else "Drawdown",
                    line=dict(color=palette[i % len(palette)], width=2),
                    fill='tozeroy' if len(cols) == 1 else None,
                    opacity=0.85 if len(cols) > 1 else 0.95
                ),
                row=1, col=2
            )

        if bmk is not None and len(bmk) > 0:
            bc = (1 + bmk).cumprod()
            rm = bc.cummax()
            bdd = (bc - rm) / rm * 100
            fig.add_trace(
                go.Scatter(
                    x=bdd.index,
                    y=bdd.values,
                    name="Benchmark Drawdown",
                    line=dict(color=self.colors.get("gray", "#888888"), width=2, dash='dot'),
                    opacity=0.9
                ),
                row=1, col=2
            )

        # -----------------------------
        # Rolling returns (row 2, col 1)
        # -----------------------------
        for i, col in enumerate(cols):
            s = returns_df[col]
            rolling_returns = s.rolling(window=252, min_periods=60).mean() * 252 * 100
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns.values,
                    name=f"{col} Rolling Return" if len(cols) > 1 else "Rolling Return",
                    line=dict(color=palette[i % len(palette)], width=2),
                    opacity=0.75 if len(cols) > 1 else 0.95
                ),
                row=2, col=1
            )

        if bmk is not None and len(bmk) > 0:
            brr = bmk.rolling(window=252, min_periods=60).mean() * 252 * 100
            fig.add_trace(
                go.Scatter(
                    x=brr.index,
                    y=brr.values,
                    name="Benchmark Rolling Return",
                    line=dict(color=self.colors.get("gray", "#888888"), width=2, dash='dash')
                ),
                row=2, col=1
            )

        # -----------------------------
        # Rolling volatility (row 2, col 2)
        # -----------------------------
        for i, col in enumerate(cols):
            s = returns_df[col]
            rolling_vol = s.rolling(window=252, min_periods=60).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    name=f"{col} Rolling Vol" if len(cols) > 1 else "Rolling Volatility",
                    line=dict(color=palette[i % len(palette)], width=2),
                    opacity=0.75 if len(cols) > 1 else 0.95
                ),
                row=2, col=2
            )

        if bmk is not None and len(bmk) > 0:
            brv = bmk.rolling(window=252, min_periods=60).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=brv.index,
                    y=brv.values,
                    name="Benchmark Rolling Vol",
                    line=dict(color=self.colors.get("gray", "#888888"), width=2, dash='dash')
                ),
                row=2, col=2
            )

        # -----------------------------
        # Returns distribution (row 3, col 1)
        # -----------------------------
        for i, col in enumerate(cols):
            s = (returns_df[col] * 100).dropna()
            if s.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=s,
                    nbinsx=50,
                    name=str(col),
                    marker_color=palette[i % len(palette)],
                    opacity=0.45 if len(cols) > 1 else 0.7
                ),
                row=3, col=1
            )

        # -----------------------------
        # QQ Plot (row 3, col 2) - per series + pooled theoretical line
        # -----------------------------
        for i, col in enumerate(cols):
            vals = returns_df[col].dropna().values
            if vals is None or len(vals) <= 10:
                continue
            try:
                qq_data = stats.probplot(vals, dist="norm")
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode='markers',
                        name=str(col),
                        marker=dict(size=6),
                        opacity=0.7 if len(cols) > 1 else 1.0
                    ),
                    row=3, col=2
                )
            except Exception:
                continue

        # Add pooled theoretical line (prevents DataFrame probplot shape issues)
        try:
            pooled = returns_df.stack().dropna().values if not returns_df.empty else np.array([])
            if pooled is not None and len(pooled) > 10:
                qq_all = stats.probplot(pooled, dist="norm")
                x_line = np.array([qq_all[0][0][0], qq_all[0][0][-1]])
                y_line = qq_all[1][0] + qq_all[1][1] * x_line
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name="Normal",
                        line=dict(color=self.colors.get("danger", "#d62728"), width=2, dash='dash')
                    ),
                    row=3, col=2
                )
        except Exception:
            pass

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=24)),
            height=1000,
            template=self.template,
            showlegend=True,
            hovermode='x unified'
        )

        # Update axes titles (consistent with subplot placement)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Volatility (%)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=3, col=2)

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
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
        
        # Initialize configuration
        self.cfg = AnalysisConfiguration()
        
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

    def _display_tracking_error(self, config: 'AnalysisConfiguration'):
        """Interactive Tracking Error analytics with institutional band zones.
        Robust implementation: always available even if earlier patch blocks were misplaced.
        """
        st.markdown("### 🎯 Tracking Error (Institutional Band Monitoring)")
        # --- Load returns
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
            avg_skew = float(returns_df.skew().mean()) if not returns_df.empty else np.nan
            avg_skew_disp = "N/A" if (avg_skew is None or (isinstance(avg_skew, float) and np.isnan(avg_skew))) else f"{avg_skew:.3f}"
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📐 Avg Skewness</div>
                <div class="metric-value">{avg_skew_disp}</div>
            </div>
            """), unsafe_allow_html=True)

        with col4:
            n_assets = int(returns_df.shape[1]) if isinstance(returns_df, pd.DataFrame) and not returns_df.empty else 0
            n_obs = int(returns_df.shape[0]) if isinstance(returns_df, pd.DataFrame) and not returns_df.empty else 0
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">📦 Assets / Obs</div>
                <div class="metric-value">{n_assets} / {n_obs}</div>
            </div>
            """), unsafe_allow_html=True)

        # Display price chart for first asset
        if st.session_state.asset_data:
            first_asset = list(st.session_state.asset_data.keys())[0]
            df = st.session_state.asset_data[first_asset]
            if not df.empty:
                fig = self.visualizer.create_price_chart(df, f"{first_asset} Price Analysis")
                st.plotly_chart(fig, use_container_width=True)

    def _display_advanced_analytics(self, config: AnalysisConfiguration):
        """Display advanced analytics tab"""
        st.markdown("### 🧠 Advanced Analytics")
        st.write("Advanced analytics features including GARCH and regime detection")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selector
        assets = list(returns_df.columns)
        selected_asset = st.selectbox("Select Asset", options=assets, index=0)
        
        if selected_asset:
            returns = returns_df[selected_asset].dropna()
            
            # GARCH Analysis
            st.subheader("📉 GARCH Volatility Analysis")
            if st.button("Run GARCH Analysis", key="garch_btn"):
                with st.spinner("Running GARCH analysis..."):
                    garch_result = self.analytics.garch_analysis(returns)
                    if garch_result.get("success", False):
                        st.success("GARCH analysis completed!")
                        st.json(garch_result, expanded=False)
                    else:
                        st.warning(f"GARCH analysis failed: {garch_result.get('message', 'Unknown error')}")
            
            # Regime Detection
            st.subheader("🧩 Regime Detection")
            if st.button("Detect Market Regimes", key="regime_btn"):
                with st.spinner("Detecting market regimes..."):
                    regime_result = self.analytics.detect_regimes(returns)
                    if regime_result.get("available", False):
                        st.success("Regime detection completed!")
                        st.json(regime_result, expanded=False)
                    else:
                        st.warning(f"Regime detection failed: {regime_result.get('message', 'Unknown error')}")

    def _display_risk_analytics(self, config: AnalysisConfiguration):
        """Display risk analytics tab"""
        st.markdown("### 🧮 Risk Analytics")
        st.write("Comprehensive risk metrics and analysis")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selector
        assets = list(returns_df.columns)
        selected_asset = st.selectbox("Select Asset", options=assets, index=0, key="risk_asset")
        
        if selected_asset:
            returns = returns_df[selected_asset].dropna()
            
            # VaR Analysis
            st.subheader("📊 Value at Risk (VaR) Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
            with col2:
                method = st.selectbox("Method", options=["historical", "parametric", "modified"])
            with col3:
                horizon = st.select_slider("Horizon (days)", options=[1, 5, 10, 20], value=1)
            
            if st.button("Calculate VaR", key="var_btn"):
                with st.spinner("Calculating VaR..."):
                    var_result = self.analytics.calculate_var(returns, confidence_level=confidence, method=method, horizon=horizon)
                    if var_result.get("success", False):
                        st.success(f"VaR Calculation Complete")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"VaR ({confidence*100}%)", f"{var_result.get('VaR', 0)*100:.2f}%")
                        with col2:
                            st.metric(f"CVaR ({confidence*100}%)", f"{var_result.get('CVaR', 0)*100:.2f}%")
                        with col3:
                            st.metric(f"ES ({confidence*100}%)", f"{var_result.get('ES', 0)*100:.2f}%")
                    else:
                        st.warning(f"VaR calculation failed: {var_result.get('message', 'Unknown error')}")

    def _display_ewma_ratio_signal(self, config: AnalysisConfiguration):
        """Display EWMA ratio signal tab"""
        st.markdown("### 📉 EWMA Volatility Ratio Signal")
        st.write("Institutional EWMA volatility ratio signal with Bollinger Bands")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selector
        assets = list(returns_df.columns)
        selected_asset = st.selectbox("Select Asset", options=assets, index=0, key="ewma_asset")
        
        if selected_asset:
            returns = returns_df[selected_asset].dropna()
            
            # EWMA Ratio Parameters
            st.subheader("⚙️ EWMA Ratio Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                span_fast = st.number_input("Fast Span", min_value=5, max_value=100, value=22)
            with col2:
                span_mid = st.number_input("Mid Span", min_value=5, max_value=200, value=33)
            with col3:
                span_slow = st.number_input("Slow Span", min_value=10, max_value=500, value=99)
            
            col4, col5 = st.columns(2)
            with col4:
                green_max = st.number_input("Green Max Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            with col5:
                red_min = st.number_input("Red Min Threshold", min_value=0.0, max_value=2.0, value=0.55, step=0.01)
            
            if st.button("Calculate EWMA Ratio", key="ewma_btn"):
                with st.spinner("Calculating EWMA ratio..."):
                    ewma_df = self.analytics.compute_ewma_volatility_ratio(
                        returns, 
                        span_fast=int(span_fast),
                        span_mid=int(span_mid),
                        span_slow=int(span_slow)
                    )
                    
                    if not ewma_df.empty:
                        fig = self.visualizer.create_ewma_ratio_signal_chart(
                            ewma_df,
                            title=f"EWMA Volatility Ratio - {selected_asset}",
                            green_max=green_max,
                            red_min=red_min
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display current signal
                        latest_ratio = ewma_df["EWMA_RATIO"].iloc[-1]
                        if latest_ratio <= green_max:
                            signal = "🟢 GREEN (Low Risk)"
                        elif latest_ratio >= red_min:
                            signal = "🔴 RED (High Risk)"
                        else:
                            signal = "🟡 ORANGE (Medium Risk)"
                        
                        st.metric("Current Signal", signal, f"Ratio: {latest_ratio:.4f}")
                    else:
                        st.warning("EWMA ratio calculation failed. Please check the data.")

    def _display_portfolio(self, config: AnalysisConfiguration):
        """Display portfolio tab"""
        st.markdown("### 📈 Portfolio Analysis")
        st.write("Portfolio optimization and analysis")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selection for portfolio
        assets = list(returns_df.columns)
        selected_assets = st.multiselect(
            "Select Assets for Portfolio",
            options=assets,
            default=assets[:min(4, len(assets))]
        )
        
        if len(selected_assets) >= 2:
            portfolio_returns = returns_df[selected_assets]
            
            # Portfolio Optimization
            st.subheader("🔧 Portfolio Optimization")
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("Optimization Method", options=["sharpe", "min_var", "max_ret"], index=0)
            with col2:
                target_return = st.number_input("Target Return (annual)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            
            if st.button("Optimize Portfolio", key="optimize_btn"):
                with st.spinner("Optimizing portfolio..."):
                    target = None if target_return == 0 else target_return
                    opt_result = self.analytics.optimize_portfolio(
                        portfolio_returns,
                        method=method,
                        target_return=target
                    )
                    
                    if opt_result.get("success", False):
                        st.success("Portfolio optimization completed!")
                        
                        # Display weights
                        weights = opt_result.get("weights", {})
                        weights_df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Display metrics
                        metrics = opt_result.get("metrics", {})
                        if metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Return", f"{metrics.get('annual_return', 0):.2f}%")
                            with col2:
                                st.metric("Volatility", f"{metrics.get('annual_volatility', 0):.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    else:
                        st.warning(f"Portfolio optimization failed: {opt_result.get('message', 'Unknown error')}")
            
            # Performance Chart
            st.subheader("📊 Portfolio Performance")
            if st.button("Show Performance Chart", key="perf_btn"):
                fig = self.visualizer.create_performance_chart(portfolio_returns, title="Portfolio Performance")
                st.plotly_chart(fig, use_container_width=True)

    def _display_rolling_beta(self, config: AnalysisConfiguration):
        """Display rolling beta tab"""
        st.markdown("### β Rolling Beta Analysis")
        st.write("Rolling beta analysis vs benchmark")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        bench_df = st.session_state.get("benchmark_returns_data", pd.DataFrame())
        
        if returns_df.empty or bench_df.empty:
            st.warning("Need both asset and benchmark data. Please load data first.")
            return
        
        # Asset and benchmark selection
        assets = list(returns_df.columns)
        benchmarks = list(bench_df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            selected_asset = st.selectbox("Select Asset", options=assets, index=0, key="beta_asset")
        with col2:
            selected_benchmark = st.selectbox("Select Benchmark", options=benchmarks, index=0, key="beta_bench")
        
        if selected_asset and selected_benchmark:
            asset_returns = returns_df[selected_asset].dropna()
            bench_returns = bench_df[selected_benchmark].dropna()
            
            # Align data
            common_idx = asset_returns.index.intersection(bench_returns.index)
            if len(common_idx) < 60:
                st.warning("Insufficient overlapping data for beta calculation.")
                return
            
            asset_returns = asset_returns.loc[common_idx]
            bench_returns = bench_returns.loc[common_idx]
            
            # Rolling window selection
            window = st.select_slider("Rolling Window (days)", options=[20, 40, 60, 90, 120, 180, 252], value=60)
            
            if st.button("Calculate Rolling Beta", key="beta_btn"):
                with st.spinner("Calculating rolling beta..."):
                    # Calculate rolling beta
                    cov = asset_returns.rolling(window).cov(bench_returns)
                    var = bench_returns.rolling(window).var()
                    beta = (cov / var).replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not beta.empty:
                        # Create chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=beta.index,
                            y=beta.values,
                            mode='lines',
                            name='Rolling Beta',
                            line=dict(width=2)
                        ))
                        fig.add_hline(y=1.0, line_dash="dash", opacity=0.6, annotation_text="Beta = 1.0")
                        fig.update_layout(
                            title=f"Rolling Beta - {selected_asset} vs {selected_benchmark}",
                            height=500,
                            xaxis_title="Date",
                            yaxis_title="Beta"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display current beta
                        current_beta = beta.iloc[-1]
                        st.metric("Current Beta", f"{current_beta:.3f}")
                    else:
                        st.warning("Beta calculation failed. Please check the data.")

    def _display_relative_risk(self, config: AnalysisConfiguration):
        """Display relative risk tab"""
        st.markdown("### 📉 Relative Risk Analysis")
        st.write("Relative VaR/CVaR/ES analysis vs benchmark")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        bench_df = st.session_state.get("benchmark_returns_data", pd.DataFrame())
        
        if returns_df.empty or bench_df.empty:
            st.warning("Need both asset and benchmark data. Please load data first.")
            return
        
        # Asset and benchmark selection
        assets = list(returns_df.columns)
        benchmarks = list(bench_df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            selected_asset = st.selectbox("Select Asset", options=assets, index=0, key="relrisk_asset")
        with col2:
            selected_benchmark = st.selectbox("Select Benchmark", options=benchmarks, index=0, key="relrisk_bench")
        
        if selected_asset and selected_benchmark:
            asset_returns = returns_df[selected_asset].dropna()
            bench_returns = bench_df[selected_benchmark].dropna()
            
            # Align data
            common_idx = asset_returns.index.intersection(bench_returns.index)
            if len(common_idx) < 60:
                st.warning("Insufficient overlapping data for relative risk calculation.")
                return
            
            asset_returns = asset_returns.loc[common_idx]
            bench_returns = bench_returns.loc[common_idx]
            
            # Calculate active returns
            active_returns = asset_returns - bench_returns
            
            # Relative risk parameters
            st.subheader("⚙️ Relative Risk Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95, key="relrisk_conf")
            with col2:
                method = st.selectbox("Method", options=["historical", "parametric", "modified"], key="relrisk_method")
            with col3:
                horizon = st.select_slider("Horizon (days)", options=[1, 5, 10, 20], value=1, key="relrisk_horizon")
            
            if st.button("Calculate Relative Risk", key="relrisk_btn"):
                with st.spinner("Calculating relative risk..."):
                    relrisk_result = self.analytics.calculate_var(
                        active_returns,
                        confidence_level=confidence,
                        method=method,
                        horizon=horizon
                    )
                    
                    if relrisk_result.get("success", False):
                        st.success("Relative risk calculation complete!")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Relative VaR ({confidence*100}%)", f"{relrisk_result.get('VaR', 0)*100:.2f}%")
                        with col2:
                            st.metric(f"Relative CVaR ({confidence*100}%)", f"{relrisk_result.get('CVaR', 0)*100:.2f}%")
                        with col3:
                            st.metric(f"Relative ES ({confidence*100}%)", f"{relrisk_result.get('ES', 0)*100:.2f}%")
                        
                        # Plot active returns
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=active_returns.index,
                            y=active_returns.values * 100,
                            mode='lines',
                            name='Active Returns (%)',
                            line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"Active Returns - {selected_asset} vs {selected_benchmark}",
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Active Return (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Relative risk calculation failed: {relrisk_result.get('message', 'Unknown error')}")

    def _display_stress_testing(self, config: AnalysisConfiguration):
        """Display stress testing tab"""
        st.markdown("### 🧪 Stress Testing")
        st.write("Stress testing and scenario analysis")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selector
        assets = list(returns_df.columns)
        selected_asset = st.selectbox("Select Asset", options=assets, index=0, key="stress_asset")
        
        if selected_asset:
            returns = returns_df[selected_asset].dropna()
            
            # Stress test parameters
            st.subheader("⚙️ Stress Test Parameters")
            col1, col2 = st.columns(2)
            with col1:
                shock = st.select_slider("Shock Size", 
                                       options=[-0.30, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10],
                                       value=-0.10)
            with col2:
                duration = st.select_slider("Shock Duration (days)", 
                                          options=[1, 5, 10, 20],
                                          value=5)
            
            if st.button("Run Stress Test", key="stress_btn"):
                with st.spinner("Running stress test..."):
                    stress_result = self.analytics.stress_test(
                        returns,
                        shock=float(shock),
                        duration=int(duration)
                    )
                    
                    if stress_result.get("success", False):
                        st.success("Stress test completed!")
                        
                        # Display metrics
                        metrics = stress_result.get("metrics", {})
                        if metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                            with col2:
                                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        
                        # Plot stress path
                        path = stress_result.get("path")
                        base_path = stress_result.get("base_path")
                        
                        if isinstance(path, pd.Series) and isinstance(base_path, pd.Series):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=base_path.index,
                                y=base_path.values,
                                mode='lines',
                                name='Base Path',
                                line=dict(width=2, color='blue')
                            ))
                            fig.add_trace(go.Scatter(
                                x=path.index,
                                y=path.values,
                                mode='lines',
                                name='Stress Path',
                                line=dict(width=2, color='red', dash='dash')
                            ))
                            fig.update_layout(
                                title=f"Stress Test - {selected_asset} (Shock: {shock*100}%, Duration: {duration} days)",
                                height=500,
                                xaxis_title="Date",
                                yaxis_title="Cumulative Return"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Stress test failed: {stress_result.get('message', 'Unknown error')}")

    def _display_reporting(self, config: AnalysisConfiguration):
        """Display reporting tab"""
        st.markdown("### 📑 Reporting")
        st.write("Generate comprehensive reports and exports")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Report generation options
        st.subheader("📊 Performance Report")
        
        # Asset selection for report
        assets = list(returns_df.columns)
        selected_assets = st.multiselect(
            "Select Assets for Report",
            options=assets,
            default=assets[:min(5, len(assets))]
        )
        
        if selected_assets:
            # Generate performance metrics
            if st.button("Generate Performance Report", key="report_btn"):
                with st.spinner("Generating report..."):
                    # Calculate metrics for each asset
                    report_data = []
                    for asset in selected_assets:
                        returns = returns_df[asset].dropna()
                        if len(returns) >= 20:
                            metrics = self.analytics.calculate_performance_metrics(returns)
                            metrics['Asset'] = asset
                            report_data.append(metrics)
                    
                    if report_data:
                        # Create DataFrame
                        report_df = pd.DataFrame(report_data)
                        report_df.set_index('Asset', inplace=True)
                        
                        # Display report
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Export options
                        st.subheader("📤 Export Options")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV Export
                            csv = report_df.to_csv()
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="performance_report.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Excel Export
                            buffer = BytesIO()
                            writer, engine = icd_safe_excel_writer(buffer)
                            if writer:
                                report_df.to_excel(writer, sheet_name='Performance Report')
                                writer.close()
                                st.download_button(
                                    label="Download Excel",
                                    data=buffer.getvalue(),
                                    file_name="performance_report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.info("Excel export requires openpyxl or xlsxwriter")
                    else:
                        st.warning("Insufficient data to generate report.")

    def _display_settings(self, config: AnalysisConfiguration):
        """Display settings tab"""
        st.markdown("### ⚙️ Settings")
        st.write("Configure application settings and parameters")
        
        # Configuration settings
        st.subheader("🔧 Configuration Parameters")
        
        # Risk-free rate
        new_rf = st.number_input(
            "Risk-Free Rate (annual)",
            min_value=0.0,
            max_value=0.2,
            value=float(config.risk_free_rate),
            step=0.001,
            format="%.3f"
        )
        
        # Annual trading days
        new_trading_days = st.number_input(
            "Annual Trading Days",
            min_value=200,
            max_value=365,
            value=int(config.annual_trading_days),
            step=1
        )
        
        # Confidence levels
        st.subheader("📊 Confidence Levels")
        conf_levels = st.multiselect(
            "Select Confidence Levels",
            options=[0.90, 0.95, 0.99, 0.995],
            default=list(config.confidence_levels)
        )
        
        # GARCH parameters
        st.subheader("📉 GARCH Parameters")
        col1, col2 = st.columns(2)
        with col1:
            garch_p_min = st.number_input("GARCH p min", min_value=1, max_value=5, value=config.garch_p_range[0])
            garch_p_max = st.number_input("GARCH p max", min_value=1, max_value=5, value=config.garch_p_range[1])
        with col2:
            garch_q_min = st.number_input("GARCH q min", min_value=1, max_value=5, value=config.garch_q_range[0])
            garch_q_max = st.number_input("GARCH q max", min_value=1, max_value=5, value=config.garch_q_range[1])
        
        # Regime detection
        st.subheader("🧩 Regime Detection")
        n_regimes = st.number_input(
            "Number of Regimes",
            min_value=2,
            max_value=5,
            value=config.regime_states,
            step=1
        )
        
        # Save settings
        if st.button("Save Settings", key="save_settings"):
            config.risk_free_rate = new_rf
            config.annual_trading_days = new_trading_days
            config.confidence_levels = tuple(sorted(conf_levels))
            config.garch_p_range = (garch_p_min, garch_p_max)
            config.garch_q_range = (garch_q_min, garch_q_max)
            config.regime_states = n_regimes
            
            # Update analytics engine
            self.analytics.risk_free_rate = new_rf
            self.analytics.annual_trading_days = new_trading_days
            
            st.success("Settings saved successfully!")
            
            # Display current settings
            st.subheader("📋 Current Settings")
            settings_dict = {
                "Risk-Free Rate": f"{config.risk_free_rate:.3f}",
                "Annual Trading Days": config.annual_trading_days,
                "Confidence Levels": config.confidence_levels,
                "GARCH p Range": config.garch_p_range,
                "GARCH q Range": config.garch_q_range,
                "Regime States": config.regime_states,
                "Backtest Window": config.backtest_window,
                "Rolling Window": config.rolling_window,
                "Monte Carlo Simulations": config.monte_carlo_simulations
            }
            st.json(settings_dict, expanded=False)

    def _display_portfolio_lab(self, config: AnalysisConfiguration):
        """Display portfolio lab tab with PyPortfolioOpt integration"""
        st.markdown("### 🧰 Portfolio Lab (PyPortfolioOpt)")
        st.write("Advanced portfolio optimization using PyPortfolioOpt")
        
        # Check if data is loaded
        if not st.session_state.get("data_loaded", False):
            st.info("Please load data first from the sidebar.")
            return
        
        # Get returns data
        returns_df = st.session_state.get("returns_data", pd.DataFrame())
        if returns_df.empty:
            st.warning("No returns data available. Please load data first.")
            return
        
        # Asset selection for portfolio
        assets = list(returns_df.columns)
        selected_assets = st.multiselect(
            "Select Assets for Portfolio",
            options=assets,
            default=assets[:min(6, len(assets))]
        )
        
        if len(selected_assets) >= 2:
            portfolio_returns = returns_df[selected_assets].dropna()
            
            # Check if PyPortfolioOpt is available
            try:
                from pypfopt import expected_returns, risk_models
                from pypfopt.efficient_frontier import EfficientFrontier
                
                pypfopt_available = True
            except ImportError:
                pypfopt_available = False
                st.warning("PyPortfolioOpt is not installed. Using internal optimizer.")
            
            if pypfopt_available:
                st.subheader("🔧 PyPortfolioOpt Optimization")
                
                # Optimization method selection
                method = st.selectbox(
                    "Optimization Method",
                    options=["Max Sharpe", "Min Volatility", "Efficient Risk", "Efficient Return"],
                    index=0
                )
                
                # Additional parameters
                if method == "Efficient Risk":
                    target_risk = st.number_input(
                        "Target Risk (annual volatility)",
                        min_value=0.05,
                        max_value=1.0,
                        value=0.20,
                        step=0.01
                    )
                elif method == "Efficient Return":
                    target_return = st.number_input(
                        "Target Return (annual)",
                        min_value=-0.2,
                        max_value=1.0,
                        value=0.15,
                        step=0.01
                    )
                
                if st.button("Optimize with PyPortfolioOpt", key="pypfopt_btn"):
                    with st.spinner("Optimizing portfolio with PyPortfolioOpt..."):
                        try:
                            # Calculate expected returns and covariance matrix
                            mu = expected_returns.mean_historical_return(portfolio_returns)
                            S = risk_models.sample_cov(portfolio_returns)
                            
                            # Create efficient frontier
                            ef = EfficientFrontier(mu, S)
                            
                            # Perform optimization based on selected method
                            if method == "Max Sharpe":
                                ef.max_sharpe()
                            elif method == "Min Volatility":
                                ef.min_volatility()
                            elif method == "Efficient Risk":
                                ef.efficient_risk(target_volatility=target_risk)
                            elif method == "Efficient Return":
                                ef.efficient_return(target_return=target_return)
                            
                            # Get optimized weights
                            weights = ef.clean_weights()
                            
                            # Display results
                            st.success("PyPortfolioOpt optimization completed!")
                            
                            # Display weights
                            weights_df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
                            st.dataframe(weights_df, use_container_width=True)
                            
                            # Display portfolio performance
                            perf = ef.portfolio_performance(verbose=False)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Return", f"{perf[0]*100:.2f}%")
                            with col2:
                                st.metric("Volatility", f"{perf[1]*100:.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                                
                        except Exception as e:
                            st.error(f"PyPortfolioOpt optimization failed: {str(e)}")
            
            # Fallback to internal optimizer
            st.subheader("🔧 Internal Optimizer (Fallback)")
            if st.button("Optimize with Internal Engine", key="internal_opt_btn"):
                with st.spinner("Optimizing portfolio with internal engine..."):
                    opt_result = self.analytics.optimize_portfolio(
                        portfolio_returns,
                        method="sharpe"
                    )
                    
                    if opt_result.get("success", False):
                        st.success("Internal optimization completed!")
                        
                        # Display weights
                        weights = opt_result.get("weights", {})
                        weights_df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Display metrics
                        metrics = opt_result.get("metrics", {})
                        if metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Return", f"{metrics.get('annual_return', 0):.2f}%")
                            with col2:
                                st.metric("Volatility", f"{metrics.get('annual_volatility', 0):.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    else:
                        st.warning(f"Internal optimization failed: {opt_result.get('message', 'Unknown error')}")

# =============================================================================
# 🧭 APPLICATION ROUTER — Mode selector
# =============================================================================

def _run_app_router():
    """Main application router"""
    import streamlit as st

    st.sidebar.markdown("### 🧭 Platform Mode")
    mode = st.sidebar.radio(
        "Select application layer",
        options=[
            "🏛️ Institutional Commodities Platform (v6.x)",
        ],
        index=0,
        key="app_mode_selector"
    )

    # Run the selected platform
    if mode == "🏛️ Institutional Commodities Platform (v6.x)":
        try:
            dashboard = InstitutionalCommoditiesDashboard()
            dashboard.run()
        except Exception as e:
            st.error(f"Institutional dashboard failed to start: {e}")
            st.exception(e)

# Execute router (Streamlit entrypoint)
if __name__ == "__main__":
    _run_app_router()
