"""
🏛️ ENIGMA Commodities Analytics Platform v6.0
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • Machine Learning • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance
ENHANCED VERSION WITH ADVANCED TECHNIQUES
"""

import os
import math
import warnings
import json
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from enum import Enum
from pathlib import Path
import pickle
import contextlib
import sys
import importlib

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats, optimize, signal, special
import seaborn as sns
from io import BytesIO, StringIO
import base64
import joblib
from numba import jit, prange
import numba

# -----------------------------------------------------------------------------
# yfinance download compatibility helper (Streamlit Cloud safe)
# -----------------------------------------------------------------------------
def yf_download_safe(params: Dict[str, Any]) -> pd.DataFrame:
    """Call yfinance.download with fallbacks for version/arg compatibility."""
    try:
        return yf.download(**params)
    except TypeError as e:
        # Some yfinance versions don't accept these args
        p = dict(params)
        p.pop("threads", None)
        p.pop("timeout", None)
        # Backward compatibility: if someone accidentally uses 'symbol'
        if "tickers" not in p and "symbol" in p:
            p["tickers"] = p.pop("symbol")
        try:
            return yf.download(**p)
        except Exception as e2:
            st.warning(f"yFinance download failed: {str(e2)[:100]}")
            return pd.DataFrame()

# =============================================================================
# ADVANCED OPTIMIZATION DECORATORS
# =============================================================================


# =============================================================================
# OPTIONAL DEPENDENCY MANAGER (STREAMLIT CLOUD-SAFE)
# =============================================================================

class DependencyManager:
    """Lightweight optional dependency manager.

    Keeps the app resilient on Streamlit Cloud where some heavy packages (arch, hmmlearn)
    may not be installed. Code can query availability safely without crashing.
    """

    def __init__(self):
        self.dependencies: Dict[str, Dict[str, Any]] = {}
        self._load_defaults()

    @staticmethod
    def _safe_import(module_name: str):
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    def _load_defaults(self):
        # --- ARCH / GARCH (arch) ---
        arch_mod = self._safe_import("arch")
        if arch_mod is not None:
            try:
                from arch import arch_model as _arch_model
                self.dependencies["arch"] = {
                    "available": True,
                    "module": arch_mod,
                    "arch_model": _arch_model
                }
            except Exception:
                self.dependencies["arch"] = {
                    "available": False,
                    "module": None,
                    "arch_model": None
                }
        else:
            self.dependencies["arch"] = {
                "available": False,
                "module": None,
                "arch_model": None
            }

        # --- HMM (hmmlearn) ---
        hmm_mod = self._safe_import("hmmlearn")
        if hmm_mod is not None:
            try:
                from hmmlearn.hmm import GaussianHMM as _GaussianHMM
                self.dependencies["hmmlearn"] = {
                    "available": True,
                    "module": hmm_mod,
                    "GaussianHMM": _GaussianHMM
                }
            except Exception:
                self.dependencies["hmmlearn"] = {
                    "available": False,
                    "module": None,
                    "GaussianHMM": None
                }
        else:
            self.dependencies["hmmlearn"] = {
                "available": False,
                "module": None,
                "GaussianHMM": None
            }

    def is_available(self, name: str) -> bool:
        info = self.dependencies.get(name, {})
        return bool(info.get("available", False))


# Global singleton used across the app
dep_manager = DependencyManager()

class PerformanceOptimizer:
    """Advanced performance optimization utilities"""
    
    @staticmethod
    def vectorized(func):
        """Decorator to vectorize functions for NumPy arrays"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert pandas Series to numpy arrays if needed
            new_args = []
            for arg in args:
                if isinstance(arg, pd.Series):
                    new_args.append(arg.values)
                elif isinstance(arg, pd.DataFrame):
                    new_args.append(arg.values)
                else:
                    new_args.append(arg)
            return func(*new_args, **kwargs)
        return wrapper
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def calculate_rolling_statistics_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """Numba-accelerated rolling statistics calculation"""
        n = len(prices)
        result = np.empty(n)
        result[:window-1] = np.nan
        
        for i in prange(window-1, n):
            window_data = prices[i-window+1:i+1]
            result[i] = np.std(window_data)
        
        return result
    
    @staticmethod
    def cache_with_invalidation(max_age_seconds: int = 3600):
        """Advanced caching with automatic invalidation"""
        def decorator(func):
            cache = {}
            timestamps = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = hashlib.sha256(
                    str(args).encode() + str(kwargs).encode()
                ).hexdigest()
                
                current_time = datetime.now().timestamp()
                
                # Check if cache entry exists and is fresh
                if key in cache:
                    if current_time - timestamps[key] < max_age_seconds:
                        return cache[key]
                
                # Compute and cache
                result = func(*args, **kwargs)
                cache[key] = result
                timestamps[key] = current_time
                
                # Clean old entries
                keys_to_delete = [
                    k for k, t in timestamps.items()
                    if current_time - t > max_age_seconds * 2
                ]
                for k in keys_to_delete:
                    cache.pop(k, None)
                    timestamps.pop(k, None)
                
                return result
            
            return wrapper
        return decorator

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Advanced environment optimization
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Configure NumPy for better performance
np.seterr(all='ignore')
np.set_printoptions(precision=4, suppress=True)

# Streamlit configuration with enhanced settings
st.set_page_config(
    page_title="Institutional Commodities Platform v6.0 Enhanced",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/institutional-commodities',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v6.0 Enhanced
                    Advanced analytics platform for institutional commodity trading
                    © 2024 Institutional Trading Analytics"""
    }
)


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
        },
        "corporate": {
            "primary": "#003366",
            "secondary": "#0066cc",
            "accent": "#0099ff",
            "success": "#00cc66",
            "warning": "#ff9900",
            "danger": "#ff3333",
            "dark": "#1a1a1a",
            "light": "#f5f5f5",
            "gray": "#666666",
            "background": "#ffffff"
        },
        "trading": {
            "primary": "#0d47a1",
            "secondary": "#1976d2",
            "accent": "#2196f3",
            "success": "#4caf50",
            "warning": "#ff9800",
            "danger": "#f44336",
            "dark": "#121212",
            "light": "#f5f5f5",
            "gray": "#757575",
            "background": "#ffffff"
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

            /* Enhanced Metric Cards */
            .metric-card.positive .metric-value {{
                color: var(--success);
                background: linear-gradient(135deg, var(--success), #059669);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .metric-card.negative .metric-value {{
                color: var(--danger);
                background: linear-gradient(135deg, var(--danger), #dc2626);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .metric-card.neutral .metric-value {{
                color: var(--warning);
                background: linear-gradient(135deg, var(--warning), #d97706);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
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

                .stTabs [data-baseweb="tab-list"] {{
                    flex-direction: column;
                }}
            }}

            /* Dark mode adjustments */
            @media (prefers-color-scheme: dark) {{
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
                }}
            }}
        </style>
        """

# =============================================================================
# SAFE HELPERS (Streamlit Cloud / Pandas Truthiness)
# =============================================================================

def _is_empty_data(obj: Any) -> bool:
    """Return True if obj is None/empty DataFrame/empty Series/empty dict/list."""
    try:
        if obj is None:
            return True
        if isinstance(obj, pd.DataFrame):
            return obj.empty
        if isinstance(obj, pd.Series):
            return obj.dropna().empty
        if isinstance(obj, (dict, list, tuple, set)):
            return len(obj) == 0
        if isinstance(obj, np.ndarray):
            return obj.size == 0
        return False
    except Exception:
        return True


# =============================================================================
# ENHANCED DATA STRUCTURES & CONFIGURATION
# =============================================================================

class AssetCategory(Enum):
    """Asset categories for classification with enhanced metadata"""
    PRECIOUS_METALS = "Precious Metals"
    INDUSTRIAL_METALS = "Industrial Metals"
    ENERGY = "Energy"
    AGRICULTURE = "Agriculture"
    BENCHMARK = "Benchmark"
    CRYPTO = "Cryptocurrency"
    CURRENCY = "Currency"

@dataclass
class AssetMetadata:
    """Enhanced metadata for assets with comprehensive details"""
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
    risk_level: str = "Medium"
    liquidity_score: float = 0.8  # 0-1 scale
    volatility_profile: str = "Medium"
    correlation_group: str = "Metals"
    data_source: str = "Yahoo Finance"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def formatted_name(self) -> str:
        return f"{self.symbol} - {self.name}"

@dataclass
class AnalysisConfiguration:
    """Comprehensive analysis configuration with validation"""
    start_date: datetime
    end_date: datetime
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
    optimization_method: str = "sharpe"
    covariance_estimator: str = "ledoit_wolf"  # ledoit_wolf, oas, empirical
    risk_model: str = "semi_variance"  # variance, semi_variance, cvar
    confidence_interval: float = 0.95
    max_iterations: int = 1000
    tolerance: float = 1e-8
    
    def validate(self) -> Tuple[bool, str]:
        """Enhanced validation with detailed error messages"""
        errors = []
        
        if self.start_date >= self.end_date:
            errors.append("Start date must be before end date")
        
        if not (0 <= self.risk_free_rate <= 0.5):
            errors.append("Risk-free rate must be between 0% and 50%")
        
        if not all(0.5 <= cl <= 0.999 for cl in self.confidence_levels):
            errors.append("Confidence levels must be between 0.5 and 0.999")
        
        if self.monte_carlo_simulations < 1000:
            errors.append("Monte Carlo simulations should be at least 1000")
        
        if self.backtest_window < 60:
            errors.append("Backtest window should be at least 60 days")
        
        return len(errors) == 0, "; ".join(errors) if errors else "Configuration valid"
    
    def to_json(self) -> str:
        """Serialize configuration to JSON"""
        config_dict = asdict(self)
        config_dict['start_date'] = self.start_date.isoformat()
        config_dict['end_date'] = self.end_date.isoformat()
        return json.dumps(config_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create configuration from JSON"""
        data = json.loads(json_str)
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['end_date'] = datetime.fromisoformat(data['end_date'])
        return cls(**data)

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
            risk_level="Low",
            liquidity_score=0.95,
            volatility_profile="Low",
            correlation_group="SafeHaven"
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
            risk_level="Medium",
            liquidity_score=0.85,
            volatility_profile="Medium",
            correlation_group="PreciousMetals"
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
            risk_level="High",
            liquidity_score=0.70,
            volatility_profile="High",
            correlation_group="PreciousMetals"
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
            risk_level="Medium",
            liquidity_score=0.80,
            volatility_profile="Medium",
            correlation_group="IndustrialMetals"
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
            risk_level="High",
            liquidity_score=0.65,
            volatility_profile="High",
            correlation_group="IndustrialMetals"
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
            risk_level="High",
            liquidity_score=0.90,
            volatility_profile="High",
            correlation_group="Energy"
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
            risk_level="High",
            liquidity_score=0.75,
            volatility_profile="VeryHigh",
            correlation_group="Energy"
        ),
        "BZ=F": AssetMetadata(
            symbol="BZ=F",
            name="Brent Crude",
            category=AssetCategory.ENERGY,
            color="#2F4F4F",
            description="ICE Brent Crude Futures",
            exchange="ICE",
            contract_size="1,000 barrels",
            margin_requirement=0.085,
            tick_size=0.01,
            risk_level="High",
            liquidity_score=0.85,
            volatility_profile="High",
            correlation_group="Energy"
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
            risk_level="Medium",
            liquidity_score=0.80,
            volatility_profile="Medium",
            correlation_group="Grains"
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
            risk_level="Medium",
            liquidity_score=0.75,
            volatility_profile="Medium",
            correlation_group="Grains"
        ),
        "ZS=F": AssetMetadata(
            symbol="ZS=F",
            name="Soybean Futures",
            category=AssetCategory.AGRICULTURE,
            color="#8B4513",
            description="CBOT Soybean Futures (5,000 bushels)",
            exchange="CBOT",
            contract_size="5,000 bushels",
            margin_requirement=0.070,
            tick_size=0.0025,
            risk_level="Medium",
            liquidity_score=0.78,
            volatility_profile="Medium",
            correlation_group="Oilseeds"
        ),
    }
}

# Enhanced benchmarks with additional metadata
BENCHMARKS = {
    "^GSPC": AssetMetadata(
        symbol="^GSPC",
        name="S&P 500 Index",
        category=AssetCategory.BENCHMARK,
        color="#1E90FF",
        description="S&P 500 Equity Index",
        exchange="NYSE",
        risk_level="Medium",
        liquidity_score=0.99
    ),
    "DX-Y.NYB": AssetMetadata(
        symbol="DX-Y.NYB",
        name="US Dollar Index",
        category=AssetCategory.CURRENCY,
        color="#32CD32",
        description="US Dollar Currency Index",
        exchange="ICE",
        risk_level="Low",
        liquidity_score=0.95
    ),
    "TLT": AssetMetadata(
        symbol="TLT",
        name="20+ Year Treasury ETF",
        category=AssetCategory.BENCHMARK,
        color="#8A2BE2",
        description="Long-term US Treasury Bonds",
        exchange="NYSE",
        risk_level="Low",
        liquidity_score=0.90
    ),
    "GLD": AssetMetadata(
        symbol="GLD",
        name="SPDR Gold Shares",
        category=AssetCategory.BENCHMARK,
        color="#FFD700",
        description="Gold-backed ETF",
        exchange="NYSE",
        risk_level="Low",
        liquidity_score=0.92
    ),
    "DBC": AssetMetadata(
        symbol="DBC",
        name="Invesco DB Commodity Index",
        category=AssetCategory.BENCHMARK,
        color="#FF6347",
        description="Broad Commodities ETF",
        exchange="NYSE",
        risk_level="Medium",
        liquidity_score=0.85
    ),
    "BCOM": AssetMetadata(
        symbol="BCOM",
        name="Bloomberg Commodity Index",
        category=AssetCategory.BENCHMARK,
        color="#9370DB",
        description="Broad Commodities Index",
        exchange="NYSE",
        risk_level="Medium",
        liquidity_score=0.80
    )
}

# =============================================================================
# ADVANCED CACHING SYSTEM WITH PERSISTENCE
# =============================================================================

class AdvancedCache:
    """Advanced caching with TTL, persistence, and compression"""
    
    def __init__(self, max_size_mb: int = 100, ttl_hours: int = 24):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    @staticmethod
    def generate_advanced_key(*args, **kwargs) -> str:
        """Generate deterministic cache key with type awareness"""
        import pickle
        key_parts = []
        
        def process_value(val):
            if isinstance(val, (str, int, float, bool, type(None))):
                return str(val)
            elif isinstance(val, (datetime, pd.Timestamp)):
                return val.isoformat()
            elif isinstance(val, pd.DataFrame):
                # Use content-based hashing for DataFrames
                try:
                    # Hash structure and content
                    struct_hash = hashlib.sha256(
                        pickle.dumps((val.shape, val.columns.tolist(), val.dtypes))
                    ).hexdigest()
                    content_hash = hashlib.sha256(
                        pd.util.hash_pandas_object(val).values.tobytes()
                    ).hexdigest()
                    return f"df_{struct_hash}_{content_hash}"
                except:
                    return str(hash(str(val)))
            elif isinstance(val, np.ndarray):
                return hashlib.sha256(val.tobytes()).hexdigest()
            elif isinstance(val, (list, tuple, dict)):
                return hashlib.sha256(json.dumps(val, sort_keys=True).encode()).hexdigest()
            else:
                try:
                    return hashlib.sha256(pickle.dumps(val)).hexdigest()
                except:
                    return str(hash(str(val)))
        
        # Process all arguments
        for arg in args:
            key_parts.append(process_value(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{process_value(v)}")
        
        # Create final key
        key_string = "_".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def cache_persistent(self, func_name: str = None, compress: bool = True):
        """Decorator for persistent caching"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.generate_advanced_key(func.__name__, *args, **kwargs)
                cache_file = self.cache_dir / f"{func_name or func.__name__}_{key}.pkl"
                
                # Check if valid cache exists
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            timestamp, result = pickle.load(f)
                        
                        # Check TTL
                        if (datetime.now().timestamp() - timestamp) < self.ttl_seconds:
                            return result
                    except:
                        pass  # Cache corrupted, recompute
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Save to cache
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump((datetime.now().timestamp(), result), f)
                except:
                    pass  # Failed to cache, but return result anyway
                
                # Cleanup old cache files
                self._cleanup_cache()
                
                return result
            
            return wrapper
        return decorator
    
    def _cleanup_cache(self):
        """Remove old cache files to stay within size limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            if total_size > self.max_size_bytes:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                
                for cache_file in cache_files:
                    total_size -= cache_file.stat().st_size
                    cache_file.unlink()
                    
                    if total_size <= self.max_size_bytes * 0.8:  # Keep some buffer
                        break
        except:
            pass  # Silently fail on cache cleanup

# Initialize advanced cache
advanced_cache = AdvancedCache(max_size_mb=200, ttl_hours=48)

# =============================================================================
# ENHANCED DATA MANAGER WITH ADVANCED TECHNIQUES
# =============================================================================

class EnhancedDataManager:
    """Advanced data management with machine learning preprocessing"""
    
    def __init__(self):
        self.cache = advanced_cache
        self._setup_advanced_indicators()
    
    def _setup_advanced_indicators(self):
        """Setup technical indicator parameters"""
        self.indicators_config = {
            'trend': {
                'ema_periods': [8, 21, 55, 200],
                'sma_periods': [20, 50, 100, 200],
                'wma_periods': [10, 20, 50],
                'dema_period': 20,
                'tema_period': 20
            },
            'momentum': {
                'rsi_period': 14,
                'stoch_period': 14,
                'stoch_smooth': 3,
                'williams_period': 14,
                'awesome_oscillator': True,
                'ultimate_oscillator': [7, 14, 28]
            },
            'volatility': {
                'bb_period': 20,
                'bb_std': 2.0,
                'atr_period': 14,
                'keltner_period': 20,
                'donchian_period': 20
            },
            'volume': {
                'obv': True,
                'volume_profile': True,
                'vwap': True,
                'mfi_period': 14
            },
            'advanced': {
                'ichimoku': True,
                'heikin_ashi': True,
                'renko': False,
                'pivot_points': True
            }
        }
    
    @advanced_cache.cache_persistent(func_name="fetch_asset_data", compress=True)
    def fetch_asset_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        retries: int = 3,
        fallback_source: bool = True
    ) -> pd.DataFrame:
        """Advanced data fetching with multiple fallback strategies"""
        
        # Check for cached data first
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        for attempt in range(retries):
            try:
                # Strategy 1: Standard yfinance with enhanced parameters
                if attempt == 0:
                    download_params = {
                        'tickers': symbol,
                        'start': start_date,
                        'end': end_date,
                        'interval': interval,
                        'auto_adjust': True,
                        'back_adjust': True,
                        'repair': True,
                        'keepna': False,
                        'progress': False,
                        'timeout': 30,
                        'threads': True
                    }
                    
                    df = yf_download_safe(download_params)
                
                # Strategy 2: Try with period instead of dates
                elif attempt == 1:
                    days_diff = (end_date - start_date).days
                    period = "max" if days_diff > 365*5 else f"{max(1, days_diff)}d"
                    
                    download_params = {
                        'tickers': symbol,
                        'period': period,
                        'interval': interval,
                        'auto_adjust': True,
                        'repair': True
                    }
                    
                    df = yf_download_safe(download_params)
                    if not df.empty:
                        df = df[df.index >= pd.Timestamp(start_date)]
                        df = df[df.index <= pd.Timestamp(end_date)]
                
                # Strategy 3: Try different interval
                else:
                    download_params = {
                        'tickers': symbol,
                        'start': start_date - timedelta(days=30),  # Buffer
                        'end': end_date,
                        'interval': "1d" if interval != "1d" else "1wk",
                        'auto_adjust': True
                    }
                    
                    df = yf_download_safe(download_params)
                    df = df[df.index >= pd.Timestamp(start_date)]
                
                # Enhanced validation and cleaning
                df = self._enhanced_clean_dataframe(df, symbol)
                
                if self._validate_dataframe(df, symbol):
                    # Add metadata
                    df.attrs['symbol'] = symbol
                    df.attrs['fetch_time'] = datetime.now()
                    df.attrs['source'] = 'yfinance'
                    
                    return df
                
            except Exception as e:
                if attempt == retries - 1:
                    st.warning(f"Failed to fetch {symbol} after {retries} attempts: {str(e)[:150]}")
                
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
                continue
        
        # Fallback to synthetic data or return empty
        if fallback_source:
            return self._create_synthetic_data(symbol, start_date, end_date)
        
        return pd.DataFrame()
    
    def _enhanced_clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Advanced data cleaning with imputation and validation"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'Adj Close': 'Adj_Close',
            'Adj_Close': 'Adj_Close',
            'Close': 'Close',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Volume': 'Volume'
        }
        
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['Close', 'Open', 'High', 'Low']
        
        # If no price columns, try to infer from available data
        if not any(col in df.columns for col in required_columns):
            if len(df.columns) >= 1:
                df['Close'] = df.iloc[:, 0]
                df['Open'] = df['Close']
                df['High'] = df['Close']
                df['Low'] = df['Close']
            else:
                return pd.DataFrame()
        
        # Ensure Close exists
        if 'Close' not in df.columns:
            if 'Adj_Close' in df.columns:
                df['Close'] = df['Adj_Close']
            elif 'Last' in df.columns:
                df['Close'] = df['Last']
            elif len(df.columns) > 0:
                df['Close'] = df.iloc[:, -1]
        
        # Fill missing OHLC
        for col in ['Open', 'High', 'Low']:
            if col not in df.columns:
                df[col] = df['Close']
        
        # Ensure Adj_Close exists
        if 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Handle Volume
        if 'Volume' not in df.columns:
            df['Volume'] = np.nan
        
        # Clean index
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Advanced imputation for missing values
        df = self._advanced_imputation(df)
        
        # Validate price monotonicity
        df = self._validate_price_monotonicity(df)
        
        # Add derived columns
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Weighted_Close'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4
        
        return df
    
    def _advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value imputation using multiple techniques"""
        df = df.copy()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Adj_Close']:
            if col in df.columns:
                # Forward fill for small gaps
                df[col] = df[col].ffill(limit=5)
                
                # Backward fill if needed
                df[col] = df[col].bfill(limit=5)
                
                # Linear interpolation for remaining gaps
                if df[col].isna().sum() > 0:
                    df[col] = df[col].interpolate(
                        method='linear',
                        limit_direction='both'
                    )
        
        # For volume, use median imputation
        if 'Volume' in df.columns:
            median_volume = df['Volume'].median()
            df['Volume'] = df['Volume'].fillna(median_volume)
        
        return df
    
    def _validate_price_monotonicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct price anomalies"""
        df = df.copy()
        
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
        price_cols = [col for col in price_cols if col in df.columns]
        
        for col in price_cols:
            # Remove extreme outliers (beyond 10 standard deviations)
            price_series = df[col]
            z_scores = np.abs((price_series - price_series.mean()) / price_series.std())
            outliers = z_scores > 10
            
            if outliers.any():
                # Replace outliers with interpolated values
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate(method='linear')
            
            # Ensure High >= Low and High >= Close and Low <= Close
            if col == 'High':
                df[col] = np.maximum(df[col], df['Low'])
                df[col] = np.maximum(df[col], df['Close'])
            elif col == 'Low':
                df[col] = np.minimum(df[col], df['High'])
                df[col] = np.minimum(df[col], df['Close'])
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str) -> bool:
        """Comprehensive dataframe validation"""
        if df.empty:
            return False
        
        # Check minimum rows
        if len(df) < 10:
            return False
        
        # Check required columns
        required_cols = ['Close', 'Open', 'High', 'Low']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for NaN in critical columns
        critical_cols = ['Close', 'Adj_Close']
        critical_cols = [col for col in critical_cols if col in df.columns]
        
        if any(df[col].isna().all() for col in critical_cols):
            return False
        
        # Check price consistency
        price_checks = (
            (df['High'] >= df['Low']).all() and
            (df['High'] >= df['Close']).all() and
            (df['Low'] <= df['Close']).all()
        )
        
        if not price_checks:
            return False
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            return False
        
        return True
    
    def _create_synthetic_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create synthetic data for testing when real data is unavailable"""
        days = (end_date - start_date).days + 1
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create realistic price series with GBM
        np.random.seed(hash(symbol) % 10000)
        
        # Base price based on symbol
        base_prices = {
            'GC=F': 1800,
            'SI=F': 22,
            'CL=F': 75,
            'HG=F': 3.8,
            'ZC=F': 4.5,
            'default': 100
        }
        
        base_price = base_prices.get(symbol, base_prices['default'])
        daily_vol = 0.015  # 1.5% daily volatility
        
        # Generate log returns
        returns = np.random.normal(0, daily_vol, days)
        prices = base_price * np.exp(np.cumsum(returns - 0.5 * daily_vol**2))
        
        # Create OHLC data
        df = pd.DataFrame(index=dates[:len(prices)])
        df['Close'] = prices
        
        # Add some noise to create OHLC
        noise_scale = 0.002
        df['Open'] = df['Close'].shift(1).fillna(df['Close'] * (1 + np.random.normal(0, noise_scale)))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, noise_scale/2)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, noise_scale/2)))
        df['Adj_Close'] = df['Close']
        df['Volume'] = np.random.lognormal(14, 1, len(df))  # Realistic volume distribution
        
        # Fill any remaining NaNs
        df = df.ffill().bfill()
        
        df.attrs['symbol'] = symbol
        df.attrs['synthetic'] = True
        df.attrs['warning'] = 'Using synthetic data - real data unavailable'
        
        return df
    
    def calculate_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators using advanced techniques"""
        df = df.copy()
        
        # Ensure required columns
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        
        # 1. PRICE TRANSFORMS
        df['Log_Price'] = np.log(df[price_col])
        df['Returns'] = df[price_col].pct_change()
        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 2. TREND INDICATORS
        # Multiple EMAs
        for period in self.indicators_config['trend']['ema_periods']:
            df[f'EMA_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        
        # DEMA (Double EMA)
        dema_period = self.indicators_config['trend']['dema_period']
        ema = df[price_col].ewm(span=dema_period, adjust=False).mean()
        ema_of_ema = ema.ewm(span=dema_period, adjust=False).mean()
        df[f'DEMA_{dema_period}'] = 2 * ema - ema_of_ema
        
        # TEMA (Triple EMA)
        tema_period = self.indicators_config['trend']['tema_period']
        ema1 = df[price_col].ewm(span=tema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=tema_period, adjust=False).mean()
        ema3 = ema2.ewm(span=tema_period, adjust=False).mean()
        df[f'TEMA_{tema_period}'] = 3 * ema1 - 3 * ema2 + ema3
        
        # 3. MOMENTUM INDICATORS
        # RSI with different periods
        for period in [7, 14, 21]:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        stoch_period = self.indicators_config['momentum']['stoch_period']
        stoch_smooth = self.indicators_config['momentum']['stoch_smooth']
        
        lowest_low = df['Low'].rolling(window=stoch_period).min()
        highest_high = df['High'].rolling(window=stoch_period).max()
        df['Stoch_%K'] = 100 * ((df[price_col] - lowest_low) / (highest_high - lowest_low))
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=stoch_smooth).mean()
        
        # Awesome Oscillator
        if self.indicators_config['momentum']['awesome_oscillator']:
            median_price = (df['High'] + df['Low']) / 2
            df['Awesome_OSC'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        
        # 4. VOLATILITY INDICATORS
        # Bollinger Bands with multiple deviations
        bb_period = self.indicators_config['volatility']['bb_period']
        bb_middle = df[price_col].rolling(window=bb_period).mean()
        bb_std = df[price_col].rolling(window=bb_period).std()
        
        for std_dev in [1, 1.5, 2, 2.5]:
            df[f'BB_Upper_{std_dev}'] = bb_middle + (bb_std * std_dev)
            df[f'BB_Lower_{std_dev}'] = bb_middle - (bb_std * std_dev)
        
        df['BB_Width'] = (df['BB_Upper_2'] - df['BB_Lower_2']) / bb_middle
        df['BB_%B'] = (df[price_col] - df['BB_Lower_2']) / (df['BB_Upper_2'] - df['BB_Lower_2'])
        
        # ATR (Average True Range)
        atr_period = self.indicators_config['volatility']['atr_period']
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df[price_col].shift())
        low_close = np.abs(df['Low'] - df[price_col].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=atr_period).mean()
        df['ATR_Pct'] = df['ATR'] / df[price_col] * 100
        
        # Keltner Channels
        kc_period = self.indicators_config['volatility']['keltner_period']
        kc_middle = df[price_col].ewm(span=kc_period).mean()
        kc_range = df['ATR'] * 2
        df['KC_Upper'] = kc_middle + kc_range
        df['KC_Lower'] = kc_middle - kc_range
        
        # 5. VOLUME INDICATORS
        if 'Volume' in df.columns:
            # On-Balance Volume
            df['OBV'] = (np.sign(df['Returns'].fillna(0)) * df['Volume']).cumsum()
            
            # Volume Weighted Average Price
            df['VWAP'] = (df['Volume'] * df['Typical_Price']).cumsum() / df['Volume'].cumsum()
            
            # Money Flow Index
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(df['Typical_Price'] > df['Typical_Price'].shift(1), 0)
            negative_flow = money_flow.where(df['Typical_Price'] < df['Typical_Price'].shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            money_ratio = positive_mf / negative_mf
            df['MFI'] = 100 - (100 / (1 + money_ratio))
            
            # Volume Profile
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Volume_Spike'] = df['Volume_Ratio'] > 2.0
        
        # 6. ADVANCED INDICATORS
        # Ichimoku Cloud
        if self.indicators_config['advanced']['ichimoku']:
            # Tenkan-sen (Conversion Line)
            period9_high = df['High'].rolling(window=9).max()
            period9_low = df['Low'].rolling(window=9).min()
            df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = df['High'].rolling(window=26).max()
            period26_low = df['Low'].rolling(window=26).min()
            df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            period52_high = df['High'].rolling(window=52).max()
            period52_low = df['Low'].rolling(window=52).min()
            df['Ichimoku_Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Heikin-Ashi
        if self.indicators_config['advanced']['heikin_ashi']:
            df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
            df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
            df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
            df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        # 7. STATISTICAL FEATURES
        # Rolling statistics
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            df[f'Rolling_Mean_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df[price_col].rolling(window=window).std()
            df[f'Rolling_Skew_{window}'] = df[price_col].rolling(window=window).skew()
            df[f'Rolling_Kurt_{window}'] = df[price_col].rolling(window=window).kurt()
            df[f'Rolling_Min_{window}'] = df[price_col].rolling(window=window).min()
            df[f'Rolling_Max_{window}'] = df[price_col].rolling(window=window).max()
        
        # Volatility measures
        df['Volatility_20D'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_60D'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
        df['Volatility_120D'] = df['Returns'].rolling(window=120).std() * np.sqrt(252)
        
        # Realized volatility (Parkinson estimator)
        df['Realized_Vol_Parkinson'] = np.sqrt(
            (1 / (4 * np.log(2))) * ((np.log(df['High'] / df['Low'])) ** 2).rolling(window=20).mean()
        ) * np.sqrt(252)
        
        # 8. CYCLE ANALYSIS
        # Hilbert transform for cycle detection
        analytic_signal = signal.hilbert(df[price_col].fillna(method='ffill').values)
        df['Hilbert_Amplitude'] = np.abs(analytic_signal)
        df['Hilbert_Phase'] = np.angle(analytic_signal)
        
        # 9. MACHINE LEARNING FEATURES
        # Price position in range
        df['Price_Position'] = (df[price_col] - df['Low'].rolling(20).min()) / \
                              (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Trend strength (correlation with time)
        def trend_strength(x):
            if len(x) < 2:
                return 0
            return np.corrcoef(np.arange(len(x)), x)[0, 1]
        
        df['Trend_Strength_20'] = df[price_col].rolling(window=20).apply(trend_strength, raw=True)
        
        # Rate of change
        for period in [5, 10, 20, 50]:
            df[f'ROC_{period}'] = ((df[price_col] - df[price_col].shift(period)) / 
                                  df[price_col].shift(period)) * 100
        
        # 10. RISK METRICS
        # Value at Risk (historical)
        for confidence in [0.90, 0.95, 0.99]:
            df[f'VaR_{int(confidence*100)}'] = df['Returns'].rolling(window=252).apply(
                lambda x: np.percentile(x, (1-confidence)*100) if len(x) > 50 else np.nan
            )
        
        # Expected Shortfall
        df['ES_95'] = df['Returns'].rolling(window=252).apply(
            lambda x: x[x <= np.percentile(x, 5)].mean() if len(x) > 50 else np.nan
        )
        
        # Drop rows with insufficient data for calculations
        df = df.dropna(subset=['Returns', 'Volatility_20D'])
        
        # Add metadata
        df.attrs['technical_features_generated'] = True
        df.attrs['feature_count'] = len(df.columns)
        
        return df

# =============================================================================
# ADVANCED ANALYTICS ENGINE WITH MACHINE LEARNING
# =============================================================================

class InstitutionalAnalytics:
    """Institutional-grade analytics engine with ML and advanced methods"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.annual_trading_days = 252
        self._setup_advanced_models()
    
    def _setup_advanced_models(self):
        """Initialize advanced models and estimators"""
        self.models = {}
        self.scalers = {}
        
        # Check for ML dependencies
        try:
            from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
            from sklearn.preprocessing import RobustScaler, PowerTransformer
            from sklearn.decomposition import PCA, KernelPCA
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            
            self.models['covariance'] = {
                'ledoit_wolf': LedoitWolf,
                'oas': OAS,
                'empirical': EmpiricalCovariance
            }
            
            self.models['scalers'] = {
                'robust': RobustScaler,
                'power': PowerTransformer
            }
            
            self.models['pca'] = PCA
            self.models['kpca'] = KernelPCA
            self.models['isolation_forest'] = IsolationForest
            self.models['lof'] = LocalOutlierFactor
            
        except ImportError:
            self.models['available'] = False
    
    def calculate_advanced_performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics with advanced statistics"""
        returns = returns.dropna()
        
        if len(returns) < 20:
            return {}
        
        # Use cached calculations if available
        cache_key = f"metrics_{hashlib.sha256(returns.values.tobytes()).hexdigest()}"
        
        # 1. BASIC PERFORMANCE METRICS
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualization
        years = len(returns) / self.annual_trading_days
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 2. RISK METRICS
        annual_vol = returns.std() * np.sqrt(self.annual_trading_days)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.annual_trading_days) if len(downside_returns) > 1 else 0
        
        # Advanced risk metrics
        upside_returns = returns[returns > 0]
        upside_vol = upside_returns.std() * np.sqrt(self.annual_trading_days) if len(upside_returns) > 1 else 0
        
        # 3. RISK-ADJUSTED RETURNS
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Upside potential ratio
        upside_potential = upside_returns.mean() * self.annual_trading_days if len(upside_returns) > 0 else 0
        upside_potential_ratio = upside_potential / downside_vol if downside_vol > 0 else 0
        
        # 4. DRAWDOWN ANALYSIS
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Advanced drawdown metrics
        dd_duration = self._calculate_drawdown_duration(drawdown)
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        recovery_factor = -total_return / max_dd if max_dd < 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
        
        # 5. HIGHER MOMENTS AND DISTRIBUTION
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
        
        # 6. VAR AND EXPECTED SHORTFALL
        var_levels = [0.90, 0.95, 0.99]
        var_metrics = {}
        
        for level in var_levels:
            var_historical = np.percentile(returns, (1 - level) * 100)
            
            # Parametric VaR (Cornish-Fisher)
            z = stats.norm.ppf(1 - level)
            z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skewness**2 / 36
            var_parametric = returns.mean() + returns.std() * z_cf
            
            # CVaR/ES
            cvar = returns[returns <= var_historical].mean()
            
            var_metrics[f'var_{int(level*100)}'] = var_historical * 100
            var_metrics[f'var_parametric_{int(level*100)}'] = var_parametric * 100
            var_metrics[f'cvar_{int(level*100)}'] = cvar * 100
        
        # 7. GAIN/LOSS METRICS
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')
        
        # 8. BENCHMARK COMPARISON (if available)
        alpha = beta = treynor = information_ratio = tracking_error = 0
        omega_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns
            aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
            if len(aligned) > 20:
                asset_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]
                
                # Regression analysis
                X = sm.add_constant(bench_ret)
                model = sm.OLS(asset_ret, X).fit()
                alpha = model.params[0] * self.annual_trading_days
                beta = model.params[1]
                
                # Advanced metrics
                excess_returns = asset_ret - bench_ret
                tracking_error = excess_returns.std() * np.sqrt(self.annual_trading_days)
                information_ratio = excess_returns.mean() * self.annual_trading_days / tracking_error if tracking_error > 0 else 0
                treynor = (annual_return - self.risk_free_rate) / beta if beta != 0 else 0
                
                # Omega Ratio
                threshold = self.risk_free_rate / self.annual_trading_days
                gains = asset_ret[asset_ret > threshold].sum()
                losses = abs(asset_ret[asset_ret <= threshold].sum())
                omega_ratio = gains / losses if losses > 0 else float('inf')
        
        # 9. TIME-SERIES PROPERTIES
        # Autocorrelation
        autocorr = returns.autocorr(lag=1)
        
        # Runs test for randomness
        from statsmodels.sandbox.stats.runs import runstest_1samp
        try:
            runs_stat, runs_p = runstest_1samp(returns > returns.mean())
        except:
            runs_stat = runs_p = 0
        
        # 10. COMPOSITE METRICS
        # Burke Ratio
        burke_ratio = (annual_return - self.risk_free_rate) / np.sqrt((drawdown ** 2).sum() / len(drawdown)) \
            if len(drawdown) > 0 and (drawdown ** 2).sum() > 0 else 0
        
        # Sterling Ratio
        avg_max_dd = drawdown.rolling(window=3).min().mean()  # Average of 3 worst drawdowns
        sterling_ratio = (annual_return - self.risk_free_rate) / abs(avg_max_dd) if avg_max_dd != 0 else 0
        
        # Return to VaR ratio
        return_to_var = (annual_return - self.risk_free_rate) / abs(var_metrics.get('var_95', 1)) \
            if var_metrics.get('var_95', 0) != 0 else 0
        
        # 11. COMPREHENSIVE RESULTS
        results = {
            # Basic metrics
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'downside_volatility': downside_vol * 100,
            'upside_volatility': upside_vol * 100,
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'upside_potential_ratio': upside_potential_ratio,
            'omega_ratio': omega_ratio if omega_ratio != float('inf') else 1000,
            'burke_ratio': burke_ratio,
            'sterling_ratio': sterling_ratio,
            'return_to_var_ratio': return_to_var,
            'calmar_ratio': (annual_return - self.risk_free_rate) / abs(max_dd) if max_dd != 0 else 0,
            
            # Drawdown metrics
            'max_drawdown': max_dd * 100,
            'avg_drawdown': avg_dd * 100,
            'max_dd_duration': dd_duration['max_duration'],
            'avg_dd_duration': dd_duration['avg_duration'],
            'recovery_factor': recovery_factor,
            'ulcer_index': ulcer_index,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jarque_bera_stat,
            'jarque_bera_p': jarque_bera_p,
            
            # Gain/Loss metrics
            'win_rate': win_rate * 100,
            'avg_gain': avg_gain * 100,
            'avg_loss': avg_loss * 100,
            'gain_loss_ratio': gain_loss_ratio if gain_loss_ratio != float('inf') else 1000,
            'profit_factor': profit_factor if profit_factor != float('inf') else 1000,
            'positive_returns': len(positive_returns),
            'negative_returns': len(negative_returns),
            'total_trades': len(returns),
            
            # Time-series properties
            'autocorrelation': autocorr,
            'runs_test_stat': runs_stat,
            'runs_test_p': runs_p,
            
            # Benchmark metrics (if available)
            'alpha': alpha * 100,
            'beta': beta,
            'treynor_ratio': treynor,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error * 100,
            
            # VaR and CVaR metrics
            **var_metrics,
            
            # Metadata
            'years_data': years,
            'data_points': len(returns),
            'start_date': returns.index[0].strftime('%Y-%m-%d'),
            'end_date': returns.index[-1].strftime('%Y-%m-%d'),
            'risk_free_rate': self.risk_free_rate * 100
        }
        
        return results
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict[str, Any]:
        """Calculate detailed drawdown duration statistics"""
        if drawdown.empty:
            return {'max_duration': 0, 'avg_duration': 0, 'durations': []}
        
        current_duration = 0
        durations = []
        in_drawdown = False
        
        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        # Handle case where drawdown continues to the end
        if in_drawdown:
            durations.append(current_duration)
        
        if durations:
            return {
                'max_duration': max(durations),
                'avg_duration': np.mean(durations),
                'durations': durations,
                'count': len(durations)
            }
        else:
            return {'max_duration': 0, 'avg_duration': 0, 'durations': [], 'count': 0}
    
    def optimize_portfolio_advanced(
        self,
        returns_df: pd.DataFrame,
        method: str = 'sharpe',
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        covariance_method: str = 'ledoit_wolf',
        risk_model: str = 'semi_variance'
    ) -> Dict[str, Any]:
        """Advanced portfolio optimization with multiple covariance estimators and risk models"""
        
        if returns_df.empty or len(returns_df) < 60:
            return {'success': False, 'message': 'Insufficient data', 'code': 'INSUFFICIENT_DATA'}
        
        n_assets = returns_df.shape[1]
        assets = returns_df.columns.tolist()
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'sum_to_one': True,
                'long_only': True,
                'max_concentration': 0.3,
                'turnover_limit': 0.5
            }
        
        # Set bounds
        if constraints.get('long_only', True):
            bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                          for _ in range(n_assets))
        else:
            bounds = tuple((-constraints['max_weight'], constraints['max_weight']) 
                          for _ in range(n_assets))
        
        # Initial weights (equal weight or previous weights)
        init_weights = np.ones(n_assets) / n_assets
        
        # Define constraints
        opt_constraints = []
        
        # Sum to one constraint
        if constraints.get('sum_to_one', True):
            opt_constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Target return constraint
        if target_return is not None:
            opt_constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(returns_df.mean() * w) * self.annual_trading_days - target_return
            })
        
        # Concentration constraint
        if 'max_concentration' in constraints:
            max_conc = constraints['max_concentration']
            opt_constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_conc - np.max(np.abs(w))
            })
        
        # Advanced covariance estimation
        cov_matrix = self._estimate_covariance(returns_df, method=covariance_method)
        mean_returns = returns_df.mean() * self.annual_trading_days
        
        # Define objective functions based on risk model
        if risk_model == 'semi_variance':
            objective = self._create_semivariance_objective(returns_df, mean_returns)
        elif risk_model == 'cvar':
            objective = self._create_cvar_objective(returns_df, mean_returns, alpha=0.05)
        else:  # variance
            objective = self._create_variance_objective(cov_matrix, mean_returns, method)
        
        # Perform optimization with advanced settings
        try:
            result = optimize.minimize(
                objective,
                x0=init_weights,
                bounds=bounds,
                constraints=opt_constraints,
                method='SLSQP',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-10,
                    'eps': 1e-8,
                    'disp': False
                },
                callback=self._optimization_callback
            )
            
            if result.success:
                optimized_weights = result.x
                optimized_weights = optimized_weights / np.sum(np.abs(optimized_weights))  # Normalize
                
                # Calculate portfolio metrics
                portfolio_returns = returns_df @ optimized_weights
                metrics = self.calculate_advanced_performance_metrics(portfolio_returns)
                
                # Advanced risk decomposition
                risk_decomp = self._advanced_risk_decomposition(
                    returns_df, optimized_weights, cov_matrix, risk_model
                )
                
                # Calculate diversification metrics
                diversification = self._calculate_diversification_metrics(
                    returns_df, optimized_weights, cov_matrix
                )
                
                # Calculate transaction costs
                transaction_costs = self._estimate_transaction_costs(
                    init_weights, optimized_weights, returns_df
                )
                
                # Stability analysis
                stability = self._analyze_portfolio_stability(
                    returns_df, optimized_weights, window=60
                )
                
                return {
                    'success': True,
                    'weights': dict(zip(assets, optimized_weights)),
                    'metrics': metrics,
                    'risk_decomposition': risk_decomp,
                    'diversification': diversification,
                    'transaction_costs': transaction_costs,
                    'stability': stability,
                    'optimization_info': {
                        'iterations': result.nit,
                        'function_calls': result.nfev,
                        'objective_value': result.fun,
                        'method': method,
                        'risk_model': risk_model,
                        'covariance_method': covariance_method
                    }
                }
            else:
                return {
                    'success': False,
                    'message': f"Optimization failed: {result.message}",
                    'code': 'OPTIMIZATION_FAILED'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Optimization error: {str(e)}",
                'code': 'OPTIMIZATION_ERROR',
                'exception': str(e)
            }
    
    def _estimate_covariance(self, returns_df: pd.DataFrame, method: str = 'ledoit_wolf') -> np.ndarray:
        """Advanced covariance matrix estimation"""
        
        if method == 'empirical':
            return returns_df.cov().values * self.annual_trading_days
        
        elif method == 'ledoit_wolf' and self.models.get('available', False):
            lw = self.models['covariance']['ledoit_wolf'](assume_centered=True)
            lw.fit(returns_df.values)
            return lw.covariance_ * self.annual_trading_days
        
        elif method == 'oas' and self.models.get('available', False):
            oas = self.models['covariance']['oas'](assume_centered=True)
            oas.fit(returns_df.values)
            return oas.covariance_ * self.annual_trading_days
        
        else:
            # Fallback to empirical with shrinkage
            empirical_cov = returns_df.cov().values * self.annual_trading_days
            n = len(returns_df)
            p = len(returns_df.columns)
            
            # Simple shrinkage towards identity
            shrinkage = min(1, p / n)
            target = np.eye(p) * np.trace(empirical_cov) / p
            
            return (1 - shrinkage) * empirical_cov + shrinkage * target
    
    def _create_variance_objective(self, cov_matrix: np.ndarray, mean_returns: np.ndarray, method: str):
        """Create objective function for variance-based optimization"""
        
        if method == 'sharpe':
            def objective(weights):
                port_return = np.sum(mean_returns * weights)
                port_var = weights.T @ cov_matrix @ weights
                port_vol = np.sqrt(port_var)
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 1e6
        
        elif method == 'min_variance':
            def objective(weights):
                return weights.T @ cov_matrix @ weights
        
        elif method == 'max_return':
            def objective(weights):
                return -np.sum(mean_returns * weights)
        
        elif method == 'risk_parity':
            def objective(weights):
                # Risk parity objective: equal risk contribution
                risk_contributions = (cov_matrix @ weights) * weights / (weights.T @ cov_matrix @ weights)
                return np.sum((risk_contributions - 1/len(weights))**2)
        
        else:
            def objective(weights):
                port_return = np.sum(mean_returns * weights)
                port_var = weights.T @ cov_matrix @ weights
                port_vol = np.sqrt(port_var)
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 1e6
        
        return objective
    
    def _create_semivariance_objective(self, returns_df: pd.DataFrame, mean_returns: np.ndarray):
        """Create objective function for semi-variance optimization"""
        
        def objective(weights):
            port_returns = returns_df @ weights
            downside_returns = port_returns[port_returns < 0]
            
            if len(downside_returns) > 1:
                semi_variance = np.var(downside_returns) * self.annual_trading_days
            else:
                semi_variance = 1e-6
            
            port_return = np.sum(mean_returns * weights)
            return -port_return / np.sqrt(semi_variance) if semi_variance > 0 else 1e6
        
        return objective
    
    def _create_cvar_objective(self, returns_df: pd.DataFrame, mean_returns: np.ndarray, alpha: float = 0.05):
        """Create objective function for CVaR optimization"""
        
        def objective(weights):
            port_returns = returns_df @ weights
            var = np.percentile(port_returns, alpha * 100)
            cvar = port_returns[port_returns <= var].mean()
            
            port_return = np.sum(mean_returns * weights)
            return -port_return / abs(cvar) if cvar < 0 else 1e6
        
        return objective
    
    def _advanced_risk_decomposition(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_model: str
    ) -> Dict[str, Any]:
        """Advanced risk decomposition analysis"""
        
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        if portfolio_variance <= 0:
            return {'marginal_risk': {}, 'component_risk': {}, 'percentage_risk': {}}
        
        # Marginal risk contributions
        marginal_risk = (cov_matrix @ weights) / portfolio_vol
        
        # Component risk contributions
        component_risk = marginal_risk * weights
        
        # Percentage contributions
        percentage_risk = (component_risk / portfolio_vol) * 100
        
        # Additional risk metrics
        assets = returns_df.columns.tolist()
        
        # Standalone risk
        standalone_risk = np.sqrt(np.diag(cov_matrix))
        
        # Diversification benefit
        diversification_benefit = (np.sum(np.abs(weights) * standalone_risk) - portfolio_vol) / portfolio_vol
        
        # Risk concentration (Herfindahl index)
        risk_concentration = np.sum(percentage_risk ** 2) / 10000
        
        return {
            'marginal_risk': dict(zip(assets, marginal_risk)),
            'component_risk': dict(zip(assets, component_risk)),
            'percentage_risk': dict(zip(assets, percentage_risk)),
            'standalone_risk': dict(zip(assets, standalone_risk)),
            'diversification_benefit': diversification_benefit * 100,
            'risk_concentration': risk_concentration * 100,
            'portfolio_volatility': portfolio_vol * 100
        }
    
    def _calculate_diversification_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive diversification metrics"""
        
        # Diversification ratio
        standalone_vols = np.sqrt(np.diag(cov_matrix))
        weighted_standalone_vol = np.sum(np.abs(weights) * standalone_vols)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        diversification_ratio = weighted_standalone_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Effective number of bets (based on risk contributions)
        risk_contributions = (cov_matrix @ weights) * weights / (weights.T @ cov_matrix @ weights)
        effective_bets = 1 / np.sum(risk_contributions ** 2) if np.sum(risk_contributions ** 2) > 0 else len(weights)
        
        # Correlation-based diversification
        corr_matrix = returns_df.corr().values
        avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        # Maximum drawdown diversification
        # (This would require more complex simulation)
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_bets': effective_bets,
            'avg_correlation': avg_correlation,
            'concentration_index': 1 / effective_bets,
            'diversification_score': (diversification_ratio - 1) * 100  # Percentage benefit
        }
    
    def _estimate_transaction_costs(
        self,
        initial_weights: np.ndarray,
        target_weights: np.ndarray,
        returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Estimate transaction costs for portfolio rebalancing"""
        
        # Turnover
        turnover = np.sum(np.abs(target_weights - initial_weights)) / 2
        
        # Assume costs (can be customized)
        commission_rate = 0.001  # 0.1%
        spread_cost = 0.0005     # 0.05%
        
        # Total cost estimate
        total_cost = turnover * (commission_rate + spread_cost)
        
        # Annualized cost assuming quarterly rebalancing
        annual_cost = total_cost * 4
        
        return {
            'turnover': turnover * 100,  # Percentage
            'estimated_cost': total_cost * 100,
            'annualized_cost': annual_cost * 100,
            'commission_rate': commission_rate * 100,
            'spread_cost': spread_cost * 100
        }
    
    def _analyze_portfolio_stability(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        window: int = 60
    ) -> Dict[str, Any]:
        """Analyze portfolio stability through time"""
        
        n_periods = len(returns_df)
        if n_periods < window * 2:
            return {'available': False, 'message': 'Insufficient data for stability analysis'}
        
        # Rolling portfolio returns
        rolling_returns = []
        weight_stability = []
        
        for i in range(window, n_periods):
            # Use same weights for simplicity (in reality would re-optimize)
            period_returns = returns_df.iloc[i-window:i] @ weights
            rolling_returns.append(period_returns.mean() * self.annual_trading_days)
            
            # Could add re-optimization here for true stability analysis
        
        rolling_returns = np.array(rolling_returns)
        
        # Stability metrics
        returns_std = np.std(rolling_returns)
        returns_range = np.ptp(rolling_returns)
        
        # Information ratio of rolling returns
        if len(rolling_returns) > 1:
            info_ratio = np.mean(rolling_returns) / np.std(rolling_returns) if np.std(rolling_returns) > 0 else 0
        else:
            info_ratio = 0
        
        return {
            'available': True,
            'rolling_return_mean': np.mean(rolling_returns) if len(rolling_returns) > 0 else 0,
            'rolling_return_std': returns_std,
            'rolling_return_range': returns_range,
            'stability_ratio': np.mean(rolling_returns) / returns_std if returns_std > 0 else 0,
            'max_drawdown_stability': np.min(rolling_returns) - np.max(rolling_returns),
            'information_ratio_stability': info_ratio,
            'n_periods_analyzed': len(rolling_returns)
        }
    
    def _optimization_callback(self, xk):
        """Callback function for optimization progress"""
        # Can be used to track optimization progress
        pass
    
    # GARCH modeling with enhancements
    def garch_analysis_advanced(
        self,
        returns: pd.Series,
        p_range: Tuple[int, int] = (1, 3),
        q_range: Tuple[int, int] = (1, 3),
        distributions: List[str] = None,
        include_egarch: bool = True,
        include_gjr: bool = True,
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """Advanced GARCH analysis with multiple model types"""
        if not dep_manager.is_available('arch'):
            return {'available': False, 'message': 'ARCH package not available'}
        
        if distributions is None:
            distributions = ['normal', 't', 'skewt']
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 300:
            return {'available': False, 'message': 'Insufficient data for GARCH (min 300 obs required)'}
        
        # Scale returns
        returns_scaled = returns_clean * 100
        
        results = []
        arch_model = dep_manager.dependencies['arch']['arch_model']
        
        # Test different GARCH specifications
        garch_specs = []
        
        # Standard GARCH
        for p in range(p_range[0], p_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                garch_specs.append(('GARCH', p, q))
        
        # EGARCH
        if include_egarch:
            for p in range(1, min(3, p_range[1]) + 1):
                for q in range(1, min(3, q_range[1]) + 1):
                    garch_specs.append(('EGARCH', p, q))
        
        # GJR-GARCH
        if include_gjr:
            for p in range(1, min(3, p_range[1]) + 1):
                for q in range(1, min(3, q_range[1]) + 1):
                    garch_specs.append(('GARCH', p, q))  # GJR is special case
        
        # Fit models
        for spec in garch_specs:
            model_type, p, q = spec
            
            for dist in distributions:
                try:
                    if model_type == 'EGARCH':
                        model = arch_model(
                            returns_scaled,
                            mean='Constant',
                            vol='EGARCH',
                            p=p,
                            q=q,
                            dist=dist,
                            power=2.0
                        )
                    elif model_type == 'GJR':
                        model = arch_model(
                            returns_scaled,
                            mean='Constant',
                            vol='GARCH',
                            p=p,
                            q=q,
                            o=1,  # GJR parameter
                            dist=dist
                        )
                    else:  # Standard GARCH
                        model = arch_model(
                            returns_scaled,
                            mean='Constant',
                            vol='GARCH',
                            p=p,
                            q=q,
                            dist=dist
                        )
                    
                    fit = model.fit(disp='off', show_warning=False, options={'maxiter': 1000})
                    
                    # Calculate diagnostics
                    std_resid = fit.resid / fit.conditional_volatility
                    
                    # Calculate persistence
                    if model_type == 'GARCH':
                        persistence = np.sum(fit.params[p+1:p+q+1])  # GARCH parameters
                    elif model_type == 'EGARCH':
                        persistence = 0.9  # Placeholder
                    else:
                        persistence = 0.9
                    
                    # Calculate information criteria
                    n_params = len(fit.params)
                    n_obs = len(returns_scaled)
                    
                    aic = fit.aic
                    bic = fit.bic
                    hqic = 2 * fit.loglikelihood + 2 * n_params * np.log(np.log(n_obs))  # Hannan-Quinn
                    
                    # Store results
                    model_result = {
                        'type': model_type,
                        'p': p,
                        'q': q,
                        'distribution': dist,
                        'aic': aic,
                        'bic': bic,
                        'hqic': hqic,
                        'log_likelihood': fit.loglikelihood,
                        'converged': fit.convergence_flag == 0,
                        'persistence': persistence,
                        'params': dict(fit.params),
                        'conditional_volatility': fit.conditional_volatility / 100,
                        'residuals': fit.resid / 100,
                        'std_residuals': std_resid,
                        'information_criteria': {
                            'aic': aic,
                            'bic': bic,
                            'hqic': hqic
                        }
                    }
                    
                    # Add forecasts if converged
                    if fit.convergence_flag == 0:
                        try:
                            forecast = fit.forecast(horizon=forecast_horizon, reindex=False)
                            model_result['volatility_forecast'] = np.sqrt(forecast.variance.values[-1, :]) / 100
                        except:
                            model_result['volatility_forecast'] = None
                    
                    results.append(model_result)
                    
                except Exception as e:
                    continue
        
        if not results:
            return {'available': False, 'message': 'No GARCH models converged'}
        
        # Select best model based on BIC
        results_df = pd.DataFrame(results)
        best_model = results_df.loc[results_df['bic'].idxmin()].to_dict()
        
        # Calculate model confidence set (simplified)
        top_models = results_df.nsmallest(5, 'bic')
        
        return {
            'available': True,
            'best_model': best_model,
            'top_models': top_models.to_dict('records'),
            'all_models': results,
            'n_models_tested': len(results),
            'model_types_tested': list(set([r['type'] for r in results])),
            'distributions_tested': distributions,
            'returns': returns_clean,
            'summary': {
                'best_type': best_model['type'],
                'best_order': f"{best_model['p']},{best_model['q']}",
                'best_distribution': best_model['distribution'],
                'best_bic': best_model['bic'],
                'persistence': best_model.get('persistence', 0)
            }
        }



    # -------------------------------------------------------------------------
    # Regime Detection (HMM / fallback)
    # -------------------------------------------------------------------------
    def detect_regimes(
        self,
        returns: pd.Series,
        n_regimes: int = 3,
        features: Optional[List[str]] = None,
        random_state: int = 42,
        lookback: int = 21,
        n_iter: int = 500
    ) -> Dict[str, Any]:
        """Detect market regimes using HMM (preferred) with robust fallbacks.

        Parameters
        ----------
        returns : pd.Series
            Return series (daily). Index must be datetime-like.
        n_regimes : int
            Number of latent regimes to detect.
        features : list[str]
            Feature names from: returns, volatility, momentum, range, volume.
            (volume is not available if only returns are provided; it will be ignored.)
        """
        if features is None:
            features = ["returns", "volatility"]

        if returns is None:
            return {"available": False, "message": "No returns provided"}

        rets = pd.Series(returns).dropna()
        if len(rets) < max(200, lookback * 10):
            return {"available": False, "message": "Insufficient data for regime detection (min ~200 obs required)"}

        # Build features from returns only (volume not available here)
        feat = pd.DataFrame(index=rets.index)

        if "returns" in features:
            feat["returns"] = rets

        if "volatility" in features:
            feat["volatility"] = rets.rolling(lookback).std()

        if "momentum" in features:
            feat["momentum"] = rets.rolling(lookback).mean()

        if "range" in features:
            feat["range"] = rets.rolling(lookback).max() - rets.rolling(lookback).min()

        if "volume" in features:
            # Not available when only returns are provided
            feat["volume"] = np.nan

        feat = feat.dropna()
        if feat.empty or len(feat) < 50:
            return {"available": False, "message": "Feature matrix is empty after preprocessing"}

        # Standardize features (robust to zero variance)
        mu = feat.mean()
        sigma = feat.std(ddof=0).replace(0, np.nan)
        X = ((feat - mu) / sigma).dropna()
        if X.empty:
            return {"available": False, "message": "Feature matrix became empty after standardization"}

        # Preferred: HMM (hmmlearn)
        hmm_cls = dep_manager.dependencies.get("hmmlearn", {}).get("GaussianHMM")
        if hmm_cls is None:
            try:
                from hmmlearn.hmm import GaussianHMM as hmm_cls  # type: ignore
            except Exception:
                hmm_cls = None

        if hmm_cls is None:
            # Fallback: KMeans clustering (if sklearn exists)
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
                states = km.fit_predict(X.values)
                model = km
                probs = None
            except Exception:
                return {"available": False, "message": "Neither hmmlearn nor sklearn KMeans is available"}
        else:
            try:
                model = hmm_cls(
                    n_components=int(n_regimes),
                    covariance_type="full",
                    n_iter=int(n_iter),
                    random_state=int(random_state)
                )
                model.fit(X.values)
                states = model.predict(X.values)
                try:
                    probs = model.predict_proba(X.values)
                except Exception:
                    probs = None
            except Exception as e:
                return {"available": False, "message": f"Regime model fitting failed: {str(e)[:160]}"}

        states_series = pd.Series(states, index=X.index, name="regime_raw")

        # Relabel regimes by mean return (low -> high) for stable interpretation
        aligned_rets = rets.loc[states_series.index]
        state_means = {}
        for s in sorted(states_series.unique()):
            subset = aligned_rets[states_series == s]
            state_means[int(s)] = float(subset.mean()) if len(subset) else -1e9

        ordered = sorted(state_means.keys(), key=lambda k: state_means[k])
        mapping = {old: new for new, old in enumerate(ordered)}
        regimes = states_series.map(mapping).astype(int)
        regimes.name = "regime"

        # Regime stats (values in % where UI expects)
        stats_records: List[Dict[str, Any]] = []
        for r in sorted(regimes.unique()):
            subset = aligned_rets[regimes == r]
            if subset.empty:
                continue

            mean_ann_pct = float(subset.mean() * self.annual_trading_days * 100)
            vol_ann_pct = float(subset.std(ddof=0) * np.sqrt(self.annual_trading_days) * 100)
            sharpe = (mean_ann_pct / 100 - self.risk_free_rate) / (vol_ann_pct / 100) if vol_ann_pct > 0 else np.nan
            freq_pct = float(len(subset) / len(aligned_rets) * 100)

            # Historical daily 95% VaR (5th percentile) in %
            var_95_pct = float(np.percentile(subset.values, 5) * 100)

            stats_records.append({
                "regime": int(r),
                "frequency": freq_pct,
                "mean_return": mean_ann_pct,
                "volatility": vol_ann_pct,
                "sharpe": float(sharpe) if sharpe == sharpe else np.nan,  # NaN-safe
                "var_95": var_95_pct,
                "n_obs": int(len(subset))
            })

        result = {
            "available": True,
            "regimes": regimes,
            "regime_stats": stats_records,
            "model": model
        }

        if probs is not None:
            try:
                prob_df = pd.DataFrame(probs, index=X.index)
                prob_df.columns = [f"state_{i}" for i in range(prob_df.shape[1])]
                result["probabilities"] = prob_df
            except Exception:
                pass

        # Also return features used (non-standardized) for transparency
        result["features_df"] = feat.loc[X.index]

        return result

# =============================================================================
# ENHANCED VISUALIZATION ENGINE
# =============================================================================

class InstitutionalVisualizer:
    """Professional visualization engine with advanced techniques"""
    
    def __init__(self, theme: str = "default"):
        self.theme = theme
        self.colors = ThemeManager.THEMES.get(theme, ThemeManager.THEMES["default"])
        
        # Advanced Plotly template
        self.template = go.layout.Template(
            layout=go.Layout(
                font_family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                title_font_size=24,
                title_font_color=self.colors['dark'],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.colors['dark'],
                    font_size=12,
                    font_family="Inter",
                    bordercolor='white'
                ),
                colorway=self._get_colorway(),
                xaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    zerolinecolor='rgba(0,0,0,0.1)',
                    zerolinewidth=1,
                    showgrid=True,
                    tickformat='%Y-%m-%d'
                ),
                yaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    zerolinecolor='rgba(0,0,0,0.1)',
                    zerolinewidth=1,
                    showgrid=True,
                    tickformat=',.2f'
                ),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font_size=12,
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                transition=dict(
                    duration=500,
                    easing='cubic-in-out'
                )
            )
        )
    
    def _get_colorway(self) -> List[str]:
        """Get advanced colorway for visualizations"""
        return [
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['accent'],
            self.colors['success'],
            self.colors['warning'],
            self.colors['danger'],
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
    
    def create_advanced_price_chart(
        self,
        df: pd.DataFrame,
        title: str,
        show_advanced_indicators: bool = True,
        show_volume_profile: bool = True,
        show_order_flow: bool = False
    ) -> go.Figure:
        """Create advanced price chart with multiple indicators and analytics"""
        
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        
        # Determine subplot configuration
        n_rows = 3  # Base rows: Price, Volume, RSI
        
        if show_advanced_indicators:
            n_rows += 2  # Add MACD and ATR
        if show_volume_profile:
            n_rows += 1
        
        row_heights = [0.4] + [0.15] * (n_rows - 1)  # Price chart gets most space
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(
                f"{title} - Advanced Analysis",
                "Volume",
                "RSI",
                *(["MACD", "ATR", "Volume Profile"] if show_advanced_indicators else [])
            )[:n_rows]
        )
        
        # 1. PRICE CHART WITH ADVANCED FEATURES
        # Candlestick or line chart
        if len(df) < 500:  # Use candlesticks for smaller datasets
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df[price_col],
                    name='OHLC',
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger'],
                    showlegend=False
                ),
                row=1, col=1
            )
        else:
            # Line chart for large datasets (better performance)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[price_col],
                    name='Price',
                    line=dict(color=self.colors['primary'], width=1.5),
                    fill='tozeroy',
                    fillcolor=f"rgba({int(self.colors['primary'][1:3], 16)}, "
                             f"{int(self.colors['primary'][3:5], 16)}, "
                             f"{int(self.colors['primary'][5:7], 16)}, 0.1)"
                ),
                row=1, col=1
            )
        
        # Add moving averages
        for period, color, width in [(20, self.colors['secondary'], 1.5), 
                                    (50, self.colors['accent'], 1.5),
                                    (200, self.colors['gray'], 2)]:
            if f'EMA_{period}' in df.columns or f'SMA_{period}' in df.columns:
                ma_col = f'EMA_{period}' if f'EMA_{period}' in df.columns else f'SMA_{period}'
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma_col],
                        name=f'MA {period}',
                        line=dict(color=color, width=width),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper_2', 'BB_Lower_2']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper_2'],
                    name='BB Upper',
                    line=dict(color=self.colors['gray'], width=1, dash='dot'),
                    opacity=0.6,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower_2'],
                    name='BB Lower',
                    line=dict(color=self.colors['gray'], width=1, dash='dot'),
                    opacity=0.6,
                    showlegend=False,
                    fill='tonexty',
                    fillcolor=f"rgba({int(self.colors['gray'][1:3], 16)}, "
                             f"{int(self.colors['gray'][3:5], 16)}, "
                             f"{int(self.colors['gray'][5:7], 16)}, 0.1)"
                ),
                row=1, col=1
            )
        
        # 2. VOLUME
        if 'Volume' in df.columns:
            colors = [self.colors['success'] if close >= open_ else self.colors['danger']
                     for close, open_ in zip(df[price_col], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    marker_line_width=0
                ),
                row=2, col=1
            )
            
            # Add volume moving average
            if 'Volume_SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Volume_SMA_20'],
                        name='Volume MA 20',
                        line=dict(color=self.colors['accent'], width=1.5),
                        opacity=0.8
                    ),
                    row=2, col=1
                )
        
        # 3. RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
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
        
        # 4. ADVANCED INDICATORS
        current_row = 4
        
        # MACD
        if show_advanced_indicators and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                row=current_row, col=1
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
                    opacity=0.6,
                    marker_line_width=0
                ),
                row=current_row, col=1
            )
            
            current_row += 1
        
        # ATR
        if show_advanced_indicators and 'ATR' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ATR'],
                    name='ATR',
                    line=dict(color=self.colors['warning'], width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba({int(self.colors['warning'][1:3], 16)}, "
                             f"{int(self.colors['warning'][3:5], 16)}, "
                             f"{int(self.colors['warning'][5:7], 16)}, 0.2)"
                ),
                row=current_row, col=1
            )
            
            current_row += 1
        
        # Volume Profile
        if show_volume_profile and 'Volume' in df.columns:
            # Simplified volume profile (price vs volume histogram)
            price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
            volume_profile = []
            
            for i in range(len(price_bins)-1):
                mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i+1])
                volume_sum = df.loc[mask, 'Volume'].sum() if mask.any() else 0
                volume_profile.append(volume_sum)
            
            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=(price_bins[:-1] + price_bins[1:]) / 2,
                    name='Volume Profile',
                    marker_color=self.colors['gray'],
                    opacity=0.7,
                    orientation='h'
                ),
                row=current_row, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=24, color=self.colors['dark'])
            ),
            height=150 + n_rows * 150,
            template=self.template,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        if show_advanced_indicators:
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            fig.update_yaxes(title_text="ATR", row=5, col=1)
        
        if show_volume_profile:
            fig.update_xaxes(title_text="Volume", row=current_row, col=1)
            fig.update_yaxes(title_text="Price", row=current_row, col=1)
        
        return fig
    
    def create_interactive_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Advanced Correlation Analysis",
        show_clusters: bool = True,
        show_network: bool = False
    ) -> go.Figure:
        """Create interactive correlation matrix with clustering"""
        
        # Apply clustering if requested
        if show_clusters and len(corr_matrix) > 2:
            try:
                from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
                
                # Perform hierarchical clustering
                dist_matrix = 1 - np.abs(corr_matrix.values)  # Convert to distance
                np.fill_diagonal(dist_matrix, 0)
                
                linkage_matrix = linkage(dist_matrix, method='average')
                leaves = leaves_list(linkage_matrix)
                
                # Reorder matrix
                corr_matrix = corr_matrix.iloc[leaves, leaves]
                
                # Create dendrogram
                dendro_fig = ff.create_dendrogram(
                    dist_matrix,
                    orientation='left',
                    labels=corr_matrix.columns.tolist(),
                    color_threshold=0.7
                )
                
            except ImportError:
                show_clusters = False
        
        # Create heatmap
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
            textfont=dict(size=10),
            hoverinfo='x+y+z',
            colorbar=dict(
                title='Correlation',
                                tickformat='.2f',
                len=0.75
            ),
            xgap=1,
            ygap=1
        ))
        
        # Add annotations for significant correlations
        significant_mask = np.abs(corr_matrix.values) > 0.7
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i != j and significant_mask[i, j]:
                    fig.add_annotation(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.index[i],
                        text=f"{corr_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(size=9, color='white' if np.abs(corr_matrix.iloc[i, j]) > 0.8 else 'black'),
                        bgcolor='rgba(0,0,0,0.3)' if np.abs(corr_matrix.iloc[i, j]) > 0.8 else 'rgba(255,255,255,0.5)'
                    )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20, color=self.colors['dark'])
            ),
            height=max(600, len(corr_matrix) * 40),
            width=max(800, len(corr_matrix) * 40),
            template=self.template,
            xaxis_tickangle=45,
            xaxis=dict(
                side="bottom",
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=10)
            )
        )
        
        return fig


    # -------------------------------------------------------------------------
    # Portfolio / Asset Performance Chart (vs Benchmark) - robust
    # -------------------------------------------------------------------------
    def create_performance_chart(
        self,
        asset_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Performance Analysis",
        rolling_window: int = 21
    ) -> go.Figure:
        """Create a robust performance chart (cumulative + drawdown)."""
        ar = pd.Series(asset_returns).dropna()
        if ar.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (no data)", template=self.template, height=450)
            return fig

        br = None
        if benchmark_returns is not None:
            br = pd.Series(benchmark_returns).dropna()

        # Align
        if br is not None and not br.empty:
            df = pd.concat([ar.rename("asset"), br.rename("bench")], axis=1).dropna()
            ar = df["asset"]
            br = df["bench"]

        # Cumulative returns
        cum_a = (1 + ar).cumprod() - 1
        dd_a = (cum_a + 1) / (cum_a + 1).cummax() - 1

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("Cumulative Return", "Drawdown")
        )

        fig.add_trace(
            go.Scatter(
                x=cum_a.index, y=cum_a.values,
                name="Asset",
                mode="lines",
                line=dict(width=2.5, color=self.colors["primary"])
            ),
            row=1, col=1
        )

        if br is not None and not br.empty:
            cum_b = (1 + br).cumprod() - 1
            dd_b = (cum_b + 1) / (cum_b + 1).cummax() - 1

            fig.add_trace(
                go.Scatter(
                    x=cum_b.index, y=cum_b.values,
                    name="Benchmark",
                    mode="lines",
                    line=dict(width=2.0, color=self.colors["secondary"], dash="dash")
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=dd_b.index, y=dd_b.values,
                    name="Benchmark DD",
                    mode="lines",
                    line=dict(width=1.6, color=self.colors["gray"], dash="dot"),
                    showlegend=False
                ),
                row=2, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=dd_a.index, y=dd_a.values,
                name="Drawdown",
                mode="lines",
                line=dict(width=2.0, color=self.colors["danger"]),
                showlegend=False
            ),
            row=2, col=1
        )

        # Rolling Sharpe (optional hover)
        try:
            rmean = ar.rolling(rolling_window).mean() * 252
            rstd = ar.rolling(rolling_window).std() * np.sqrt(252)
            rsh = (rmean - 0.0) / rstd
            fig.add_trace(
                go.Scatter(
                    x=rsh.index, y=rsh.values,
                    name=f"Rolling Sharpe ({rolling_window}d)",
                    mode="lines",
                    line=dict(width=1.5, color=self.colors["accent"]),
                    opacity=0.35
                ),
                row=1, col=1
            )
        except Exception:
            pass

        fig.update_layout(
            title=title,
            height=650,
            template=self.template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified"
        )
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        return fig

    # -------------------------------------------------------------------------
    # GARCH Volatility Chart - robust (in-sample + forecast + realized)
    # -------------------------------------------------------------------------
    def create_garch_volatility(
        self,
        returns: pd.Series,
        conditional_volatility: pd.Series,
        volatility_forecast: Optional[Any] = None,
        title: str = "GARCH Volatility"
    ) -> go.Figure:
        rets = pd.Series(returns).dropna()
        cv = pd.Series(conditional_volatility).dropna()

        if rets.empty or cv.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (no data)", template=self.template, height=450)
            return fig

        # Align by index where possible
        df = pd.concat([rets.rename("returns"), cv.rename("cond_vol")], axis=1).dropna()
        rets = df["returns"]
        cv = df["cond_vol"]

        # If cv looks like percent (typical arch output), annualize accordingly
        cv_vals = cv.values.astype(float)
        if np.nanmedian(np.abs(cv_vals)) > 1.0:
            vol_daily = cv_vals / 100.0
        else:
            vol_daily = cv_vals

        vol_ann = vol_daily * np.sqrt(252)

        # Realized volatility (21d)
        rv = rets.rolling(21).std() * np.sqrt(252)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cv.index, y=vol_ann,
                name="GARCH Vol (ann.)",
                mode="lines",
                line=dict(width=2.5, color=self.colors["primary"])
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rv.index, y=rv.values,
                name="Realized Vol (21d, ann.)",
                mode="lines",
                line=dict(width=1.8, color=self.colors["secondary"], dash="dash"),
                opacity=0.9
            )
        )

        # Forecast (if provided)
        try:
            if volatility_forecast is not None:
                if isinstance(volatility_forecast, (list, tuple, np.ndarray, pd.Series)):
                    f = np.array(volatility_forecast, dtype=float).flatten()
                elif isinstance(volatility_forecast, dict) and "forecast" in volatility_forecast:
                    f = np.array(volatility_forecast["forecast"], dtype=float).flatten()
                else:
                    f = None

                if f is not None and len(f) > 0:
                    # Assume forecast is daily vol (same scale as vol_daily)
                    if np.nanmedian(np.abs(f)) > 1.0:
                        f_daily = f / 100.0
                    else:
                        f_daily = f
                    f_ann = f_daily * np.sqrt(252)

                    last_dt = cv.index.max()
                    # create forward dates (business days)
                    f_idx = pd.bdate_range(last_dt, periods=len(f_ann) + 1, closed="right")
                    fig.add_trace(
                        go.Scatter(
                            x=f_idx, y=f_ann,
                            name="Forecast (ann.)",
                            mode="lines",
                            line=dict(width=2.2, color=self.colors["accent"], dash="dot")
                        )
                    )
        except Exception:
            pass

        fig.update_layout(
            title=title,
            height=520,
            template=self.template,
            yaxis_title="Annualized Volatility",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig.update_yaxes(tickformat=".1%")
        return fig

    # -------------------------------------------------------------------------
    # Regime Chart (price + regime shading) - robust
    # -------------------------------------------------------------------------
    def create_regime_chart(
        self,
        price: pd.Series,
        regimes: pd.Series,
        regime_labels: Optional[Dict[int, Dict[str, Any]]] = None,
        title: str = "Market Regimes"
    ) -> go.Figure:
        p = pd.Series(price).dropna()
        r = pd.Series(regimes).dropna()

        if p.empty or r.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (no data)", template=self.template, height=450)
            return fig

        # Align by intersection
        df = pd.concat([p.rename("price"), r.rename("regime")], axis=1).dropna()
        p = df["price"]
        r = df["regime"].astype(int)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=p.index, y=p.values,
                name="Price",
                mode="lines",
                line=dict(width=2.2, color=self.colors["primary"])
            )
        )

        # Shading by consecutive regime segments
        if regime_labels is None:
            regime_labels = {}

        # Choose colors
        default_palette = [
            self.colors["danger"],
            self.colors["warning"],
            self.colors["success"],
            self.colors["secondary"],
            self.colors["accent"]
        ]

        def _color_for(regime_id: int):
            lab = regime_labels.get(int(regime_id), {})
            return lab.get("color", default_palette[int(regime_id) % len(default_palette)])

        # Build segments
        segments = []
        curr = int(r.iloc[0])
        seg_start = r.index[0]
        for dt, reg in r.iloc[1:].items():
            reg = int(reg)
            if reg != curr:
                segments.append((seg_start, dt, curr))
                seg_start = dt
                curr = reg
        segments.append((seg_start, r.index[-1], curr))

        for (a, b, reg) in segments:
            fig.add_vrect(
                x0=a, x1=b,
                fillcolor=_color_for(reg),
                opacity=0.10,
                line_width=0,
                layer="below"
            )

        # Annotate legend-like markers
        uniq = sorted(r.unique())
        for u in uniq:
            name = regime_labels.get(int(u), {}).get("name", f"Regime {int(u)}")
            fig.add_trace(
                go.Scatter(
                    x=[p.index[0]], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=_color_for(u)),
                    name=name
                )
            )

        fig.update_layout(
            title=title,
            height=520,
            template=self.template,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        return fig


# =============================================================================
# ENHANCED DASHBOARD METHODS
# =============================================================================

# Fix method signatures by removing config parameter where not needed
# Update the dashboard methods to match the correct signatures

class InstitutionalCommoditiesDashboard:
    """Main dashboard class with enhanced architecture"""
    
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
            'advanced_metrics': {},
            
            # Configuration
            'analysis_config': AnalysisConfiguration(
                start_date=datetime.now() - timedelta(days=1095),
                end_date=datetime.now()
            ),
            
            # UI state
            'current_tab': 'dashboard',
            'last_update': datetime.now(),
            'error_log': [],
            'last_selection_hash': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _log_error(self, error: Exception, context: str = ""):
        """Enhanced error logging"""
        error_entry = {
            'timestamp': datetime.now(),
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc()[:500],  # Limit traceback length
            'session_state': {
                'assets': st.session_state.get('selected_assets', []),
                'data_loaded': st.session_state.get('data_loaded', False)
            }
        }
        st.session_state.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(st.session_state.error_log) > 100:
            st.session_state.error_log = st.session_state.error_log[-100:]
    
    def display_header(self):
        """Display enhanced institutional header"""
        st.components.v1.html(f"""
        <div style="
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            padding: 2rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            position: relative;
            overflow: hidden;
        ">
            <div style="position: absolute; top: 0; right: 0; width: 300px; height: 300px; 
                        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                        background-size: 30px 30px; opacity: 0.3; transform: rotate(30deg);"></div>
            
            <div style="position: relative; z-index: 1;">
                <div style="font-size: 2.8rem; font-weight: 850; line-height: 1.1; margin-bottom: 0.5rem;">
                    🏛️ Institutional Commodities Analytics v6.0 Enhanced
                </div>
                <div style="font-size: 1.1rem; opacity: 0.9; font-weight: 400;">
                    Advanced Analytics • Machine Learning • Institutional Grade
                </div>
                <div style="display: flex; gap: 1rem; margin-top: 1rem; font-size: 0.9rem;">
                    <span>📈 Real-time Analytics</span>
                    <span>⚡ High Performance</span>
                    <span>🔒 Enterprise Security</span>
                </div>
            </div>
        </div>
        """, height=180)
    
    def _render_sidebar_controls(self):
        """Enhanced sidebar controls with advanced options"""
        with st.sidebar:
            st.markdown("## ⚙️ Advanced Controls")
            
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["default", "dark"],
                key="theme_select",
                help="Select application theme"
            )
            
            # Apply theme
            if theme:
                st.markdown(ThemeManager.get_styles(theme), unsafe_allow_html=True)
            
            # Universe / Asset selection
            categories = list(COMMODITIES_UNIVERSE.keys())
            selected_categories = st.multiselect(
                "Commodity Groups",
                options=categories,
                default=[AssetCategory.PRECIOUS_METALS.value, AssetCategory.ENERGY.value],
                key="sidebar_groups",
                help="Select commodity groups to populate asset list"
            )
            
            # Build asset options
            ticker_to_label = {}
            for cat in selected_categories:
                for t, meta in COMMODITIES_UNIVERSE.get(cat, {}).items():
                    ticker_to_label[t] = f"{t} — {meta.name} ({meta.risk_level} Risk)"
            
            asset_options = list(ticker_to_label.keys())
            
            # Smart default selection
            default_assets = ["GC=F", "SI=F", "CL=F", "HG=F", "ZC=F"]
            default_assets = [t for t in default_assets if t in asset_options]
            
            selected_assets = st.multiselect(
                "Assets",
                options=asset_options,
                default=default_assets,
                format_func=lambda x: ticker_to_label.get(x, x),
                key="sidebar_assets",
                help="Select assets for analysis"
            )
            
            # Benchmark selection
            bench_options = list(BENCHMARKS.keys())
            bench_to_label = {k: f"{k} — {v.name}" for k, v in BENCHMARKS.items()}
            
            selected_benchmarks = st.multiselect(
                "Benchmarks",
                options=bench_options,
                default=["^GSPC", "GLD"],
                format_func=lambda x: bench_to_label.get(x, x),
                key="sidebar_benchmarks",
                help="Select benchmarks for comparison"
            )
            
            st.markdown("---")
            
            # Date range with presets
            st.markdown("### 📅 Date Range")
            
            # Quick presets
            preset = st.selectbox(
                "Quick Presets",
                ["Custom", "1 Year", "2 Years", "5 Years", "10 Years", "Max"],
                key="date_preset"
            )
            
            today = datetime.now().date()
            
            if preset == "1 Year":
                default_start = today - timedelta(days=365)
            elif preset == "2 Years":
                default_start = today - timedelta(days=365*2)
            elif preset == "5 Years":
                default_start = today - timedelta(days=365*5)
            elif preset == "10 Years":
                default_start = today - timedelta(days=365*10)
            elif preset == "Max":
                default_start = today - timedelta(days=365*20)
            else:
                # Custom - use previous or default
                prev_cfg = st.session_state.get("analysis_config")
                default_start = prev_cfg.start_date.date() if prev_cfg else today - timedelta(days=365*2)
            
            c1, c2 = st.columns(2)
            start_date = c1.date_input(
                "Start Date",
                value=default_start,
                key="sidebar_start_date"
            )
            end_date = c2.date_input(
                "End Date",
                value=today,
                key="sidebar_end_date"
            )
            
            # Advanced options
            with st.expander("⚡ Advanced Options"):
                # Data interval
                interval = st.selectbox(
                    "Data Interval",
                    ["1d", "1wk", "1mo"],
                    key="data_interval",
                    help="Select data frequency"
                )
                
                # Risk-free rate
                risk_free_rate = st.slider(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    key="risk_free_rate",
                    help="Annual risk-free rate for calculations"
                ) / 100
                
                # Update configuration
                if 'analysis_config' in st.session_state:
                    st.session_state.analysis_config.risk_free_rate = risk_free_rate
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                load_clicked = st.button(
                    "🚀 Load Data",
                    use_container_width=True,
                    type="primary",
                    key="sidebar_load_btn"
                )
            
            with col2:
                clear_cache = st.button(
                    "🧹 Clear Cache",
                    use_container_width=True,
                    key="sidebar_clear_cache_btn"
                )
            
            if clear_cache:
                try:
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache cleared successfully!")
                except Exception as e:
                    st.warning(f"Cache clear attempted: {str(e)[:100]}")
            
            # Auto-reload option
            auto_reload = st.checkbox(
                "Auto-reload on changes",
                value=True,
                key="sidebar_autoreload",
                help="Automatically reload data when selections change"
            )
            
            return {
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "start_date": start_date,
                "end_date": end_date,
                "auto_reload": auto_reload,
                "load_clicked": load_clicked,
                "interval": interval,
                "risk_free_rate": risk_free_rate
            }
    
    def _load_sidebar_selection(self, sidebar_state: dict):
        """Enhanced data loading with progress tracking"""
        selected_assets = sidebar_state.get("selected_assets", [])
        selected_benchmarks = sidebar_state.get("selected_benchmarks", [])
        start_date = sidebar_state.get("start_date")
        end_date = sidebar_state.get("end_date")
        interval = sidebar_state.get("interval", "1d")
        
        if not selected_assets:
            st.warning("Please select at least one asset from the sidebar.")
            st.session_state.data_loaded = False
            return
        
        # Validate dates
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        
        if end_dt <= start_dt:
            st.error("❌ End date must be after start date.")
            st.session_state.data_loaded = False
            return
        
        # Check date range
        days_diff = (end_dt - start_dt).days
        if days_diff < 30:
            st.warning("⚠️ Date range is less than 30 days. Some analytics may not work properly.")
        elif days_diff > 365 * 20:
            st.warning("⚠️ Date range exceeds 20 years. Consider using a smaller range for better performance.")
        
        # Create selection fingerprint
        selection_fingerprint = json.dumps(
            {
                "assets": sorted(selected_assets),
                "benchmarks": sorted(selected_benchmarks),
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "interval": interval
            },
            sort_keys=True,
        )
        selection_hash = hashlib.sha256(selection_fingerprint.encode()).hexdigest()
        
        # Skip if same selection and data already loaded
        if (st.session_state.get("last_selection_hash") == selection_hash and 
            st.session_state.get("data_loaded", False)):
            st.sidebar.info("✅ Using cached data")
            return
        
        # Update session state
        st.session_state.last_selection_hash = selection_hash
        st.session_state.selected_assets = selected_assets
        st.session_state.selected_benchmarks = selected_benchmarks
        
        # Update configuration
        cfg = st.session_state.get("analysis_config", AnalysisConfiguration(
            start_date=start_dt,
            end_date=end_dt
        ))
        cfg.start_date = start_dt
        cfg.end_date = end_dt
        cfg.risk_free_rate = sidebar_state.get("risk_free_rate", 0.02)
        st.session_state.analysis_config = cfg
        
        # Load data with progress tracking
        with st.spinner("📊 Loading market data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Fetch asset data
                status_text.text("Fetching asset data...")
                asset_data = {}
                missing_assets = []
                
                for i, symbol in enumerate(selected_assets):
                    progress_bar.progress(i / (len(selected_assets) + len(selected_benchmarks)))
                    
                    df = self.data_manager.fetch_asset_data(
                        symbol, start_dt, end_dt, interval
                    )
                    
                    if df is None or df.empty:
                        missing_assets.append(symbol)
                        continue
                    
                    # Calculate technical features
                    df_feat = self.data_manager.calculate_advanced_technical_features(df)
                    asset_data[symbol] = df_feat
                
                # Step 2: Fetch benchmark data
                status_text.text("Fetching benchmark data...")
                benchmark_data = {}
                missing_benchmarks = []
                
                for i, symbol in enumerate(selected_benchmarks, start=len(selected_assets)):
                    progress_bar.progress(i / (len(selected_assets) + len(selected_benchmarks)))
                    
                    df = self.data_manager.fetch_asset_data(
                        symbol, start_dt, end_dt, interval
                    )
                    
                    if df is None or df.empty:
                        missing_benchmarks.append(symbol)
                        continue
                    
                    df_feat = self.data_manager.calculate_advanced_technical_features(df)
                    benchmark_data[symbol] = df_feat
                
                progress_bar.progress(1.0)
                status_text.text("Processing data...")
                
                # Check if we have any data
                if not asset_data:
                    st.session_state.data_loaded = False
                    st.error("❌ No valid market data could be loaded. Please try different assets or date range.")
                    
                    if missing_assets:
                        st.info(f"Missing assets: {', '.join(missing_assets)}")
                    
                    return
                
                # Build returns matrices
                returns_data = {}
                for symbol, df in asset_data.items():
                    if 'Returns' in df.columns:
                        returns_data[symbol] = df['Returns']
                
                returns_df = pd.DataFrame(returns_data).dropna(how='all')
                
                # Benchmark returns
                benchmark_returns_data = {}
                for symbol, df in benchmark_data.items():
                    if 'Returns' in df.columns:
                        benchmark_returns_data[symbol] = df['Returns']
                
                benchmark_returns_df = pd.DataFrame(benchmark_returns_data).dropna(how='all')
                
                # Update session state
                st.session_state.asset_data = asset_data
                st.session_state.benchmark_data = benchmark_data
                st.session_state.returns_data = returns_df
                st.session_state.benchmark_returns_data = benchmark_returns_df
                st.session_state.data_loaded = True
                
                # Show summary
                status_text.text("✅ Data loaded successfully!")
                
                # Show missing data warnings
                if missing_assets:
                    st.sidebar.warning(f"⚠️ {len(missing_assets)} assets missing data")
                
                if missing_benchmarks:
                    st.sidebar.warning(f"⚠️ {len(missing_benchmarks)} benchmarks missing data")
                
                # Show data summary
                with st.sidebar.expander("📊 Data Summary", expanded=False):
                    st.write(f"**Assets loaded:** {len(asset_data)}")
                    st.write(f"**Date range:** {start_date} to {end_date}")
                    st.write(f"**Data points:** {len(returns_df)}")
                    st.write(f"**Interval:** {interval}")
                
            except Exception as e:
                self._log_error(e, context="data_load")
                st.session_state.data_loaded = False
                st.error(f"❌ Data load failed: {str(e)[:200]}")
                
                # Provide troubleshooting tips
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    1. Check your internet connection
                    2. Try a smaller date range
                    3. Select fewer assets
                    4. Check if symbols are valid
                    5. Wait a few moments and retry
                    """)
    
    def run(self):
        """Enhanced main application runner"""
        try:
            # Apply global styles
            st.markdown(ThemeManager.get_styles("default"), unsafe_allow_html=True)
            
            # Display header
            self.display_header()
            
            # Render sidebar and get state
            sidebar_state = self._render_sidebar_controls()
            
            # Handle auto-reload or manual load
            if sidebar_state.get("auto_reload", False) or sidebar_state.get("load_clicked", False):
                self._load_sidebar_selection(sidebar_state)
            
            # Check if data is loaded
            if not st.session_state.get("data_loaded", False):
                self._display_welcome()
                return
            
            # Create enhanced tabs
            tab_labels = [
                "📊 Dashboard",
                "⚡ Advanced Analytics",
                "⚖️ Risk Analytics",
                "🧺 Portfolio",
                "🎲 Monte Carlo",
                "📈 GARCH Analysis",
                "🔄 Regime Detection",
                "📋 Reporting",
                "⚙️ Settings"
            ]
            
            tabs = st.tabs(tab_labels)
            
            # Dashboard
            with tabs[0]:
                self._display_dashboard()
            
            # Advanced Analytics
            with tabs[1]:
                self._display_advanced_analytics()
            
            # Risk Analytics
            with tabs[2]:
                self._display_risk_analytics()
            
            # Portfolio
            with tabs[3]:
                self._display_portfolio()
            
            # Monte Carlo
            with tabs[4]:
                self._display_monte_carlo()
            
            # GARCH Analysis
            with tabs[5]:
                self._display_garch_analysis()
            
            # Regime Detection
            with tabs[6]:
                self._display_regime_analysis()
            
            # Reporting
            with tabs[7]:
                self._display_reporting()
            
            # Settings
            with tabs[8]:
                self._display_settings()
            
        except Exception as e:
            self._log_error(e, context="run")
            st.error(f"🚨 Application Error: {str(e)[:500]}")
            
            # Show detailed error for debugging
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc(), language="python")
            
            # Recovery options
            st.markdown("### 🔧 Recovery Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Restart Application", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            with col2:
                if st.button("📋 View Error Log", use_container_width=True):
                    if st.session_state.error_log:
                        st.write("### Recent Errors")
                        for error in st.session_state.error_log[-5:]:
                            st.json(error)
    
    def _display_welcome(self):
        """Enhanced welcome screen"""
        st.markdown("## 🏛️ Welcome to Institutional Commodities Analytics")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Select Assets** - Choose from the sidebar
            2. **Set Date Range** - Use presets or custom range
            3. **Click Load Data** - Fetch market data
            4. **Explore Analytics** - Use the tabs above
            
            ### 📈 Available Features
            
            - **Real-time Market Data** - Live commodity prices
            - **Advanced Analytics** - GARCH, Regime Detection, ML
            - **Portfolio Optimization** - Mean-variance, risk parity
            - **Risk Management** - VaR, CVaR, Stress Testing
            - **Professional Reporting** - HTML, PDF, JSON exports
            
            ### ⚡ Quick Tips
            
            - Start with 2-5 assets for optimal performance
            - Use 1-3 year range for most analyses
            - Enable auto-reload for real-time updates
            - Clear cache if experiencing issues
            """)
        
        with col2:
            st.markdown("### 🔥 Quick Start")
            
            if st.button("🚀 Load Sample Portfolio", use_container_width=True, type="primary"):
                # Set sample configuration
                st.session_state.selected_assets = ["GC=F", "CL=F", "HG=F", "ZC=F"]
                st.session_state.selected_benchmarks = ["^GSPC", "GLD"]
                st.session_state.analysis_config.start_date = datetime.now() - timedelta(days=365*2)
                st.session_state.analysis_config.end_date = datetime.now()
                st.rerun()
            
            st.markdown("---")
            
            st.markdown("### 📊 Platform Stats")
            st.metric("Available Assets", sum(len(cat) for cat in COMMODITIES_UNIVERSE.values()))
            st.metric("Analytics Methods", "15+")
            st.metric("Max Data Points", "100,000+")
            
            st.markdown("---")
            
            st.markdown("### 🆘 Need Help?")
            st.markdown("""
            - Check the [Documentation](#)
            - View [Examples](#)
            - Contact [Support](#)
            """)
    
    def _display_dashboard(self):
        """Enhanced dashboard display"""
        st.markdown('<div class="section-header"><h2>📊 Institutional Dashboard</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        # Quick metrics dashboard
        self._display_quick_metrics(config)
        
        # Asset performance table
        self._display_asset_performance()
        
        # Individual asset analysis
        self._display_individual_analysis()
        
        # Correlation analysis
        self._display_correlation_analysis()
        
        # Market overview
        self._display_market_overview()
    
    def _display_quick_metrics(self, config: AnalysisConfiguration):
        """Display quick metrics dashboard"""
        st.markdown("### 📈 Market Overview")
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("No return data available")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        # Calculate metrics
        with st.spinner("Calculating market metrics..."):
            metrics = self._calculate_market_metrics(returns_df)
        
        # Display metrics in a grid
        cols = st.columns(4)
        
        metric_configs = [
            ("Market Return", f"{metrics['avg_return']:.2f}%", 
             "📈", "positive" if metrics['avg_return'] > 0 else "negative",
             "Average annual return across all assets"),
            
            ("Market Volatility", f"{metrics['avg_volatility']:.2f}%",
             "📉", "neutral", "Average annual volatility"),
            
            ("Avg Sharpe", f"{metrics['avg_sharpe']:.2f}",
             "⚡", "positive" if metrics['avg_sharpe'] > 0 else "negative",
             "Average Sharpe ratio"),
            
            ("Correlation", f"{metrics['avg_correlation']:.3f}",
             "🔗", "neutral" if abs(metrics['avg_correlation']) < 0.3 else 
                   "positive" if metrics['avg_correlation'] > 0 else "negative",
             "Average correlation between assets")
        ]
        
        for (label, value, icon, color_class, tooltip), col in zip(metric_configs, cols):
            with col:
                st.markdown(f"""
                <div class="metric-card custom-tooltip" data-tooltip="{tooltip}">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value {color_class}">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional metrics
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        additional_metrics = [
            ("Best Performer", f"{metrics['best_asset']} ({metrics['best_return']:.2f}%)",
             "🥇", "success"),
            
            ("Worst Performer", f"{metrics['worst_asset']} ({metrics['worst_return']:.2f}%)",
             "📉", "danger"),
            
            ("Highest Volatility", f"{metrics['highest_vol_asset']} ({metrics['highest_vol']:.2f}%)",
             "⚡", "warning"),
            
            ("Trading Days", f"{metrics['trading_days']:,}",
             "📅", "info")
        ]
        
        cols2 = st.columns(4)
        for (label, value, icon, color_class), col in zip(additional_metrics, cols2):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _calculate_market_metrics(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive market metrics"""
        if returns_df.empty:
            return {}
        
        # Calculate individual asset metrics
        asset_metrics = {}
        for asset in returns_df.columns:
            returns = returns_df[asset].dropna()
            if len(returns) > 0:
                metrics = self.analytics.calculate_advanced_performance_metrics(returns)
                asset_metrics[asset] = metrics
        
        # Aggregate metrics
        avg_return = np.mean([m.get('annual_return', 0) for m in asset_metrics.values()])
        avg_volatility = np.mean([m.get('annual_volatility', 0) for m in asset_metrics.values()])
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in asset_metrics.values()])
        
        # Find extremes
        returns_by_asset = {a: m.get('annual_return', 0) for a, m in asset_metrics.items()}
        volatilities_by_asset = {a: m.get('annual_volatility', 0) for a, m in asset_metrics.items()}
        
        best_asset = max(returns_by_asset.items(), key=lambda x: x[1])[0] if returns_by_asset else "N/A"
        worst_asset = min(returns_by_asset.items(), key=lambda x: x[1])[0] if returns_by_asset else "N/A"
        highest_vol_asset = max(volatilities_by_asset.items(), key=lambda x: x[1])[0] if volatilities_by_asset else "N/A"
        
        # Correlation
        if len(returns_df.columns) > 1:
            corr_matrix = returns_df.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
        else:
            avg_correlation = 0
        
        return {
            'avg_return': avg_return,
            'avg_volatility': avg_volatility,
            'avg_sharpe': avg_sharpe,
            'avg_correlation': avg_correlation,
            'best_asset': best_asset,
            'worst_asset': worst_asset,
            'highest_vol_asset': highest_vol_asset,
            'best_return': returns_by_asset.get(best_asset, 0),
            'worst_return': returns_by_asset.get(worst_asset, 0),
            'highest_vol': volatilities_by_asset.get(highest_vol_asset, 0),
            'trading_days': len(returns_df),
            'n_assets': len(returns_df.columns)
        }
    
    def _display_asset_performance(self):
        """Display enhanced asset performance table"""
        st.markdown("### 📊 Asset Performance Overview")
        
        if not st.session_state.asset_data:
            st.warning("No asset data available")
            return
        
        # Prepare performance data
        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if 'Returns' in df.columns:
                returns = df['Returns'].dropna()
                if len(returns) > 0:
                    # Get metadata
                    metadata = self._get_asset_metadata(symbol)
                    
                    # Calculate metrics
                    metrics = self.analytics.calculate_advanced_performance_metrics(returns)
                    
                    performance_data.append({
                        'Asset': symbol,
                        'Name': metadata.name,
                        'Category': metadata.category.value,
                        'Risk': metadata.risk_level,
                        'Current Price': df['Adj_Close'].iloc[-1] if 'Adj_Close' in df.columns else df['Close'].iloc[-1],
                        '1D Return': df['Returns'].iloc[-1] * 100 if len(df) > 1 else 0,
                        '1M Return': ((df['Adj_Close'].iloc[-1] / df['Adj_Close'].iloc[-22] - 1) * 100) if len(df) > 22 else 0,
                        'Annual Return': metrics.get('annual_return', 0),
                        'Annual Vol': metrics.get('annual_volatility', 0),
                        'Sharpe': metrics.get('sharpe_ratio', 0),
                        'Max DD': metrics.get('max_drawdown', 0),
                        'Sortino': metrics.get('sortino_ratio', 0),
                        'Color': metadata.color
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Style the dataframe
            styled_df = perf_df.style.format({
                'Current Price': '{:.2f}',
                '1D Return': '{:.2f}%',
                '1M Return': '{:.2f}%',
                'Annual Return': '{:.2f}%',
                'Annual Vol': '{:.2f}%',
                'Sharpe': '{:.3f}',
                'Max DD': '{:.2f}%',
                'Sortino': '{:.3f}'
            }).background_gradient(
                subset=['Annual Return', 'Sharpe', 'Sortino'],
                cmap='RdYlGn',
                vmin=-2, vmax=2
            ).background_gradient(
                subset=['Annual Vol', 'Max DD'],
                cmap='RdYlGn_r'
            )
            
            # Display with column configuration
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(400, len(performance_data) * 35 + 100),
                column_config={
                    'Color': st.column_config.Column(disabled=True),
                    'Name': st.column_config.Column(width="medium"),
                    'Asset': st.column_config.Column(width="small"),
                    'Category': st.column_config.Column(width="small"),
                    'Risk': st.column_config.Column(width="small")
                }
            )
            
            # Export option
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📥 Export Performance Data", use_container_width=True):
                    csv = perf_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"asset_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    def _get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Get asset metadata from universe or benchmarks"""
        # Check commodities universe
        for category in COMMODITIES_UNIVERSE.values():
            if symbol in category:
                return category[symbol]
        
        # Check benchmarks
        if symbol in BENCHMARKS:
            return BENCHMARKS[symbol]
        
        # Default metadata
        return AssetMetadata(
            symbol=symbol,
            name=symbol,
            category=AssetCategory.BENCHMARK,
            color="#666666"
        )
    
    def _display_individual_analysis(self):
        """Display individual asset analysis"""
        st.markdown("### 📉 Individual Asset Analysis")
        
        if not st.session_state.selected_assets:
            return
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed Analysis",
            options=st.session_state.selected_assets,
            key="dashboard_asset_select",
            format_func=lambda x: f"{x} - {self._get_asset_metadata(x).name}"
        )
        
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            metadata = self._get_asset_metadata(selected_asset)
            
            # Display asset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${df['Adj_Close'].iloc[-1]:.2f}" 
                         if 'Adj_Close' in df.columns else f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Category", metadata.category.value)
            with col3:
                st.metric("Risk Level", metadata.risk_level)
            
            # Create tabs for different analyses
            chart_tabs = st.tabs([
                "📊 Price Analysis",
                "📈 Returns Analysis",
                "⚡ Technical Indicators",
                "📊 Statistics"
            ])
            
            with chart_tabs[0]:
                fig = self.visualizer.create_advanced_price_chart(
                    df,
                    f"{selected_asset} - Comprehensive Analysis",
                    show_advanced_indicators=True,
                    show_volume_profile=True
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            with chart_tabs[1]:
                returns = df['Returns'].dropna()
                
                if len(returns) > 0:
                    # Get benchmark returns for comparison
                    benchmark_returns = None
                    if st.session_state.benchmark_returns_data is not None and not st.session_state.benchmark_returns_data.empty:
                        # Use first benchmark
                        first_benchmark = st.session_state.benchmark_returns_data.columns[0]
                        benchmark_returns = st.session_state.benchmark_returns_data[first_benchmark]
                    
                    # Create performance chart
                    try:
                        fig = self.visualizer.create_performance_chart(
                            returns,
                            benchmark_returns,
                            f"{selected_asset} - Performance Analysis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create performance chart: {str(e)[:100]}")
            
            with chart_tabs[2]:
                # Technical indicators summary
                self._display_technical_indicators(df, selected_asset)
            
            with chart_tabs[3]:
                # Statistical summary
                self._display_statistical_summary(df, selected_asset)
    
    def _display_technical_indicators(self, df: pd.DataFrame, symbol: str):
        """Display technical indicators summary"""
        st.markdown("#### ⚡ Technical Indicators")
        
        # Group indicators by type
        indicator_groups = {
            'Trend': ['EMA_20', 'EMA_50', 'EMA_200', 'SMA_20', 'SMA_50', 'SMA_200'],
            'Momentum': ['RSI_14', 'Stoch_%K', 'Stoch_%D', 'MACD', 'MACD_Signal'],
            'Volatility': ['ATR', 'ATR_Pct', 'BB_Width', 'BB_%B'],
            'Volume': ['Volume_Ratio', 'OBV', 'MFI']
        }
        
        # Filter to available indicators
        available_groups = {}
        for group, indicators in indicator_groups.items():
            available = [ind for ind in indicators if ind in df.columns]
            if available:
                available_groups[group] = available
        
        # Display indicators in columns
        n_cols = min(4, len(available_groups))
        cols = st.columns(n_cols)
        
        for i, (group, indicators) in enumerate(available_groups.items()):
            with cols[i % n_cols]:
                st.markdown(f"**{group}**")
                for indicator in indicators[:3]:  # Show top 3
                    if indicator in df.columns:
                        value = df[indicator].iloc[-1]
                        
                        # Determine status and color
                        status, color = self._get_indicator_status(indicator, value)
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 0.9rem;">{indicator}</span>
                                <span style="font-weight: 600; color: {color};">{value:.2f}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: {color};">{status}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    def _get_indicator_status(self, indicator: str, value: float) -> Tuple[str, str]:
        """Get status and color for technical indicator"""
        if 'RSI' in indicator:
            if value > 70:
                return 'Overbought', self.visualizer.colors['danger']
            elif value < 30:
                return 'Oversold', self.visualizer.colors['success']
            else:
                return 'Neutral', self.visualizer.colors['gray']
        
        elif 'BB_%B' in indicator:
            if value > 0.8:
                return 'Upper Band', self.visualizer.colors['warning']
            elif value < 0.2:
                return 'Lower Band', self.visualizer.colors['warning']
            else:
                return 'Within Bands', self.visualizer.colors['success']
        
        elif 'Stoch' in indicator:
            if value > 80:
                return 'Overbought', self.visualizer.colors['danger']
            elif value < 20:
                return 'Oversold', self.visualizer.colors['success']
            else:
                return 'Neutral', self.visualizer.colors['gray']
        
        elif 'MACD' in indicator:
            if value > 0:
                return 'Bullish', self.visualizer.colors['success']
            else:
                return 'Bearish', self.visualizer.colors['danger']
        
        else:
            return 'Current', self.visualizer.colors['primary']
    
    def _display_statistical_summary(self, df: pd.DataFrame, symbol: str):
        """Display statistical summary of asset"""
        st.markdown("#### 📊 Statistical Summary")
        
        if 'Returns' not in df.columns:
            st.warning("No return data available")
            return
        
        returns = df['Returns'].dropna()
        
        if len(returns) < 20:
            st.warning("Insufficient data for statistical analysis")
            return
        
        # Calculate statistics
        stats_data = {
            'Mean Return': f"{returns.mean() * 100:.4f}%",
            'Standard Deviation': f"{returns.std() * 100:.4f}%",
            'Skewness': f"{returns.skew():.4f}",
            'Kurtosis': f"{returns.kurtosis():.4f}",
            'Min Return': f"{returns.min() * 100:.4f}%",
            'Max Return': f"{returns.max() * 100:.4f}%",
            'Median Return': f"{returns.median() * 100:.4f}%",
            'Number of Observations': f"{len(returns):,}",
            'Positive Days': f"{(returns > 0).sum()} ({(returns > 0).mean() * 100:.1f}%)",
            'Negative Days': f"{(returns < 0).sum()} ({(returns < 0).mean() * 100:.1f}%)"
        }
        
        # Display in two columns
        col1, col2 = st.columns(2)
        
        items = list(stats_data.items())
        half = len(items) // 2
        
        with col1:
            for key, value in items[:half]:
                st.metric(key, value)
        
        with col2:
            for key, value in items[half:]:
                st.metric(key, value)
        
        # Distribution plot
        st.markdown("##### Returns Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color=self.visualizer.colors['primary'],
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        y_norm = stats.norm.pdf(x_norm, returns.mean() * 100, returns.std() * 100) * len(returns) * (x_norm[1] - x_norm[0])
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            name='Normal Distribution',
            line=dict(color=self.visualizer.colors['danger'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{symbol} - Returns Distribution",
            height=400,
            template=self.visualizer.template,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_correlation_analysis(self):
        """Display correlation analysis"""
        if _is_empty_data(st.session_state.get("returns_data")) or (isinstance(st.session_state.get("returns_data"), pd.DataFrame) and len(st.session_state.get("returns_data").columns) < 2):
            return
        
        st.markdown("### 🔗 Correlation Analysis")
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty or len(returns_df.columns) < 2:
            st.warning("Insufficient data for correlation analysis")
            return
        
        corr_matrix = returns_df.corr()
        
        # Create correlation matrix with clustering
        fig = self.visualizer.create_interactive_correlation_matrix(
            corr_matrix,
            "Asset Correlations",
            show_clusters=True
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Additional correlation metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
            st.metric("Average Correlation", f"{avg_corr:.3f}")
        
        with col2:
            max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max()
            st.metric("Maximum Correlation", f"{max_corr:.3f}")
        
        with col3:
            min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].min()
            st.metric("Minimum Correlation", f"{min_corr:.3f}")
    
    def _display_market_overview(self):
        """Display market overview"""
        st.markdown("### 🌐 Market Overview")
        
        if not st.session_state.asset_data:
            return
        
        # Create small multiples of price charts
        assets = list(st.session_state.asset_data.keys())[:4]  # Show first 4
        
        if not assets:
            return
        
        cols = st.columns(2)
        
        for i, asset in enumerate(assets):
            with cols[i % 2]:
                df = st.session_state.asset_data[asset]
                if len(df) > 0:
                    # Simple price chart
                    price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[price_col],
                        name=asset,
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=asset,
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False,
                        xaxis_showticklabels=False,
                        yaxis_showticklabels=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def _display_advanced_analytics(self):
        """Display advanced analytics section"""
        st.markdown('<div class="section-header"><h2>⚡ Advanced Analytics</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("⚠️ Please load data first")
            return
        
        # Create advanced analytics tabs
        adv_tabs = st.tabs([
            "📊 Rolling Statistics",
            "📈 Factor Analysis",
            "🔄 Cycle Analysis",
            "🧮 Advanced Metrics"
        ])
        
        with adv_tabs[0]:
            self._display_rolling_statistics(config)
        
        with adv_tabs[1]:
            self._display_factor_analysis(config)
        
        with adv_tabs[2]:
            self._display_cycle_analysis(config)
        
        with adv_tabs[3]:
            self._display_advanced_metrics(config)
    
    def _display_rolling_statistics(self, config: AnalysisConfiguration):
        """Display rolling statistics analysis"""
        st.markdown("#### 📊 Rolling Statistics Analysis")
        
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
                    ["Mean", "Volatility", "Sharpe", "Skewness", "Kurtosis", "VaR", "CVaR"],
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
            elif stat_type == "Kurtosis":
                rolling_stat = returns.rolling(window).kurt()
                y_title = "Kurtosis"
            elif stat_type == "VaR":
                rolling_stat = returns.rolling(window).apply(
                    lambda x: np.percentile(x, 5) * 100 if len(x) > window//2 else np.nan
                )
                y_title = "VaR (95%) (%)"
            elif stat_type == "CVaR":
                rolling_stat = returns.rolling(window).apply(
                    lambda x: x[x <= np.percentile(x, 5)].mean() * 100 if len(x) > window//2 else np.nan
                )
                y_title = "CVaR (95%) (%)"
            
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
            
            # Add rolling mean of the statistic
            if len(rolling_stat) > window:
                rolling_mean_stat = rolling_stat.rolling(window//2).mean()
                fig.add_trace(go.Scatter(
                    x=rolling_mean_stat.index,
                    y=rolling_mean_stat.values,
                    name=f"{stat_type} Trend",
                    line=dict(width=2, color=self.visualizer.colors['danger'], dash='dash')
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
            
            # Statistics summary
            if not rolling_stat.isna().all():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current", f"{rolling_stat.iloc[-1]:.3f}" if not np.isnan(rolling_stat.iloc[-1]) else "N/A")
                
                with col2:
                    st.metric("Mean", f"{rolling_stat.mean():.3f}")
                
                with col3:
                    st.metric("Std", f"{rolling_stat.std():.3f}")
                
                with col4:
                    st.metric("Range", f"{rolling_stat.max() - rolling_stat.min():.3f}")
    
    def _display_factor_analysis(self, config: AnalysisConfiguration):
        """Display factor analysis"""
        st.markdown("#### 📈 Factor Analysis")
        
        if _is_empty_data(st.session_state.get("returns_data")) or (isinstance(st.session_state.get("returns_data"), pd.DataFrame) and len(st.session_state.get("returns_data").columns) < 2):
            st.warning("Need at least 2 assets for factor analysis")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty or len(returns_df) < 60:
            st.warning("Insufficient data for factor analysis")
            return
        
        # Perform PCA
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize returns
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_df)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(returns_scaled)
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Plot explained variance
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(explained_variance))],
                y=explained_variance * 100,
                name='Explained Variance',
                marker_color=self.visualizer.colors['primary']
            ))
            
            fig.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
                y=cumulative_variance * 100,
                name='Cumulative Variance',
                mode='lines+markers',
                line=dict(color=self.visualizer.colors['danger'], width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='PCA - Explained Variance',
                height=400,
                template=self.visualizer.template,
                yaxis=dict(
                    title='Explained Variance (%)',
                    range=[0, max(explained_variance) * 100 * 1.1]
                ),
                yaxis2=dict(
                    title='Cumulative Variance (%)',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                legend=dict(x=0.7, y=0.95)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display PCA results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Principal Components")
                n_components = st.slider(
                    "Number of components to display",
                    1, min(10, len(returns_df.columns)),
                    3, 1
                )
                
                # Create loadings heatmap
                loadings = pca.components_[:n_components]
                loadings_df = pd.DataFrame(
                    loadings.T,
                    columns=[f"PC{i+1}" for i in range(n_components)],
                    index=returns_df.columns
                )
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=loadings_df.values,
                    x=loadings_df.columns,
                    y=loadings_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=loadings_df.round(3).values,
                    texttemplate='%{text}'
                ))
                
                fig2.update_layout(
                    title=f'PCA Loadings (First {n_components} Components)',
                    height=300,
                    width=600
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.markdown("##### Variance Explained")
                variance_df = pd.DataFrame({
                    'PC': [f"PC{i+1}" for i in range(len(explained_variance))],
                    'Variance %': explained_variance * 100,
                    'Cumulative %': cumulative_variance * 100
                })
                
                st.dataframe(
                    variance_df.head(10).style.format({
                        'Variance %': '{:.2f}%',
                        'Cumulative %': '{:.2f}%'
                    }),
                    use_container_width=True,
                    height=300
                )
                
                # Summary metrics
                st.metric(
                    "Components for 95% variance",
                    f"{(cumulative_variance > 0.95).sum() + 1}"
                )
                
                st.metric(
                    "First component variance",
                    f"{explained_variance[0] * 100:.1f}%"
                )
        
        except ImportError:
            st.warning("scikit-learn not available for PCA analysis")
        except Exception as e:
            st.error(f"PCA analysis failed: {str(e)[:100]}")
    
    def _display_cycle_analysis(self, config: AnalysisConfiguration):
        """Display cycle analysis"""
        st.markdown("#### 🔄 Cycle Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Cycle Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="cycle_asset"
        )
        
        if selected_asset and selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
            prices = df[price_col].dropna()
            
            if len(prices) < 100:
                st.warning("Need at least 100 data points for cycle analysis")
                return
            
            # Perform spectral analysis
            try:
                # Detrend the data
                from scipy import signal
                
                # Remove linear trend
                prices_detrended = signal.detrend(prices.values)
                
                # Perform FFT
                n = len(prices_detrended)
                fft_values = np.fft.fft(prices_detrended)
                frequencies = np.fft.fftfreq(n)
                
                # Get power spectrum
                power_spectrum = np.abs(fft_values) ** 2
                
                # Filter positive frequencies
                positive_freq = frequencies[:n//2]
                positive_power = power_spectrum[:n//2]
                
                # Convert frequencies to periods (in days)
                periods = 1 / positive_freq[1:]  # Skip zero frequency
                periods_power = positive_power[1:]
                
                # Find dominant cycles
                dominant_idx = np.argsort(periods_power)[-5:]  # Top 5
                dominant_periods = periods[dominant_idx]
                dominant_power = periods_power[dominant_idx]
                
                # Plot power spectrum
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=periods_power,
                    name='Power Spectrum',
                    line=dict(width=2, color=self.visualizer.colors['primary']),
                    fill='tozeroy',
                    fillcolor=f"rgba({int(self.visualizer.colors['primary'][1:3], 16)}, "
                             f"{int(self.visualizer.colors['primary'][3:5], 16)}, "
                             f"{int(self.visualizer.colors['primary'][5:7], 16)}, 0.2)"
                ))
                
                # Add markers for dominant cycles
                fig.add_trace(go.Scatter(
                    x=dominant_periods,
                    y=dominant_power,
                    mode='markers',
                    name='Dominant Cycles',
                    marker=dict(
                        size=10,
                        color=self.visualizer.colors['danger'],
                        symbol='circle'
                    )
                ))
                
                fig.update_layout(
                    title=f'{selected_asset} - Spectral Analysis',
                    height=400,
                    template=self.visualizer.template,
                    xaxis_title='Period (days)',
                    yaxis_title='Power',
                    xaxis_type='log',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display dominant cycles
                st.markdown("##### Dominant Cycles")
                
                cycles_df = pd.DataFrame({
                    'Period (days)': dominant_periods,
                    'Power': dominant_power,
                    'Type': ['Very Short' if p < 20 else 
                            'Short' if p < 60 else 
                            'Medium' if p < 120 else 
                            'Long' if p < 250 else 
                            'Very Long' for p in dominant_periods]
                }).sort_values('Period (days)')
                
                st.dataframe(
                    cycles_df.style.format({
                        'Period (days)': '{:.1f}',
                        'Power': '{:.2e}'
                    }),
                    use_container_width=True
                )
                
                # Add cycle overlay on price
                if len(dominant_periods) > 0:
                    # Use the strongest cycle
                    strongest_period = dominant_periods[np.argmax(dominant_power)]
                    
                    # Create synthetic cycle
                    t = np.arange(len(prices))
                    cycle = np.sin(2 * np.pi * t / strongest_period)
                    
                    # Scale cycle to price range
                    price_range = prices.max() - prices.min()
                    cycle_scaled = cycle * (price_range * 0.3) + prices.mean()
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=prices.index,
                        y=prices.values,
                        name='Price',
                        line=dict(width=2, color=self.visualizer.colors['primary'])
                    ))
                    
                    fig2.add_trace(go.Scatter(
                        x=prices.index,
                        y=cycle_scaled,
                        name=f'Cycle ({strongest_period:.1f} days)',
                        line=dict(width=1.5, color=self.visualizer.colors['danger'], dash='dash')
                    ))
                    
                    fig2.update_layout(
                        title=f'{selected_asset} - Price with Dominant Cycle',
                        height=400,
                        template=self.visualizer.template,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            except Exception as e:
                st.error(f"Cycle analysis failed: {str(e)[:100]}")
    
    def _display_advanced_metrics(self, config: AnalysisConfiguration):
        """Display advanced performance metrics"""
        st.markdown("#### 🧮 Advanced Performance Metrics")
        
        selected_asset = st.selectbox(
            "Select Asset",
            options=list(st.session_state.returns_data.keys()),
            key="advanced_metrics_asset"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            if len(returns) < 60:
                st.warning("Need at least 60 data points for advanced metrics")
                return
            
            # Calculate advanced metrics
            metrics = self.analytics.calculate_advanced_performance_metrics(returns)
            
            if not metrics:
                st.warning("Could not calculate advanced metrics")
                return
            
            # Display in categorized sections
            st.markdown("##### Risk-Adjusted Returns")
            
            risk_cols = st.columns(4)
            risk_metrics = [
                ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
                ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
                ('Omega Ratio', 'omega_ratio', '{:.3f}'),
                ('Calmar Ratio', 'calmar_ratio', '{:.3f}')
            ]
            
            for (label, key, fmt), col in zip(risk_metrics, risk_cols):
                with col:
                    value = metrics.get(key, 0)
                    st.metric(label, fmt.format(value))
            
            st.markdown("##### Drawdown Analysis")
            
            dd_cols = st.columns(4)
            dd_metrics = [
                ('Max Drawdown', 'max_drawdown', '{:.2f}%'),
                ('Avg Drawdown', 'avg_drawdown', '{:.2f}%'),
                ('Ulcer Index', 'ulcer_index', '{:.2f}'),
                ('Recovery Factor', 'recovery_factor', '{:.3f}')
            ]
            
            for (label, key, fmt), col in zip(dd_metrics, dd_cols):
                with col:
                    value = metrics.get(key, 0)
                    st.metric(label, fmt.format(value))
            
            st.markdown("##### Distribution Statistics")
            
            dist_cols = st.columns(4)
            dist_metrics = [
                ('Skewness', 'skewness', '{:.3f}'),
                ('Kurtosis', 'kurtosis', '{:.3f}'),
                ('Jarque-Bera p-value', 'jarque_bera_p', '{:.4f}'),
                ('Autocorrelation', 'autocorrelation', '{:.3f}')
            ]
            
            for (label, key, fmt), col in zip(dist_metrics, dist_cols):
                with col:
                    value = metrics.get(key, 0)
                    st.metric(label, fmt.format(value))
            
            st.markdown("##### Value at Risk (VaR)")
            
            var_cols = st.columns(3)
            var_metrics = [
                ('VaR (90%)', 'var_90', '{:.2f}%'),
                ('VaR (95%)', 'var_95', '{:.2f}%'),
                ('VaR (99%)', 'var_99', '{:.2f}%')
            ]
            
            for (label, key, fmt), col in zip(var_metrics, var_cols):
                with col:
                    value = metrics.get(key, 0)
                    st.metric(label, fmt.format(value))
            
            st.markdown("##### Conditional VaR (CVaR)")
            
            cvar_cols = st.columns(3)
            cvar_metrics = [
                ('CVaR (90%)', 'cvar_90', '{:.2f}%'),
                ('CVaR (95%)', 'cvar_95', '{:.2f}%'),
                ('CVaR (99%)', 'cvar_99', '{:.2f}%')
            ]
            
            for (label, key, fmt), col in zip(cvar_metrics, cvar_cols):
                with col:
                    value = metrics.get(key, 0)
                    st.metric(label, fmt.format(value))
    
    def _display_risk_analytics(self):
        """Display risk analytics section"""
        st.markdown('<div class="section-header"><h2>⚖️ Risk Analytics</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("⚠️ Please load data first")
            return
        
        # Create risk analytics tabs
        risk_tabs = st.tabs([
            "⚠️ Value at Risk",
            "🧪 Stress Testing",
            "📊 Risk Decomposition",
            "🎯 Risk Limits"
        ])
        
        with risk_tabs[0]:
            self._display_var_analysis(config)
        
        with risk_tabs[1]:
            self._display_stress_testing(config)
        
        with risk_tabs[2]:
            self._display_risk_decomposition(config)
        
        with risk_tabs[3]:
            self._display_risk_limits(config)
    
    def _display_var_analysis(self, config: AnalysisConfiguration):
        """Display Value at Risk analysis"""
        st.markdown("#### ⚠️ Value at Risk (VaR) Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for VaR Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="var_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_method = st.selectbox(
                    "VaR Method",
                    ["historical", "parametric", "modified", "monte_carlo"],
                    key="var_method",
                    help="Historical: based on empirical distribution\n"
                         "Parametric: assumes normal distribution\n"
                         "Modified: Cornish-Fisher adjustment for skewness/kurtosis\n"
                         "Monte Carlo: simulation-based"
                )
            
            with col2:
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.99, 0.995],
                    value=0.95,
                    key="var_confidence"
                )
            
            with col3:
                if st.button("📊 Calculate VaR", type="primary", use_container_width=True, key="calculate_var"):
                    with st.spinner("Calculating VaR..."):
                        # Calculate VaR using different methods
                        var_results = {}
                        
                        # Historical VaR
                        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
                        var_results['Historical'] = var_historical * 100
                        
                        # Parametric VaR (Normal)
                        mean = returns.mean()
                        std = returns.std()
                        z = stats.norm.ppf(1 - confidence_level)
                        var_parametric = mean + std * z
                        var_results['Parametric (Normal)'] = var_parametric * 100
                        
                        # Modified VaR (Cornish-Fisher)
                        skew = returns.skew()
                        kurt = returns.kurtosis()
                        z_cf = (z + 
                               (z**2 - 1) * skew / 6 +
                               (z**3 - 3*z) * kurt / 24 -
                               (2*z**3 - 5*z) * skew**2 / 36)
                        var_modified = mean + std * z_cf
                        var_results['Modified (Cornish-Fisher)'] = var_modified * 100
                        
                        # Calculate CVaR for historical method
                        cvar_historical = returns[returns <= var_historical].mean()
                        
                        # Display results
                        st.markdown("##### VaR Results")
                        
                        results_cols = st.columns(4)
                        
                        with results_cols[0]:
                            st.metric(
                                f"Historical VaR ({confidence_level:.1%})",
                                f"{var_historical * 100:.2f}%",
                                help="Based on empirical distribution"
                            )
                        
                        with results_cols[1]:
                            st.metric(
                                f"Parametric VaR ({confidence_level:.1%})",
                                f"{var_parametric * 100:.2f}%",
                                help="Assumes normal distribution"
                            )
                        
                        with results_cols[2]:
                            st.metric(
                                f"Modified VaR ({confidence_level:.1%})",
                                f"{var_modified * 100:.2f}%",
                                help="Adjusts for skewness and kurtosis"
                            )
                        
                        with results_cols[3]:
                            st.metric(
                                f"CVaR ({confidence_level:.1%})",
                                f"{cvar_historical * 100:.2f}%",
                                help="Expected Shortfall (Conditional VaR)"
                            )
                        
                        # VaR comparison chart
                        methods = list(var_results.keys())
                        values = list(var_results.values())
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=methods,
                                y=values,
                                marker_color=[self.visualizer.colors['primary'],
                                            self.visualizer.colors['secondary'],
                                            self.visualizer.colors['accent']],
                                text=[f"{v:.2f}%" for v in values],
                                textposition='auto'
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"{selected_asset} - VaR Comparison ({confidence_level:.0%} Confidence)",
                            height=400,
                            template=self.visualizer.template,
                            yaxis_title="VaR (%)",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Historical VaR backtest
                        st.markdown("##### VaR Backtest")
                        
                        # Calculate exceedances
                        var_threshold = var_historical
                        exceedances = returns < var_threshold
                        n_exceedances = exceedances.sum()
                        expected_exceedances = len(returns) * (1 - confidence_level)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Actual Exceedances", n_exceedances)
                        
                        with col2:
                            st.metric("Expected Exceedances", f"{expected_exceedances:.1f}")
                        
                        with col3:
                            violation_ratio = n_exceedances / expected_exceedances if expected_exceedances > 0 else 0
                            st.metric("Violation Ratio", f"{violation_ratio:.2f}")
                        
                        # Kupiec test for VaR validity
                        if n_exceedances > 0 and expected_exceedances > 0:
                            from scipy.stats import binomtest
                            
                            # Two-sided binomial test
                            test_result = binomtest(
                                n_exceedances,
                                len(returns),
                                1 - confidence_level,
                                alternative='two-sided'
                            )
                            
                            st.metric("Kupiec Test p-value", f"{test_result.pvalue:.4f}")
                            
                            if test_result.pvalue > 0.05:
                                st.success("✅ VaR model appears valid (p > 0.05)")
                            else:
                                st.warning("⚠️ VaR model may be invalid (p ≤ 0.05)")
    
    def _display_stress_testing(self, config: AnalysisConfiguration):
        """Display stress testing analysis"""
        st.markdown("#### 🧪 Stress Testing")
        
        selected_asset = st.selectbox(
            "Select Asset for Stress Testing",
            options=list(st.session_state.returns_data.keys()),
            key="stress_asset"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            # Define stress scenarios
            st.markdown("##### Define Stress Scenarios")
            
            col1, col2 = st.columns(2)
            
            with col1:
                shock_type = st.radio(
                    "Shock Type",
                    ["Percentage", "Standard Deviations", "Historical"],
                    key="shock_type"
                )
            
            with col2:
                if shock_type == "Percentage":
                    shocks = st.text_input(
                        "Shocks (%)",
                        "-1, -2, -5, -10",
                        help="Comma-separated percentage shocks"
                    )
                    shocks = [float(s.strip()) / 100 for s in shocks.split(",")]
                
                elif shock_type == "Standard Deviations":
                    n_std = st.slider(
                        "Number of Standard Deviations",
                        1.0, 5.0, 2.0, 0.5,
                        key="n_std"
                    )
                    std_dev = returns.std()
                    shocks = [-std_dev * n_std]
                
                else:  # Historical
                    historical_periods = st.selectbox(
                        "Historical Period",
                        ["2008 Crisis", "2020 COVID", "Worst Month", "Worst Week"],
                        key="historical_period"
                    )
                    
                    # Placeholder shocks - in reality would use actual historical data
                    shocks = [-0.05, -0.08, -0.15, -0.20]
            
            if st.button("⚡ Run Stress Test", type="primary", use_container_width=True, key="run_stress_test"):
                with st.spinner("Running stress tests..."):
                    stress_results = []
                    
                    for shock in shocks:
                        # Apply shock to returns
                        shocked_returns = returns + shock
                        
                        # Calculate metrics for shocked returns
                        metrics = self.analytics.calculate_advanced_performance_metrics(shocked_returns)
                        
                        # Calculate portfolio impact
                        initial_value = 100
                        shocked_value = initial_value * (1 + shocked_returns.sum())
                        loss = initial_value - shocked_value
                        
                        stress_results.append({
                            'shock': shock * 100,
                            'shocked_return': shocked_returns.mean() * 252 * 100,
                            'shocked_volatility': shocked_returns.std() * np.sqrt(252) * 100,
                            'shocked_sharpe': metrics.get('sharpe_ratio', 0),
                            'max_drawdown': metrics.get('max_drawdown', 0),
                            'var_95': metrics.get('var_95', 0),
                            'loss': loss,
                            'recovery_time': self._estimate_recovery_time(shocked_returns)
                        })
                    
                    # Display results
                    stress_df = pd.DataFrame(stress_results)
                    
                    st.markdown("##### Stress Test Results")
                    st.dataframe(
                        stress_df.style.format({
                            'shock': '{:.1f}%',
                            'shocked_return': '{:.2f}%',
                            'shocked_volatility': '{:.2f}%',
                            'shocked_sharpe': '{:.3f}',
                            'max_drawdown': '{:.2f}%',
                            'var_95': '{:.2f}%',
                            'loss': '{:.2f}',
                            'recovery_time': '{:.0f} days'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualize stress test results
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=stress_df['shock'],
                        y=stress_df['loss'],
                        name='Portfolio Loss',
                        mode='lines+markers',
                        line=dict(width=3, color=self.visualizer.colors['danger']),
                        marker=dict(size=10)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=stress_df['shock'],
                        y=stress_df['max_drawdown'],
                        name='Max Drawdown',
                        mode='lines+markers',
                        line=dict(width=2, color=self.visualizer.colors['warning'], dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_asset} - Stress Test Results",
                        height=500,
                        template=self.visualizer.template,
                        xaxis_title="Shock (%)",
                        yaxis_title="Impact (%)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _estimate_recovery_time(self, returns: pd.Series) -> float:
        """Estimate recovery time from drawdown"""
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        # Find recovery periods
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = i
            else:
                if in_drawdown:
                    recovery_times.append(i - drawdown_start)
                    in_drawdown = False
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _display_risk_decomposition(self, config: AnalysisConfiguration):
        """Display risk decomposition analysis"""
        st.markdown("#### 📊 Risk Decomposition")
        
        if _is_empty_data(st.session_state.get("returns_data")) or (isinstance(st.session_state.get("returns_data"), pd.DataFrame) and len(st.session_state.get("returns_data").columns) < 2):
            st.warning("Need at least 2 assets for risk decomposition")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty:
            st.warning("No return data available")
            return
        
        # Get portfolio weights
        assets = returns_df.columns.tolist()
        
        st.markdown("##### Portfolio Weights")
        
        # Create weight inputs
        weights = {}
        total_weight = 0
        
        cols = st.columns(min(4, len(assets)))
        
        for i, asset in enumerate(assets):
            with cols[i % len(cols)]:
                default_weight = 1.0 / len(assets)
                weight = st.slider(
                    asset,
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.01,
                    key=f"risk_weight_{asset}"
                )
                weights[asset] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        weight_array = np.array([weights[asset] for asset in assets])
        
        # Calculate risk decomposition
        if st.button("📊 Decompose Risk", type="primary", use_container_width=True):
            with st.spinner("Calculating risk decomposition..."):
                # Calculate covariance matrix
                cov_matrix = returns_df.cov().values * config.annual_trading_days
                portfolio_variance = weight_array.T @ cov_matrix @ weight_array
                portfolio_vol = np.sqrt(portfolio_variance)
                
                if portfolio_variance <= 0:
                    st.warning("Portfolio variance is zero or negative")
                    return
                
                # Marginal risk contributions
                marginal_risk = (cov_matrix @ weight_array) / portfolio_vol
                
                # Component risk contributions
                component_risk = marginal_risk * weight_array
                
                # Percentage contributions
                percentage_risk = (component_risk / portfolio_vol) * 100
                
                # Create results dataframe
                risk_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [weights[a] for a in assets],
                    'Marginal Risk': marginal_risk,
                    'Component Risk': component_risk,
                    'Risk %': percentage_risk
                })
                
                st.markdown("##### Risk Contribution Breakdown")
                
                # Display as table
                st.dataframe(
                    risk_df.style.format({
                        'Weight': '{:.1%}',
                        'Marginal Risk': '{:.4f}',
                        'Component Risk': '{:.4f}',
                        'Risk %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Create visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=risk_df['Asset'],
                    y=risk_df['Risk %'],
                    name='Risk Contribution',
                    marker_color=self.visualizer.colors['primary'],
                    text=risk_df['Risk %'].round(1).astype(str) + '%',
                    textposition='auto'
                ))
                
                fig.add_trace(go.Scatter(
                    x=risk_df['Asset'],
                    y=risk_df['Weight'] * 100,
                    name='Weight %',
                    mode='lines+markers',
                    line=dict(width=2, color=self.visualizer.colors['danger'], dash='dash'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Risk Contribution vs Weight',
                    height=500,
                    template=self.visualizer.template,
                    yaxis=dict(
                        title='Risk Contribution (%)',
                        range=[0, risk_df['Risk %'].max() * 1.1]
                    ),
                    yaxis2=dict(
                        title='Weight (%)',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    ),
                    legend=dict(x=0.7, y=0.95)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk concentration metrics
                st.markdown("##### Risk Concentration Metrics")
                
                # Herfindahl-Hirschman Index for risk concentration
                hhi = np.sum((percentage_risk / 100) ** 2)
                
                # Effective number of risk factors
                effective_factors = 1 / hhi if hhi > 0 else len(assets)
                
                # Diversification benefit
                standalone_vols = np.sqrt(np.diag(cov_matrix))
                weighted_standalone_vol = np.sum(np.abs(weight_array) * standalone_vols)
                diversification_benefit = (weighted_standalone_vol - portfolio_vol) / portfolio_vol
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Portfolio Volatility", f"{portfolio_vol * 100:.2f}%")
                
                with col2:
                    st.metric("Risk Concentration (HHI)", f"{hhi:.3f}")
                
                with col3:
                    st.metric("Effective Risk Factors", f"{effective_factors:.1f}")
                
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric("Diversification Benefit", f"{diversification_benefit * 100:.1f}%")
                
                with col5:
                    max_risk_contrib = risk_df['Risk %'].max()
                    st.metric("Max Risk Contribution", f"{max_risk_contrib:.1f}%")
                
                with col6:
                    risk_inequality = risk_df['Risk %'].std() / risk_df['Risk %'].mean()
                    st.metric("Risk Inequality", f"{risk_inequality:.3f}")
    
    def _display_risk_limits(self, config: AnalysisConfiguration):
        """Display risk limits monitoring"""
        st.markdown("#### 🎯 Risk Limits Monitoring")
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("No return data available")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        # Define risk limits
        st.markdown("##### Define Risk Limits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_limit = st.number_input(
                "VaR Limit (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                key="var_limit"
            ) / 100
        
        with col2:
            vol_limit = st.number_input(
                "Volatility Limit (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="vol_limit"
            ) / 100
        
        with col3:
            dd_limit = st.number_input(
                "Drawdown Limit (%)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                key="dd_limit"
            ) / 100
        
        # Calculate current risk metrics
        if st.button("📊 Check Risk Limits", type="primary", use_container_width=True):
            with st.spinner("Checking risk limits..."):
                limit_violations = []
                warning_assets = []
                
                for asset in returns_df.columns:
                    returns = returns_df[asset].dropna()
                    
                    if len(returns) < 20:
                        continue
                    
                    # Calculate metrics
                    vol = returns.std() * np.sqrt(config.annual_trading_days)
                    var_95 = np.percentile(returns, 5)
                    
                    # Calculate max drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative - running_max) / running_max
                    max_dd = drawdown.min()
                    
                    # Check limits
                    violations = []
                    
                    if abs(var_95) > var_limit:
                        violations.append(f"VaR ({abs(var_95*100):.1f}% > {var_limit*100:.1f}%)")
                    
                    if vol > vol_limit:
                        violations.append(f"Volatility ({vol*100:.1f}% > {vol_limit*100:.1f}%)")
                    
                    if abs(max_dd) > dd_limit:
                        violations.append(f"Drawdown ({abs(max_dd*100):.1f}% > {dd_limit*100:.1f}%)")
                    
                    if violations:
                        limit_violations.append({
                            'Asset': asset,
                            'Violations': ', '.join(violations),
                            'Volatility': vol * 100,
                            'VaR (95%)': var_95 * 100,
                            'Max DD': max_dd * 100
                        })
                    elif vol > vol_limit * 0.8 or abs(var_95) > var_limit * 0.8 or abs(max_dd) > dd_limit * 0.8:
                        warning_assets.append(asset)
                
                # Display results
                if limit_violations:
                    st.error("🚨 Risk Limit Violations Detected!")
                    
                    violations_df = pd.DataFrame(limit_violations)
                    
                    st.dataframe(
                        violations_df.style.format({
                            'Volatility': '{:.1f}%',
                            'VaR (95%)': '{:.1f}%',
                            'Max DD': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Suggest actions
                    with st.expander("🔧 Recommended Actions"):
                        st.markdown("""
                        1. **Reduce position sizes** in violating assets
                        2. **Add hedging instruments** to reduce risk
                        3. **Increase diversification** across uncorrelated assets
                        4. **Implement stop-loss orders**
                        5. **Review risk limits** and adjust if appropriate
                        """)
                
                elif warning_assets:
                    st.warning(f"⚠️ Risk levels approaching limits for: {', '.join(warning_assets)}")
                    
                    # Monitor these assets
                    st.info("""
                    **Monitoring Recommendations:**
                    - Increase monitoring frequency
                    - Prepare contingency plans
                    - Consider partial profit-taking
                    - Review correlation assumptions
                    """)
                
                else:
                    st.success("✅ All assets within risk limits")
                    
                    # Display risk dashboard
                    st.markdown("##### Current Risk Dashboard")
                    
                    risk_metrics = []
                    for asset in returns_df.columns:
                        returns = returns_df[asset].dropna()
                        if len(returns) > 20:
                            vol = returns.std() * np.sqrt(config.annual_trading_days) * 100
                            var_95 = np.percentile(returns, 5) * 100
                            
                            risk_metrics.append({
                                'Asset': asset,
                                'Volatility': vol,
                                'VaR (95%)': var_95,
                                'Vol Limit %': (vol / (vol_limit * 100)) * 100,
                                'VaR Limit %': (abs(var_95) / (var_limit * 100)) * 100
                            })
                    
                    if risk_metrics:
                        risk_df = pd.DataFrame(risk_metrics)
                        
                        # Create risk heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=risk_df[['Vol Limit %', 'VaR Limit %']].values.T,
                            x=risk_df['Asset'],
                            y=['Volatility', 'VaR (95%)'],
                            colorscale='RdYlGn',
                            zmin=0,
                            zmax=100,
                            text=risk_df[['Volatility', 'VaR (95%)']].round(1).values.T,
                            texttemplate='%{text}%',
                            hoverinfo='x+y+z'
                        ))
                        
                        fig.update_layout(
                            title='Risk Limit Utilization',
                            height=300,
                            template=self.visualizer.template,
                            xaxis_title="Asset",
                            yaxis_title="Risk Metric"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def _display_portfolio(self):
        """Display portfolio analysis section"""
        st.markdown('<div class="section-header"><h2>🧺 Portfolio Analysis</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        if _is_empty_data(st.session_state.get("returns_data")):
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
                ["Equal Weight", "Optimized (Sharpe)", "Optimized (Min Variance)", 
                 "Optimized (Risk Parity)", "Custom Weights", "Market Cap Weighted"],
                horizontal=True,
                key="portfolio_weight_mode"
            )
            
            if weight_mode == "Custom Weights":
                st.markdown("**Set Custom Weights:**")
                
                assets = returns_df.columns.tolist()
                n_cols = min(4, len(assets))
                cols = st.columns(n_cols)
                
                weight_inputs = {}
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
                        weight_inputs[asset] = weight
                
                total_weight = sum(weight_inputs.values())
                if total_weight > 0:
                    weights = np.array([weight_inputs[a] / total_weight for a in assets])
                else:
                    weights = np.ones(len(assets)) / len(assets)
                
                st.session_state.portfolio_weights = dict(zip(assets, weights))
            
            elif weight_mode.startswith("Optimized"):
                optimization_type = weight_mode.split("(")[1].rstrip(")")
                
                # Advanced optimization options
                with st.expander("⚡ Advanced Optimization Settings"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        covariance_method = st.selectbox(
                            "Covariance Estimation",
                            ["ledoit_wolf", "oas", "empirical", "shrunk"],
                            key="covariance_method"
                        )
                    
                    with col2:
                        risk_model = st.selectbox(
                            "Risk Model",
                            ["variance", "semi_variance", "cvar", "mad"],
                            key="risk_model"
                        )
                
                if st.button(f"🔄 Optimize Portfolio ({optimization_type})", type="primary", use_container_width=True):
                    with st.spinner("Optimizing portfolio..."):
                        # Map optimization type
                        opt_map = {
                            "Sharpe": "sharpe",
                            "Min Variance": "min_variance",
                            "Risk Parity": "risk_parity",
                            "Max Return": "max_return"
                        }
                        
                        method = opt_map.get(optimization_type, "sharpe")
                        
                        result = self.analytics.optimize_portfolio_advanced(
                            returns_df,
                            method=method,
                            covariance_method=covariance_method,
                            risk_model=risk_model
                        )
                        
                        if result['success']:
                            weights = np.array(list(result['weights'].values()))
                            st.session_state.portfolio_weights = result['weights']
                            st.session_state.portfolio_metrics = result.get('metrics', {})
                            st.session_state.optimization_results = result
                            st.success("✅ Portfolio optimized successfully!")
                            
                            # Display optimization details
                            with st.expander("📊 Optimization Details"):
                                st.json(result.get('optimization_info', {}))
                        else:
                            st.warning(f"⚠️ Optimization failed: {result.get('message', 'Unknown error')}")
                            # Use equal weights as fallback
                            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                else:
                    # Use current weights or equal weights
                    if st.session_state.portfolio_weights:
                        weights = np.array(list(st.session_state.portfolio_weights.values()))
                    else:
                        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            
            elif weight_mode == "Market Cap Weighted":
                st.info("Market cap weighting requires additional data. Using equal weights as proxy.")
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                st.session_state.portfolio_weights = dict(zip(returns_df.columns, weights))
            
            else:  # Equal Weight
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
                st.session_state.portfolio_weights = dict(zip(returns_df.columns, weights))
        
        with col2:
            # Display current weights
            st.markdown("**Current Weights:**")
            
            if st.session_state.portfolio_weights:
                weight_data = []
                for asset, weight in st.session_state.portfolio_weights.items():
                    metadata = self._get_asset_metadata(asset)
                    weight_data.append({
                        'Asset': asset,
                        'Weight': weight,
                        'Color': metadata.color
                    })
                
                for item in sorted(weight_data, key=lambda x: x['Weight'], reverse=True):
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: {item['Color']}; font-weight: 600;">{item['Asset']}</span>
                            <span style="font-weight: 600;">{item['Weight']:.1%}</span>
                        </div>
                        <div style="background: var(--light); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: {item['Color']}; width: {item['Weight']*100}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Weight statistics
                weight_array = np.array([item['Weight'] for item in weight_data])
                st.metric("Concentration (HHI)", f"{np.sum(weight_array ** 2):.3f}")
                st.metric("Effective N", f"{1/np.sum(weight_array ** 2):.1f}")
            else:
                st.info("No weights set")
        
        # Calculate portfolio metrics if weights are available
        if st.session_state.portfolio_weights:
            weight_array = np.array(list(st.session_state.portfolio_weights.values()))
            
            # Ensure weights sum to 1
            weight_array = weight_array / np.sum(weight_array)
            
            # Calculate portfolio returns
            portfolio_returns = returns_df @ weight_array
            
            # Calculate comprehensive metrics
            with st.spinner("Calculating portfolio metrics..."):
                portfolio_metrics = self.analytics.calculate_advanced_performance_metrics(portfolio_returns)
                st.session_state.portfolio_metrics = portfolio_metrics
            
            # Display portfolio performance
            self._display_portfolio_performance(portfolio_metrics, portfolio_returns, returns_df, weight_array)
            
            # Display risk decomposition
            self._display_portfolio_risk_decomposition(returns_df, weight_array)
            
            # Display optimization results if available
            if st.session_state.optimization_results:
                self._display_optimization_results(st.session_state.optimization_results)
        else:
            st.warning("Please configure portfolio weights")
    
    def _display_portfolio_performance(self, metrics: Dict, portfolio_returns: pd.Series, 
                                      returns_df: pd.DataFrame, weights: np.ndarray):
        """Display portfolio performance metrics"""
        st.markdown("### 📊 Portfolio Performance")
        
        # Ensure config is available in this scope (Streamlit-safe)
        config = st.session_state.get("analysis_config", None)
        risk_free_rate = getattr(config, "risk_free_rate", 0.0)
        annual_days = getattr(config, "annual_trading_days", 252)

        # Create metrics grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        # Key performance metrics
        perf_configs = [
            ("Annual Return", "annual_return", "{:.2f}%", 
             "positive" if metrics.get('annual_return', 0) > 0 else "negative",
             "Total annualized return"),
            
            ("Annual Volatility", "annual_volatility", "{:.2f}%", "neutral",
             "Annualized standard deviation of returns"),
            
            ("Sharpe Ratio", "sharpe_ratio", "{:.3f}",
             "positive" if metrics.get('sharpe_ratio', 0) > 0 else "negative",
             "Risk-adjusted return (Sharpe ratio)"),
            
            ("Max Drawdown", "max_drawdown", "{:.2f}%", "negative",
             "Maximum peak-to-trough decline"),
            
            ("Sortino Ratio", "sortino_ratio", "{:.3f}",
             "positive" if metrics.get('sortino_ratio', 0) > 0 else "negative",
             "Downside risk-adjusted return"),
            
            ("Calmar Ratio", "calmar_ratio", "{:.3f}",
             "positive" if metrics.get('calmar_ratio', 0) > 0 else "negative",
             "Return to max drawdown ratio"),
            
            ("Omega Ratio", "omega_ratio", "{:.3f}",
             "positive" if metrics.get('omega_ratio', 0) > 1 else "negative",
             "Probability-weighted ratio of gains vs losses"),
            
            ("Win Rate", "win_rate", "{:.1f}%", "positive",
             "Percentage of positive return periods")
        ]
        
        # Display metrics in a grid
        cols = st.columns(4)
        for i, (label, key, fmt, color_class, tooltip) in enumerate(perf_configs):
            with cols[i % 4]:
                value = metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card custom-tooltip" data-tooltip="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk metrics
        st.markdown("### ⚖️ Risk Metrics")
        
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        risk_configs = [
            ("VaR (95%)", "var_95", "{:.2f}%", "negative", "Value at Risk (95% confidence)"),
            ("CVaR (95%)", "cvar_95", "{:.2f}%", "negative", "Conditional VaR (95% confidence)"),
            ("VaR (99%)", "var_99", "{:.2f}%", "negative", "Value at Risk (99% confidence)"),
            ("CVaR (99%)", "cvar_99", "{:.2f}%", "negative", "Conditional VaR (99% confidence)"),
            ("Ulcer Index", "ulcer_index", "{:.2f}", "negative", "Measure of downside volatility"),
            ("Skewness", "skewness", "{:.3f}", 
             "positive" if metrics.get('skewness', 0) > 0 else "negative", 
             "Asymmetry of return distribution"),
            ("Kurtosis", "kurtosis", "{:.3f}", 
             "warning" if metrics.get('kurtosis', 0) > 3 else "neutral", 
             "Tail risk measure"),
            ("Information Ratio", "information_ratio", "{:.3f}", 
             "positive" if metrics.get('information_ratio', 0) > 0 else "negative", 
             "Active return per unit of active risk")
        ]
        
        risk_cols = st.columns(4)
        for i, (label, key, fmt, color_class, tooltip) in enumerate(risk_configs):
            with risk_cols[i % 4]:
                value = metrics.get(key, 0)
                st.markdown(f"""
                <div class="metric-card custom-tooltip" data-tooltip="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{fmt.format(value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Portfolio Visualization")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Cumulative returns chart
            fig = go.Figure()
            
            # Portfolio cumulative returns
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                name="Portfolio",
                line=dict(color=self.visualizer.colors['primary'], width=3),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.visualizer.colors['primary'][1:3], 16)}, "
                         f"{int(self.visualizer.colors['primary'][3:5], 16)}, "
                         f"{int(self.visualizer.colors['primary'][5:7], 16)}, 0.1)"
            ))
            
            # Add benchmark if available
            if st.session_state.benchmark_returns_data is not None and not st.session_state.benchmark_returns_data.empty:
                benchmark_col = st.session_state.benchmark_returns_data.columns[0]
                benchmark_returns = st.session_state.benchmark_returns_data[benchmark_col]
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                # Align dates
                common_idx = portfolio_cumulative.index.intersection(benchmark_cumulative.index)
                if len(common_idx) > 0:
                    fig.add_trace(go.Scatter(
                        x=common_idx,
                        y=benchmark_cumulative.reindex(common_idx).values,
                        name=f"Benchmark: {benchmark_col}",
                        line=dict(dash='dash', width=2, color=self.visualizer.colors['gray'])
                    ))
            
            fig.update_layout(
                title="Portfolio Cumulative Returns",
                height=400,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Drawdown chart
            running_max = portfolio_cumulative.cummax()
            drawdown = (portfolio_cumulative - running_max) / running_max * 100
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line=dict(color=self.visualizer.colors['danger'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.visualizer.colors['danger'][1:3], 16)}, "
                         f"{int(self.visualizer.colors['danger'][3:5], 16)}, "
                         f"{int(self.visualizer.colors['danger'][5:7], 16)}, 0.3)"
            ))
            
            fig2.update_layout(
                title="Portfolio Drawdown",
                height=400,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title="Drawdown (%)",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Rolling metrics
        st.markdown("### 📊 Rolling Performance")
        
        rolling_col1, rolling_col2 = st.columns(2)
        
        with rolling_col1:
            # Rolling Sharpe ratio
            window = 252  # 1 year
            rolling_mean = portfolio_returns.rolling(window).mean() * annual_days
            rolling_std = portfolio_returns.rolling(window).std() * np.sqrt(annual_days)
            rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
            rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="Rolling Sharpe",
                line=dict(color=self.visualizer.colors['success'], width=2)
            ))
            
            fig3.update_layout(
                title=f"Rolling Sharpe Ratio ({window}-day window)",
                height=300,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title="Sharpe Ratio"
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with rolling_col2:
            # Rolling volatility
            rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(annual_days) * 100
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="Rolling Volatility",
                line=dict(color=self.visualizer.colors['warning'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.visualizer.colors['warning'][1:3], 16)}, "
                         f"{int(self.visualizer.colors['warning'][3:5], 16)}, "
                         f"{int(self.visualizer.colors['warning'][5:7], 16)}, 0.2)"
            ))
            
            fig4.update_layout(
                title=f"Rolling Volatility ({window}-day window)",
                height=300,
                template=self.visualizer.template,
                hovermode='x unified',
                yaxis_title="Annual Volatility (%)"
            )
            
            st.plotly_chart(fig4, use_container_width=True)
    
    def _display_portfolio_risk_decomposition(self, returns_df: pd.DataFrame, weights: np.ndarray):
        """Display portfolio risk decomposition"""
        st.markdown("### 📊 Risk Decomposition")
        
        # Calculate risk decomposition
        cov_matrix = returns_df.cov().values * st.session_state.analysis_config.annual_trading_days
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        if portfolio_variance <= 0:
            st.warning("Cannot calculate risk decomposition: portfolio variance is zero or negative")
            return
        
        # Marginal risk contributions
        marginal_risk = (cov_matrix @ weights) / portfolio_vol
        
        # Component risk contributions
        component_risk = marginal_risk * weights
        
        # Percentage contributions
        percentage_risk = (component_risk / portfolio_vol) * 100
        
        # Create risk decomposition chart
        assets = returns_df.columns.tolist()
        
        risk_df = pd.DataFrame({
            'Asset': assets,
            'Weight': weights,
            'Risk Contribution %': percentage_risk,
            'Color': [self._get_asset_metadata(a).color for a in assets]
        }).sort_values('Risk Contribution %', ascending=False)
        
        # Sunburst chart for risk decomposition
        fig = go.Figure(data=[go.Sunburst(
            labels=risk_df['Asset'].tolist() + ['Portfolio'],
            parents=['Portfolio'] * len(risk_df) + [''],
            values=risk_df['Risk Contribution %'].tolist() + [100],
            branchvalues="total",
            marker=dict(
                colors=risk_df['Color'].tolist() + [self.visualizer.colors['primary']],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Risk Contribution: %{value:.1f}%<br>',
            textinfo='label+percent entry'
        )])
        
        fig.update_layout(
            title="Risk Contribution Breakdown",
            height=500,
            template=self.visualizer.template,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk contribution table
        st.markdown("##### Detailed Risk Contributions")
        
        detailed_df = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.1%}" for w in weights],
            'Marginal Risk': marginal_risk,
            'Component Risk': component_risk,
            'Risk Contribution %': percentage_risk
        }).sort_values('Risk Contribution %', ascending=False)
        
        st.dataframe(
            detailed_df.style.format({
                'Marginal Risk': '{:.4f}',
                'Component Risk': '{:.4f}',
                'Risk Contribution %': '{:.1f}%'
            }),
            use_container_width=True,
            height=300
        )
        
        # Risk concentration metrics
        st.markdown("##### Risk Concentration Metrics")
        
        # Herfindahl-Hirschman Index for risk concentration
        risk_hhi = np.sum((percentage_risk / 100) ** 2)
        
        # Weight HHI
        weight_hhi = np.sum(weights ** 2)
        
        # Diversification metrics
        standalone_vols = np.sqrt(np.diag(cov_matrix))
        weighted_standalone_vol = np.sum(np.abs(weights) * standalone_vols)
        diversification_benefit = (weighted_standalone_vol - portfolio_vol) / portfolio_vol
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Volatility", f"{portfolio_vol * 100:.2f}%")
        
        with col2:
            st.metric("Risk Concentration", f"{risk_hhi:.3f}")
        
        with col3:
            st.metric("Weight Concentration", f"{weight_hhi:.3f}")
        
        with col4:
            st.metric("Diversification Benefit", f"{diversification_benefit * 100:.1f}%")
    
    def _display_optimization_results(self, results: Dict):
        """Display portfolio optimization results"""
        st.markdown("### ⚡ Optimization Results")
        
        with st.expander("View Optimization Details", expanded=False):
            if 'optimization_info' in results:
                st.json(results['optimization_info'])
            
            if 'diversification' in results:
                st.markdown("##### Diversification Metrics")
                diversification_df = pd.DataFrame([results['diversification']])
                st.dataframe(diversification_df, use_container_width=True)
            
            if 'transaction_costs' in results:
                st.markdown("##### Transaction Costs")
                costs_df = pd.DataFrame([results['transaction_costs']])
                st.dataframe(costs_df, use_container_width=True)
            
            if 'stability' in results and results['stability']['available']:
                st.markdown("##### Portfolio Stability")
                stability_df = pd.DataFrame([results['stability']])
                st.dataframe(stability_df, use_container_width=True)
    
    def _display_monte_carlo(self):
        """Display Monte Carlo simulation section"""
        st.markdown('<div class="section-header"><h2>🎲 Monte Carlo Simulation</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("⚠️ Please load data first")
            return
        
        # Monte Carlo configuration
        st.markdown("### ⚙️ Simulation Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_simulations = st.slider(
                "Number of Simulations",
                1000, 50000, 10000, 1000,
                key="mc_simulations"
            )
        
        with col2:
            n_days = st.slider(
                "Time Horizon (days)",
                30, 1000, 252, 30,
                key="mc_horizon"
            )
        
        with col3:
            initial_value = st.number_input(
                "Initial Portfolio Value ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                key="mc_initial_value"
            )
        
        # Select asset or portfolio
        simulation_type = st.radio(
            "Simulation Type",
            ["Single Asset", "Portfolio"],
            horizontal=True,
            key="mc_type"
        )
        
        if simulation_type == "Single Asset":
            selected_asset = st.selectbox(
                "Select Asset for Simulation",
                options=list(st.session_state.returns_data.keys()),
                key="mc_single_asset"
            )
            
            if selected_asset:
                returns = st.session_state.returns_data[selected_asset]
        else:
            # Use portfolio returns if available
            if st.session_state.portfolio_weights and not (_is_empty_data(st.session_state.get("returns_data"))):
                weights = np.array(list(st.session_state.portfolio_weights.values()))
                returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
                returns = returns_df @ weights
                selected_asset = "Portfolio"
            else:
                st.warning("Please configure portfolio weights first")
                return
        
        # Run simulation
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True, key="run_mc"):
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                # Perform Monte Carlo simulation
                mc_result = self.analytics.monte_carlo_simulation(
                    returns,
                    n_simulations=n_simulations,
                    n_days=n_days
                )
                
                if mc_result:
                    st.session_state.monte_carlo_results[selected_asset] = mc_result
                    
                    # Scale results by initial value
                    scaled_paths = mc_result['paths'] * (initial_value / 100)
                    scaled_final = mc_result['mean_final_value'] * (initial_value / 100)
                    scaled_var = mc_result['var_95_final'] * (initial_value / 100)
                    
                    # Display results
                    st.markdown("### 📊 Simulation Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Expected Final Value",
                            f"${scaled_final:,.0f}",
                            help="Mean of final portfolio values"
                        )
                    
                    with col2:
                        st.metric(
                            "VaR (95%)",
                            f"${scaled_var:,.0f}",
                            help="5th percentile of final values"
                        )
                    
                    with col3:
                        probability_loss = mc_result['probability_loss']
                        st.metric(
                            "Probability of Loss",
                            f"{probability_loss:.1f}%",
                            delta=f"-{probability_loss:.1f}%" if probability_loss > 50 else None,
                            help="Probability of ending below initial value"
                        )
                    
                    with col4:
                        expected_max = mc_result['expected_max'] * (initial_value / 100)
                        st.metric(
                            "Expected Maximum",
                            f"${expected_max:,.0f}",
                            help="Expected maximum value during simulation"
                        )
                    
                    # Additional metrics
                    st.markdown("##### Additional Statistics")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        expected_min = mc_result['expected_min'] * (initial_value / 100)
                        st.metric("Expected Minimum", f"${expected_min:,.0f}")
                    
                    with col6:
                        std_final = mc_result['std_final_value'] * (initial_value / 100)
                        st.metric("Std of Final Values", f"${std_final:,.0f}")
                    
                    with col7:
                        cvar_95 = mc_result['cvar_95_final'] * (initial_value / 100)
                        st.metric("CVaR (95%)", f"${cvar_95:,.0f}")
                    
                    with col8:
                        # Calculate probability of doubling
                        doubling_prob = (scaled_paths[:, -1] > initial_value * 2).mean() * 100
                        st.metric("Probability of Doubling", f"{doubling_prob:.1f}%")
                    
                    # Plot simulation paths
                    st.markdown("### 📈 Simulation Paths")
                    
                    fig = go.Figure()
                    
                    # Plot a subset of paths for clarity
                    n_sample_paths = min(100, n_simulations)
                    sample_paths = scaled_paths[:n_sample_paths]
                    
                    for i in range(n_sample_paths):
                        fig.add_trace(go.Scatter(
                            x=list(range(n_days)),
                            y=sample_paths[i],
                            mode='lines',
                            line=dict(width=1, color='rgba(100, 100, 100, 0.1)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Plot mean path
                    mean_path = scaled_paths.mean(axis=0)
                    fig.add_trace(go.Scatter(
                        x=list(range(n_days)),
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color=self.visualizer.colors['primary']),
                        name='Mean Path'
                    ))
                    
                    # Plot confidence intervals
                    percentile_5 = np.percentile(scaled_paths, 5, axis=0)
                    percentile_95 = np.percentile(scaled_paths, 95, axis=0)
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(n_days)) + list(range(n_days))[::-1],
                        y=list(percentile_95) + list(percentile_5)[::-1],
                        fill='toself',
                        fillcolor='rgba(38, 208, 206, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='90% Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_asset} - Monte Carlo Simulation Paths (n={n_simulations:,})",
                        height=500,
                        template=self.visualizer.template,
                        xaxis_title="Days",
                        yaxis_title="Portfolio Value ($)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Final value distribution
                    st.markdown("### 📊 Final Value Distribution")
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Histogram(
                        x=scaled_paths[:, -1],
                        nbinsx=50,
                        name='Final Values',
                        marker_color=self.visualizer.colors['primary'],
                        opacity=0.7
                    ))
                    
                    # Add vertical lines for key values
                    fig2.add_vline(x=initial_value, line_dash="dash", line_color="red", 
                                 annotation_text="Initial Value")
                    fig2.add_vline(x=scaled_final, line_dash="dash", line_color="green", 
                                 annotation_text="Mean")
                    fig2.add_vline(x=scaled_var, line_dash="dash", line_color="orange", 
                                 annotation_text="VaR (95%)")
                    
                    fig2.update_layout(
                        title='Distribution of Final Portfolio Values',
                        height=400,
                        template=self.visualizer.template,
                        xaxis_title="Final Value ($)",
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Performance metrics by percentile
                    st.markdown("### 📈 Performance by Percentile")
                    
                    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    percentile_values = np.percentile(scaled_paths[:, -1], percentiles)
                    
                    perf_df = pd.DataFrame({
                        'Percentile': [f"{p}%" for p in percentiles],
                        'Final Value': percentile_values,
                        'Return': (percentile_values / initial_value - 1) * 100
                    })
                    
                    st.dataframe(
                        perf_df.style.format({
                            'Final Value': '${:,.0f}',
                            'Return': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Export simulation results
                    st.markdown("### 💾 Export Results")
                    
                    if st.button("📥 Download Simulation Data", use_container_width=True):
                        # Create comprehensive results dataframe
                        results_data = {
                            'initial_value': initial_value,
                            'n_simulations': n_simulations,
                            'n_days': n_days,
                            'mean_final_value': scaled_final,
                            'var_95': scaled_var,
                            'probability_loss': probability_loss,
                            'paths': scaled_paths.tolist()
                        }
                        
                        results_json = json.dumps(results_data, indent=2)
                        
                        st.download_button(
                            label="Download JSON Results",
                            data=results_json,
                            file_name=f"monte_carlo_{selected_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    
    def _display_garch_analysis(self):
        """Display GARCH analysis section"""
        st.markdown('<div class="section-header"><h2>📈 GARCH Volatility Modeling</h2></div>', unsafe_allow_html=True)
        
        if not dep_manager.is_available('arch'):
            st.warning("⚠️ ARCH/GARCH modeling requires the 'arch' package. Please install it.")
            return
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("⚠️ Please load data first")
            return
        
        st.markdown("### ⚙️ GARCH Configuration")
        
        selected_asset = st.selectbox(
            "Select Asset for GARCH Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="garch_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p_max = st.slider("ARCH Order (p)", 1, 5, 1, 1, key="garch_p")
            
            with col2:
                q_max = st.slider("GARCH Order (q)", 1, 5, 1, 1, key="garch_q")
            
            with col3:
                distributions = st.multiselect(
                    "Error Distributions",
                    ["normal", "t", "skewt"],
                    default=["normal", "t"],
                    key="garch_dist"
                )
            
            # Advanced options
            with st.expander("⚡ Advanced GARCH Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    include_egarch = st.checkbox("Include EGARCH", value=True, key="include_egarch")
                    include_gjr = st.checkbox("Include GJR-GARCH", value=True, key="include_gjr")
                
                with col2:
                    forecast_horizon = st.slider(
                        "Forecast Horizon (days)",
                        5, 100, 30, 5,
                        key="garch_forecast"
                    )
            
            if st.button("🔍 Run GARCH Analysis", type="primary", use_container_width=True, key="run_garch"):
                with st.spinner("Running GARCH analysis..."):
                    result = self.analytics.garch_analysis_advanced(
                        returns,
                        p_range=(1, p_max),
                        q_range=(1, q_max),
                        distributions=distributions,
                        include_egarch=include_egarch,
                        include_gjr=include_gjr,
                        forecast_horizon=forecast_horizon
                    )
                    
                    if result.get('available', False):
                        st.session_state.garch_results[selected_asset] = result
                        st.success("✅ GARCH analysis completed!")
                        
                        # Display results
                        self._display_garch_results(result, selected_asset)
                    else:
                        st.warning(f"⚠️ {result.get('message', 'GARCH analysis failed')}")
    
    def _display_garch_results(self, result: Dict, asset: str):
        """Display GARCH analysis results"""
        best_model = result['best_model']
        
        # Model summary
        st.markdown("### 📊 Best Model Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", best_model['type'])
        
        with col2:
            st.metric("Order", f"({best_model['p']},{best_model['q']})")
        
        with col3:
            st.metric("Distribution", best_model['distribution'])
        
        with col4:
            st.metric("BIC", f"{best_model['bic']:.1f}")
        
        # Model comparison
        st.markdown("### 📈 Model Comparison")
        
        if 'top_models' in result:
            top_models_df = pd.DataFrame(result['top_models'])
            
            # Display top models
            st.dataframe(
                top_models_df[['type', 'p', 'q', 'distribution', 'aic', 'bic', 'log_likelihood']]
                .style.format({
                    'aic': '{:.1f}',
                    'bic': '{:.1f}',
                    'log_likelihood': '{:.1f}'
                }),
                use_container_width=True
            )
        
        # Volatility visualization
        st.markdown("### 📊 Volatility Analysis")
        
        if 'conditional_volatility' in best_model:
            conditional_vol = best_model['conditional_volatility']
            
            # Get returns for the same period
            returns = result['returns']
            
            # Create volatility chart
            fig = self.visualizer.create_garch_volatility(
                returns,
                conditional_vol,
                best_model.get('volatility_forecast'),
                f"{asset} - GARCH Volatility"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
        st.markdown("### 🧮 Model Parameters")
        
        if 'params' in best_model:
            params_df = pd.DataFrame.from_dict(
                best_model['params'],
                orient='index',
                columns=['Value']
            )
            
            st.dataframe(
                params_df.style.format({'Value': '{:.6f}'}),
                use_container_width=True
            )
            
            # Parameter significance
            st.markdown("##### Parameter Significance")
            
            # Calculate approximate t-stats (if we had standard errors)
            # For now, just show parameters
            
            # Volatility persistence
            if 'persistence' in best_model:
                persistence = best_model['persistence']
                st.metric("Volatility Persistence", f"{persistence:.4f}")
                
                if persistence > 0.95:
                    st.warning("⚠️ High volatility persistence - shocks have long-lasting effects")
                elif persistence < 0.5:
                    st.info("ℹ️ Low volatility persistence - shocks decay quickly")
        
        # Residual analysis
        st.markdown("### 📊 Residual Analysis")
        
        if 'std_residuals' in best_model:
            residuals = best_model['std_residuals']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # QQ plot of residuals
                qq_data = stats.probplot(residuals, dist="norm")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='Residuals',
                    marker=dict(color=self.visualizer.colors['primary'], size=6)
                ))
                
                # Add theoretical line
                x_line = np.array([qq_data[0][0][0], qq_data[0][0][-1]])
                y_line = qq_data[1][0] + qq_data[1][1] * x_line
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name='Normal',
                    line=dict(color=self.visualizer.colors['danger'], width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='QQ Plot of Standardized Residuals',
                    height=400,
                    template=self.visualizer.template,
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ACF of squared residuals
                from statsmodels.graphics.tsaplots import plot_acf
                import matplotlib.pyplot as plt
                
                squared_residuals = residuals ** 2
                
                fig2, ax = plt.subplots(figsize=(10, 6))
                plot_acf(squared_residuals, lags=20, ax=ax, title='ACF of Squared Residuals')
                plt.tight_layout()
                
                st.pyplot(fig2)
        
        # Forecast
        if 'volatility_forecast' in best_model and best_model['volatility_forecast'] is not None:
            st.markdown("### 🔮 Volatility Forecast")
            
            forecast = best_model['volatility_forecast']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(forecast) + 1)),
                y=forecast * 100,
                name='Volatility Forecast',
                line=dict(color=self.visualizer.colors['primary'], width=3),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f'{asset} - {len(forecast)}-Day Volatility Forecast',
                height=400,
                template=self.visualizer.template,
                xaxis_title="Days Ahead",
                yaxis_title="Annualized Volatility (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Forecast", f"{forecast.mean() * 100:.2f}%")
            
            with col2:
                st.metric("Max Forecast", f"{forecast.max() * 100:.2f}%")
            
            with col3:
                st.metric("Min Forecast", f"{forecast.min() * 100:.2f}%")
    
    def _display_regime_analysis(self):
        """Display regime detection analysis"""
        st.markdown('<div class="section-header"><h2>🔄 Market Regime Detection</h2></div>', unsafe_allow_html=True)
        
        if not dep_manager.is_available('hmmlearn'):
            st.warning("⚠️ Regime detection requires 'hmmlearn' package. Please install it.")
            return
        
        if _is_empty_data(st.session_state.get("returns_data")):
            st.warning("⚠️ Please load data first")
            return
        
        st.markdown("### ⚙️ Regime Detection Configuration")
        
        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=list(st.session_state.returns_data.keys()),
            key="regime_asset_select"
        )
        
        if selected_asset:
            returns = st.session_state.returns_data[selected_asset]
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_regimes = st.slider(
                    "Number of Regimes",
                    2, 5, 3, 1,
                    key="n_regimes"
                )
            
            with col2:
                features = st.multiselect(
                    "Features for Regime Detection",
                    ["returns", "volatility", "volume", "range", "momentum"],
                    default=["returns", "volatility"],
                    key="regime_features"
                )
            
            # Advanced options
            with st.expander("⚡ Advanced HMM Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    covariance_type = st.selectbox(
                        "Covariance Type",
                        ["full", "tied", "diag", "spherical"],
                        key="covariance_type"
                    )
                
                with col2:
                    n_iter = st.slider(
                        "Maximum Iterations",
                        10, 1000, 100, 10,
                        key="hmm_iterations"
                    )
            
            if st.button("🔍 Detect Regimes", type="primary", use_container_width=True, key="detect_regimes"):
                with st.spinner("Detecting market regimes..."):
                    result = self.analytics.detect_regimes(
                        returns,
                        n_regimes=n_regimes,
                        features=features
                    )
                    
                    if result.get('available', False):
                        st.session_state.regime_results[selected_asset] = result
                        st.success("✅ Regime detection completed!")
                        
                        # Display results
                        self._display_regime_results(result, selected_asset)
                    else:
                        st.warning(f"⚠️ {result.get('message', 'Regime detection failed')}")
    
    def _display_regime_results(self, result: Dict, asset: str):
        """Display regime detection results"""
        # Regime statistics
        st.markdown("### 📊 Regime Statistics")
        
        if result.get('regime_stats'):
            stats_df = pd.DataFrame(result['regime_stats'])
            
            # Format regime names if available
            if 'regime_labels' in result:
                regime_names = result['regime_labels']
                stats_df['Regime Name'] = stats_df['regime'].map(
                    lambda x: regime_names.get(int(x), {}).get('name', f'Regime {x}')
                )
                stats_df['Color'] = stats_df['regime'].map(
                    lambda x: regime_names.get(int(x), {}).get('color', '#666666')
                )
            
            st.dataframe(
                stats_df.style.format({
                    'frequency': '{:.2f}%',
                    'mean_return': '{:.4f}%',
                    'volatility': '{:.2f}%',
                    'sharpe': '{:.3f}',
                    'var_95': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            # Visualize regime statistics
            fig = go.Figure()
            
            for _, row in stats_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row.get('Regime Name', f'Regime {row["regime"]}')],
                    y=[row['mean_return']],
                    name=f'Regime {row["regime"]}',
                    marker_color=row.get('Color', self.visualizer.colors['primary']),
                    error_y=dict(
                        type='data',
                        array=[row['volatility']],
                        visible=True
                    )
                ))
            
            fig.update_layout(
                title=f'{asset} - Regime Return vs Volatility',
                height=400,
                template=self.visualizer.template,
                xaxis_title="Regime",
                yaxis_title="Mean Return (%)",
                showlegend=False,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Regime visualization
        st.markdown("### 📈 Regime Visualization")
        
        if asset in st.session_state.asset_data and 'regimes' in result:
            price_data = st.session_state.asset_data[asset]
            price_col = 'Adj_Close' if 'Adj_Close' in price_data.columns else 'Close'
            price = price_data[price_col]
            
            fig = self.visualizer.create_regime_chart(
                price,
                result['regimes'],
                result.get('regime_labels', {}),
                f"{asset} - Market Regimes"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Transition matrix
        st.markdown("### 🔄 Regime Transitions")
        
        if 'model' in result:
            try:
                model = result['model']
                transition_matrix = model.transmat_
                
                # Create heatmap
                n_regimes = transition_matrix.shape[0]
                
                fig = go.Figure(data=go.Heatmap(
                    z=transition_matrix,
                    x=[f'To {i}' for i in range(n_regimes)],
                    y=[f'From {i}' for i in range(n_regimes)],
                    colorscale='Viridis',
                    text=transition_matrix.round(3),
                    texttemplate='%{text}',
                    hoverinfo='x+y+z'
                ))
                
                fig.update_layout(
                    title='Regime Transition Probabilities',
                    height=400,
                    template=self.visualizer.template,
                    xaxis_title="To Regime",
                    yaxis_title="From Regime"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stationary distribution
                from scipy.linalg import eig
                
                # Calculate stationary distribution
                eigenvalues, eigenvectors = eig(transition_matrix.T)
                stationary_idx = np.where(np.abs(eigenvalues - 1) < 1e-10)[0]
                
                if len(stationary_idx) > 0:
                    stationary_dist = np.real(eigenvectors[:, stationary_idx[0]])
                    stationary_dist = stationary_dist / stationary_dist.sum()
                    
                    st.markdown("##### Stationary Distribution")
                    
                    stationary_df = pd.DataFrame({
                        'Regime': range(n_regimes),
                        'Probability': stationary_dist
                    })
                    
                    if 'regime_labels' in result:
                        stationary_df['Name'] = stationary_df['Regime'].map(
                            lambda x: result['regime_labels'].get(x, {}).get('name', f'Regime {x}')
                        )
                    
                    st.dataframe(
                        stationary_df.style.format({'Probability': '{:.3f}'}),
                        use_container_width=True
                    )
                    
                    # Expected regime duration
                    expected_durations = 1 / (1 - np.diag(transition_matrix))
                    
                    duration_df = pd.DataFrame({
                        'Regime': range(n_regimes),
                        'Expected Duration (days)': expected_durations
                    })
                    
                    if 'regime_labels' in result:
                        duration_df['Name'] = duration_df['Regime'].map(
                            lambda x: result['regime_labels'].get(x, {}).get('name', f'Regime {x}')
                        )
                    
                    st.dataframe(
                        duration_df.style.format({'Expected Duration (days)': '{:.1f}'}),
                        use_container_width=True
                    )
            
            except Exception as e:
                st.warning(f"Could not calculate transition matrix: {str(e)[:100]}")
        
        # Regime switching strategy
        st.markdown("### 🎯 Regime-Based Strategy")
        
        if 'regimes' in result and asset in st.session_state.asset_data:
            returns = st.session_state.returns_data[asset]
            regimes = result['regimes']
            
            # Simple regime-based strategy
            col1, col2 = st.columns(2)
            
            with col1:
                bullish_regime = st.selectbox(
                    "Bullish Regime",
                    options=list(set(regimes)),
                    key="bullish_regime",
                    help="Regime to be fully invested in"
                )
            
            with col2:
                bearish_action = st.selectbox(
                    "Action in Bearish Regimes",
                    ["Hold Cash", "Reduce 50%", "Short", "Hedge"],
                    key="bearish_action"
                )
            
            if st.button("📊 Backtest Regime Strategy", type="secondary", use_container_width=True):
                with st.spinner("Backtesting regime strategy..."):
                    # Simple backtest
                    strategy_returns = returns.copy()
                    
                    if bearish_action == "Hold Cash":
                        # Set returns to 0 in non-bullish regimes
                        strategy_returns[regimes != bullish_regime] = 0
                    elif bearish_action == "Reduce 50%":
                        # Reduce returns by 50% in non-bullish regimes
                        strategy_returns[regimes != bullish_regime] *= 0.5
                    # Add other strategies as needed
                    
                    # Calculate strategy metrics
                    strategy_metrics = self.analytics.calculate_advanced_performance_metrics(strategy_returns)
                    
                    # Compare with buy-and-hold
                    buyhold_metrics = self.analytics.calculate_advanced_performance_metrics(returns)
                    
                    # Display comparison
                    comparison_df = pd.DataFrame({
                        'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                        'Regime Strategy': [
                            strategy_metrics.get('annual_return', 0),
                            strategy_metrics.get('annual_volatility', 0),
                            strategy_metrics.get('sharpe_ratio', 0),
                            strategy_metrics.get('max_drawdown', 0)
                        ],
                        'Buy & Hold': [
                            buyhold_metrics.get('annual_return', 0),
                            buyhold_metrics.get('annual_volatility', 0),
                            buyhold_metrics.get('sharpe_ratio', 0),
                            buyhold_metrics.get('max_drawdown', 0)
                        ]
                    })
                    
                    st.dataframe(
                        comparison_df.style.format({
                            'Regime Strategy': '{:.2f}%',
                            'Buy & Hold': '{:.2f}%'
                        }).format({
                            'Regime Strategy': '{:.3f}',
                            'Buy & Hold': '{:.3f}'
                        }, subset=['Sharpe Ratio']),
                        use_container_width=True
                    )
    
    def _display_reporting(self):
        """Display reporting section"""
        st.markdown('<div class="section-header"><h2>📋 Professional Reports</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first to generate reports")
            return
        
        # Report configuration
        st.markdown("### ⚙️ Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                [
                    "Portfolio Summary",
                    "Risk Analysis", 
                    "Performance Attribution",
                    "Comprehensive Analysis",
                    "Executive Summary",
                    "Regulatory Compliance"
                ],
                key="report_type"
            )
        
        with col2:
            report_format = st.selectbox(
                "Output Format",
                ["HTML", "Markdown", "JSON", "PDF (Beta)"],
                key="report_format"
            )
        
        # Report options
        st.markdown("### 📊 Report Content")
        
        options_cols = st.columns(4)
        
        with options_cols[0]:
            include_charts = st.checkbox("Include Charts", value=True, key="include_charts")
            include_tables = st.checkbox("Include Tables", value=True, key="include_tables")
        
        with options_cols[1]:
            include_metrics = st.checkbox("Include Metrics", value=True, key="include_metrics")
            include_details = st.checkbox("Include Details", value=True, key="include_details")
        
        with options_cols[2]:
            include_analysis = st.checkbox("Include Analysis", value=True, key="include_analysis")
            include_recommendations = st.checkbox("Include Recommendations", value=True, key="include_recs")
        
        with options_cols[3]:
            confidential = st.checkbox("Confidential", value=True, key="confidential")
            include_disclaimer = st.checkbox("Include Disclaimer", value=True, key="include_disclaimer")
        
        # Generate report
        if st.button("📄 Generate Professional Report", type="primary", use_container_width=True):
            with st.spinner("Generating professional report..."):
                try:
                    # Prepare report data
                    report_data = self._prepare_report_data(
                        report_type,
                        include_charts,
                        include_tables,
                        include_metrics,
                        include_details,
                        include_analysis,
                        include_recommendations
                    )
                    
                    # Generate report based on format
                    if report_format == "HTML":
                        report_content = self._generate_html_report(report_data, confidential, include_disclaimer)
                        file_name = f"ica_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        mime_type = "text/html"
                    
                    elif report_format == "Markdown":
                        report_content = self._generate_markdown_report(report_data, confidential, include_disclaimer)
                        file_name = f"ica_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        mime_type = "text/markdown"
                    
                    elif report_format == "JSON":
                        report_content = json.dumps(report_data, indent=2)
                        file_name = f"ica_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        mime_type = "application/json"
                    
                    else:  # PDF
                        report_content = self._generate_html_report(report_data, confidential, include_disclaimer)
                        file_name = f"ica_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        mime_type = "text/html"
                        st.info("📝 PDF generation requires additional libraries. Downloading HTML version instead.")
                    
                    # Display preview for HTML
                    if report_format == "HTML":
                        st.markdown("### 📊 Report Preview")
                        with st.expander("Preview Report", expanded=True):
                            st.components.v1.html(report_content, height=600, scrolling=True)
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download {report_format.upper()} Report",
                        data=report_content,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True
                    )
                    
                    # Report metadata
                    with st.expander("📊 Report Metadata"):
                        st.metric("Report ID", hashlib.md5(str(report_data).encode()).hexdigest()[:16])
                        st.metric("Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        st.metric("Data Points", len(next(iter(st.session_state.returns_data.values()))) 
                                 if not _is_empty_data(st.session_state.get("returns_data")) else 0)
                    
                except Exception as e:
                    st.error(f"❌ Failed to generate report: {str(e)[:200]}")
                    self._log_error(e, "Report generation")
        
        # Quick snapshot
        st.markdown("### 📸 Quick Snapshot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Take System Snapshot", use_container_width=True, key="take_snapshot"):
                snapshot = self._create_system_snapshot()
                st.json(snapshot, expanded=False)
                
                st.download_button(
                    label="📥 Download JSON Snapshot",
                    data=json.dumps(snapshot, indent=2),
                    file_name=f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("Export Configuration", use_container_width=True, key="export_config"):
                config_data = {
                    'timestamp': datetime.now().isoformat(),
                    'platform_version': 'v6.0 Enhanced',
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
                    mime="application/json",
                    use_container_width=True
                )
    
    def _prepare_report_data(self, report_type: str, include_charts: bool, include_tables: bool,
                           include_metrics: bool, include_details: bool, include_analysis: bool,
                           include_recommendations: bool) -> Dict[str, Any]:
        """Prepare comprehensive report data"""
        
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_type': report_type,
                'platform_version': 'v6.0 Enhanced',
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
            },
            'configuration': {
                'include_charts': include_charts,
                'include_tables': include_tables,
                'include_metrics': include_metrics,
                'include_details': include_details,
                'include_analysis': include_analysis,
                'include_recommendations': include_recommendations
            },
            'data_summary': {
                'assets': st.session_state.selected_assets,
                'benchmarks': st.session_state.selected_benchmarks,
                'date_range': {
                    'start': st.session_state.analysis_config.start_date.strftime('%Y-%m-%d'),
                    'end': st.session_state.analysis_config.end_date.strftime('%Y-%m-%d')
                },
                'data_points': len(next(iter(st.session_state.returns_data.values()))) 
                             if not _is_empty_data(st.session_state.get("returns_data")) else 0,
                'assets_loaded': len(st.session_state.asset_data)
            }
        }
        
        # Add portfolio data if available
        if st.session_state.portfolio_weights:
            report_data['portfolio'] = {
                'weights': st.session_state.portfolio_weights,
                'metrics': st.session_state.portfolio_metrics
            }
        
        # Add analytics data if available
        if include_analysis:
            report_data['analytics'] = {
                'garch_results': st.session_state.garch_results,
                'regime_results': st.session_state.regime_results,
                'risk_results': st.session_state.risk_results,
                'monte_carlo_results': st.session_state.monte_carlo_results
            }
        
        # Add recommendations if requested
        if include_recommendations and st.session_state.portfolio_metrics:
            report_data['recommendations'] = self._generate_recommendations()
        
        return report_data
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate investment recommendations based on analysis"""
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'summary': [],
            'actions': [],
            'warnings': [],
            'opportunities': []
        }
        
        # Basic recommendations based on portfolio metrics
        if st.session_state.portfolio_metrics:
            metrics = st.session_state.portfolio_metrics
            
            # Sharpe ratio recommendations
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 1.5:
                recommendations['summary'].append("Excellent risk-adjusted returns")
            elif sharpe > 1.0:
                recommendations['summary'].append("Good risk-adjusted returns")
            elif sharpe > 0.5:
                recommendations['summary'].append("Moderate risk-adjusted returns")
            else:
                recommendations['summary'].append("Consider improving risk-adjusted returns")
                recommendations['actions'].append("Review asset allocation for better risk-return tradeoff")
            
            # Drawdown recommendations
            max_dd = abs(metrics.get('max_drawdown', 0))
            if max_dd > 30:
                recommendations['warnings'].append(f"High maximum drawdown ({max_dd:.1f}%)")
                recommendations['actions'].append("Consider adding downside protection or reducing risk")
            elif max_dd > 20:
                recommendations['warnings'].append(f"Moderate maximum drawdown ({max_dd:.1f}%)")
            
            # Concentration recommendations
            if st.session_state.portfolio_weights:
                weights = np.array(list(st.session_state.portfolio_weights.values()))
                hhi = np.sum(weights ** 2)
                if hhi > 0.3:
                    recommendations['actions'].append("Portfolio is concentrated - consider increasing diversification")
                elif hhi < 0.1:
                    recommendations['summary'].append("Well-diversified portfolio")
        
        return recommendations
    
    def _create_system_snapshot(self) -> Dict[str, Any]:
        """Create system snapshot"""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'platform_version': 'v6.0 Enhanced',
            'data': {
                'assets_loaded': st.session_state.selected_assets,
                'benchmarks_loaded': st.session_state.selected_benchmarks,
                'data_points': len(next(iter(st.session_state.returns_data.values()))) 
                             if not _is_empty_data(st.session_state.get("returns_data")) else 0,
                'data_loaded': st.session_state.data_loaded
            },
            'portfolio': {
                'weights': st.session_state.portfolio_weights,
                'metrics_summary': {
                    k: v for k, v in st.session_state.portfolio_metrics.items()
                    if k in ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
                } if st.session_state.portfolio_metrics else {}
            },
            'system': {
                'dependencies': {dep: info.get('available', False) 
                               for dep, info in dep_manager.dependencies.items()},
                'python_version': sys.version,
                'streamlit_version': st.__version__,
                'platform': sys.platform
            },
            'performance': {
                'runtime': str(datetime.now() - self.start_time),
                'memory_usage': 'Optimized',
                'cache_status': 'Active'
            }
        }
        
        return snapshot
    
    def _display_settings(self):
        """Display settings and system information"""
        st.markdown('<div class="section-header"><h2>⚙️ Settings & System Info</h2></div>', unsafe_allow_html=True)
        
        config = st.session_state.analysis_config
        
        # Platform information
        st.markdown("### 🏛️ Platform Information")
        
        info_cols = st.columns(4)
        
        with info_cols[0]:
            st.metric("Platform Version", "v6.0 Enhanced")
        
        with info_cols[1]:
            st.metric("Python Version", sys.version.split()[0])
        
        with info_cols[2]:
            st.metric("Streamlit Version", st.__version__)
        
        with info_cols[3]:
            st.metric("Architecture", "64-bit" if sys.maxsize > 2**32 else "32-bit")
        
        # Dependencies status
        st.markdown("### 📦 Dependencies Status")
        
        deps_cols = st.columns(4)
        deps_list = list(dep_manager.dependencies.items())
        
        for i, (dep_name, dep_info) in enumerate(deps_list):
            with deps_cols[i % 4]:
                available = dep_info.get('available', False)
                status = "🟢 Available" if available else "🔴 Not Available"
                st.markdown(f"**{dep_name}:** {status}")
                
                if not available and dep_name in ['arch', 'hmmlearn']:
                    st.caption("Some features disabled")
        
        # Configuration editor
        st.markdown("### ⚙️ Configuration Editor")
        
        with st.expander("Edit Analysis Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_rf_rate = st.slider(
                    "Risk-Free Rate (%)",
                    0.0, 10.0, config.risk_free_rate * 100, 0.1,
                    key="config_rf_rate"
                ) / 100
                
                new_trading_days = st.slider(
                    "Annual Trading Days",
                    200, 365, config.annual_trading_days, 1,
                    key="config_trading_days"
                )
            
            with col2:
                new_backtest_window = st.slider(
                    "Backtest Window (days)",
                    60, 500, config.backtest_window, 10,
                    key="config_backtest_window"
                )
                
                new_mc_sims = st.slider(
                    "Monte Carlo Simulations",
                    1000, 50000, config.monte_carlo_simulations, 1000,
                    key="config_mc_sims"
                )
            
            if st.button("Update Configuration", use_container_width=True):
                config.risk_free_rate = new_rf_rate
                config.annual_trading_days = new_trading_days
                config.backtest_window = new_backtest_window
                config.monte_carlo_simulations = new_mc_sims
                
                st.success("✅ Configuration updated!")
                st.rerun()
        
        # Performance monitoring
        st.markdown("### 📊 Performance Monitoring")
        
        runtime = datetime.now() - self.start_time
        assets_loaded = len(st.session_state.asset_data)
        returns_data_obj = st.session_state.get("returns_data")
        if _is_empty_data(returns_data_obj):
            data_points = 0
        elif isinstance(returns_data_obj, pd.DataFrame):
            # Count rows with at least one non-NaN value
            data_points = int(returns_data_obj.dropna(how="all").shape[0])
        elif isinstance(returns_data_obj, dict):
            first_val = next(iter(returns_data_obj.values()), None)
            data_points = int(len(first_val)) if first_val is not None else 0
        elif isinstance(returns_data_obj, (pd.Series, np.ndarray, list, tuple)):
            data_points = int(len(returns_data_obj))
        else:
            data_points = 0
        perf_cols = st.columns(4)
        
        with perf_cols[0]:
            st.metric("Runtime", f"{runtime.total_seconds():.0f}s")
        
        with perf_cols[1]:
            st.metric("Assets Loaded", assets_loaded)
        
        with perf_cols[2]:
            st.metric("Data Points", f"{data_points:,}")
        
        with perf_cols[3]:
            # Memory usage (simplified)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Error log
        if st.session_state.error_log:
            st.markdown("### ⚠️ Error Log")
            
            with st.expander("View Recent Errors", expanded=False):
                for error in st.session_state.error_log[-10:]:
                    st.markdown(f"""
                    **{error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}** - {error['context']}
                    ```python
                    {error['error'][:200]}
                    ```
                    """)
            
            if st.button("Clear Error Log", use_container_width=True, type="secondary"):
                st.session_state.error_log = []
                st.success("Error log cleared")
                st.rerun()
        
        # Reset and maintenance
        st.markdown("### 🔄 Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear All Cache", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                if hasattr(self.data_manager.cache, '_cleanup_cache'):
                    self.data_manager.cache._cleanup_cache()
                st.success("✅ All cache cleared successfully!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Application", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Application reset!")
                st.rerun()
        
        # Diagnostics
        st.markdown("### 🔧 Diagnostics")
        
        if st.button("Run Diagnostics", use_container_width=True):
            with st.spinner("Running diagnostics..."):
                diagnostics = self._run_diagnostics()
                
                with st.expander("Diagnostics Results", expanded=True):
                    st.json(diagnostics)
        
        # Export system info
        st.markdown("### 💾 Export System Information")
        
        if st.button("Export System Report", use_container_width=True):
            system_info = self._create_system_snapshot()
            system_json = json.dumps(system_info, indent=2)
            
            st.download_button(
                label="Download System Report",
                data=system_json,
                file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    def _run_diagnostics(self) -> Dict[str, Any]:
        """Run system diagnostics"""
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'python': sys.version,
                'streamlit': st.__version__,
                'platform': sys.platform,
                'processor': 'Unknown'
            },
            'dependencies': {},
            'data_health': {},
            'performance': {}
        }
        
        # Check dependencies
        for dep_name, dep_info in dep_manager.dependencies.items():
            diagnostics['dependencies'][dep_name] = {
                'available': dep_info.get('available', False),
                'version': 'Unknown'
            }
        
        # Check data health
        if st.session_state.data_loaded:
            diagnostics['data_health'] = {
                'assets_loaded': len(st.session_state.asset_data),
                'data_points': len(next(iter(st.session_state.returns_data.values()))) 
                             if not _is_empty_data(st.session_state.get("returns_data")) else 0,
                'date_range': {
                    'start': st.session_state.analysis_config.start_date.strftime('%Y-%m-%d'),
                    'end': st.session_state.analysis_config.end_date.strftime('%Y-%m-%d')
                }
            }
        
        # Performance metrics
        diagnostics['performance'] = {
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'cache_hits': 'N/A',  # Would need cache tracking
            'memory_usage': 'Optimized'
        }
        
        # Recommendations
        diagnostics['recommendations'] = []
        
        if not dep_manager.is_available('arch'):
            diagnostics['recommendations'].append("Install 'arch' package for GARCH analysis")
        
        if not dep_manager.is_available('hmmlearn'):
            diagnostics['recommendations'].append("Install 'hmmlearn' package for regime detection")
        
        if st.session_state.get("data_loaded", False) and (isinstance(st.session_state.get("returns_data"), pd.DataFrame) and len(st.session_state.get("returns_data")) < 100):
            diagnostics['recommendations'].append("Consider loading more data for better analysis")
        
        return diagnostics

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
        st.markdown("""
        <style>
            .stAlert { 
                border-radius: 10px; 
                border-left: 5px solid;
            }
            .stButton > button { 
                border-radius: 8px; 
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .stSelectbox, .stMultiselect { 
                border-radius: 8px; 
            }
            .stSlider { 
                border-radius: 8px; 
            }
            .stDataFrame {
                border-radius: 10px;
                overflow: hidden;
            }
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #1a2980, #26d0ce);
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize and run dashboard
        dashboard = InstitutionalCommoditiesDashboard()
        dashboard.run()
        
    except Exception as e:
        # Comprehensive error handling
        st.error(f"""
        ## 🚨 Application Error
        
        An unexpected error occurred in the Institutional Commodities Analytics Platform.
        
        **Error Details:** {str(e)[:500]}
        
        ### 🔧 Troubleshooting Steps:
        1. Refresh the page
        2. Clear your browser cache
        3. Check your internet connection
        4. Try selecting different assets or date ranges
        5. Clear the application cache from Settings
        
        If the problem persists, please contact support with the error details below.
        """)
        
        # Log error for debugging
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'streamlit_version': st.__version__,
            'python_version': sys.version
        }
        
        with st.expander("🔍 Technical Details"):
            st.code(json.dumps(error_log, indent=2), language='json')
        
        # Recovery button
        if st.button("🔄 Restart Application", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
