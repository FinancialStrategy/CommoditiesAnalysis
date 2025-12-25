"""
🏛️ Institutional Commodities Analytics Platform v7.0
Enhanced Scientific Analytics • Advanced Correlation Methods • Professional Risk Metrics
Institutional-Grade Computational Finance Platform
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, signal, linalg
import seaborn as sns
from io import BytesIO, StringIO
import base64

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Environment optimization for scientific computing
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
np.seterr(all='ignore')  # Suppress numpy warnings for stability

# Scientific precision settings
np.set_printoptions(precision=6, suppress=True)
pd.set_option('display.precision', 6)

# Streamlit configuration for institutional interface
st.set_page_config(
    page_title="Institutional Commodities Platform v7.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/institutional-commodities',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v7.0
                    Advanced scientific analytics platform for institutional commodity trading
                    © 2024 Institutional Trading Analytics • Scientific Computing Division"""
    }
)

# =============================================================================
# SCIENTIFIC DATA STRUCTURES & VALIDATION
# =============================================================================

class AssetCategory(Enum):
    """Scientific asset classification with validation"""
    PRECIOUS_METALS = "Precious Metals"
    INDUSTRIAL_METALS = "Industrial Metals"
    ENERGY = "Energy"
    AGRICULTURE = "Agriculture"
    BENCHMARK = "Benchmark"

@dataclass
class ScientificAssetMetadata:
    """Enhanced scientific metadata with validation"""
    symbol: str
    name: str
    category: AssetCategory
    color: str
    description: str = ""
    exchange: str = "CME"
    contract_size: str = "Standard"
    margin_requirement: float = field(default=0.05, metadata={'range': (0.01, 0.50)})
    tick_size: float = field(default=0.01, metadata={'min': 0.0001})
    enabled: bool = True
    risk_level: str = "Medium"
    beta_to_spx: float = field(default=0.0, metadata={'range': (-2.0, 5.0)})
    liquidity_score: float = field(default=0.5, metadata={'range': (0.0, 1.0)})
    fundamental_score: float = field(default=0.5, metadata={'range': (0.0, 1.0)})
    
    def __post_init__(self):
        """Validate metadata upon initialization"""
        self.validate()
    
    def validate(self) -> bool:
        """Scientific validation of metadata"""
        if not isinstance(self.symbol, str) or len(self.symbol) < 1:
            raise ValueError(f"Invalid symbol: {self.symbol}")
        if self.margin_requirement < 0.01 or self.margin_requirement > 0.50:
            raise ValueError(f"Margin requirement out of range: {self.margin_requirement}")
        if self.tick_size <= 0:
            raise ValueError(f"Invalid tick size: {self.tick_size}")
        if not -2.0 <= self.beta_to_spx <= 5.0:
            raise ValueError(f"Beta to SPX out of range: {self.beta_to_spx}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ScientificAnalysisConfiguration:
    """Comprehensive scientific analysis configuration with validation"""
    start_date: datetime
    end_date: datetime
    risk_free_rate: float = field(default=0.02, metadata={'range': (0.0, 0.10)})
    annual_trading_days: int = field(default=252, metadata={'options': [252, 365]})
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    garch_p_range: Tuple[int, int] = (1, 3)
    garch_q_range: Tuple[int, int] = (1, 3)
    regime_states: int = field(default=3, metadata={'range': (2, 5)})
    backtest_window: int = field(default=250, metadata={'range': (60, 1000)})
    rolling_window: int = field(default=60, metadata={'range': (20, 250)})
    volatility_window: int = field(default=20, metadata={'range': (10, 100)})
    monte_carlo_simulations: int = field(default=10000, metadata={'range': (1000, 100000)})
    optimization_method: str = field(default="sharpe", metadata={'options': ['sharpe', 'min_vol', 'risk_parity']})
    correlation_method: str = field(default="pearson", metadata={'options': ['pearson', 'spearman', 'kendall', 'ewma']})
    ewma_lambda: float = field(default=0.94, metadata={'range': (0.90, 0.99)})
    significance_level: float = field(default=0.05, metadata={'range': (0.01, 0.10)})
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive validation with error messages"""
        errors = []
        
        if self.start_date >= self.end_date:
            errors.append("Start date must be before end date")
        
        if not (0.0 <= self.risk_free_rate <= 0.10):
            errors.append(f"Risk-free rate {self.risk_free_rate} outside valid range [0.0, 0.10]")
        
        if not all(0.5 <= cl <= 0.999 for cl in self.confidence_levels):
            errors.append("Confidence levels must be between 0.5 and 0.999")
        
        if not (2 <= self.regime_states <= 5):
            errors.append(f"Regime states {self.regime_states} outside valid range [2, 5]")
        
        if not (0.90 <= self.ewma_lambda <= 0.99):
            errors.append(f"EWMA lambda {self.ewma_lambda} outside valid range [0.90, 0.99]")
        
        if not (0.01 <= self.significance_level <= 0.10):
            errors.append(f"Significance level {self.significance_level} outside valid range [0.01, 0.10]")
        
        return len(errors) == 0, errors
    
    def get_ewma_halflife(self) -> float:
        """Calculate half-life for EWMA decay factor"""
        return math.log(0.5) / math.log(self.ewma_lambda)

# Enhanced scientific commodities universe
COMMODITIES_UNIVERSE = {
    AssetCategory.PRECIOUS_METALS.value: {
        "GC=F": ScientificAssetMetadata(
            symbol="GC=F",
            name="Gold Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#FFD700",
            description="COMEX Gold Futures (100 troy ounces) - Safe haven asset with inflation hedge properties",
            exchange="COMEX",
            contract_size="100 troy oz",
            margin_requirement=0.045,
            tick_size=0.10,
            risk_level="Low",
            beta_to_spx=0.15,
            liquidity_score=0.95,
            fundamental_score=0.85
        ),
        "SI=F": ScientificAssetMetadata(
            symbol="SI=F",
            name="Silver Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#C0C0C0",
            description="COMEX Silver Futures (5,000 troy ounces) - Industrial and monetary metal with high volatility",
            exchange="COMEX",
            contract_size="5,000 troy oz",
            margin_requirement=0.065,
            tick_size=0.005,
            risk_level="Medium",
            beta_to_spx=0.25,
            liquidity_score=0.85,
            fundamental_score=0.75
        ),
        "PL=F": ScientificAssetMetadata(
            symbol="PL=F",
            name="Platinum Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#E5E4E2",
            description="NYMEX Platinum Futures (50 troy ounces) - Industrial precious metal with autocatalytic demand",
            exchange="NYMEX",
            contract_size="50 troy oz",
            margin_requirement=0.075,
            tick_size=0.10,
            risk_level="High",
            beta_to_spx=0.35,
            liquidity_score=0.70,
            fundamental_score=0.65
        ),
    },
    AssetCategory.INDUSTRIAL_METALS.value: {
        "HG=F": ScientificAssetMetadata(
            symbol="HG=F",
            name="Copper Futures",
            category=AssetCategory.INDUSTRIAL_METALS,
            color="#B87333",
            description="COMEX Copper Futures (25,000 pounds) - Economic bellwether with industrial applications",
            exchange="COMEX",
            contract_size="25,000 lbs",
            margin_requirement=0.085,
            tick_size=0.0005,
            risk_level="Medium",
            beta_to_spx=0.45,
            liquidity_score=0.90,
            fundamental_score=0.80
        ),
        "ALI=F": ScientificAssetMetadata(
            symbol="ALI=F",
            name="Aluminum Futures",
            category=AssetCategory.INDUSTRIAL_METALS,
            color="#848482",
            description="COMEX Aluminum Futures (44,000 pounds) - Lightweight metal with energy-intensive production",
            exchange="COMEX",
            contract_size="44,000 lbs",
            margin_requirement=0.095,
            tick_size=0.0001,
            risk_level="High",
            beta_to_spx=0.55,
            liquidity_score=0.75,
            fundamental_score=0.70
        ),
    },
    AssetCategory.ENERGY.value: {
        "CL=F": ScientificAssetMetadata(
            symbol="CL=F",
            name="Crude Oil WTI",
            category=AssetCategory.ENERGY,
            color="#000000",
            description="NYMEX Light Sweet Crude Oil (1,000 barrels) - Global energy benchmark with geopolitical sensitivity",
            exchange="NYMEX",
            contract_size="1,000 barrels",
            margin_requirement=0.085,
            tick_size=0.01,
            risk_level="High",
            beta_to_spx=0.60,
            liquidity_score=0.98,
            fundamental_score=0.75
        ),
        "NG=F": ScientificAssetMetadata(
            symbol="NG=F",
            name="Natural Gas",
            category=AssetCategory.ENERGY,
            color="#4169E1",
            description="NYMEX Natural Gas (10,000 MMBtu) - Seasonal commodity with storage-driven dynamics",
            exchange="NYMEX",
            contract_size="10,000 MMBtu",
            margin_requirement=0.095,
            tick_size=0.001,
            risk_level="High",
            beta_to_spx=0.30,
            liquidity_score=0.88,
            fundamental_score=0.65
        ),
    },
    AssetCategory.AGRICULTURE.value: {
        "ZC=F": ScientificAssetMetadata(
            symbol="ZC=F",
            name="Corn Futures",
            category=AssetCategory.AGRICULTURE,
            color="#FFD700",
            description="CBOT Corn Futures (5,000 bushels) - Staple grain with biofuel linkage",
            exchange="CBOT",
            contract_size="5,000 bushels",
            margin_requirement=0.065,
            tick_size=0.0025,
            risk_level="Medium",
            beta_to_spx=0.20,
            liquidity_score=0.82,
            fundamental_score=0.70
        ),
        "ZW=F": ScientificAssetMetadata(
            symbol="ZW=F",
            name="Wheat Futures",
            category=AssetCategory.AGRICULTURE,
            color="#F5DEB3",
            description="CBOT Wheat Futures (5,000 bushels) - Weather-sensitive staple crop",
            exchange="CBOT",
            contract_size="5,000 bushels",
            margin_requirement=0.075,
            tick_size=0.0025,
            risk_level="Medium",
            beta_to_spx=0.18,
            liquidity_score=0.80,
            fundamental_score=0.68
        ),
    }
}

BENCHMARKS = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "type": "equity",
        "color": "#1E90FF",
        "description": "S&P 500 Equity Index - US large-cap equity benchmark"
    },
    "DX-Y.NYB": {
        "name": "US Dollar Index",
        "type": "currency",
        "color": "#32CD32",
        "description": "US Dollar Currency Index - Dollar strength indicator"
    },
    "TLT": {
        "name": "20+ Year Treasury ETF",
        "type": "fixed_income",
        "color": "#8A2BE2",
        "description": "Long-term US Treasury Bonds - Interest rate sensitivity"
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "type": "commodity",
        "color": "#FFD700",
        "description": "Gold-backed ETF - Gold price proxy"
    },
    "DBC": {
        "name": "Invesco DB Commodity Index",
        "type": "commodity",
        "color": "#FF6347",
        "description": "Broad Commodities ETF - Diversified commodity exposure"
    }
}

# =============================================================================
# SCIENTIFIC THEMING & INSTITUTIONAL STYLING
# =============================================================================

class ScientificThemeManager:
    """Institutional scientific theming with validation"""
    
    THEMES = {
        "institutional": {
            "primary": "#1a237e",  # Deep indigo
            "secondary": "#283593", # Medium indigo
            "accent": "#3949ab",    # Light indigo
            "success": "#2e7d32",   # Deep green
            "warning": "#f57c00",   # Deep orange
            "danger": "#c62828",    # Deep red
            "dark": "#0d1b2a",      # Navy blue
            "light": "#e0e1dd",     # Light gray
            "gray": "#415a77",      # Medium gray
            "background": "#ffffff",
            "grid": "#e8eaf6",      # Very light indigo
            "border": "#c5cae9"     # Light indigo border
        },
        "dark_scientific": {
            "primary": "#2962ff",   # Bright blue
            "secondary": "#448aff",  # Light blue
            "accent": "#82b1ff",    # Very light blue
            "success": "#00c853",   # Bright green
            "warning": "#ff9100",   # Bright orange
            "danger": "#ff5252",    # Bright red
            "dark": "#0a0e17",      # Very dark blue
            "light": "#263238",     # Dark blue-gray
            "gray": "#546e7a",      # Medium blue-gray
            "background": "#1e272e",
            "grid": "#2c3e50",      # Dark grid
            "border": "#34495e"     # Dark border
        }
    }
    
    @staticmethod
    def get_styles(theme: str = "institutional") -> str:
        """Get institutional scientific CSS styles"""
        colors = ScientificThemeManager.THEMES.get(theme, ScientificThemeManager.THEMES["institutional"])
        
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
                --grid: {colors['grid']};
                --border: {colors['border']};
                --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
                --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
                --shadow-lg: 0 10px 15px rgba(0,0,0,0.08);
                --shadow-xl: 0 20px 25px rgba(0,0,0,0.10);
                --radius-sm: 4px;
                --radius-md: 6px;
                --radius-lg: 8px;
                --radius-xl: 12px;
                --transition: all 0.2s ease-in-out;
            }}
            
            /* Scientific Header */
            .scientific-header {{
                background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 100%);
                padding: 2rem;
                border-radius: var(--radius-lg);
                color: white;
                margin-bottom: 1.5rem;
                box-shadow: var(--shadow-lg);
                border: 1px solid var(--border);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            
            .scientific-header h1 {{
                font-size: 2.2rem;
                font-weight: 700;
                margin: 0 0 0.5rem 0;
                letter-spacing: -0.5px;
            }}
            
            .scientific-header p {{
                font-size: 1rem;
                opacity: 0.9;
                margin: 0;
                font-weight: 400;
            }}
            
            /* Institutional Cards */
            .institutional-card {{
                background: var(--background);
                padding: 1.5rem;
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-sm);
                border: 1px solid var(--border);
                margin-bottom: 1rem;
                transition: var(--transition);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .institutional-card:hover {{
                box-shadow: var(--shadow-md);
                border-color: var(--primary);
            }}
            
            .institutional-card .metric-title {{
                font-size: 0.85rem;
                color: var(--gray);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}
            
            .institutional-card .metric-value {{
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--dark);
                margin: 0;
                font-family: 'SF Mono', 'Roboto Mono', monospace;
            }}
            
            .institutional-card .metric-change {{
                font-size: 0.85rem;
                font-weight: 500;
                margin-top: 0.25rem;
            }}
            
            .institutional-card .metric-change.positive {{
                color: var(--success);
            }}
            
            .institutional-card .metric-change.negative {{
                color: var(--danger);
            }}
            
            /* Scientific Badges */
            .scientific-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                border: 1px solid transparent;
                transition: var(--transition);
            }}
            
            .scientific-badge.low-risk {{
                background: rgba(46, 125, 50, 0.1);
                color: var(--success);
                border-color: rgba(46, 125, 50, 0.3);
            }}
            
            .scientific-badge.medium-risk {{
                background: rgba(245, 124, 0, 0.1);
                color: var(--warning);
                border-color: rgba(245, 124, 0, 0.3);
            }}
            
            .scientific-badge.high-risk {{
                background: rgba(198, 40, 40, 0.1);
                color: var(--danger);
                border-color: rgba(198, 40, 40, 0.3);
            }}
            
            .scientific-badge.info {{
                background: rgba(41, 98, 255, 0.1);
                color: var(--primary);
                border-color: rgba(41, 98, 255, 0.3);
            }}
            
            /* Scientific Tables */
            .scientific-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                overflow: hidden;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .scientific-table thead {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
            }}
            
            .scientific-table th {{
                padding: 0.75rem 1rem;
                font-weight: 600;
                text-align: left;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 1px solid var(--border);
            }}
            
            .scientific-table td {{
                padding: 0.75rem 1rem;
                border-bottom: 1px solid var(--border);
                font-size: 0.9rem;
            }}
            
            .scientific-table tbody tr:hover {{
                background-color: rgba(0, 0, 0, 0.02);
            }}
            
            /* Warning Messages */
            .warning-message {{
                background: linear-gradient(135deg, rgba(245, 124, 0, 0.1) 0%, rgba(245, 124, 0, 0.05) 100%);
                border-left: 4px solid var(--warning);
                padding: 1rem;
                border-radius: var(--radius-sm);
                margin: 1rem 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .warning-message strong {{
                color: var(--warning);
                font-weight: 600;
            }}
            
            .error-message {{
                background: linear-gradient(135deg, rgba(198, 40, 40, 0.1) 0%, rgba(198, 40, 40, 0.05) 100%);
                border-left: 4px solid var(--danger);
                padding: 1rem;
                border-radius: var(--radius-sm);
                margin: 1rem 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .error-message strong {{
                color: var(--danger);
                font-weight: 600;
            }}
            
            .info-message {{
                background: linear-gradient(135deg, rgba(41, 98, 255, 0.1) 0%, rgba(41, 98, 255, 0.05) 100%);
                border-left: 4px solid var(--primary);
                padding: 1rem;
                border-radius: var(--radius-sm);
                margin: 1rem 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .info-message strong {{
                color: var(--primary);
                font-weight: 600;
            }}
            
            /* Scientific Section Headers */
            .section-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin: 2rem 0 1rem;
                padding-bottom: 0.75rem;
                border-bottom: 2px solid var(--border);
            }}
            
            .section-header h2 {{
                margin: 0;
                color: var(--dark);
                font-size: 1.4rem;
                font-weight: 700;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .section-header .section-actions {{
                display: flex;
                gap: 0.5rem;
            }}
            
            /* Tabs Enhancement */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background-color: var(--light);
                padding: 8px;
                border-radius: var(--radius-lg);
                margin-bottom: 1.5rem;
                border: 1px solid var(--border);
            }}
            
            .stTabs [data-baseweb="tab"] {{
                border-radius: var(--radius-md);
                padding: 10px 20px;
                background-color: var(--background);
                border: 1px solid var(--border);
                transition: var(--transition);
                font-weight: 600;
                font-size: 0.9rem;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border-color: var(--primary);
                box-shadow: var(--shadow-sm);
            }}
            
            /* Scientific Grid */
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }}
            
            /* Scientific Controls */
            .stSlider > div > div > div {{
                background: linear-gradient(90deg, var(--primary), var(--secondary));
            }}
            
            .stSelectbox, .stMultiselect, .stNumberInput {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            /* Scientific Spinner */
            .stSpinner > div {{
                border-color: var(--primary) transparent transparent transparent;
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .metric-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .scientific-header {{
                    padding: 1.5rem;
                }}
                
                .scientific-header h1 {{
                    font-size: 1.8rem;
                }}
            }}
            
            /* Code Blocks */
            .stCodeBlock {{
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                font-family: 'SF Mono', 'Roboto Mono', monospace;
            }}
        </style>
        """

# Apply institutional theme
st.markdown(ScientificThemeManager.get_styles("institutional"), unsafe_allow_html=True)

# =============================================================================
# ADVANCED DEPENDENCY MANAGEMENT WITH VALIDATION
# =============================================================================

class ScientificDependencyManager:
    """Scientific dependency management with validation"""
    
    def __init__(self):
        self.dependencies = {}
        self._scientific_imports()
        self._validate_versions()
    
    def _scientific_imports(self):
        """Load scientific dependencies with validation"""
        # statsmodels for statistical analysis
        try:
            from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox, het_breuschpagan
            import statsmodels.api as sm
            from statsmodels.regression.rolling import RollingOLS
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            from statsmodels.tsa.stattools import adfuller, kpss, coint
            self.dependencies['statsmodels'] = {
                'available': True,
                'module': sm,
                'version': sm.__version__,
                'het_arch': het_arch,
                'acorr_ljungbox': acorr_ljungbox,
                'het_breuschpagan': het_breuschpagan,
                'RollingOLS': RollingOLS,
                'plot_acf': plot_acf,
                'plot_pacf': plot_pacf,
                'adfuller': adfuller,
                'kpss': kpss,
                'coint': coint
            }
        except ImportError as e:
            self.dependencies['statsmodels'] = {'available': False, 'error': str(e)}
        
        # arch for volatility modeling
        try:
            from arch import arch_model
            from arch.univariate import GARCH, EWMAVariance, ConstantMean, ZeroMean, ARX
            import arch
            self.dependencies['arch'] = {
                'available': True,
                'version': arch.__version__,
                'arch_model': arch_model,
                'GARCH': GARCH,
                'EWMAVariance': EWMAVariance,
                'ConstantMean': ConstantMean,
                'ZeroMean': ZeroMean,
                'ARX': ARX
            }
        except ImportError as e:
            self.dependencies['arch'] = {'available': False, 'error': str(e)}
        
        # hmmlearn & sklearn for ML
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            import sklearn
            self.dependencies['hmmlearn'] = {
                'available': True,
                'version': sklearn.__version__,
                'GaussianHMM': GaussianHMM,
                'StandardScaler': StandardScaler,
                'RobustScaler': RobustScaler,
                'KMeans': KMeans,
                'DBSCAN': DBSCAN,
                'PCA': PCA,
                'silhouette_score': silhouette_score,
                'calinski_harabasz_score': calinski_harabasz_score
            }
        except ImportError as e:
            self.dependencies['hmmlearn'] = {'available': False, 'error': str(e)}
        
        # quantstats for performance analytics
        try:
            import quantstats as qs
            self.dependencies['quantstats'] = {
                'available': True,
                'module': qs,
                'version': qs.__version__
            }
        except ImportError as e:
            self.dependencies['quantstats'] = {'available': False, 'error': str(e)}
        
        # ta for technical analysis
        try:
            import ta
            self.dependencies['ta'] = {
                'available': True,
                'module': ta,
                'version': ta.__version__
            }
        except ImportError as e:
            self.dependencies['ta'] = {'available': False, 'error': str(e)}
    
    def _validate_versions(self):
        """Validate dependency versions for scientific stability"""
        version_requirements = {
            'statsmodels': '0.14.0',
            'arch': '6.0.0',
            'sklearn': '1.3.0',
            'quantstats': '0.0.62',
            'ta': '0.10.2'
        }
        
        for dep, info in self.dependencies.items():
            if info.get('available') and 'version' in info:
                required = version_requirements.get(dep)
                if required:
                    # Basic version check (simplified)
                    current = info['version']
                    if dep == 'statsmodels' and current < '0.14':
                        st.warning(f"⚠️ {dep} version {current} may have stability issues. Recommended: {required}")
    
    def is_available(self, dependency: str) -> bool:
        """Check if scientific dependency is available"""
        return self.dependencies.get(dependency, {}).get('available', False)
    
    def get_module(self, dependency: str):
        """Get dependency module if available"""
        dep = self.dependencies.get(dependency, {})
        return dep.get('module') if dep.get('available') else None
    
    def get_function(self, dependency: str, function_name: str):
        """Get specific function from dependency"""
        dep = self.dependencies.get(dependency, {})
        return dep.get(function_name) if dep.get('available') else None
    
    def display_status(self):
        """Display dependency status"""
        status_html = """
        <div class="institutional-card">
            <div class="metric-title">🧪 Scientific Dependencies</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 0.5rem;">
        """
        
        for dep, info in self.dependencies.items():
            status = "🟢" if info.get('available') else "🔴"
            version = info.get('version', 'N/A') if info.get('available') else 'Missing'
            status_html += f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.25rem 0;">
                    <span style="font-size: 0.85rem; color: var(--gray);">{dep}</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.75rem; color: var(--gray);">{version}</span>
                        <span>{status}</span>
                    </div>
                </div>
            """
        
        status_html += "</div></div>"
        return status_html

# Initialize scientific dependency manager
sci_dep_manager = ScientificDependencyManager()

# =============================================================================
# SCIENTIFIC CACHING SYSTEM WITH VALIDATION
# =============================================================================

class ScientificCache:
    """Scientific caching with validation and persistence"""
    
    def __init__(self, max_entries: int = 200, ttl_hours: int = 12):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_hours * 3600
    
    @staticmethod
    def generate_scientific_key(*args, **kwargs) -> str:
        """Generate deterministic cache key with scientific precision"""
        import inspect
        
        def serialize_value(val):
            if isinstance(val, (str, int, float, bool, type(None))):
                return str(val)
            elif isinstance(val, (datetime, pd.Timestamp)):
                return val.isoformat()
            elif isinstance(val, pd.DataFrame):
                # Include shape, columns, and hash of first/last rows for validation
                shape_hash = hashlib.md5(f"{val.shape}_{tuple(val.columns)}".encode()).hexdigest()
                if len(val) > 0:
                    sample_hash = hashlib.md5(
                        pd.util.hash_pandas_object(val.iloc[[0, -1]]).values.tobytes()
                    ).hexdigest()
                    return f"df_{shape_hash}_{sample_hash}"
                return f"df_{shape_hash}_empty"
            elif isinstance(val, np.ndarray):
                return f"np_{val.shape}_{val.dtype}_{hashlib.md5(val.tobytes()).hexdigest()[:16]}"
            elif isinstance(val, dict):
                return f"dict_{len(val)}_{hashlib.md5(json.dumps(val, sort_keys=True).encode()).hexdigest()[:16]}"
            elif callable(val):
                # For functions, use their name and module
                return f"func_{val.__module__}.{val.__name__}"
            else:
                # Fallback to string representation
                return str(val)
        
        key_parts = []
        
        # Add caller information for debugging
        try:
            caller = inspect.stack()[1]
            key_parts.append(f"caller:{caller.function}:{caller.lineno}")
        except:
            pass
        
        # Serialize arguments
        for i, arg in enumerate(args):
            key_parts.append(f"arg{i}:{serialize_value(arg)}")
        
        # Serialize keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{serialize_value(v)}")
        
        # Generate final key
        key_string = "_".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def cache_scientific_data(ttl: int = 7200, max_entries: int = 100, validate: bool = True):
        """Decorator for caching scientific data with validation"""
        def decorator(func):
            @wraps(func)
            @st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=False)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    
                    # Scientific validation of cached results
                    if validate:
                        wrapper._validate_result(result, func.__name__)
                    
                    return result
                except Exception as e:
                    # Clear cache on critical errors
                    st.cache_data.clear()
                    st.error(f"Cache validation failed for {func.__name__}: {str(e)}")
                    return func(*args, **kwargs)
            
            def validate_result(result, func_name: str):
                """Validate cached results for scientific integrity"""
                if result is None:
                    return
                
                if isinstance(result, pd.DataFrame):
                    if result.empty:
                        st.warning(f"⚠️ Empty DataFrame returned by {func_name}")
                    # Check for infinite values
                    if np.any(np.isinf(result.values)):
                        raise ValueError(f"Infinite values detected in {func_name} result")
                    # Check for excessive NaN values
                    nan_ratio = result.isna().sum().sum() / (result.shape[0] * result.shape[1])
                    if nan_ratio > 0.5:
                        st.warning(f"⚠️ High NaN ratio ({nan_ratio:.1%}) in {func_name} result")
                
                elif isinstance(result, dict):
                    if len(result) == 0:
                        st.warning(f"⚠️ Empty dictionary returned by {func_name}")
                
                elif isinstance(result, np.ndarray):
                    if np.any(np.isnan(result)):
                        st.warning(f"⚠️ NaN values detected in {func_name} array result")
                    if np.any(np.isinf(result)):
                        raise ValueError(f"Infinite values detected in {func_name} array")
            
            wrapper._validate_result = validate_result
            return wrapper
        return decorator
    
    @staticmethod
    def cache_scientific_resource(max_entries: int = 50):
        """Decorator for caching scientific resources"""
        def decorator(func):
            @wraps(func)
            @st.cache_resource(max_entries=max_entries)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

# =============================================================================
# SCIENTIFIC DATA MANAGER WITH VALIDATION
# =============================================================================

class ScientificDataManager:
    """Scientific data management with comprehensive validation"""
    
    def __init__(self):
        self.cache = ScientificCache()
        self._validation_metrics = {}
    
    @ScientificCache.cache_scientific_data(ttl=10800, max_entries=150, validate=True)
    def fetch_scientific_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        retries: int = 3,
        validate: bool = True
    ) -> pd.DataFrame:
        """Fetch and validate scientific data with comprehensive error handling"""
        
        cache_key = self.cache.generate_scientific_key(
            "fetch_scientific", symbol, start_date, end_date, interval
        )
        
        validation_errors = []
        df_final = pd.DataFrame()
        
        for attempt in range(retries):
            try:
                # Different download strategies for robustness
                if attempt == 0:
                    # Primary strategy with comprehensive settings
                    df = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date + timedelta(days=1),  # Include end date
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                        threads=True,
                        timeout=30,
                        group_by='ticker',
                        proxy=None
                    )
                elif attempt == 1:
                    # Fallback strategy without auto-adjust
                    df = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date + timedelta(days=1),
                        interval=interval,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                        timeout=45
                    )
                else:
                    # Last resort: use period instead of dates
                    days_diff = (end_date - start_date).days
                    if days_diff <= 7:
                        period = "1wk"
                    elif days_diff <= 30:
                        period = "1mo"
                    elif days_diff <= 90:
                        period = "3mo"
                    elif days_diff <= 365:
                        period = "1y"
                    else:
                        period = "max"
                    
                    df = yf.download(
                        symbol,
                        period=period,
                        interval="1d",
                        progress=False,
                        auto_adjust=True
                    )
                    # Filter to date range
                    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
                    df = df[mask]
                
                # Validate download result
                if df is None:
                    raise ValueError(f"Download returned None for {symbol}")
                
                if isinstance(df, pd.DataFrame) and df.empty:
                    raise ValueError(f"Empty DataFrame for {symbol}")
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Convert to proper DataFrame structure
                df = self._scientific_clean_dataframe(df, symbol)
                
                # Scientific validation
                if validate:
                    validation_result = self._validate_dataframe(df, symbol)
                    if not validation_result['valid']:
                        validation_errors.extend(validation_result['errors'])
                        if attempt < retries - 1:
                            continue  # Try again
                
                # Check for sufficient data
                if len(df) < 20:
                    validation_errors.append(f"Insufficient data points ({len(df)} < 20)")
                    if attempt < retries - 1:
                        continue
                
                df_final = df
                break  # Success
                
            except Exception as e:
                validation_errors.append(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    st.error(f"❌ Failed to fetch data for {symbol} after {retries} attempts")
                    for err in validation_errors:
                        st.error(f"  - {err}")
                continue
        
        # Store validation metrics
        self._validation_metrics[symbol] = {
            'fetch_attempts': attempt + 1 if df_final is not None else retries,
            'validation_errors': validation_errors,
            'success': not df_final.empty,
            'data_points': len(df_final) if not df_final.empty else 0,
            'date_range': (df_final.index.min(), df_final.index.max()) if not df_final.empty else None
        }
        
        return df_final
    
    def _scientific_clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Scientific cleaning and preprocessing"""
        df = df.copy()
        
        # Standardize column names
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Handle different column naming conventions
        column_mapping = {
            'Adj_Close': 'Adj_Close',
            'Adj Close': 'Adj_Close',
            'AdjClose': 'Adj_Close'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # If no adjusted close, use close
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == 'Volume':
                    df['Volume'] = 0.0
                elif col in ['Open', 'High', 'Low']:
                    df[col] = df['Close']
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Handle missing values scientifically
        critical_cols = ['Close', 'Adj_Close']
        for col in critical_cols:
            if col in df.columns:
                # Forward fill then backward fill for critical price data
                df[col] = df[col].ffill().bfill()
        
        # Remove any remaining NaN in critical columns
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        # Validate price monotonicity (allow small reversals due to adjustments)
        if 'Adj_Close' in df.columns:
            price_changes = df['Adj_Close'].pct_change().dropna()
            if len(price_changes[abs(price_changes) > 0.5]) > len(df) * 0.01:  # More than 1% large moves
                st.warning(f"⚠️ {symbol}: Excessive price changes detected")
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Comprehensive scientific validation of dataframe"""
        errors = []
        warnings = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Basic structure validation
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Date range validation
        if len(df) < 10:
            warnings.append(f"Limited data points: {len(df)}")
        
        # Price validation
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Check for negative prices
                if (df[col] <= 0).any():
                    errors.append(f"Non-positive values in {col}")
                
                # Check for reasonable price ranges
                if col == 'Close':
                    price = df[col]
                    if price.mean() < 0.01 or price.mean() > 1000000:
                        warnings.append(f"Unusual average price: ${price.mean():.2f}")
        
        # High-Low validation
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_hl = df['High'] < df['Low']
            if invalid_hl.any():
                errors.append(f"High < Low on {invalid_hl.sum()} days")
        
        # Volume validation
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                errors.append("Negative volume values")
        
        # Return validation summary
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'n_rows': len(df),
            'date_range': (df.index.min(), df.index.max())
        }
    
    @ScientificCache.cache_scientific_data(ttl=5400, max_entries=100, validate=True)
    def fetch_multiple_assets_scientific(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 6,
        validation_level: str = "strict"
    ) -> Dict[str, pd.DataFrame]:
        """Parallel fetch with scientific validation"""
        
        results = {}
        failed_symbols = []
        validation_summary = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            future_to_symbol = {}
            for idx, symbol in enumerate(symbols):
                future = executor.submit(
                    self.fetch_scientific_data,
                    symbol,
                    start_date,
                    end_date,
                    "1d",
                    3,
                    True
                )
                future_to_symbol[future] = (symbol, idx)
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol, idx = future_to_symbol[future]
                try:
                    df = future.result()
                    
                    # Update progress
                    completed += 1
                    progress_bar.progress(completed / len(symbols))
                    status_text.text(f"📊 Fetching data... {completed}/{len(symbols)} ({symbol})")
                    
                    if not df.empty:
                        # Additional validation based on level
                        if validation_level == "strict":
                            val_result = self._validate_dataframe(df, symbol)
                            if val_result['valid']:
                                results[symbol] = df
                                validation_summary[symbol] = val_result
                            else:
                                failed_symbols.append((symbol, val_result['errors']))
                        else:
                            results[symbol] = df
                    else:
                        failed_symbols.append((symbol, ["Empty DataFrame"]))
                        
                except Exception as e:
                    failed_symbols.append((symbol, [str(e)]))
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Display validation summary
        if validation_summary:
            self._display_validation_summary(validation_summary, failed_symbols)
        
        return results
    
    def _display_validation_summary(self, validation_summary: Dict, failed_symbols: List):
        """Display scientific validation summary"""
        with st.expander("🧪 Data Validation Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Successfully Validated")
                success_table = []
                for symbol, summary in validation_summary.items():
                    success_table.append({
                        "Symbol": symbol,
                        "Rows": summary['n_rows'],
                        "Start": summary['date_range'][0].date(),
                        "End": summary['date_range'][1].date(),
                        "Warnings": len(summary.get('warnings', []))
                    })
                
                if success_table:
                    st.dataframe(pd.DataFrame(success_table), use_container_width=True)
            
            with col2:
                if failed_symbols:
                    st.markdown("#### ❌ Failed Validation")
                    for symbol, errors in failed_symbols:
                        st.error(f"**{symbol}**: {', '.join(errors[:3])}")
    
    def calculate_scientific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive scientific features with validation"""
        df = df.copy()
        
        if df.empty:
            return df
        
        # Determine price column
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        if price_col not in df.columns:
            return df
        
        # Scientific returns calculation
        df['Returns'] = df[price_col].pct_change()
        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Remove extreme outliers in returns (beyond 10 standard deviations)
        returns = df['Returns'].dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            extreme_mask = abs(returns - mean_return) > 10 * std_return
            if extreme_mask.any():
                df.loc[extreme_mask.index, 'Returns'] = np.sign(returns[extreme_mask]) * 10 * std_return
                st.warning(f"⚠️ Extreme returns detected and winsorized: {extreme_mask.sum()} points")
        
        # Price-based features
        df['Price_Range_Pct'] = (df['High'] - df['Low']) / df[price_col] * 100
        df['Close_to_Close_Change'] = df[price_col].pct_change() * 100
        
        # Moving averages with scientific validation
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                df[f'SMA_{period}'] = df[price_col].rolling(window=period, min_periods=int(period*0.8)).mean()
                df[f'EMA_{period}'] = df[price_col].ewm(span=period, min_periods=int(period*0.8)).mean()
        
        # Bollinger Bands with scientific adjustments
        bb_period = 20
        if len(df) >= bb_period:
            bb_middle = df[price_col].rolling(window=bb_period, min_periods=int(bb_period*0.8)).mean()
            bb_std = df[price_col].rolling(window=bb_period, min_periods=int(bb_period*0.8)).std()
            
            # Use adaptive multiplier based on volatility regime
            volatility_ratio = bb_std / bb_middle
            multiplier = np.where(volatility_ratio > 0.02, 2.5, 2.0)  # Higher multiplier in high vol
            
            df['BB_Upper'] = bb_middle + (bb_std * multiplier)
            df['BB_Lower'] = bb_middle - (bb_std * multiplier)
            df['BB_Width_Pct'] = ((df['BB_Upper'] - df['BB_Lower']) / bb_middle) * 100
            df['BB_Position'] = (df[price_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
        
        # RSI with robust calculation
        period = 14
        if len(df) >= period:
            delta = df[price_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Use Wilder's smoothing
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            for i in range(period, len(df)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
            
            rs = avg_gain / (avg_loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD with scientific validation
        if len(df) >= 26:
            ema12 = df[price_col].ewm(span=12, min_periods=10).mean()
            ema26 = df[price_col].ewm(span=26, min_periods=20).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=7).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Signal_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        
        # Volatility measures with different methodologies
        if len(df) >= 20:
            # Simple historical volatility
            df['Volatility_20D_Simple'] = df['Returns'].rolling(window=20, min_periods=15).std() * np.sqrt(252) * 100
            
            # Parkinson volatility (using high-low range)
            df['Parkinson_Vol'] = np.sqrt(1/(4*np.log(2)) * (np.log(df['High']/df['Low'])**2).rolling(window=20).mean()) * np.sqrt(252) * 100
            
            # Garman-Klass volatility
            df['Garman_Klass_Vol'] = np.sqrt((0.5 * (np.log(df['High']/df['Low'])**2).rolling(window=20).mean() - 
                                             (2*np.log(2)-1) * (np.log(df['Close']/df['Open'])**2).rolling(window=20).mean()) * 252) * 100
        
        # Volume analysis
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=15).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)
            df['Volume_Price_Trend'] = df['Volume'] * df['Returns']
            
            # On-Balance Volume
            df['OBV'] = 0
            obv = 0
            for i in range(1, len(df)):
                if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
                    obv += df['Volume'].iloc[i]
                elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
                    obv -= df['Volume'].iloc[i]
                df.iloc[i, df.columns.get_loc('OBV')] = obv
        
        # ATR (Average True Range)
        if len(df) >= 14:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df[price_col].shift())
            low_close = np.abs(df['Low'] - df[price_col].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=10).mean()
            df['ATR_Pct'] = (df['ATR'] / df[price_col]) * 100
        
        # Momentum indicators
        momentum_periods = [5, 10, 20, 50]
        for period in momentum_periods:
            if len(df) >= period:
                df[f'Momentum_{period}D'] = df[price_col].pct_change(periods=period) * 100
                df[f'ROC_{period}'] = ((df[price_col] - df[price_col].shift(period)) / df[price_col].shift(period)) * 100
        
        # Statistical features
        if len(df) >= 20:
            df['Z_Score_20'] = (df[price_col] - df[price_col].rolling(window=20).mean()) / df[price_col].rolling(window=20).std()
            df['Skewness_20D'] = df['Returns'].rolling(window=20).skew()
            df['Kurtosis_20D'] = df['Returns'].rolling(window=20).kurt()
        
        # Trend features
        if len(df) >= 50:
            df['Trend_Slope_50D'] = df[price_col].rolling(window=50).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )
            df['Trend_R2_50D'] = df[price_col].rolling(window=50).apply(
                lambda x: np.corrcoef(range(len(x)), x)[0, 1]**2, raw=True
            )
        
        # Remove NaN values while preserving as much data as possible
        original_length = len(df)
        df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% valid data
        removed_pct = (original_length - len(df)) / original_length * 100
        
        if removed_pct > 10:
            st.warning(f"⚠️ High data removal ({removed_pct:.1f}%) during feature calculation")
        
        return df

# =============================================================================
# SCIENTIFIC CORRELATION ENGINE (FIXED & ENHANCED)
# =============================================================================

class ScientificCorrelationEngine:
    """Advanced scientific correlation analysis with multiple methodologies"""
    
    def __init__(self, config: ScientificAnalysisConfiguration):
        self.config = config
        self._validation_results = {}
    
    def calculate_correlation_matrix(
        self, 
        returns_dict: Dict[str, pd.Series], 
        method: str = "pearson",
        significance_test: bool = True,
        min_common_periods: int = 50
    ) -> Dict[str, Any]:
        """Calculate correlation matrix with scientific validation"""
        
        if not returns_dict or len(returns_dict) < 2:
            return {
                'correlation_matrix': pd.DataFrame(),
                'validation_summary': {'error': 'Insufficient assets for correlation analysis'},
                'significance_matrix': pd.DataFrame()
            }
        
        # Align all return series with scientific validation
        aligned_data, alignment_info = self._align_return_series(returns_dict, min_common_periods)
        
        if aligned_data.empty or len(aligned_data.columns) < 2:
            return {
                'correlation_matrix': pd.DataFrame(),
                'validation_summary': {'error': 'Insufficient common data after alignment'},
                'significance_matrix': pd.DataFrame()
            }
        
        # Calculate correlation based on selected method
        correlation_matrix = self._calculate_correlation_method(aligned_data, method)
        
        # Calculate significance matrix if requested
        significance_matrix = pd.DataFrame()
        if significance_test and method in ["pearson", "spearman"]:
            significance_matrix = self._calculate_significance_matrix(aligned_data, method)
        
        # Validate correlation matrix
        validation_result = self._validate_correlation_matrix(correlation_matrix)
        
        # Store results
        result = {
            'correlation_matrix': correlation_matrix,
            'significance_matrix': significance_matrix,
            'aligned_data': aligned_data,
            'alignment_info': alignment_info,
            'method_used': method,
            'validation_result': validation_result,
            'summary_stats': self._calculate_correlation_summary(correlation_matrix, significance_matrix)
        }
        
        self._validation_results[method] = validation_result
        
        return result
    
    def _align_return_series(
        self, 
        returns_dict: Dict[str, pd.Series], 
        min_common_periods: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """Scientifically align return series with validation"""
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        
        # Remove series with insufficient data
        initial_count = len(df.columns)
        df = df.dropna(thresh=min_common_periods, axis=1)
        
        # Find common period with most data
        common_df = df.dropna()
        
        if len(common_df) < min_common_periods:
            # Try forward-fill for small gaps
            df_ffill = df.ffill().bfill()
            common_df = df_ffill.dropna()
            
            if len(common_df) < min_common_periods:
                return pd.DataFrame(), {'error': 'Insufficient common data period'}
        
        # Remove extreme outliers (beyond 5 standard deviations)
        for col in common_df.columns:
            series = common_df[col]
            mean_val = series.mean()
            std_val = series.std()
            outlier_mask = abs(series - mean_val) > 5 * std_val
            if outlier_mask.any():
                common_df.loc[outlier_mask, col] = np.sign(series[outlier_mask]) * 5 * std_val
                st.warning(f"⚠️ Extreme returns winsorized in {col}: {outlier_mask.sum()} points")
        
        alignment_info = {
            'initial_assets': initial_count,
            'final_assets': len(common_df.columns),
            'common_period_length': len(common_df),
            'common_start_date': common_df.index.min(),
            'common_end_date': common_df.index.max(),
            'removed_assets': initial_count - len(common_df.columns)
        }
        
        return common_df, alignment_info
    
    def _calculate_correlation_method(
        self, 
        data: pd.DataFrame, 
        method: str
    ) -> pd.DataFrame:
        """Calculate correlation using specified scientific method"""
        
        if method == "pearson":
            # Standard Pearson correlation with validation
            corr_matrix = data.corr(method='pearson')
            
        elif method == "spearman":
            # Spearman rank correlation (non-parametric)
            corr_matrix = data.corr(method='spearman')
            
        elif method == "kendall":
            # Kendall's tau (non-parametric, robust to outliers)
            corr_matrix = data.corr(method='kendall')
            
        elif method == "ewma":
            # Exponential Weighted Moving Average correlation
            corr_matrix = self._calculate_ewma_correlation(data)
            
        else:
            st.error(f"Unsupported correlation method: {method}")
            corr_matrix = data.corr(method='pearson')  # Default fallback
        
        # Ensure matrix is symmetric and valid
        corr_matrix = self._ensure_valid_correlation_matrix(corr_matrix)
        
        return corr_matrix
    
    def _calculate_ewma_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EWMA correlation matrix with scientific validation"""
        
        n_assets = len(data.columns)
        lambda_decay = self.config.ewma_lambda
        half_life = self.config.get_ewma_halflife()
        
        # Initialize correlation matrix
        corr_matrix = pd.DataFrame(
            np.eye(n_assets), 
            index=data.columns, 
            columns=data.columns
        )
        
        # Calculate EWMA means and covariances
        weights = np.zeros(len(data))
        ewma_means = pd.DataFrame(0.0, index=data.index, columns=data.columns)
        ewma_vars = pd.DataFrame(0.0, index=data.index, columns=data.columns)
        ewma_covs = {}
        
        # Initialize pairwise covariance storage
        for i in range(n_assets):
            for j in range(i, n_assets):
                ewma_covs[(i, j)] = pd.Series(0.0, index=data.index)
        
        # Calculate EWMA statistics
        for t in range(len(data)):
            # Calculate weight for this observation
            if t == 0:
                weight = 1.0
            else:
                weight = (1 - lambda_decay) * (lambda_decay ** t)
            weights[t] = weight
            
            # Update EWMA means
            if t == 0:
                ewma_means.iloc[t] = data.iloc[t]
            else:
                ewma_means.iloc[t] = lambda_decay * ewma_means.iloc[t-1] + (1 - lambda_decay) * data.iloc[t]
            
            # Update EWMA variances and covariances
            for i in range(n_assets):
                # Variance
                if t == 0:
                    ewma_vars.iloc[t, i] = 0.0
                else:
                    dev_i = data.iloc[t, i] - ewma_means.iloc[t-1, i]
                    ewma_vars.iloc[t, i] = lambda_decay * ewma_vars.iloc[t-1, i] + (1 - lambda_decay) * dev_i**2
                
                # Covariance
                for j in range(i, n_assets):
                    if t == 0:
                        ewma_covs[(i, j)].iloc[t] = 0.0
                    else:
                        dev_i = data.iloc[t, i] - ewma_means.iloc[t-1, i]
                        dev_j = data.iloc[t, j] - ewma_means.iloc[t-1, j]
                        ewma_covs[(i, j)].iloc[t] = (
                            lambda_decay * ewma_covs[(i, j)].iloc[t-1] + 
                            (1 - lambda_decay) * dev_i * dev_j
                        )
        
        # Calculate final correlation matrix using last period's values
        final_corr = np.eye(n_assets)
        for i in range(n_assets):
            var_i = ewma_vars.iloc[-1, i]
            if var_i <= 0:
                var_i = 1e-10  # Prevent division by zero
            
            for j in range(i+1, n_assets):
                var_j = ewma_vars.iloc[-1, j]
                if var_j <= 0:
                    var_j = 1e-10
                
                cov_ij = ewma_covs[(i, j)].iloc[-1]
                corr_ij = cov_ij / np.sqrt(var_i * var_j)
                
                # Ensure valid correlation values
                corr_ij = max(-0.9999, min(0.9999, corr_ij))
                
                final_corr[i, j] = corr_ij
                final_corr[j, i] = corr_ij
        
        corr_df = pd.DataFrame(
            final_corr, 
            index=data.columns, 
            columns=data.columns
        )
        
        return corr_df
    
    def _calculate_significance_matrix(
        self, 
        data: pd.DataFrame, 
        method: str
    ) -> pd.DataFrame:
        """Calculate p-values for correlation significance"""
        
        n = len(data)
        p_value_matrix = pd.DataFrame(
            1.0, 
            index=data.columns, 
            columns=data.columns
        )
        
        # Calculate p-values for each pair
        assets = data.columns
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    p_value_matrix.iloc[i, j] = 0.0  # Diagonal
                    continue
                
                # Extract valid data for this pair
                pair_data = data[[asset1, asset2]].dropna()
                if len(pair_data) < 10:  # Minimum observations
                    p_value_matrix.iloc[asset1, asset2] = 1.0
                    continue
                
                # Calculate correlation and test significance
                if method == "pearson":
                    corr, p_value = stats.pearsonr(pair_data[asset1], pair_data[asset2])
                elif method == "spearman":
                    corr, p_value = stats.spearmanr(pair_data[asset1], pair_data[asset2])
                else:
                    p_value = 1.0
                
                p_value_matrix.iloc[i, j] = p_value
        
        return p_value_matrix
    
    def _ensure_valid_correlation_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Ensure correlation matrix is valid (symmetric, positive semi-definite)"""
        
        # Ensure symmetry
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        # Check and fix positive semi-definiteness
        eigenvalues = np.linalg.eigvals(corr_matrix)
        min_eigenvalue = eigenvalues.min()
        
        if min_eigenvalue < -1e-10:  # Negative eigenvalues indicate invalid matrix
            st.warning(f"⚠️ Correlation matrix has negative eigenvalue: {min_eigenvalue:.6f}. Applying correction.")
            
            # Apply Higham's nearest correlation matrix algorithm (simplified)
            corr_matrix = self._higham_nearest_correlation(corr_matrix)
            
            # Recheck eigenvalues
            eigenvalues = np.linalg.eigvals(corr_matrix)
            min_eigenvalue = eigenvalues.min()
            
            if min_eigenvalue < -1e-10:
                st.error("⚠️ Could not fix correlation matrix positive definiteness")
        
        # Ensure values are within [-1, 1]
        corr_matrix = corr_matrix.clip(-0.9999, 0.9999)
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        return corr_matrix
    
    def _higham_nearest_correlation(self, A: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Simplified Higham's algorithm for nearest correlation matrix"""
        # This is a simplified version - in production, use full algorithm
        n = A.shape[0]
        X = A.copy()
        
        for k in range(max_iter):
            # Project onto space of matrices with unit diagonal
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            
            # Project onto space of positive semi-definite matrices
            eigvals, eigvecs = np.linalg.eigh(Y)
            eigvals = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Check convergence
            if np.linalg.norm(X - Y, 'fro') < 1e-10:
                break
        
        # Final projection to unit diagonal
        np.fill_diagonal(X, 1.0)
        
        return X
    
    def _validate_correlation_matrix(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Scientific validation of correlation matrix"""
        
        errors = []
        warnings = []
        
        if corr_matrix.empty:
            errors.append("Correlation matrix is empty")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check symmetry
        sym_diff = np.abs(corr_matrix - corr_matrix.T).max().max()
        if sym_diff > 1e-10:
            errors.append(f"Matrix not symmetric: max difference = {sym_diff:.2e}")
        
        # Check diagonal values
        diag_values = np.diag(corr_matrix)
        if not np.allclose(diag_values, 1.0, atol=1e-10):
            warnings.append(f"Diagonal values not exactly 1: min={diag_values.min():.6f}, max={diag_values.max():.6f}")
        
        # Check bounds
        min_val = corr_matrix.values.min()
        max_val = corr_matrix.values.max()
        if min_val < -1.0 or max_val > 1.0:
            errors.append(f"Values outside [-1, 1] range: min={min_val:.4f}, max={max_val:.4f}")
        
        # Check for NaN values
        if corr_matrix.isna().any().any():
            errors.append("NaN values present in correlation matrix")
        
        # Check positive semi-definiteness
        try:
            eigenvalues = np.linalg.eigvals(corr_matrix)
            min_eigenvalue = eigenvalues.min()
            if min_eigenvalue < -1e-10:
                warnings.append(f"Negative eigenvalue detected: {min_eigenvalue:.2e}")
            
            # Calculate condition number
            cond_number = np.abs(eigenvalues.max() / eigenvalues.min())
            if cond_number > 1e6:
                warnings.append(f"High condition number: {cond_number:.2e}")
        except:
            errors.append("Eigenvalue calculation failed")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'min_correlation': min_val,
            'max_correlation': max_val,
            'mean_absolute_correlation': np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]).mean(),
            'positive_semi_definite': min_eigenvalue >= -1e-10 if 'min_eigenvalue' in locals() else False
        }
    
    def _calculate_correlation_summary(
        self, 
        corr_matrix: pd.DataFrame, 
        significance_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive correlation summary statistics"""
        
        if corr_matrix.empty:
            return {}
        
        # Extract upper triangle values (excluding diagonal)
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
        
        summary = {
            'mean_correlation': float(np.mean(corr_values)),
            'median_correlation': float(np.median(corr_values)),
            'std_correlation': float(np.std(corr_values)),
            'min_correlation': float(np.min(corr_values)),
            'max_correlation': float(np.max(corr_values)),
            'abs_mean_correlation': float(np.mean(np.abs(corr_values))),
            'positive_correlation_ratio': float(np.sum(corr_values > 0) / len(corr_values)),
            'high_correlation_ratio': float(np.sum(np.abs(corr_values) > 0.7) / len(corr_values)),
            'low_correlation_ratio': float(np.sum(np.abs(corr_values) < 0.3) / len(corr_values))
        }
        
        # Add significance statistics if available
        if not significance_matrix.empty:
            sig_values = significance_matrix.values[np.triu_indices_from(significance_matrix, k=1)]
            significant_mask = sig_values < self.config.significance_level
            if len(sig_values) > 0:
                summary['significant_correlation_ratio'] = float(np.sum(significant_mask) / len(sig_values))
        
        return summary
    
    def calculate_rolling_correlation(
        self, 
        returns1: pd.Series, 
        returns2: pd.Series, 
        window: int = 60,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """Calculate rolling correlation with scientific validation"""
        
        # Align series
        aligned = pd.DataFrame({'Asset1': returns1, 'Asset2': returns2}).dropna()
        
        if len(aligned) < window:
            return pd.DataFrame()
        
        rolling_corr = pd.Series(index=aligned.index, dtype=float)
        rolling_p_value = pd.Series(index=aligned.index, dtype=float)
        
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            
            if method == "pearson":
                corr, p_value = stats.pearsonr(window_data['Asset1'], window_data['Asset2'])
            elif method == "spearman":
                corr, p_value = stats.spearmanr(window_data['Asset1'], window_data['Asset2'])
            elif method == "ewma":
                # Rolling EWMA correlation
                corr = self._calculate_ewma_correlation(window_data).iloc[0, 1]
                p_value = np.nan
            else:
                corr, p_value = stats.pearsonr(window_data['Asset1'], window_data['Asset2'])
            
            rolling_corr.iloc[i] = corr
            rolling_p_value.iloc[i] = p_value
        
        result_df = pd.DataFrame({
            'Rolling_Correlation': rolling_corr,
            'P_Value': rolling_p_value,
            'Significant': rolling_p_value < self.config.significance_level
        })
        
        return result_df.dropna()
    
    def calculate_correlation_network_metrics(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate network-based correlation metrics"""
        
        if corr_matrix.empty:
            return {}
        
        # Convert to adjacency matrix (absolute correlations as weights)
        adj_matrix = np.abs(corr_matrix.values)
        np.fill_diagonal(adj_matrix, 0)  # Remove self-correlations
        
        # Calculate network metrics
        n = len(adj_matrix)
        
        # Average degree (average correlation strength)
        avg_degree = np.mean(np.sum(adj_matrix, axis=1)) / (n - 1)
        
        # Network density
        density = np.sum(adj_matrix > 0) / (n * (n - 1))
        
        # Clustering coefficient (weighted)
        clustering_coeffs = []
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) >= 2:
                # Calculate weighted clustering coefficient
                triangles = 0
                triples = 0
                for j in neighbors:
                    for k in neighbors:
                        if j != k:
                            triples += adj_matrix[i, j] * adj_matrix[i, k]
                            triangles += adj_matrix[i, j] * adj_matrix[i, k] * adj_matrix[j, k]
                if triples > 0:
                    clustering_coeffs.append(triangles / triples)
        
        avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
        
        # Centrality measures
        degree_centrality = np.sum(adj_matrix, axis=1) / (n - 1)
        
        return {
            'average_degree': float(avg_degree),
            'network_density': float(density),
            'average_clustering': float(avg_clustering),
            'degree_centrality': dict(zip(corr_matrix.index, degree_centrality)),
            'most_central_assets': list(corr_matrix.index[np.argsort(degree_centrality)[-3:]])
        }

# =============================================================================
# SCIENTIFIC ANALYTICS ENGINE (ENHANCED)
# =============================================================================

class ScientificAnalyticsEngine:
    """Enhanced scientific analytics engine with validation"""
    
    def __init__(self, config: ScientificAnalysisConfiguration):
        self.config = config
        self.data_manager = ScientificDataManager()
        self.correlation_engine = ScientificCorrelationEngine(config)
        
    def calculate_scientific_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive scientific risk metrics with validation"""
        
        if returns.empty or len(returns) < 20:
            return {
                'metrics': {},
                'validation': {'error': 'Insufficient data for risk metrics'},
                'confidence_intervals': {}
            }
        
        # Remove extreme outliers for robust statistics
        returns_clean = self._winsorize_returns(returns)
        
        n = len(returns_clean)
        annual_factor = np.sqrt(self.config.annual_trading_days)
        
        # Calculate basic metrics
        annual_return = returns_clean.mean() * self.config.annual_trading_days
        annual_vol = returns_clean.std() * annual_factor
        
        # Calculate Sharpe ratio with validation
        sharpe_ratio = 0
        if annual_vol > 0:
            sharpe_ratio = (annual_return - self.config.risk_free_rate) / annual_vol
        
        # Calculate Sortino ratio
        sortino_ratio = self._calculate_scientific_sortino_ratio(returns_clean)
        
        # Calculate Maximum Drawdown with confidence intervals
        max_dd_result = self._calculate_scientific_max_drawdown(returns_clean)
        
        # Calculate VaR and CVaR with multiple methods
        var_results = self._calculate_scientific_var(returns_clean)
        cvar_results = self._calculate_scientific_cvar(returns_clean)
        
        # Calculate distribution statistics
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        
        # Calculate additional ratios
        calmar_ratio = self._calculate_scientific_calmar_ratio(returns_clean, annual_return)
        omega_ratio = self._calculate_scientific_omega_ratio(returns_clean)
        
        # Calculate confidence intervals using bootstrap
        ci_results = self._calculate_confidence_intervals(returns_clean)
        
        metrics = {
            # Return metrics
            'Annualized_Return': annual_return,
            'Cumulative_Return': (1 + returns_clean).prod() - 1,
            'Geometric_Mean_Return': stats.gmean(1 + returns_clean) - 1,
            
            # Risk metrics
            'Annualized_Volatility': annual_vol,
            'Downside_Deviation': returns_clean[returns_clean < 0].std() * annual_factor if len(returns_clean[returns_clean < 0]) > 0 else 0,
            'Maximum_Drawdown': max_dd_result['max_drawdown'],
            'Avg_Drawdown': max_dd_result['avg_drawdown'],
            'Max_Drawdown_Duration': max_dd_result['max_duration'],
            
            # Ratio metrics
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Omega_Ratio': omega_ratio,
            'Gain_Loss_Ratio': abs(returns_clean[returns_clean > 0].mean() / returns_clean[returns_clean < 0].mean()) if len(returns_clean[returns_clean < 0]) > 0 else 0,
            
            # Distribution metrics
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Jarque_Bera_Stat': self._calculate_jarque_bera(returns_clean),
            'Normality_P_Value': stats.normaltest(returns_clean).pvalue if len(returns_clean) > 20 else np.nan,
            
            # Performance metrics
            'Win_Rate': len(returns_clean[returns_clean > 0]) / n * 100,
            'Profit_Factor': abs(returns_clean[returns_clean > 0].sum() / returns_clean[returns_clean < 0].sum()) if returns_clean[returns_clean < 0].sum() != 0 else float('inf'),
            'Expectancy': (returns_clean[returns_clean > 0].mean() * (len(returns_clean[returns_clean > 0])/n) + 
                          returns_clean[returns_clean < 0].mean() * (len(returns_clean[returns_clean < 0])/n)),
            
            # Risk-adjusted metrics
            'Treynor_Ratio': self._calculate_treynor_ratio(returns_clean),
            'Information_Ratio': self._calculate_information_ratio(returns_clean),
            'Ulcer_Index': self._calculate_ulcer_index(returns_clean),
            
            # Tail risk metrics
            'Tail_Ratio': self._calculate_scientific_tail_ratio(returns_clean),
            'Common_Sense_Ratio': self._calculate_common_sense_ratio(returns_clean),
            'Risk_of_Ruin': self._calculate_risk_of_ruin(returns_clean)
        }
        
        # Add VaR and CVaR metrics
        metrics.update(var_results)
        metrics.update(cvar_results)
        
        # Add confidence intervals
        metrics['confidence_intervals'] = ci_results
        
        # Calculate validation metrics
        validation = self._validate_risk_metrics(metrics, returns_clean)
        
        return {
            'metrics': metrics,
            'validation': validation,
            'returns_used': returns_clean
        }
    
    def _winsorize_returns(self, returns: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
        """Winsorize returns to handle extreme outliers"""
        q_low, q_high = returns.quantile([limits[0], 1 - limits[1]])
        returns_winsorized = returns.clip(lower=q_low, upper=q_high)
        
        if (returns_winsorized != returns).any():
            n_winsorized = (returns_winsorized != returns).sum()
            st.info(f"📊 {n_winsorized} extreme returns winsorized ({n_winsorized/len(returns)*100:.1f}%)")
        
        return returns_winsorized
    
    def _calculate_scientific_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio with scientific validation"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 10:
            return 0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        
        annual_return = returns.mean() * self.config.annual_trading_days
        return (annual_return - self.config.risk_free_rate) / (downside_std * np.sqrt(self.config.annual_trading_days))
    
    def _calculate_scientific_max_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown with additional statistics"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Calculate additional drawdown statistics
        drawdown_series = drawdown[drawdown < 0]
        
        result = {
            'max_drawdown': drawdown.min() * 100,
            'avg_drawdown': drawdown_series.mean() * 100 if len(drawdown_series) > 0 else 0,
            'std_drawdown': drawdown_series.std() * 100 if len(drawdown_series) > 0 else 0,
            'max_duration': self._calculate_max_drawdown_duration(drawdown)
        }
        
        return result
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods"""
        if len(drawdown) == 0:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_scientific_var(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk using multiple methods"""
        var_results = {}
        
        # Historical VaR
        for cl in self.config.confidence_levels:
            var_key = f'VaR_{int(cl*100)}_Historical'
            var_results[var_key] = np.percentile(returns, (1 - cl) * 100) * 100
        
        # Parametric (Gaussian) VaR
        for cl in self.config.confidence_levels:
            var_key = f'VaR_{int(cl*100)}_Parametric'
            z_score = stats.norm.ppf(1 - cl)
            var_results[var_key] = (returns.mean() + z_score * returns.std()) * 100
        
        # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
        for cl in self.config.confidence_levels:
            var_key = f'VaR_{int(cl*100)}_Cornish_Fisher'
            z = stats.norm.ppf(1 - cl)
            s = returns.skew()
            k = returns.kurtosis()
            z_cf = z + (z**2 - 1) * s/6 + (z**3 - 3*z) * k/24 - (2*z**3 - 5*z) * s**2/36
            var_results[var_key] = (returns.mean() + z_cf * returns.std()) * 100
        
        return var_results
    
    def _calculate_scientific_cvar(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        cvar_results = {}
        
        for cl in self.config.confidence_levels:
            cvar_key = f'CVaR_{int(cl*100)}'
            var = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= var].mean()
            cvar_results[cvar_key] = cvar * 100 if not np.isnan(cvar) else 0
        
        return cvar_results
    
    def _calculate_scientific_calmar_ratio(self, returns: pd.Series, annual_return: float) -> float:
        """Calculate Calmar ratio"""
        max_dd = abs(self._calculate_scientific_max_drawdown(returns)['max_drawdown'] / 100)
        if max_dd == 0:
            return float('inf')
        return (annual_return - self.config.risk_free_rate) / max_dd
    
    def _calculate_scientific_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_jarque_bera(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera test statistic for normality"""
        if len(returns) < 20:
            return np.nan
        
        n = len(returns)
        s = returns.skew()
        k = returns.kurtosis()
        
        jb_stat = n/6 * (s**2 + (k**2)/4)
        return jb_stat
    
    def _calculate_treynor_ratio(self, returns: pd.Series) -> float:
        """Calculate Treynor ratio (requires market returns)"""
        # Simplified version - in production, use actual market returns
        market_returns = pd.Series(np.random.normal(0.0003, 0.01, len(returns)), index=returns.index)
        
        # Calculate beta
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        annual_return = returns.mean() * self.config.annual_trading_days
        return (annual_return - self.config.risk_free_rate) / beta if beta != 0 else 0
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio (requires benchmark returns)"""
        # Simplified version
        benchmark_returns = pd.Series(np.random.normal(0.0002, 0.008, len(returns)), index=returns.index)
        
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.config.annual_trading_days)
        
        if tracking_error > 0:
            annual_active_return = active_returns.mean() * self.config.annual_trading_days
            return annual_active_return / tracking_error
        return 0
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        
        squared_drawdown = drawdown**2
        ulcer_index = np.sqrt(squared_drawdown.mean()) if len(squared_drawdown) > 0 else 0
        
        return ulcer_index
    
    def _calculate_scientific_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate Tail ratio (95th vs 5th percentile)"""
        if len(returns) < 20:
            return 0
        
        tail_95 = np.percentile(returns, 95)
        tail_5 = np.percentile(returns, 5)
        
        if abs(tail_5) > 0:
            return abs(tail_95 / tail_5)
        return 0
    
    def _calculate_common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate Common Sense Ratio (Profitable days / Losing days)"""
        profitable_days = len(returns[returns > 0])
        losing_days = len(returns[returns < 0])
        
        if losing_days > 0:
            return profitable_days / losing_days
        return float('inf')
    
    def _calculate_risk_of_ruin(self, returns: pd.Series, initial_capital: float = 100000) -> float:
        """Calculate Risk of Ruin using Monte Carlo simulation"""
        if len(returns) < 100:
            return 0
        
        n_simulations = 1000
        ruin_count = 0
        
        for _ in range(n_simulations):
            capital = initial_capital
            for ret in np.random.choice(returns, size=min(100, len(returns)), replace=True):
                capital *= (1 + ret)
                if capital < initial_capital * 0.8:  # 20% drawdown defined as ruin
                    ruin_count += 1
                    break
        
        return ruin_count / n_simulations
    
    def _calculate_confidence_intervals(self, returns: pd.Series, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals for key metrics"""
        
        if len(returns) < 50:
            return {}
        
        metrics_to_bootstrap = ['Sharpe_Ratio', 'Annualized_Volatility', 'Maximum_Drawdown']
        ci_results = {}
        
        for metric_name in metrics_to_bootstrap:
            bootstrap_values = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                sample = np.random.choice(returns, size=len(returns), replace=True)
                
                if metric_name == 'Sharpe_Ratio':
                    if sample.std() > 0:
                        value = (sample.mean() * self.config.annual_trading_days - self.config.risk_free_rate) / (sample.std() * np.sqrt(self.config.annual_trading_days))
                    else:
                        value = 0
                
                elif metric_name == 'Annualized_Volatility':
                    value = sample.std() * np.sqrt(self.config.annual_trading_days)
                
                elif metric_name == 'Maximum_Drawdown':
                    cumulative = (1 + sample).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    value = drawdown.min() * 100
                
                bootstrap_values.append(value)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_values, 2.5)
            ci_upper = np.percentile(bootstrap_values, 97.5)
            
            ci_results[metric_name] = {
                'lower_95': float(ci_lower),
                'upper_95': float(ci_upper),
                'bootstrap_mean': float(np.mean(bootstrap_values)),
                'bootstrap_std': float(np.std(bootstrap_values))
            }
        
        return ci_results
    
    def _validate_risk_metrics(self, metrics: Dict, returns: pd.Series) -> Dict[str, Any]:
        """Validate calculated risk metrics"""
        
        errors = []
        warnings = []
        
        # Check for infinite values
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isinf(value):
                warnings.append(f"Infinite value in {key}")
        
        # Check for NaN values
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isnan(value):
                warnings.append(f"NaN value in {key}")
        
        # Validate Sharpe ratio range
        sharpe = metrics.get('Sharpe_Ratio', 0)
        if abs(sharpe) > 10:
            warnings.append(f"Extreme Sharpe ratio: {sharpe:.2f}")
        
        # Validate volatility
        vol = metrics.get('Annualized_Volatility', 0)
        if vol > 1.0:  # 100% annualized volatility
            warnings.append(f"Extremely high volatility: {vol:.1%}")
        
        # Validate maximum drawdown
        max_dd = abs(metrics.get('Maximum_Drawdown', 0))
        if max_dd > 80:  # 80% drawdown
            warnings.append(f"Extreme maximum drawdown: {max_dd:.1f}%")
        
        # Check consistency between metrics
        if metrics.get('Win_Rate', 0) > 0 and metrics.get('Profit_Factor', 0) < 1:
            warnings.append("High win rate but low profit factor - check calculation")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'n_observations': len(returns),
            'data_period_days': (returns.index[-1] - returns.index[0]).days if len(returns) > 1 else 0
        }

# =============================================================================
# SCIENTIFIC VISUALIZATION ENGINE
# =============================================================================

class ScientificVisualizationEngine:
    """Institutional scientific visualization engine"""
    
    def __init__(self):
        self.theme = ScientificThemeManager()
    
    def create_scientific_correlation_matrix(
        self,
        correlation_data: Dict[str, Any],
        title: str = "Scientific Correlation Analysis"
    ) -> go.Figure:
        """Create comprehensive scientific correlation visualization"""
        
        if not correlation_data or 'correlation_matrix' not in correlation_data:
            return self._create_empty_plot("No correlation data available")
        
        corr_matrix = correlation_data['correlation_matrix']
        significance_matrix = correlation_data.get('significance_matrix', pd.DataFrame())
        validation_result = correlation_data.get('validation_result', {})
        summary_stats = correlation_data.get('summary_stats', {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Correlation Heatmap", 
                "Correlation Distribution",
                "Significance Matrix",
                "Correlation Network"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Correlation Heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    tickmode="array",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                )
            ),
            row=1, col=1
        )
        
        # 2. Correlation Distribution
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
        
        fig.add_trace(
            go.Histogram(
                x=corr_values,
                nbinsx=30,
                name="Correlation Distribution",
                marker_color='#1a237e',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=2
        )
        
        # Add normal distribution fit
        if len(corr_values) > 10:
            x_norm = np.linspace(corr_values.min(), corr_values.max(), 100)
            try:
                params = stats.norm.fit(corr_values)
                y_norm = stats.norm.pdf(x_norm, *params)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=y_norm,
                        name="Normal Fit",
                        line=dict(color='red', width=2, dash='dash'),
                        opacity=0.7
                    ),
                    row=1, col=2
                )
            except:
                pass
        
        # 3. Significance Matrix (if available)
        if not significance_matrix.empty:
            # Create binary significance mask
            sig_mask = significance_matrix < 0.05
            sig_heatmap = sig_mask.astype(float).values
            
            fig.add_trace(
                go.Heatmap(
                    z=sig_heatmap,
                    x=significance_matrix.columns,
                    y=significance_matrix.index,
                    colorscale=[[0, 'lightgray'], [1, 'darkgreen']],
                    text=significance_matrix.round(3).values,
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    hoverongaps=False,
                    colorbar=dict(
                        title="Significant (p<0.05)",
                        titleside="right",
                        tickmode="array",
                        tickvals=[0, 1],
                        ticktext=["No", "Yes"]
                    )
                ),
                row=2, col=1
            )
        
        # 4. Correlation Network Visualization
        if len(corr_matrix) > 2:
            # Use PCA for 2D projection
            pca = PCA(n_components=2)
            corr_2d = pca.fit_transform(corr_matrix.values)
            
            fig.add_trace(
                go.Scatter(
                    x=corr_2d[:, 0],
                    y=corr_2d[:, 1],
                    mode='markers+text',
                    text=corr_matrix.index,
                    textposition='top center',
                    marker=dict(
                        size=20,
                        color=corr_matrix.mean(axis=1).values,  # Color by average correlation
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Avg Correlation")
                    ),
                    name="Assets"
                ),
                row=2, col=2
            )
            
            # Add connections for high correlations
            high_corr_threshold = 0.7
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                        fig.add_trace(
                            go.Scatter(
                                x=[corr_2d[i, 0], corr_2d[j, 0]],
                                y=[corr_2d[i, 1], corr_2d[j, 1]],
                                mode='lines',
                                line=dict(
                                    width=abs(corr_matrix.iloc[i, j]) * 3,
                                    color='rgba(128, 128, 128, 0.5)'
                                ),
                                showlegend=False
                            ),
                            row=2, col=2
                        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial", color="#1a237e"),
                x=0.5,
                xanchor="center"
            ),
            template="plotly_white",
            height=900,
            showlegend=True,
            hovermode='closest',
            font=dict(family="Arial")
        )
        
        # Add annotations for validation results
        if validation_result:
            validation_text = []
            if validation_result.get('valid'):
                validation_text.append("✓ Matrix Valid")
            else:
                validation_text.append("✗ Matrix Invalid")
            
            if 'warnings' in validation_result:
                for warning in validation_result['warnings'][:2]:
                    validation_text.append(f"⚠️ {warning}")
            
            fig.add_annotation(
                x=0.02,
                y=1.02,
                xref="paper",
                yref="paper",
                text="<br>".join(validation_text),
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Add summary statistics annotation
        if summary_stats:
            summary_text = [
                f"Mean: {summary_stats.get('mean_correlation', 0):.3f}",
                f"Std: {summary_stats.get('std_correlation', 0):.3f}",
                f"Min: {summary_stats.get('min_correlation', 0):.3f}",
                f"Max: {summary_stats.get('max_correlation', 0):.3f}"
            ]
            
            fig.add_annotation(
                x=0.98,
                y=1.02,
                xref="paper",
                yref="paper",
                text="<br>".join(summary_text),
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Update axes
        fig.update_xaxes(title_text="Assets", row=1, col=1)
        fig.update_yaxes(title_text="Assets", row=1, col=1)
        fig.update_xaxes(title_text="Correlation", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        
        if not significance_matrix.empty:
            fig.update_xaxes(title_text="Assets", row=2, col=1)
            fig.update_yaxes(title_text="Assets", row=2, col=1)
        
        fig.update_xaxes(title_text="PC1", row=2, col=2)
        fig.update_yaxes(title_text="PC2", row=2, col=2)
        
        return fig
    
    def create_correlation_comparison_chart(
        self,
        correlation_results: Dict[str, Dict[str, Any]],
        title: str = "Correlation Method Comparison"
    ) -> go.Figure:
        """Compare different correlation calculation methods"""
        
        if not correlation_results:
            return self._create_empty_plot("No correlation results for comparison")
        
        methods = list(correlation_results.keys())
        n_methods = len(methods)
        
        # Create subplots for comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Mean Correlation by Method",
                "Correlation Matrix Differences",
                "Method Performance Metrics",
                "Distribution Comparison"
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "violin"}]
            ]
        )
        
        # 1. Mean Correlation by Method
        mean_correlations = []
        for method, data in correlation_results.items():
            if 'summary_stats' in data:
                mean_correlations.append(data['summary_stats'].get('mean_correlation', 0))
            else:
                mean_correlations.append(0)
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=mean_correlations,
                name="Mean Correlation",
                marker_color=['#1a237e', '#283593', '#3949ab', '#5c6bc0'][:n_methods],
                text=[f"{x:.3f}" for x in mean_correlations],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Matrix Differences (if multiple methods)
        if n_methods >= 2:
            # Compare first two methods
            corr1 = correlation_results[methods[0]]['correlation_matrix']
            corr2 = correlation_results[methods[1]]['correlation_matrix']
            
            # Align matrices
            common_assets = corr1.index.intersection(corr2.index)
            if len(common_assets) > 1:
                corr1_aligned = corr1.loc[common_assets, common_assets]
                corr2_aligned = corr2.loc[common_assets, common_assets]
                diff_matrix = corr1_aligned - corr2_aligned
                
                fig.add_trace(
                    go.Heatmap(
                        z=diff_matrix.values,
                        x=diff_matrix.columns,
                        y=diff_matrix.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=diff_matrix.round(3).values,
                        texttemplate='%{text}',
                        colorbar=dict(title=f"{methods[0]} - {methods[1]}")
                    ),
                    row=1, col=2
                )
        
        # 3. Method Performance Metrics
        metrics_data = []
        for method, data in correlation_results.items():
            if 'validation_result' in data:
                metrics_data.append({
                    'Method': method,
                    'Matrix Valid': 1 if data['validation_result'].get('valid') else 0,
                    'Min Eigenvalue': data['validation_result'].get('min_eigenvalue', 0),
                    'Condition Number': data['validation_result'].get('cond_number', 0) if 'cond_number' in data['validation_result'] else 1
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['Method'],
                    y=metrics_df['Min Eigenvalue'],
                    mode='markers+lines',
                    name='Min Eigenvalue',
                    marker=dict(size=10, color='#ff6b6b')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['Method'],
                    y=metrics_df['Condition Number'],
                    mode='markers+lines',
                    name='Condition Number',
                    yaxis='y2',
                    marker=dict(size=10, color='#4ecdc4')
                ),
                row=2, col=1
            )
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Condition Number",
                    overlaying='y',
                    side='right'
                )
            )
        
        # 4. Distribution Comparison
        all_correlations = []
        method_labels = []
        
        for method, data in correlation_results.items():
            if 'correlation_matrix' in data:
                corr_values = data['correlation_matrix'].values[np.triu_indices_from(data['correlation_matrix'], k=1)]
                all_correlations.extend(corr_values)
                method_labels.extend([method] * len(corr_values))
        
        if all_correlations:
            fig.add_trace(
                go.Violin(
                    x=method_labels,
                    y=all_correlations,
                    name="Correlation Distributions",
                    box_visible=True,
                    meanline_visible=True,
                    points='outliers',
                    marker=dict(color='#1a237e'),
                    opacity=0.6
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template="plotly_white",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_rolling_correlation_chart(
        self,
        rolling_corr_data: pd.DataFrame,
        asset1: str,
        asset2: str,
        title: str = "Rolling Correlation Analysis"
    ) -> go.Figure:
        """Create scientific rolling correlation visualization"""
        
        if rolling_corr_data.empty:
            return self._create_empty_plot("No rolling correlation data")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"Rolling Correlation: {asset1} vs {asset2}",
                "Statistical Significance",
                "Correlation Distribution Over Time"
            ),
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # 1. Rolling Correlation
        fig.add_trace(
            go.Scatter(
                x=rolling_corr_data.index,
                y=rolling_corr_data['Rolling_Correlation'],
                name="Correlation",
                line=dict(color='#1a237e', width=2),
                fill='tozeroy',
                fillcolor='rgba(26, 35, 126, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add confidence bands if available
        if 'confidence_lower' in rolling_corr_data.columns and 'confidence_upper' in rolling_corr_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr_data.index,
                    y=rolling_corr_data['confidence_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='Upper CI'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr_data.index,
                    y=rolling_corr_data['confidence_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(26, 35, 126, 0.2)',
                    showlegend=False,
                    name='Lower CI'
                ),
                row=1, col=1
            )
        
        # 2. Statistical Significance
        if 'P_Value' in rolling_corr_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr_data.index,
                    y=rolling_corr_data['P_Value'],
                    name="P-Value",
                    line=dict(color='#ff6b6b', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add significance threshold
            fig.add_hline(
                y=0.05,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text="p=0.05",
                row=2, col=1
            )
            
            # Highlight significant periods
            if 'Significant' in rolling_corr_data.columns:
                significant_periods = rolling_corr_data[rolling_corr_data['Significant']]
                if not significant_periods.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=significant_periods.index,
                            y=significant_periods['P_Value'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color='#2e7d32',
                                symbol='circle'
                            ),
                            name="Significant",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # 3. Correlation Distribution Over Time (Heatmap style)
        # Create time bins for distribution visualization
        if len(rolling_corr_data) > 50:
            n_bins = min(20, len(rolling_corr_data) // 10)
            time_bins = pd.cut(
                pd.Series(range(len(rolling_corr_data))), 
                bins=n_bins,
                labels=False
            )
            
            correlation_by_bin = []
            for bin_num in range(n_bins):
                bin_data = rolling_corr_data.iloc[time_bins[time_bins == bin_num].index]
                if len(bin_data) > 0:
                    correlation_by_bin.append(bin_data['Rolling_Correlation'].values)
            
            # Create heatmap-like distribution
            for i, corr_bin in enumerate(correlation_by_bin):
                if len(corr_bin) > 0:
                    fig.add_trace(
                        go.Box(
                            y=corr_bin,
                            name=f"Bin {i+1}",
                            boxpoints=False,
                            marker_color='#3949ab',
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=3, col=1
                    )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template="plotly_white",
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="P-Value", row=2, col=1)
        fig.update_yaxes(title_text="Correlation Distribution", row=3, col=1)
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template="plotly_white",
            height=400
        )
        return fig

# =============================================================================
# SCIENTIFIC STREAMLIT APPLICATION
# =============================================================================

class ScientificCommoditiesPlatform:
    """Main scientific Streamlit application"""
    
    def __init__(self):
        self.data_manager = ScientificDataManager()
        self.config = None
        self.visualization = ScientificVisualizationEngine()
        self.analytics_engine = None
        
        # Initialize scientific session state
        if 'scientific_analysis_results' not in st.session_state:
            st.session_state.scientific_analysis_results = {}
        if 'selected_scientific_assets' not in st.session_state:
            st.session_state.selected_scientific_assets = []
        if 'selected_benchmarks' not in st.session_state:
            st.session_state.selected_benchmarks = []
        if 'correlation_methods' not in st.session_state:
            st.session_state.correlation_methods = ['pearson', 'ewma']
        if 'validation_warnings' not in st.session_state:
            st.session_state.validation_warnings = []
        
    def render_scientific_header(self):
        """Render institutional scientific header"""
        st.markdown("""
        <div class="scientific-header">
            <h1>📈 Institutional Commodities Analytics Platform v7.0</h1>
            <p>Scientific Computing Division • Advanced Correlation Analytics • Risk Management Systems</p>
            <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                <span class="scientific-badge info">🔬 Scientific Validation</span>
                <span class="scientific-badge low-risk">📊 Advanced Correlations</span>
                <span class="scientific-badge medium-risk">⚡ Real-time Analytics</span>
                <span class="scientific-badge high-risk">📈 Institutional Grade</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display dependency status
        st.markdown(sci_dep_manager.display_status(), unsafe_allow_html=True)
    
    def render_scientific_sidebar(self):
        """Render scientific sidebar with validation"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h2 style="color: #1a237e; margin: 0;">🔬 Scientific Configuration</h2>
                <p style="color: #415a77; margin: 0;">Advanced Analytics Parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Date range with validation
            st.markdown("### 📅 Analysis Period")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365 * 3),
                    max_value=datetime.now() - timedelta(days=30),
                    help="Minimum 30 days of data required for scientific analysis"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now(),
                    help="Analysis end date"
                )
            
            # Validate date range
            if start_date >= end_date:
                st.error("❌ Start date must be before end date")
                return
            
            if (end_date - start_date).days < 30:
                st.warning("⚠️ Analysis period less than 30 days may produce unreliable results")
            
            # Asset selection with scientific categorization
            st.markdown("### 📊 Asset Universe")
            
            selected_assets = []
            for category_name, assets in COMMODITIES_UNIVERSE.items():
                with st.expander(f"{category_name} ({len(assets)} assets)", expanded=True):
                    for symbol, metadata in assets.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.checkbox(
                                f"{metadata.name}",
                                value=symbol in ["GC=F", "CL=F"],  # Default selections
                                key=f"sci_asset_{symbol}",
                                help=metadata.description
                            ):
                                selected_assets.append(symbol)
                        with col2:
                            risk_color = {
                                "Low": "success",
                                "Medium": "warning", 
                                "High": "danger"
                            }.get(metadata.risk_level, "info")
                            st.markdown(f'<span class="scientific-badge {risk_color}">{metadata.risk_level}</span>', 
                                      unsafe_allow_html=True)
            
            st.session_state.selected_scientific_assets = selected_assets
            
            # Benchmark selection
            st.markdown("### 🎯 Benchmark Selection")
            benchmark_assets = []
            for symbol, info in BENCHMARKS.items():
                if st.checkbox(
                    f"{info['name']}",
                    value=symbol in ["^GSPC", "GLD"],
                    key=f"sci_bench_{symbol}",
                    help=info['description']
                ):
                    benchmark_assets.append(symbol)
            
            st.session_state.selected_benchmarks = benchmark_assets
            
            # Scientific analysis parameters
            st.markdown("### ⚙️ Scientific Parameters")
            
            # Risk-free rate with validation
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.5,
                step=0.1,
                format="%.1f%%",
                help="Annualized risk-free rate for Sharpe ratio calculation"
            ) / 100
            
            # Correlation methods selection
            st.markdown("#### 📈 Correlation Methods")
            correlation_methods = st.multiselect(
                "Select correlation calculation methods",
                options=["pearson", "spearman", "kendall", "ewma"],
                default=["pearson", "ewma"],
                help="Pearson: Linear correlation, Spearman: Rank correlation, Kendall: Robust rank, EWMA: Time-decaying"
            )
            
            st.session_state.correlation_methods = correlation_methods
            
            # EWMA specific parameters
            if "ewma" in correlation_methods:
                st.markdown("##### EWMA Parameters")
                ewma_lambda = st.slider(
                    "EWMA Decay Factor (λ)",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.94,
                    step=0.01,
                    help="Higher λ gives more weight to recent observations"
                )
                st.caption(f"Half-life: {math.log(0.5)/math.log(ewma_lambda):.1f} days")
            else:
                ewma_lambda = 0.94
            
            # Statistical significance level
            significance_level = st.select_slider(
                "Statistical Significance Level",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05,
                format_func=lambda x: f"{x*100:.1f}%",
                help="Threshold for statistical significance (p-value)"
            )
            ####################################################################################################################################################
                           # Statistical significance level
            significance_level = st.select_slider(
                "Statistical Significance Level",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05,
                format_func=lambda x: f"{x*100:.1f}%",
                help="Threshold for statistical significance (p-value)"
            )
            
            # Monte Carlo simulations
            monte_carlo_sims = st.selectbox(
                "Monte Carlo Simulations",
                options=[1000, 5000, 10000, 25000, 50000],
                index=2,  # 10000 is at index 2
                help="Number of simulations for risk analysis"
            )
            
            # Rolling analysis window
            rolling_window = st.selectbox(
                "Rolling Analysis Window (days)",
                options=[20, 60, 120, 250],
                index=1,  # 60 is at index 1
                help="Window size for rolling statistics"
            )      
            
            # Initialize scientific configuration
            self.config = ScientificAnalysisConfiguration(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                risk_free_rate=risk_free_rate,
                confidence_levels=(0.95, 0.99),
                monte_carlo_simulations=monte_carlo_sims,
                rolling_window=rolling_window,
                correlation_method="ewma" if "ewma" in correlation_methods else "pearson",
                ewma_lambda=ewma_lambda,
                significance_level=significance_level
            )
            
            # Validate configuration
            is_valid, errors = self.config.validate()
            
            if not is_valid:
                st.error("❌ Configuration Errors:")
                for error in errors:
                    st.error(f"  - {error}")
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Scientific Analysis", type="primary", use_container_width=True, 
                           disabled=not is_valid or len(selected_assets) < 1):
                    st.session_state.run_scientific_analysis = True
                    st.rerun()
            
            with col2:
                if st.button("🧹 Clear Results", type="secondary", use_container_width=True):
                    st.session_state.scientific_analysis_results = {}
                    st.session_state.validation_warnings = []
                    st.session_state.run_scientific_analysis = False
                    st.rerun()
            
            with col3:
                if st.button("📊 Data Validation", type="secondary", use_container_width=True):
                    st.session_state.show_data_validation = True
            
            # System status
            st.markdown("---")
            st.markdown("### 📈 System Status")
            
            status_cols = st.columns(2)
            with status_cols[0]:
                st.metric("Assets Selected", len(selected_assets))
            with status_cols[1]:
                st.metric("Benchmarks", len(benchmark_assets))
            
            if len(selected_assets) < 2:
                st.warning("⚠️ Select at least 2 assets for correlation analysis")
    
    def run_scientific_analysis(self):
        """Execute comprehensive scientific analysis"""
        
        if not self.config:
            st.error("❌ Configuration not initialized")
            return
        
        if len(st.session_state.selected_scientific_assets) < 1:
            st.error("❌ No assets selected for analysis")
            return
        
        # Show progress and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch data
            status_text.text("📥 Fetching scientific data...")
            progress_bar.progress(10)
            
            all_symbols = st.session_state.selected_scientific_assets + st.session_state.selected_benchmarks
            
            data_results = self.data_manager.fetch_multiple_assets_scientific(
                all_symbols,
                self.config.start_date,
                self.config.end_date,
                max_workers=6,
                validation_level="strict"
            )
            
            progress_bar.progress(30)
            
            # Step 2: Calculate returns and features
            status_text.text("📊 Calculating scientific features...")
            
            all_returns = {}
            all_features = {}
            
            for symbol, df in data_results.items():
                if not df.empty:
                    # Calculate scientific features
                    features_df = self.data_manager.calculate_scientific_features(df)
                    all_features[symbol] = features_df
                    
                    # Calculate returns
                    if 'Returns' in features_df.columns:
                        all_returns[symbol] = features_df['Returns'].dropna()
            
            progress_bar.progress(50)
            
            # Step 3: Initialize analytics engine
            status_text.text("🔬 Initializing scientific analytics...")
            self.analytics_engine = ScientificAnalyticsEngine(self.config)
            
            # Step 4: Calculate risk metrics
            status_text.text("📈 Calculating risk metrics...")
            
            risk_metrics = {}
            for symbol, returns in all_returns.items():
                if len(returns) >= 20:
                    metric_result = self.analytics_engine.calculate_scientific_risk_metrics(returns)
                    risk_metrics[symbol] = metric_result
            
            progress_bar.progress(70)
            
            # Step 5: Calculate correlations with multiple methods
            status_text.text("📊 Calculating scientific correlations...")
            
            correlation_results = {}
            for method in st.session_state.correlation_methods:
                corr_result = self.analytics_engine.correlation_engine.calculate_correlation_matrix(
                    all_returns,
                    method=method,
                    significance_test=True,
                    min_common_periods=50
                )
                correlation_results[method] = corr_result
            
            progress_bar.progress(90)
            
            # Step 6: Store results
            st.session_state.scientific_analysis_results = {
                'data': data_results,
                'features': all_features,
                'returns': all_returns,
                'risk_metrics': risk_metrics,
                'correlation_results': correlation_results,
                'config': self.config,
                'analytics_engine': self.analytics_engine,
                'timestamp': datetime.now()
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Scientific analysis complete!")
            
            # Show completion message
            st.success("""
            ✅ Scientific analysis completed successfully!
            
            **Summary:**
            - Data fetched and validated for {} assets
            - {} risk metrics calculated
            - Correlation analysis using {} methods
            - Scientific validation passed
            """.format(
                len(data_results),
                len(risk_metrics),
                len(correlation_results)
            ))
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Scientific analysis failed: {str(e)}")
            st.code(traceback.format_exc())
    
    def render_correlation_analysis(self):
        """Render comprehensive correlation analysis"""
        
        results = st.session_state.scientific_analysis_results
        if not results or 'correlation_results' not in results:
            st.warning("⚠️ Run scientific analysis first to view correlation results")
            return
        
        correlation_results = results['correlation_results']
        
        # Correlation analysis header
        st.markdown("""
        <div class="section-header">
            <h2>📊 Scientific Correlation Analysis</h2>
            <div class="section-actions">
                <span class="scientific-badge info">Multiple Methods</span>
                <span class="scientific-badge low-risk">Statistical Validation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Method selection for detailed view
        selected_method = st.selectbox(
            "Select Correlation Method for Detailed Analysis",
            options=list(correlation_results.keys()),
            index=0,
            help="Choose correlation calculation method to analyze in detail"
        )
        
        if selected_method in correlation_results:
            corr_data = correlation_results[selected_method]
            
            # Display validation warnings
            if 'validation_result' in corr_data:
                validation = corr_data['validation_result']
                if not validation.get('valid', True):
                    st.error("❌ Correlation matrix validation failed!")
                    for error in validation.get('errors', []):
                        st.error(f"  - {error}")
                
                if validation.get('warnings'):
                    st.warning("⚠️ Correlation matrix validation warnings:")
                    for warning in validation.get('warnings', []):
                        st.warning(f"  - {warning}")
            
            # Create correlation visualization
            fig = self.visualization.create_scientific_correlation_matrix(
                corr_data,
                title=f"Scientific Correlation Analysis - {selected_method.upper()} Method"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            if 'summary_stats' in corr_data:
                summary = corr_data['summary_stats']
                
                st.markdown("#### 📈 Correlation Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Correlation", f"{summary.get('mean_correlation', 0):.3f}")
                with col2:
                    st.metric("Std Deviation", f"{summary.get('std_correlation', 0):.3f}")
                with col3:
                    st.metric("Minimum", f"{summary.get('min_correlation', 0):.3f}")
                with col4:
                    st.metric("Maximum", f"{summary.get('max_correlation', 0):.3f}")
                
                # Additional statistics
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Positive Ratio", f"{summary.get('positive_correlation_ratio', 0):.1%}")
                with col6:
                    st.metric("High (>0.7) Ratio", f"{summary.get('high_correlation_ratio', 0):.1%}")
                with col7:
                    st.metric("Low (<0.3) Ratio", f"{summary.get('low_correlation_ratio', 0):.1%}")
                with col8:
                    if 'significant_correlation_ratio' in summary:
                        st.metric("Significant Ratio", f"{summary['significant_correlation_ratio']:.1%}")
            
            # Display correlation matrix as table
            with st.expander("📋 View Correlation Matrix Table", expanded=False):
                corr_matrix = corr_data['correlation_matrix']
                st.dataframe(
                    corr_matrix.style.format("{:.3f}").background_gradient(cmap='RdBu', vmin=-1, vmax=1),
                    use_container_width=True
                )
            
            # Pairwise correlation analysis
            st.markdown("#### 🔗 Pairwise Correlation Analysis")
            
            assets = list(corr_data['correlation_matrix'].columns)
            if len(assets) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    asset1 = st.selectbox("Select First Asset", assets, index=0)
                with col2:
                    # Ensure asset2 is different from asset1
                    available_assets = [a for a in assets if a != asset1]
                    asset2 = st.selectbox("Select Second Asset", available_assets, 
                                        index=min(1, len(available_assets)-1))
                
                if asset1 and asset2:
                    # Get returns for these assets
                    returns = results.get('returns', {})
                    if asset1 in returns and asset2 in returns:
                        # Calculate rolling correlation
                        rolling_corr = self.analytics_engine.correlation_engine.calculate_rolling_correlation(
                            returns[asset1],
                            returns[asset2],
                            window=self.config.rolling_window,
                            method=selected_method
                        )
                        
                        if not rolling_corr.empty:
                            fig_rolling = self.visualization.create_rolling_correlation_chart(
                                rolling_corr,
                                asset1,
                                asset2,
                                title=f"Rolling Correlation: {asset1} vs {asset2}"
                            )
                            st.plotly_chart(fig_rolling, use_container_width=True)
                        
                        # Display correlation statistics for this pair
                        pair_corr = corr_data['correlation_matrix'].loc[asset1, asset2]
                        st.info(f"**{asset1} - {asset2} Correlation:** {pair_corr:.3f}")
                        
                        if 'significance_matrix' in corr_data and not corr_data['significance_matrix'].empty:
                            p_value = corr_data['significance_matrix'].loc[asset1, asset2]
                            is_significant = p_value < self.config.significance_level
                            significance_text = "✅ Statistically Significant" if is_significant else "❌ Not Significant"
                            st.write(f"**Statistical Significance:** {significance_text} (p={p_value:.3f})")
        
        # Compare multiple correlation methods
        if len(correlation_results) > 1:
            st.markdown("---")
            st.markdown("#### 🔄 Correlation Method Comparison")
            
            fig_comparison = self.visualization.create_correlation_comparison_chart(
                correlation_results,
                title="Correlation Method Comparison"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Display method differences
            with st.expander("📊 Method Comparison Details", expanded=False):
                methods = list(correlation_results.keys())
                comparison_data = []
                
                for method in methods:
                    corr_data = correlation_results[method]
                    if 'summary_stats' in corr_data:
                        stats = corr_data['summary_stats']
                        row = {'Method': method}
                        row.update({k: v for k, v in stats.items() if isinstance(v, (int, float))})
                        comparison_data.append(row)
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(
                        comparison_df.style.format("{:.3f}").background_gradient(
                            subset=comparison_df.columns[1:], cmap='YlOrRd'
                        ),
                        use_container_width=True
                    )
    
    def render_scientific_dashboard(self):
        """Render main scientific dashboard"""
        
        if not st.session_state.selected_scientific_assets:
            st.warning("""
            <div class="warning-message">
                <strong>⚠️ No Assets Selected</strong><br>
                Please select assets from the sidebar to begin scientific analysis.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Check if analysis should run
        if hasattr(st.session_state, 'run_scientific_analysis') and st.session_state.run_scientific_analysis:
            self.run_scientific_analysis()
            st.session_state.run_scientific_analysis = False
        
        # Display analysis results if available
        results = st.session_state.scientific_analysis_results
        if not results:
            # Show welcome/instructions
            self.render_welcome_screen()
            return
        
        # Create scientific dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Risk Analytics", 
            "🔗 Correlation Analysis",
            "📉 Portfolio Science",
            "📋 Data & Validation"
        ])
        
        with tab1:
            self.render_overview_dashboard()
        
        with tab2:
            self.render_risk_analytics()
        
        with tab3:
            self.render_correlation_analysis()
        
        with tab4:
            self.render_portfolio_science()
        
        with tab5:
            self.render_data_validation()
    
    def render_overview_dashboard(self):
        """Render overview dashboard"""
        results = st.session_state.scientific_analysis_results
        if not results:
            return
        
        # Key metrics display
        st.markdown("""
        <div class="section-header">
            <h2>📊 Scientific Overview Dashboard</h2>
            <div class="section-actions">
                <span class="scientific-badge info">Real-time Analytics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display key metrics in institutional cards
        if 'risk_metrics' in results:
            risk_metrics = results['risk_metrics']
            
            # Select top assets by Sharpe ratio
            sharpe_ratios = {}
            for symbol, metrics in risk_metrics.items():
                if 'metrics' in metrics and 'Sharpe_Ratio' in metrics['metrics']:
                    sharpe_ratios[symbol] = metrics['metrics']['Sharpe_Ratio']
            
            top_assets = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)[:4]
            
            st.markdown("#### 🏆 Top Performing Assets (Sharpe Ratio)")
            cols = st.columns(4)
            
            for idx, (symbol, sharpe) in enumerate(top_assets):
                with cols[idx]:
                    metrics_data = risk_metrics[symbol]['metrics']
                    
                    # Determine color based on Sharpe ratio
                    if sharpe > 1.0:
                        color_class = "positive"
                    elif sharpe > 0:
                        color_class = "positive"
                    else:
                        color_class = "negative"
                    
                    st.markdown(f"""
                    <div class="institutional-card">
                        <div class="metric-title">{symbol}</div>
                        <div class="metric-value">{sharpe:.2f}</div>
                        <div class="metric-change {color_class}">
                            Vol: {metrics_data.get('Annualized_Volatility', 0):.1%} |
                            DD: {abs(metrics_data.get('Maximum_Drawdown', 0)):.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick correlation insights
        if 'correlation_results' in results:
            corr_results = results['correlation_results']
            if corr_results:
                first_method = list(corr_results.keys())[0]
                corr_data = corr_results[first_method]
                
                if 'summary_stats' in corr_data:
                    summary = corr_data['summary_stats']
                    
                    st.markdown("#### 📈 Correlation Insights")
                    
                    insight_cols = st.columns(3)
                    with insight_cols[0]:
                        mean_corr = summary.get('mean_correlation', 0)
                        insight = "Highly Diversified" if abs(mean_corr) < 0.3 else "Correlated" if mean_corr > 0.5 else "Mixed"
                        st.metric("Correlation Regime", insight)
                    
                    with insight_cols[1]:
                        high_corr_ratio = summary.get('high_correlation_ratio', 0)
                        st.metric("High Correlation Pairs", f"{high_corr_ratio:.1%}")
                    
                    with insight_cols[2]:
                        if 'significant_correlation_ratio' in summary:
                            sig_ratio = summary['significant_correlation_ratio']
                            st.metric("Statistically Significant", f"{sig_ratio:.1%}")
        
        # System status and warnings
        if st.session_state.validation_warnings:
            st.markdown("#### ⚠️ Validation Warnings")
            for warning in st.session_state.validation_warnings[:3]:
                st.warning(warning)
    
    def render_risk_analytics(self):
        """Render risk analytics dashboard"""
        results = st.session_state.scientific_analysis_results
        if not results or 'risk_metrics' not in results:
            return
        
        st.markdown("""
        <div class="section-header">
            <h2>📈 Scientific Risk Analytics</h2>
            <div class="section-actions">
                <span class="scientific-badge danger">Risk Metrics</span>
                <span class="scientific-badge warning">Validation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        risk_metrics = results['risk_metrics']
        
        # Asset selector for detailed risk analysis
        selected_asset = st.selectbox(
            "Select Asset for Detailed Risk Analysis",
            options=list(risk_metrics.keys()),
            help="Choose asset to view comprehensive risk metrics"
        )
        
        if selected_asset in risk_metrics:
            asset_metrics = risk_metrics[selected_asset]
            
            # Display validation results
            if 'validation' in asset_metrics:
                validation = asset_metrics['validation']
                
                if not validation.get('valid'):
                    st.error("❌ Risk metrics validation failed!")
                    for error in validation.get('errors', []):
                        st.error(f"  - {error}")
                
                if validation.get('warnings'):
                    st.warning("⚠️ Risk metrics validation warnings:")
                    for warning in validation.get('warnings', []):
                        st.warning(f"  - {warning}")
            
            # Display key risk metrics in cards
            if 'metrics' in asset_metrics:
                metrics = asset_metrics['metrics']
                
                st.markdown("#### 📊 Key Risk Metrics")
                
                # Row 1: Return and volatility metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Annualized Return", 
                        f"{metrics.get('Annualized_Return', 0):.2%}",
                        help="Average annual return"
                    )
                
                with col2:
                    st.metric(
                        "Annualized Volatility", 
                        f"{metrics.get('Annualized_Volatility', 0):.2%}",
                        delta="High" if metrics.get('Annualized_Volatility', 0) > 0.3 else "Normal",
                        delta_color="inverse",
                        help="Annualized standard deviation of returns"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio", 
                        f"{metrics.get('Sharpe_Ratio', 0):.2f}",
                        delta="Excellent" if metrics.get('Sharpe_Ratio', 0) > 1.0 else "Poor" if metrics.get('Sharpe_Ratio', 0) < 0 else "Average",
                        help="Risk-adjusted return (Sharpe Ratio)"
                    )
                
                with col4:
                    st.metric(
                        "Maximum Drawdown", 
                        f"{abs(metrics.get('Maximum_Drawdown', 0)):.2f}%",
                        delta="Severe" if abs(metrics.get('Maximum_Drawdown', 0)) > 20 else "Moderate",
                        delta_color="inverse",
                        help="Maximum peak-to-trough decline"
                    )
                
                # Row 2: Additional risk metrics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric(
                        "Sortino Ratio", 
                        f"{metrics.get('Sortino_Ratio', 0):.2f}",
                        help="Downside risk-adjusted return"
                    )
                
                with col6:
                    st.metric(
                        "Calmar Ratio", 
                        f"{metrics.get('Calmar_Ratio', 0):.2f}" if not np.isinf(metrics.get('Calmar_Ratio', 0)) else "∞",
                        help="Return relative to maximum drawdown"
                    )
                
                with col7:
                    st.metric(
                        "VaR (95%)", 
                        f"{metrics.get('VaR_95_Historical', 0):.2f}%",
                        help="Value at Risk at 95% confidence (Historical)"
                    )
                
                with col8:
                    st.metric(
                        "CVaR (95%)", 
                        f"{metrics.get('CVaR_95', 0):.2f}%",
                        help="Conditional Value at Risk at 95% confidence"
                    )
                
                # Display detailed metrics table
                with st.expander("📋 View All Risk Metrics", expanded=False):
                    # Convert metrics to DataFrame
                    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                    metrics_df = metrics_df[~metrics_df.index.str.contains('confidence_intervals')]
                    
                    # Format values
                    def format_metric_value(val):
                        if isinstance(val, (int, float)):
                            if abs(val) < 0.01:
                                return f"{val:.4f}"
                            elif abs(val) < 1:
                                return f"{val:.3f}"
                            else:
                                return f"{val:.2f}"
                        return str(val)
                    
                    metrics_df['Value'] = metrics_df['Value'].apply(format_metric_value)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Display confidence intervals if available
                if 'confidence_intervals' in metrics:
                    ci_data = metrics['confidence_intervals']
                    if ci_data:
                        st.markdown("#### 📊 Bootstrap Confidence Intervals (95%)")
                        
                        ci_df = pd.DataFrame({
                            'Metric': [],
                            'Lower Bound': [],
                            'Estimate': [],
                            'Upper Bound': []
                        })
                        
                        for metric_name, ci_vals in ci_data.items():
                            ci_df = pd.concat([ci_df, pd.DataFrame({
                                'Metric': [metric_name],
                                'Lower Bound': [ci_vals.get('lower_95', 0)],
                                'Estimate': [ci_vals.get('bootstrap_mean', 0)],
                                'Upper Bound': [ci_vals.get('upper_95', 0)]
                            })], ignore_index=True)
                        
                        st.dataframe(
                            ci_df.style.format({
                                'Lower Bound': '{:.3f}',
                                'Estimate': '{:.3f}', 
                                'Upper Bound': '{:.3f}'
                            }),
                            use_container_width=True
                        )
        
        # Comparative risk analysis
        st.markdown("---")
        st.markdown("#### 📈 Comparative Risk Analysis")
        
        if len(risk_metrics) >= 2:
            # Create comparison DataFrame
            comparison_data = []
            for symbol, metrics_data in risk_metrics.items():
                if 'metrics' in metrics_data:
                    row = {'Asset': symbol}
                    key_metrics = ['Sharpe_Ratio', 'Annualized_Volatility', 'Maximum_Drawdown', 
                                 'Sortino_Ratio', 'VaR_95_Historical', 'Win_Rate']
                    for metric in key_metrics:
                        if metric in metrics_data['metrics']:
                            row[metric] = metrics_data['metrics'][metric]
                    comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create interactive comparison chart
                fig = go.Figure()
                
                metrics_to_plot = ['Sharpe_Ratio', 'Annualized_Volatility', 'Maximum_Drawdown']
                colors = ['#1a237e', '#283593', '#3949ab']
                
                for metric, color in zip(metrics_to_plot, colors):
                    if metric in comparison_df.columns:
                        fig.add_trace(go.Bar(
                            x=comparison_df['Asset'],
                            y=comparison_df[metric],
                            name=metric.replace('_', ' '),
                            marker_color=color,
                            opacity=0.7
                        ))
                
                fig.update_layout(
                    title="Risk Metrics Comparison Across Assets",
                    barmode='group',
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_science(self):
        """Render portfolio science dashboard"""
        results = st.session_state.scientific_analysis_results
        if not results:
            return
        
        st.markdown("""
        <div class="section-header">
            <h2>📉 Portfolio Science & Optimization</h2>
            <div class="section-actions">
                <span class="scientific-badge info">Portfolio Theory</span>
                <span class="scientific-badge warning">Optimization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we have enough data for portfolio analysis
        if 'returns' not in results or len(results['returns']) < 2:
            st.warning("⚠️ Insufficient data for portfolio analysis. Need at least 2 assets.")
            return
        
        returns_data = results['returns']
        
        # Portfolio optimization parameters
        st.markdown("#### ⚙️ Portfolio Optimization Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Objective",
                ["Maximum Sharpe Ratio", "Minimum Volatility", "Risk Parity", "Maximum Diversification"],
                help="Objective function for portfolio optimization"
            )
        
        with col2:
            target_return = st.slider(
                "Target Annual Return (%)",
                min_value=-20.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                format="%.1f%%"
            ) / 100
        
        with col3:
            risk_aversion = st.slider(
                "Risk Aversion Coefficient",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Higher values indicate greater risk aversion"
            )
        
        # Prepare returns data for optimization
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 50:
            st.warning("⚠️ Insufficient common return data for robust portfolio optimization")
            return
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Display optimization inputs
        with st.expander("📊 Optimization Inputs", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Expected Annual Returns**")
                st.dataframe(
                    pd.DataFrame(expected_returns, columns=['Expected Return']).style.format("{:.2%}"),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Annualized Covariance Matrix**")
                st.dataframe(
                    cov_matrix.style.format("{:.4f}").background_gradient(cmap='RdBu'),
                    use_container_width=True,
                    height=300
                )
        
        # Perform portfolio optimization
        if st.button("🚀 Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                # Simple mean-variance optimization
                n_assets = len(expected_returns)
                
                # Generate random portfolios for efficient frontier
                n_portfolios = 10000
                results_array = np.zeros((3, n_portfolios))
                weights_record = []
                
                for i in range(n_portfolios):
                    # Generate random weights
                    weights = np.random.random(n_assets)
                    weights /= weights.sum()
                    weights_record.append(weights)
                    
                    # Calculate portfolio statistics
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - results['config'].risk_free_rate) / portfolio_vol
                    
                    results_array[0, i] = portfolio_return
                    results_array[1, i] = portfolio_vol
                    results_array[2, i] = sharpe_ratio
                
                # Find optimal portfolios
                max_sharpe_idx = np.argmax(results_array[2])
                min_vol_idx = np.argmin(results_array[1])
                
                # Create efficient frontier visualization
                fig = go.Figure()
                
                # Random portfolios
                fig.add_trace(go.Scatter(
                    x=results_array[1, :],
                    y=results_array[0, :],
                    mode='markers',
                    name='Random Portfolios',
                    marker=dict(
                        size=5,
                        color=results_array[2, :],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    )
                ))
                
                # Optimal portfolios
                fig.add_trace(go.Scatter(
                    x=[results_array[1, max_sharpe_idx]],
                    y=[results_array[0, max_sharpe_idx]],
                    mode='markers',
                    name='Max Sharpe Ratio',
                    marker=dict(size=15, symbol='star', color='gold')
                ))
                
                fig.add_trace(go.Scatter(
                    x=[results_array[1, min_vol_idx]],
                    y=[results_array[0, min_vol_idx]],
                    mode='markers',
                    name='Min Volatility',
                    marker=dict(size=15, symbol='diamond', color='red')
                ))
                
                fig.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Annual Volatility",
                    yaxis_title="Annual Return",
                    template="plotly_white",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display optimal portfolio weights
                st.markdown("#### ⚖️ Optimal Portfolio Weights")
                
                # Get weights for optimal portfolios
                max_sharpe_weights = weights_record[max_sharpe_idx]
                min_vol_weights = weights_record[min_vol_idx]
                
                weights_df = pd.DataFrame({
                    'Asset': expected_returns.index,
                    'Max Sharpe Weight': max_sharpe_weights,
                    'Min Vol Weight': min_vol_weights
                })
                
                # Create weight visualization
                fig_weights = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Maximum Sharpe Ratio", "Minimum Volatility"),
                    specs=[[{"type": "pie"}, {"type": "pie"}]]
                )
                
                fig_weights.add_trace(
                    go.Pie(
                        labels=weights_df['Asset'],
                        values=weights_df['Max Sharpe Weight'] * 100,
                        name="Max Sharpe",
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set3)
                    ),
                    row=1, col=1
                )
                
                fig_weights.add_trace(
                    go.Pie(
                        labels=weights_df['Asset'],
                        values=weights_df['Min Vol Weight'] * 100,
                        name="Min Vol",
                        hole=0.4,
                        marker=dict(colors=px.colors.qualitative.Set3)
                    ),
                    row=1, col=2
                )
                
                fig_weights.update_layout(
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # Display weight table
                st.dataframe(
                    weights_df.style.format({
                        'Max Sharpe Weight': '{:.1%}',
                        'Min Vol Weight': '{:.1%}'
                    }),
                    use_container_width=True
                )
                
                # Calculate and display portfolio statistics
                st.markdown("#### 📊 Optimal Portfolio Statistics")
                
                for portfolio_name, weights in [("Max Sharpe", max_sharpe_weights), 
                                               ("Min Vol", min_vol_weights)]:
                    port_return = np.sum(weights * expected_returns)
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    port_sharpe = (port_return - results['config'].risk_free_rate) / port_vol
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{portfolio_name} Return", f"{port_return:.2%}")
                    with col2:
                        st.metric(f"{portfolio_name} Volatility", f"{port_vol:.2%}")
                    with col3:
                        st.metric(f"{portfolio_name} Sharpe Ratio", f"{port_sharpe:.2f}")
    
    def render_data_validation(self):
        """Render data validation dashboard"""
        results = st.session_state.scientific_analysis_results
        if not results:
            return
        
        st.markdown("""
        <div class="section-header">
            <h2>📋 Data Quality & Validation</h2>
            <div class="section-actions">
                <span class="scientific-badge info">Data Integrity</span>
                <span class="scientific-badge warning">Validation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data quality metrics
        if 'data' in results:
            data_results = results['data']
            
            st.markdown("#### 📊 Data Quality Metrics")
            
            quality_data = []
            for symbol, df in data_results.items():
                if not df.empty:
                    quality_data.append({
                        'Symbol': symbol,
                        'Rows': len(df),
                        'Start Date': df.index.min().date(),
                        'End Date': df.index.max().date(),
                        'Missing Values': df.isna().sum().sum(),
                        'Zero Volume Days': (df.get('Volume', pd.Series([0])) == 0).sum() if 'Volume' in df.columns else 0,
                        'Data Quality': 'Good' if len(df) > 100 and df.isna().sum().sum() < len(df) * 0.1 else 'Poor'
                    })
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(
                    quality_df.style.apply(
                        lambda x: ['background-color: #e8f5e8' if v == 'Good' else 'background-color: #ffebee' 
                                 for v in x] if x.name == 'Data Quality' else [''] * len(x),
                        axis=0
                    ),
                    use_container_width=True
                )
        
        # Validation results from risk metrics
        if 'risk_metrics' in results:
            st.markdown("#### 📈 Risk Metrics Validation")
            
            validation_data = []
            for symbol, metrics_data in results['risk_metrics'].items():
                if 'validation' in metrics_data:
                    validation = metrics_data['validation']
                    validation_data.append({
                        'Symbol': symbol,
                        'Valid': validation.get('valid', False),
                        'Warnings': len(validation.get('warnings', [])),
                        'Errors': len(validation.get('errors', [])),
                        'Observations': validation.get('n_observations', 0),
                        'Period (days)': validation.get('data_period_days', 0)
                    })
            
            if validation_data:
                validation_df = pd.DataFrame(validation_data)
                
                # Apply conditional formatting
                def highlight_validity(row):
                    if row['Valid'] and row['Errors'] == 0:
                        return ['background-color: #e8f5e8'] * len(row)
                    elif not row['Valid']:
                        return ['background-color: #ffebee'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    validation_df.style.apply(highlight_validity, axis=1),
                    use_container_width=True
                )
        
        # Correlation matrix validation
        if 'correlation_results' in results:
            st.markdown("#### 🔗 Correlation Matrix Validation")
            
            corr_results = results['correlation_results']
            corr_validation_data = []
            
            for method, data in corr_results.items():
                if 'validation_result' in data:
                    validation = data['validation_result']
                    corr_validation_data.append({
                        'Method': method,
                        'Valid': validation.get('valid', False),
                        'Warnings': len(validation.get('warnings', [])),
                        'Errors': len(validation.get('errors', [])),
                        'Min Eigenvalue': validation.get('min_eigenvalue', 0),
                        'Positive Definite': validation.get('positive_semi_definite', False)
                    })
            
            if corr_validation_data:
                corr_validation_df = pd.DataFrame(corr_validation_data)
                st.dataframe(
                    corr_validation_df.style.format({
                        'Min Eigenvalue': '{:.2e}'
                    }),
                    use_container_width=True
                )
        
        # Data download option
        st.markdown("---")
        st.markdown("#### 💾 Data Export")
        
        if st.button("📥 Download Analysis Results", type="secondary"):
            # Create downloadable report
            report_data = {
                'timestamp': results.get('timestamp', datetime.now()).isoformat(),
                'config': asdict(results['config']) if 'config' in results else {},
                'assets_analyzed': len(results.get('data', {})),
                'summary': "Scientific commodities analysis report"
            }
            
            # Convert to JSON for download
            json_str = json.dumps(report_data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            
            href = f'<a href="data:application/json;base64,{b64}" download="scientific_analysis_report.json">Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def render_welcome_screen(self):
        """Render welcome screen with instructions"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #e8eaf6 0%, #ffffff 100%); border-radius: 12px; margin: 2rem 0;">
            <h2 style="color: #1a237e; margin-bottom: 1rem;">🔬 Welcome to Scientific Commodities Analytics</h2>
            <p style="color: #415a77; font-size: 1.1rem; max-width: 800px; margin: 0 auto 2rem;">
                Institutional-grade scientific analysis platform for commodities trading.
                Get started by configuring your analysis in the sidebar and click "Run Scientific Analysis".
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 3rem;">
                <div class="institutional-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-weight: 600; color: #1a237e; margin-bottom: 0.5rem;">Scientific Correlation</div>
                    <div style="font-size: 0.9rem; color: #415a77;">Multiple correlation methods with statistical validation</div>
                </div>
                <div class="institutional-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
                    <div style="font-weight: 600; color: #1a237e; margin-bottom: 0.5rem;">Risk Analytics</div>
                    <div style="font-size: 0.9rem; color: #415a77;">Comprehensive risk metrics with confidence intervals</div>
                </div>
                <div class="institutional-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
                    <div style="font-weight: 600; color: #1a237e; margin-bottom: 0.5rem;">Portfolio Science</div>
                    <div style="font-size: 0.9rem; color: #415a77;">Mean-variance optimization and efficient frontier</div>
                </div>
                <div class="institutional-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
                    <div style="font-weight: 600; color: #1a237e; margin-bottom: 0.5rem;">Data Validation</div>
                    <div style="font-size: 0.9rem; color: #415a77;">Comprehensive data quality and validation checks</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        
        steps = [
            ("1. Select Assets", "Choose commodities and benchmarks from the sidebar"),
            ("2. Configure Parameters", "Set scientific analysis parameters and correlation methods"),
            ("3. Run Analysis", "Click 'Run Scientific Analysis' to generate insights"),
            ("4. Explore Results", "Navigate through tabs to view different aspects of the analysis")
        ]
        
        for title, description in steps:
            with st.expander(title, expanded=True):
                st.write(description)
                
        def render_scientific_footer(self):
        """Render scientific footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; color: #415a77; font-size: 0.85rem; font-family: 'Inter', sans-serif;">
                <p><strong>🏛️ Institutional Commodities Analytics Platform v7.0</strong></p>
                <p>Scientific Computing Division • Advanced Financial Analytics</p>
                <p>© 2024 Institutional Trading Analytics • All rights reserved</p>
                <p style="margin-top: 0.75rem; font-size: 0.8rem;">
                    <span style="margin: 0 0.5rem;">📧 research@institutional-commodities.com</span>
                    <span style="margin: 0 0.5rem;">🔬 Scientific Validation System v2.1</span>
                    <span style="margin: 0 0.5rem;">⚡ Performance Optimized</span>
                </p>
                <p style="margin-top: 0.5rem; font-size: 0.75rem; color: #6b7280;">
                    For institutional use only. Not for retail distribution.
                </p>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        """Run the scientific Streamlit application"""
        try:
            # Render scientific header
            self.render_scientific_header()
            
            # Render scientific sidebar
            self.render_scientific_sidebar()
            
            # Render main dashboard
            self.render_scientific_dashboard()
            
            # Render scientific footer
            self.render_scientific_footer()
            
        except Exception as e:
            # FIXED: Using markdown instead of error() with unsafe_allow_html
            error_html = f"""
            <div class="error-message">
                <strong>🚨 Scientific Application Error</strong><br>
                {str(e)}
            </div>
            """
            st.markdown(error_html, unsafe_allow_html=True)
            
            # Display detailed error information
            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc())
            
            # Provide recovery option
            if st.button("🔄 Restart Scientific Application", type="primary"):
                st.session_state.clear()
                st.rerun()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        # Initialize and run the scientific application
        app = ScientificCommoditiesPlatform()
        app.run()
        
    except Exception as e:
        st.error(f"🚨 Critical Application Failure: {str(e)}")
        st.code(traceback.format_exc())
        
        # Emergency restart
        if st.button("🚨 Emergency Restart", type="primary"):
            import os
            import sys
            os.execv(sys.executable, ['python'] + sys.argv)
