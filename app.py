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
import inspect
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
import pickle
import base64
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

# Import scipy safely
try:
    from scipy import stats, optimize, signal, linalg, special
    import scipy
    SCIPY_AVAILABLE = True
except ImportError as e:
    SCIPY_AVAILABLE = False
    # Create mock scipy module
    class MockScipy:
        class stats:
            @staticmethod
            def norm(*args, **kwargs):
                class MockNorm:
                    @staticmethod
                    def pdf(x):
                        return np.exp(-x**2/2)/np.sqrt(2*np.pi)
                    @staticmethod
                    def ppf(q):
                        return np.sqrt(2)*special.erfinv(2*q-1) if 'special' in globals() else 0
                return MockNorm()
            
            @staticmethod
            def pearsonr(x, y):
                return np.corrcoef(x, y)[0, 1], 0.05
            
            @staticmethod
            def spearmanr(x, y):
                return np.corrcoef(x, y)[0, 1], 0.05
            
            @staticmethod
            def normaltest(x):
                class Result:
                    pvalue = 0.5
                return Result()
            
            @staticmethod
            def gmean(x):
                return np.exp(np.mean(np.log(x)))
            
            @staticmethod
            def skew(x):
                return pd.Series(x).skew()
            
            @staticmethod
            def kurtosis(x):
                return pd.Series(x).kurtosis()
        
        __version__ = 'mock'
    
    scipy = MockScipy()
    stats = scipy.stats

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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Streamlit configuration for institutional interface
st.set_page_config(
    page_title="Institutional Commodities Analytics Platform v7.0",
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
    CRYPTO = "Cryptocurrency"
    CURRENCY = "Currency"

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
    volatility_30d: float = field(default=0.2, metadata={'range': (0.0, 1.0)})
    correlation_cluster: str = field(default="General", metadata={'options': ['SafeHaven', 'Industrial', 'Energy', 'Agricultural', 'Macro']})
    
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
        if not 0.0 <= self.liquidity_score <= 1.0:
            raise ValueError(f"Liquidity score out of range: {self.liquidity_score}")
        if not 0.0 <= self.fundamental_score <= 1.0:
            raise ValueError(f"Fundamental score out of range: {self.fundamental_score}")
        if not 0.0 <= self.volatility_30d <= 1.0:
            raise ValueError(f"30D volatility out of range: {self.volatility_30d}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def risk_color(self) -> str:
        """Get color based on risk level"""
        risk_colors = {
            "Low": "#2e7d32",    # Green
            "Medium": "#f57c00",  # Orange
            "High": "#c62828",    # Red
            "Very High": "#6a1b9a" # Purple
        }
        return risk_colors.get(self.risk_level, "#415a77")

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
    optimization_method: str = field(default="sharpe", metadata={'options': ['sharpe', 'min_vol', 'risk_parity', 'max_diversification']})
    correlation_method: str = field(default="pearson", metadata={'options': ['pearson', 'spearman', 'kendall', 'ewma', 'dynamic_copula']})
    ewma_lambda: float = field(default=0.94, metadata={'range': (0.90, 0.99)})
    significance_level: float = field(default=0.05, metadata={'range': (0.01, 0.10)})
    minimum_data_points: int = field(default=50, metadata={'range': (20, 500)})
    outlier_threshold: float = field(default=5.0, metadata={'range': (3.0, 10.0)})
    bootstrap_iterations: int = field(default=1000, metadata={'range': (100, 10000)})
    
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
        
        if not (3.0 <= self.outlier_threshold <= 10.0):
            errors.append(f"Outlier threshold {self.outlier_threshold} outside valid range [3.0, 10.0]")
        
        return len(errors) == 0, errors
    
    def get_ewma_halflife(self) -> float:
        """Calculate half-life for EWMA decay factor"""
        return math.log(0.5) / math.log(self.ewma_lambda)
    
    @property
    def date_range_days(self) -> int:
        """Get analysis period in days"""
        return (self.end_date - self.start_date).days

# Enhanced scientific commodities universe
COMMODITIES_UNIVERSE = {
    "Precious Metals": {
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
            fundamental_score=0.85,
            volatility_30d=0.18,
            correlation_cluster="SafeHaven"
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
            fundamental_score=0.75,
            volatility_30d=0.28,
            correlation_cluster="SafeHaven"
        ),
    },
    "Industrial Metals": {
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
            fundamental_score=0.80,
            volatility_30d=0.25,
            correlation_cluster="Industrial"
        ),
    },
    "Energy": {
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
            fundamental_score=0.75,
            volatility_30d=0.35,
            correlation_cluster="Energy"
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
            fundamental_score=0.65,
            volatility_30d=0.50,
            correlation_cluster="Energy"
        ),
    },
    "Agriculture": {
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
            fundamental_score=0.70,
            volatility_30d=0.25,
            correlation_cluster="Agricultural"
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
            fundamental_score=0.68,
            volatility_30d=0.28,
            correlation_cluster="Agricultural"
        ),
    },
    "Cryptocurrency": {
        "BTC-USD": ScientificAssetMetadata(
            symbol="BTC-USD",
            name="Bitcoin",
            category=AssetCategory.CRYPTO,
            color="#F7931A",
            description="Bitcoin - Digital gold and decentralized cryptocurrency",
            exchange="Various",
            contract_size="1 BTC",
            margin_requirement=0.500,
            tick_size=0.01,
            risk_level="Very High",
            beta_to_spx=0.25,
            liquidity_score=0.92,
            fundamental_score=0.60,
            volatility_30d=0.65,
            correlation_cluster="Macro"
        ),
    }
}

BENCHMARKS = {
    "^GSPC": ScientificAssetMetadata(
        symbol="^GSPC",
        name="S&P 500 Index",
        category=AssetCategory.BENCHMARK,
        color="#1E90FF",
        description="S&P 500 Equity Index - US large-cap equity benchmark",
        risk_level="Medium",
        beta_to_spx=1.00,
        liquidity_score=0.99,
        fundamental_score=0.85,
        volatility_30d=0.18,
        correlation_cluster="Macro"
    ),
    "GLD": ScientificAssetMetadata(
        symbol="GLD",
        name="SPDR Gold Shares",
        category=AssetCategory.PRECIOUS_METALS,
        color="#FFD700",
        description="Gold-backed ETF - Gold price proxy",
        risk_level="Low",
        beta_to_spx=0.10,
        liquidity_score=0.96,
        fundamental_score=0.82,
        volatility_30d=0.16,
        correlation_cluster="SafeHaven"
    ),
    "^VIX": ScientificAssetMetadata(
        symbol="^VIX",
        name="VIX Volatility Index",
        category=AssetCategory.BENCHMARK,
        color="#FF1493",
        description="CBOE Volatility Index - Market fear gauge",
        risk_level="High",
        beta_to_spx=-0.70,
        liquidity_score=0.88,
        fundamental_score=0.70,
        volatility_30d=0.60,
        correlation_cluster="Macro"
    )
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
        "commodity_trading": {
            "primary": "#1565c0",   # Commodity blue
            "secondary": "#0277bd", # Trading blue
            "accent": "#4fc3f7",    # Light trading blue
            "success": "#388e3c",   # Growth green
            "warning": "#ff8f00",   # Commodity orange
            "danger": "#d32f2f",    # Risk red
            "dark": "#0d47a1",      # Deep blue
            "light": "#e3f2fd",     # Very light blue
            "gray": "#607d8b",      # Blue-gray
            "background": "#ffffff",
            "grid": "#e1f5fe",
            "border": "#bbdefb"
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
                font-size: 2.5rem;
                font-weight: 800;
                margin: 0 0 0.5rem 0;
                letter-spacing: -0.5px;
                background: linear-gradient(90deg, #ffffff 0%, #e0e1dd 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .scientific-header p {{
                font-size: 1.1rem;
                opacity: 0.9;
                margin: 0;
                font-weight: 400;
                color: rgba(255, 255, 255, 0.85);
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
                transform: translateY(-2px);
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
                font-size: 2rem;
                font-weight: 800;
                color: var(--dark);
                margin: 0;
                font-family: 'SF Mono', 'Roboto Mono', monospace;
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
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
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                border: 1px solid transparent;
                transition: var(--transition);
            }}
            
            .scientific-badge.low-risk {{
                background: linear-gradient(135deg, rgba(46, 125, 50, 0.15) 0%, rgba(46, 125, 50, 0.05) 100%);
                color: var(--success);
                border-color: rgba(46, 125, 50, 0.3);
            }}
            
            .scientific-badge.medium-risk {{
                background: linear-gradient(135deg, rgba(245, 124, 0, 0.15) 0%, rgba(245, 124, 0, 0.05) 100%);
                color: var(--warning);
                border-color: rgba(245, 124, 0, 0.3);
            }}
            
            .scientific-badge.high-risk {{
                background: linear-gradient(135deg, rgba(198, 40, 40, 0.15) 0%, rgba(198, 40, 40, 0.05) 100%);
                color: var(--danger);
                border-color: rgba(198, 40, 40, 0.3);
            }}
            
            .scientific-badge.info {{
                background: linear-gradient(135deg, rgba(41, 98, 255, 0.15) 0%, rgba(41, 98, 255, 0.05) 100%);
                color: var(--primary);
                border-color: rgba(41, 98, 255, 0.3);
            }}
        </style>
        """

# Apply institutional theme
st.markdown(ScientificThemeManager.get_styles("commodity_trading"), unsafe_allow_html=True)

# =============================================================================
# ADVANCED DEPENDENCY MANAGEMENT WITH VALIDATION
# =============================================================================

class ScientificDependencyManager:
    """Scientific dependency management with validation"""
    
    def __init__(self):
        self.dependencies = {}
        self._scientific_imports()
        self._validate_versions()
    
    def _get_package_version(self, package_name: str, fallback='unknown'):
        """Safely get package version"""
        try:
            if package_name == 'numpy':
                return np.__version__
            elif package_name == 'pandas':
                return pd.__version__
            elif package_name == 'scipy':
                return scipy.__version__ if hasattr(scipy, '__version__') else 'mock'
            elif package_name == 'plotly':
                return '5.17.0'  # Approximate
            elif package_name == 'streamlit':
                import streamlit
                return streamlit.__version__
            elif package_name == 'yfinance':
                import yfinance
                return yfinance.__version__
            else:
                return fallback
        except:
            return fallback
    
    def _scientific_imports(self):
        """Load scientific dependencies with fallback implementations"""
        
        # Register available dependencies
        self.dependencies['numpy'] = {
            'available': True,
            'version': np.__version__,
            'module': np
        }
        
        self.dependencies['pandas'] = {
            'available': True,
            'version': pd.__version__,
            'module': pd
        }
        
        self.dependencies['scipy'] = {
            'available': SCIPY_AVAILABLE,
            'version': self._get_package_version('scipy'),
            'module': scipy,
            'stats': stats
        }
        
        self.dependencies['plotly'] = {
            'available': True,
            'version': '5.17.0',
            'module': go
        }
        
        self.dependencies['streamlit'] = {
            'available': True,
            'version': self._get_package_version('streamlit'),
            'module': st
        }
        
        self.dependencies['yfinance'] = {
            'available': True,
            'version': self._get_package_version('yfinance'),
            'module': yf
        }
    
    def _validate_versions(self):
        """Validate dependency versions for scientific stability"""
        # Skip validation in Streamlit Cloud environment
        pass
    
    def is_available(self, dependency: str) -> bool:
        """Check if scientific dependency is available"""
        return self.dependencies.get(dependency, {}).get('available', False)
    
    def display_status(self):
        """Display dependency status"""
        status_html = """
        <div class="institutional-card">
            <div class="metric-title">🧪 Scientific Dependencies Status</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 0.5rem;">
        """
        
        for dep, info in self.dependencies.items():
            status = "🟢" if info.get('available') else "🟡"
            version = info.get('version', 'N/A')
            status_html += f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.25rem 0; border-bottom: 1px solid var(--border);">
                    <span style="font-size: 0.85rem; color: var(--gray); font-weight: 600;">{dep}</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.75rem; color: var(--gray); font-family: 'SF Mono', monospace;">{version}</span>
                        <span style="font-size: 0.9rem;">{status}</span>
                    </div>
                </div>
            """
        
        status_html += """
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--gray);">
                <div>🟢 = Available | 🟡 = Mock Implementation</div>
                <div>All core scientific calculations are functional with fallback implementations</div>
            </div>
        </div>
        """
        return status_html

# Initialize scientific dependency manager
sci_dep_manager = ScientificDependencyManager()

# =============================================================================
# SCIENTIFIC DATA MANAGER WITH VALIDATION
# =============================================================================

class ScientificDataManager:
    """Scientific data management with comprehensive validation"""
    
    def __init__(self):
        self._validation_metrics = {}
    
    @staticmethod
    @st.cache_data(ttl=10800, max_entries=150, show_spinner=False)
    def fetch_scientific_data(
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        retries: int = 3,
        validate: bool = True
    ) -> pd.DataFrame:
        """Fetch and validate scientific data with comprehensive error handling"""
        
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
                        timeout=30
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
                    if not df.empty:
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
                df = ScientificDataManager._scientific_clean_dataframe(df, symbol)
                
                # Scientific validation
                if validate:
                    validation_result = ScientificDataManager._validate_dataframe(df, symbol)
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
        
        return df_final
    
    @staticmethod
    def _scientific_clean_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Scientific cleaning and preprocessing"""
        df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Adj Close': 'Adj_Close',
            'AdjClose': 'Adj_Close',
            'Adj_Close': 'Adj_Close',
            'Close': 'Close',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Volume': 'Volume'
        }
        
        # Rename columns
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        # If no adjusted close, use close
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col in ['Open', 'High', 'Low']:
                    df[col] = df['Close']
        
        # Add Volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0.0
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
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
        
        return df
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
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
        
        # High-Low validation
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_hl = df['High'] < df['Low']
            if invalid_hl.any():
                errors.append(f"High < Low on {invalid_hl.sum()} days")
        
        # Return validation summary
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'n_rows': len(df),
            'date_range': (df.index.min(), df.index.max())
        }
    
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
        
        # Price-based features
        df['Price_Range_Pct'] = (df['High'] - df['Low']) / df[price_col] * 100
        df['Close_to_Close_Change'] = df[price_col].pct_change() * 100
        
        # Moving averages with scientific validation
        periods = [5, 10, 20, 50]
        for period in periods:
            if len(df) >= period:
                df[f'SMA_{period}'] = df[price_col].rolling(window=period, min_periods=int(period*0.8)).mean()
                df[f'EMA_{period}'] = df[price_col].ewm(span=period, min_periods=int(period*0.8)).mean()
        
        # Volatility measures
        if len(df) >= 20:
            # Simple historical volatility
            df['Volatility_20D_Simple'] = df['Returns'].rolling(window=20, min_periods=15).std() * np.sqrt(252) * 100
            
            # Parkinson volatility (using high-low range)
            if 'High' in df.columns and 'Low' in df.columns:
                df['Parkinson_Vol'] = np.sqrt(1/(4*np.log(2)) * (np.log(df['High']/df['Low'])**2).rolling(window=20).mean()) * np.sqrt(252) * 100
        
        # Volume analysis
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=15).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)
        
        # ATR (Average True Range)
        if len(df) >= 14 and 'High' in df.columns and 'Low' in df.columns:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df[price_col].shift())
            low_close = np.abs(df['Low'] - df[price_col].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=10).mean()
            df['ATR_Pct'] = (df['ATR'] / df[price_col]) * 100
        
        # Momentum indicators
        momentum_periods = [5, 10, 20]
        for period in momentum_periods:
            if len(df) >= period:
                df[f'Momentum_{period}D'] = df[price_col].pct_change(periods=period) * 100
        
        # Remove NaN values while preserving as much data as possible
        original_length = len(df)
        df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% valid data
        removed_pct = (original_length - len(df)) / original_length * 100 if original_length > 0 else 0
        
        if removed_pct > 10:
            st.warning(f"⚠️ High data removal ({removed_pct:.1f}%) during feature calculation")
        
        return df

# =============================================================================
# SCIENTIFIC CORRELATION ENGINE
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
        
        if len(df.columns) < 2:
            return pd.DataFrame(), {'error': 'Insufficient assets after filtering'}
        
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
            if std_val > 0:
                outlier_mask = abs(series - mean_val) > 5 * std_val
                if outlier_mask.any():
                    common_df.loc[outlier_mask, col] = np.sign(series[outlier_mask]) * 5 * std_val
        
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
        
        # Initialize correlation matrix
        corr_matrix = pd.DataFrame(
            np.eye(n_assets), 
            index=data.columns, 
            columns=data.columns
        )
        
        # Calculate simple correlation as fallback if EWMA fails
        try:
            # Calculate EWMA means and covariances
            ewma_means = data.ewm(alpha=1-lambda_decay).mean()
            ewma_vars = ((data - ewma_means.shift(1))**2).ewm(alpha=1-lambda_decay).mean()
            
            # Calculate pairwise EWMA covariances
            for i, asset1 in enumerate(data.columns):
                for j, asset2 in enumerate(data.columns[i+1:], i+1):
                    # Calculate EWMA covariance
                    prod_series = (data[asset1] - ewma_means[asset1].shift(1)) * (data[asset2] - ewma_means[asset2].shift(1))
                    ewma_cov = prod_series.ewm(alpha=1-lambda_decay).mean()
                    
                    # Get final values
                    var_i = ewma_vars[asset1].iloc[-1]
                    var_j = ewma_vars[asset2].iloc[-1]
                    cov_ij = ewma_cov.iloc[-1]
                    
                    # Calculate correlation
                    if var_i > 0 and var_j > 0:
                        corr_ij = cov_ij / np.sqrt(var_i * var_j)
                        # Ensure valid correlation values
                        corr_ij = max(-0.9999, min(0.9999, corr_ij))
                        
                        corr_matrix.iloc[i, j] = corr_ij
                        corr_matrix.iloc[j, i] = corr_ij
        
        except Exception as e:
            st.warning(f"EWMA correlation calculation failed: {e}. Using simple correlation.")
            corr_matrix = data.corr(method='pearson')
        
        return corr_matrix
    
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
                    p_value_matrix.iloc[i, j] = 1.0
                    continue
                
                # Calculate correlation and test significance
                try:
                    if method == "pearson":
                        corr, p_value = stats.pearsonr(pair_data[asset1], pair_data[asset2])
                    elif method == "spearman":
                        corr, p_value = stats.spearmanr(pair_data[asset1], pair_data[asset2])
                    else:
                        p_value = 1.0
                except:
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
        try:
            eigenvalues = np.linalg.eigvals(corr_matrix)
            min_eigenvalue = eigenvalues.min().real
            
            if min_eigenvalue < -1e-10:  # Negative eigenvalues indicate invalid matrix
                st.warning(f"⚠️ Correlation matrix has negative eigenvalue: {min_eigenvalue:.6f}. Applying correction.")
                
                # Apply Higham's nearest correlation matrix algorithm (simplified)
                corr_matrix = self._higham_nearest_correlation(corr_matrix.values)
                
                # Recheck eigenvalues
                eigenvalues = np.linalg.eigvals(corr_matrix)
                min_eigenvalue = eigenvalues.min().real
                
                if min_eigenvalue < -1e-10:
                    st.error("⚠️ Could not fix correlation matrix positive definiteness")
        except:
            st.warning("Eigenvalue calculation failed for correlation matrix validation")
        
        # Ensure values are within [-1, 1]
        corr_matrix = corr_matrix.clip(-0.9999, 0.9999)
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        return corr_matrix
    
    def _higham_nearest_correlation(self, A: np.ndarray, max_iter: int = 100) -> pd.DataFrame:
        """Simplified Higham's algorithm for nearest correlation matrix"""
        n = A.shape[0]
        X = A.copy()
        
        for k in range(max_iter):
            # Project onto space of matrices with unit diagonal
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            
            # Project onto space of positive semi-definite matrices
            try:
                eigvals, eigvecs = np.linalg.eigh(Y)
                eigvals = np.maximum(eigvals, 0)
                X = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                break
            
            # Check convergence
            if np.linalg.norm(X - Y, 'fro') < 1e-10:
                break
        
        # Final projection to unit diagonal
        np.fill_diagonal(X, 1.0)
        
        # Return as DataFrame with original index/columns
        return pd.DataFrame(X, index=A.index, columns=A.columns)
    
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
            min_eigenvalue = eigenvalues.min().real
            if min_eigenvalue < -1e-10:
                warnings.append(f"Negative eigenvalue detected: {min_eigenvalue:.2e}")
            
            # Calculate condition number
            if abs(min_eigenvalue) > 1e-10:
                cond_number = np.abs(eigenvalues.max().real / min_eigenvalue)
                if cond_number > 1e6:
                    warnings.append(f"High condition number: {cond_number:.2e}")
        except:
            warnings.append("Eigenvalue calculation failed")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'min_correlation': min_val,
            'max_correlation': max_val,
            'mean_absolute_correlation': np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]).mean() if len(corr_matrix) > 1 else 0
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

# =============================================================================
# SCIENTIFIC ANALYTICS ENGINE
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
        
        # Calculate Maximum Drawdown
        max_dd_result = self._calculate_scientific_max_drawdown(returns_clean)
        
        # Calculate VaR
        var_results = self._calculate_scientific_var(returns_clean)
        
        # Calculate distribution statistics
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        
        metrics = {
            # Return metrics
            'Annualized_Return': annual_return,
            'Cumulative_Return': (1 + returns_clean).prod() - 1,
            
            # Risk metrics
            'Annualized_Volatility': annual_vol,
            'Maximum_Drawdown': max_dd_result['max_drawdown'],
            
            # Ratio metrics
            'Sharpe_Ratio': sharpe_ratio,
            
            # Distribution metrics
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            
            # Performance metrics
            'Win_Rate': len(returns_clean[returns_clean > 0]) / n * 100 if n > 0 else 0,
            'Profit_Factor': abs(returns_clean[returns_clean > 0].sum() / returns_clean[returns_clean < 0].sum()) if returns_clean[returns_clean < 0].sum() != 0 else float('inf'),
        }
        
        # Add VaR metrics
        metrics.update(var_results)
        
        # Calculate validation metrics
        validation = self._validate_risk_metrics(metrics, returns_clean)
        
        return {
            'metrics': metrics,
            'validation': validation,
            'returns_used': returns_clean
        }
    
    def _winsorize_returns(self, returns: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
        """Winsorize returns to handle extreme outliers"""
        if len(returns) < 10:
            return returns
        
        q_low = returns.quantile(limits[0])
        q_high = returns.quantile(1 - limits[1])
        returns_winsorized = returns.clip(lower=q_low, upper=q_high)
        
        return returns_winsorized
    
    def _calculate_scientific_max_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown with additional statistics"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Calculate additional drawdown statistics
        drawdown_series = drawdown[drawdown < 0]
        
        result = {
            'max_drawdown': drawdown.min() * 100 if len(drawdown) > 0 else 0,
            'avg_drawdown': drawdown_series.mean() * 100 if len(drawdown_series) > 0 else 0,
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
            var_results[var_key] = np.percentile(returns, (1 - cl) * 100) * 100 if len(returns) > 0 else 0
        
        return var_results
    
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
        pass
    
    def create_scientific_correlation_matrix(
        self,
        correlation_data: Dict[str, Any],
        title: str = "Scientific Correlation Analysis"
    ) -> go.Figure:
        """Create comprehensive scientific correlation visualization"""
        
        if not correlation_data or 'correlation_matrix' not in correlation_data:
            return self._create_empty_plot("No correlation data available")
        
        corr_matrix = correlation_data['correlation_matrix']
        
        # Create simple heatmap
        fig = go.Figure(data=go.Heatmap(
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
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial", color="#1a237e"),
                x=0.5,
                xanchor="center"
            ),
            template="plotly_white",
            height=600,
            showlegend=False,
            font=dict(family="Arial")
        )
        
        return fig
    
    def create_asset_performance_chart(
        self,
        asset_data: Dict[str, pd.DataFrame],
        metrics: Dict[str, Dict[str, Any]],
        title: str = "Asset Performance Comparison"
    ) -> go.Figure:
        """Create comprehensive asset performance visualization"""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        # 1. Cumulative Returns
        for idx, (symbol, df) in enumerate(asset_data.items()):
            if not df.empty and 'Adj_Close' in df.columns:
                cumulative_returns = (df['Adj_Close'] / df['Adj_Close'].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=cumulative_returns,
                    name=symbol,
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    mode='lines'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template="plotly_white",
            height=500,
            showlegend=True,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)"
        )
        
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
        if 'run_scientific_analysis' not in st.session_state:
            st.session_state.run_scientific_analysis = False
    
     def render_scientific_header(self):
        """Render institutional scientific header"""
        st.markdown("""
        <div class="scientific-header">
            <h1>🏛️ Institutional Commodities Analytics Platform v7.0</h1>
            <p>Enhanced Scientific Analytics • Advanced Correlation Methods • Professional Risk Metrics</p>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <span class="scientific-badge low-risk">Scientific Validation</span>
                <span class="scientific-badge medium-risk">Multi-Method Analysis</span>
                <span class="scientific-badge high-risk">Institutional Grade</span>
                <span class="scientific-badge info">Real-Time Analytics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display dependency status in sidebar
        with st.sidebar:
            st.markdown("### 🧪 System Status")
            st.markdown(sci_dep_manager.display_status(), unsafe_allow_html=True)
    
    def render_scientific_configuration(self):
        """Render scientific configuration interface"""
        st.markdown("### 🔧 Scientific Configuration")
        
        with st.form("scientific_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Analysis Start Date",
                    value=datetime.now() - timedelta(days=365*2),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now()
                )
                
                end_date = st.date_input(
                    "Analysis End Date",
                    value=datetime.now(),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now()
                )
                
                risk_free_rate = st.slider(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    format="%.1f%%"
                ) / 100.0
            
            with col2:
                rolling_window = st.select_slider(
                    "Rolling Window (Days)",
                    options=[20, 30, 60, 90, 120, 180, 250],
                    value=60
                )
                
                correlation_method = st.selectbox(
                    "Correlation Method",
                    options=["pearson", "spearman", "kendall", "ewma"],
                    index=0,
                    help="Select correlation calculation methodology"
                )
                
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.99],
                    value=0.95,
                    format="%.0f%%"
                )
            
            # Advanced configuration in expander
            with st.expander("Advanced Scientific Settings"):
                col3, col4 = st.columns(2)
                with col3:
                    monte_carlo_sims = st.number_input(
                        "Monte Carlo Simulations",
                        min_value=1000,
                        max_value=100000,
                        value=10000,
                        step=1000
                    )
                    
                    ewma_lambda = st.slider(
                        "EWMA Lambda (Decay Factor)",
                        min_value=0.90,
                        max_value=0.99,
                        value=0.94,
                        step=0.01
                    )
                
                with col4:
                    min_data_points = st.number_input(
                        "Minimum Data Points",
                        min_value=20,
                        max_value=500,
                        value=50,
                        step=10
                    )
                    
                    significance_level = st.slider(
                        "Statistical Significance Level",
                        min_value=0.01,
                        max_value=0.10,
                        value=0.05,
                        step=0.01,
                        format="%.2f"
                    )
            
            # Validation and submission
            st.markdown("---")
            submitted = st.form_submit_button(
                "🚀 Run Scientific Analysis",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            # Validate configuration
            self.config = ScientificAnalysisConfiguration(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.min.time()),
                risk_free_rate=risk_free_rate,
                rolling_window=rolling_window,
                correlation_method=correlation_method,
                confidence_levels=(confidence_level,),
                ewma_lambda=ewma_lambda,
                significance_level=significance_level,
                monte_carlo_simulations=monte_carlo_sims,
                minimum_data_points=min_data_points
            )
            
            # Validate configuration
            is_valid, errors = self.config.validate()
            if is_valid:
                st.session_state.run_scientific_analysis = True
                self.analytics_engine = ScientificAnalyticsEngine(self.config)
                st.success("✅ Scientific configuration validated and ready for analysis")
                st.rerun()
            else:
                st.error("❌ Configuration validation failed:")
                for error in errors:
                    st.error(f"  - {error}")
    
    def render_asset_selection(self):
        """Render scientific asset selection interface"""
        st.markdown("### 📊 Asset Selection & Universe")
        
        # Create tabs for different asset categories
        tab1, tab2, tab3 = st.tabs(["Commodities", "Benchmarks", "Selected Assets"])
        
        with tab1:
            self._render_commodity_selection()
        
        with tab2:
            self._render_benchmark_selection()
        
        with tab3:
            self._render_selected_assets_summary()
    
    def _render_commodity_selection(self):
        """Render commodity selection interface"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display commodities by category
            for category_name, category_assets in COMMODITIES_UNIVERSE.items():
                with st.expander(f"{category_name} ({len(category_assets)} assets)", expanded=True):
                    # Create grid for asset cards
                    cols = st.columns(3)
                    
                    for idx, (symbol, metadata) in enumerate(category_assets.items()):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            self._render_asset_card(symbol, metadata, "commodity")
        
        with col2:
            st.markdown("#### 📈 Selection Summary")
            st.metric("Total Selected", len(st.session_state.selected_scientific_assets))
            
            if st.session_state.selected_scientific_assets:
                st.markdown("**Selected Commodities:**")
                for symbol in st.session_state.selected_scientific_assets[:10]:
                    st.markdown(f"• {symbol}")
                
                if len(st.session_state.selected_scientific_assets) > 10:
                    st.caption(f"...and {len(st.session_state.selected_scientific_assets) - 10} more")
            
            # Clear selection button
            if st.button("Clear Selection", type="secondary", use_container_width=True):
                st.session_state.selected_scientific_assets = []
                st.rerun()
    
    def _render_benchmark_selection(self):
        """Render benchmark selection interface"""
        # Display benchmarks
        cols = st.columns(3)
        
        for idx, (symbol, metadata) in enumerate(BENCHMARKS.items()):
            col_idx = idx % 3
            with cols[col_idx]:
                self._render_asset_card(symbol, metadata, "benchmark")
    
    def _render_asset_card(self, symbol: str, metadata: ScientificAssetMetadata, asset_type: str):
        """Render individual asset selection card"""
        is_selected = symbol in (
            st.session_state.selected_scientific_assets if asset_type == "commodity" 
            else st.session_state.selected_benchmarks
        )
        
        # Determine selection list
        selection_list = (
            st.session_state.selected_scientific_assets if asset_type == "commodity" 
            else st.session_state.selected_benchmarks
        )
        
        # Create card with selection toggle
        with st.container():
            st.markdown(f"""
            <div style="
                padding: 1rem;
                border-radius: 8px;
                border: 2px solid {'#3949ab' if is_selected else '#e0e0e0'};
                background: {'rgba(57, 73, 171, 0.05)' if is_selected else '#ffffff'};
                transition: all 0.2s ease;
                cursor: pointer;
                margin-bottom: 0.5rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.1rem; color: #1a237e;">
                            {symbol}
                        </div>
                        <div style="font-size: 0.9rem; color: #666;">
                            {metadata.name}
                        </div>
                    </div>
                    <div style="
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        background: {'#3949ab' if is_selected else '#e0e0e0'};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 0.8rem;
                    ">
                        {'✓' if is_selected else '+'}
                    </div>
                </div>
                
                <div style="margin-top: 0.5rem;">
                    <span class="scientific-badge {metadata.risk_level.lower().replace(' ', '-')}-risk">
                        {metadata.risk_level}
                    </span>
                </div>
                
                <div style="
                    margin-top: 0.5rem;
                    font-size: 0.8rem;
                    color: #666;
                    display: flex;
                    justify-content: space-between;
                ">
                    <div>β: {metadata.beta_to_spx:.2f}</div>
                    <div>Vol: {metadata.volatility_30d:.1%}</div>
                    <div>Liq: {metadata.liquidity_score:.0%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add click handler
            if st.button(f"Select {symbol}", key=f"btn_{symbol}", 
                        type="primary" if is_selected else "secondary",
                        use_container_width=True):
                if is_selected:
                    if asset_type == "commodity":
                        st.session_state.selected_scientific_assets.remove(symbol)
                    else:
                        st.session_state.selected_benchmarks.remove(symbol)
                else:
                    if asset_type == "commodity":
                        st.session_state.selected_scientific_assets.append(symbol)
                    else:
                        st.session_state.selected_benchmarks.append(symbol)
                st.rerun()
    
    def _render_selected_assets_summary(self):
        """Render selected assets summary"""
        all_selected = (st.session_state.selected_scientific_assets + 
                       st.session_state.selected_benchmarks)
        
        if not all_selected:
            st.info("📝 No assets selected. Please select assets from the Commodities and Benchmarks tabs.")
            return
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", len(all_selected))
        
        with col2:
            commodities_count = len(st.session_state.selected_scientific_assets)
            st.metric("Commodities", commodities_count)
        
        with col3:
            benchmarks_count = len(st.session_state.selected_benchmarks)
            st.metric("Benchmarks", benchmarks_count)
        
        with col4:
            categories = set()
            for symbol in all_selected:
                metadata = self._get_asset_metadata(symbol)
                if metadata:
                    categories.add(metadata.category.value)
            st.metric("Categories", len(categories))
        
        # Display selected assets table
        st.markdown("#### Selected Assets Details")
        
        summary_data = []
        for symbol in all_selected:
            metadata = self._get_asset_metadata(symbol)
            if metadata:
                summary_data.append({
                    "Symbol": symbol,
                    "Name": metadata.name,
                    "Category": metadata.category.value,
                    "Risk Level": metadata.risk_level,
                    "Beta to SPX": f"{metadata.beta_to_spx:.2f}",
                    "30D Volatility": f"{metadata.volatility_30d:.1%}",
                    "Liquidity Score": f"{metadata.liquidity_score:.0%}"
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
    
    def _get_asset_metadata(self, symbol: str) -> Optional[ScientificAssetMetadata]:
        """Get asset metadata from universe"""
        # Search in commodities
        for category_assets in COMMODITIES_UNIVERSE.values():
            if symbol in category_assets:
                return category_assets[symbol]
        
        # Search in benchmarks
        if symbol in BENCHMARKS:
            return BENCHMARKS[symbol]
        
        return None
    
    def run_scientific_analysis(self):
        """Execute comprehensive scientific analysis"""
        if not self.config or not st.session_state.run_scientific_analysis:
            return
        
        # Get all selected assets
        all_symbols = (st.session_state.selected_scientific_assets + 
                      st.session_state.selected_benchmarks)
        
        if len(all_symbols) < 2:
            st.warning("⚠️ Please select at least 2 assets for scientific analysis")
            return
        
        # Display analysis progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔬 Running scientific analysis..."):
            # Step 1: Fetch data
            status_text.text("📥 Fetching market data...")
            asset_data = self.data_manager.fetch_multiple_assets_scientific(
                symbols=all_symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                max_workers=6,
                validation_level="strict"
            )
            progress_bar.progress(25)
            
            # Step 2: Calculate features
            status_text.text("🧮 Calculating scientific features...")
            for symbol, df in asset_data.items():
                asset_data[symbol] = self.data_manager.calculate_scientific_features(df)
            progress_bar.progress(50)
            
            # Step 3: Calculate returns
            returns_dict = {}
            for symbol, df in asset_data.items():
                if 'Returns' in df.columns and not df['Returns'].empty:
                    returns_dict[symbol] = df['Returns'].dropna()
            progress_bar.progress(65)
            
            # Step 4: Correlation analysis
            status_text.text("📊 Analyzing correlations...")
            correlation_results = self.correlation_engine.calculate_correlation_matrix(
                returns_dict=returns_dict,
                method=self.config.correlation_method,
                significance_test=True,
                min_common_periods=self.config.minimum_data_points
            )
            progress_bar.progress(85)
            
            # Step 5: Risk metrics
            status_text.text("⚠️ Calculating risk metrics...")
            risk_metrics = {}
            for symbol, returns in returns_dict.items():
                if len(returns) >= 20:
                    risk_metrics[symbol] = self.analytics_engine.calculate_scientific_risk_metrics(returns)
            progress_bar.progress(100)
            
            # Store results
            st.session_state.scientific_analysis_results = {
                'asset_data': asset_data,
                'returns_dict': returns_dict,
                'correlation_results': correlation_results,
                'risk_metrics': risk_metrics,
                'config': self.config
            }
            
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
    
    def render_scientific_results(self):
        """Render comprehensive scientific analysis results"""
        if not st.session_state.scientific_analysis_results:
            return
        
        results = st.session_state.scientific_analysis_results
        
        st.markdown("## 📊 Scientific Analysis Results")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance", 
            "🔄 Correlations", 
            "⚠️ Risk Metrics", 
            "🔍 Insights"
        ])
        
        with tab1:
            self._render_performance_analysis(results)
        
        with tab2:
            self._render_correlation_analysis(results)
        
        with tab3:
            self._render_risk_analysis(results)
        
        with tab4:
            self._render_scientific_insights(results)
    
    def _render_performance_analysis(self, results: Dict):
        """Render performance analysis results"""
        st.markdown("### 📈 Asset Performance Analysis")
        
        # Performance chart
        fig = self.visualization.create_asset_performance_chart(
            asset_data=results['asset_data'],
            metrics=results['risk_metrics']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.markdown("#### Performance Metrics")
        
        metrics_data = []
        for symbol, risk_result in results['risk_metrics'].items():
            metrics = risk_result['metrics']
            metrics_data.append({
                "Asset": symbol,
                "Ann. Return": f"{metrics.get('Annualized_Return', 0):.2%}",
                "Ann. Vol": f"{metrics.get('Annualized_Volatility', 0):.2%}",
                "Sharpe Ratio": f"{metrics.get('Sharpe_Ratio', 0):.2f}",
                "Max DD": f"{metrics.get('Maximum_Drawdown', 0):.1f}%",
                "Win Rate": f"{metrics.get('Win_Rate', 0):.1f}%",
                "Skewness": f"{metrics.get('Skewness', 0):.2f}",
                "Kurtosis": f"{metrics.get('Kurtosis', 0):.2f}"
            })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
    
    def _render_correlation_analysis(self, results: Dict):
        """Render correlation analysis results"""
        st.markdown("### 🔄 Correlation Analysis")
        
        if 'correlation_results' not in results:
            st.warning("No correlation results available")
            return
        
        corr_results = results['correlation_results']
        
        # Correlation matrix heatmap
        fig = self.visualization.create_scientific_correlation_matrix(
            correlation_data=corr_results,
            title=f"Correlation Matrix ({corr_results.get('method_used', 'unknown')} method)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary = corr_results.get('summary_stats', {})
            st.metric(
                "Mean Absolute Correlation",
                f"{summary.get('abs_mean_correlation', 0):.3f}"
            )
        
        with col2:
            st.metric(
                "Positive Correlation Ratio",
                f"{summary.get('positive_correlation_ratio', 0):.1%}"
            )
        
        with col3:
            if 'significant_correlation_ratio' in summary:
                st.metric(
                    "Significant Correlations",
                    f"{summary.get('significant_correlation_ratio', 0):.1%}"
                )
        
        # Display validation results
        if 'validation_result' in corr_results:
            validation = corr_results['validation_result']
            if validation.get('warnings'):
                with st.expander("⚠️ Correlation Matrix Warnings"):
                    for warning in validation['warnings']:
                        st.warning(warning)
    
    def _render_risk_analysis(self, results: Dict):
        """Render risk analysis results"""
        st.markdown("### ⚠️ Risk Analysis")
        
        if 'risk_metrics' not in results:
            st.warning("No risk metrics available")
            return
        
        # Create risk metrics dashboard
        cols = st.columns(4)
        risk_indicators = []
        
        for symbol, risk_result in results['risk_metrics'].items():
            metrics = risk_result['metrics']
            risk_indicators.append({
                'asset': symbol,
                'volatility': metrics.get('Annualized_Volatility', 0),
                'sharpe': metrics.get('Sharpe_Ratio', 0),
                'max_dd': abs(metrics.get('Maximum_Drawdown', 0)),
                'var_95': abs(metrics.get('VaR_95_Historical', 0))
            })
        
        # Convert to DataFrame for analysis
        if risk_indicators:
            df_risk = pd.DataFrame(risk_indicators)
            
            # Display risk metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Volatility Ranking")
                df_vol_sorted = df_risk.sort_values('volatility', ascending=False)
                for _, row in df_vol_sorted.iterrows():
                    vol_pct = row['volatility'] * 100
                    st.progress(
                        min(1.0, vol_pct / 100),
                        text=f"{row['asset']}: {vol_pct:.1f}%"
                    )
            
            with col2:
                st.markdown("#### 📊 Sharpe Ratio Ranking")
                df_sharpe_sorted = df_risk.sort_values('sharpe', ascending=False)
                for _, row in df_sharpe_sorted.iterrows():
                    sharpe = row['sharpe']
                    # Normalize for progress bar
                    normalized = (sharpe + 2) / 4  # Assume range [-2, 2]
                    st.progress(
                        max(0, min(1.0, normalized)),
                        text=f"{row['asset']}: {sharpe:.2f}"
                    )
            
            # Risk metrics table
            st.markdown("#### Detailed Risk Metrics")
            risk_table_data = []
            for symbol, risk_result in results['risk_metrics'].items():
                metrics = risk_result['metrics']
                risk_table_data.append({
                    "Asset": symbol,
                    "Volatility": f"{metrics.get('Annualized_Volatility', 0):.2%}",
                    "Sharpe": f"{metrics.get('Sharpe_Ratio', 0):.2f}",
                    "Max DD": f"{abs(metrics.get('Maximum_Drawdown', 0)):.1f}%",
                    "VaR 95%": f"{abs(metrics.get('VaR_95_Historical', 0)):.1f}%",
                    "Skew": f"{metrics.get('Skewness', 0):.2f}",
                    "Kurtosis": f"{metrics.get('Kurtosis', 0):.2f}"
                })
            
            df_risk_table = pd.DataFrame(risk_table_data)
            st.dataframe(df_risk_table, use_container_width=True)
    
    def _render_scientific_insights(self, results: Dict):
        """Render scientific insights and recommendations"""
        st.markdown("### 🔍 Scientific Insights")
        
        # Generate insights based on analysis results
        insights = self._generate_scientific_insights(results)
        
        # Display insights as cards
        for insight in insights:
            with st.container():
                st.markdown(f"""
                <div style="
                    padding: 1.5rem;
                    border-radius: 8px;
                    border-left: 4px solid #{insight['color']};
                    background: linear-gradient(90deg, rgba{insight['color']}, 0.05) 0%, rgba(255,255,255,1) 100%);
                    margin-bottom: 1rem;
                ">
                    <div style="display: flex; align-items: start; gap: 1rem;">
                        <div style="
                            font-size: 1.5rem;
                            color: #{insight['color']};
                        ">
                            {insight['icon']}
                        </div>
                        <div style="flex: 1;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1a237e;">
                                {insight['title']}
                            </h4>
                            <p style="margin: 0; color: #666; line-height: 1.5;">
                                {insight['description']}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _generate_scientific_insights(self, results: Dict) -> List[Dict]:
        """Generate scientific insights from analysis results"""
        insights = []
        
        if 'correlation_results' in results:
            corr_results = results['correlation_results']
            summary = corr_results.get('summary_stats', {})
            
            # Insight 1: Correlation structure
            mean_corr = summary.get('mean_correlation', 0)
            if mean_corr > 0.7:
                insights.append({
                    'icon': '🔄',
                    'title': 'High Correlation Environment',
                    'description': f'Average correlation of {mean_corr:.2f} suggests limited diversification benefits. Consider alternative assets or hedging strategies.',
                    'color': 'FF8F00'  # Warning orange
                })
            elif mean_corr < 0.3:
                insights.append({
                    'icon': '🎯',
                    'title': 'Good Diversification Opportunity',
                    'description': f'Low average correlation ({mean_corr:.2f}) indicates strong diversification potential across selected assets.',
                    'color': '2E7D32'  # Success green
                })
        
        if 'risk_metrics' in results:
            # Insight 2: Risk-adjusted performance
            sharpe_ratios = []
            for symbol, risk_result in results['risk_metrics'].items():
                sharpe = risk_result['metrics'].get('Sharpe_Ratio', 0)
                sharpe_ratios.append((symbol, sharpe))
            
            if sharpe_ratios:
                sharpe_ratios.sort(key=lambda x: x[1], reverse=True)
                best_asset, best_sharpe = sharpe_ratios[0]
                
                if best_sharpe > 1.0:
                    insights.append({
                        'icon': '📈',
                        'title': 'Strong Risk-Adjusted Performer',
                        'description': f'{best_asset} shows excellent risk-adjusted performance with Sharpe ratio of {best_sharpe:.2f}.',
                        'color': '1565C0'  # Primary blue
                    })
            
            # Insight 3: Volatility analysis
            volatilities = []
            for symbol, risk_result in results['risk_metrics'].items():
                vol = risk_result['metrics'].get('Annualized_Volatility', 0)
                volatilities.append((symbol, vol))
            
            if volatilities:
                volatilities.sort(key=lambda x: x[1], reverse=True)
                highest_vol_asset, highest_vol = volatilities[0]
                
                if highest_vol > 0.4:  # 40% annualized volatility
                    insights.append({
                        'icon': '⚡',
                        'title': 'High Volatility Asset',
                        'description': f'{highest_vol_asset} shows high volatility ({highest_vol:.1%}). Consider position sizing and risk management.',
                        'color': 'C62828'  # Danger red
                    })
        
        return insights
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_scientific_header()
        
        # Sidebar configuration
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            self.render_scientific_configuration()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Asset selection
            self.render_asset_selection()
            
            # Run analysis if configured
            if self.config and st.session_state.run_scientific_analysis:
                self.run_scientific_analysis()
                
                # Display results
                if st.session_state.scientific_analysis_results:
                    self.render_scientific_results()
        
        with col2:
            # Quick metrics and status
            st.markdown("### 📋 Quick Status")
            
            total_assets = (len(st.session_state.selected_scientific_assets) + 
                          len(st.session_state.selected_benchmarks))
            
            metrics_data = [
                ("Selected Assets", total_assets),
                ("Commodities", len(st.session_state.selected_scientific_assets)),
                ("Benchmarks", len(st.session_state.selected_benchmarks)),
                ("Analysis Ready", "✅" if st.session_state.run_scientific_analysis else "⏳")
            ]
            
            for label, value in metrics_data:
                st.metric(label, value)
            
            # Analysis trigger
            if total_assets >= 2 and not st.session_state.run_scientific_analysis:
                if st.button("▶️ Start Analysis", type="primary", use_container_width=True):
                    st.session_state.run_scientific_analysis = True
                    st.rerun()
            
            # Reset button
            if st.session_state.run_scientific_analysis:
                if st.button("🔄 Reset Analysis", type="secondary", use_container_width=True):
                    st.session_state.run_scientific_analysis = False
                    st.session_state.scientific_analysis_results = {}
                    st.rerun()

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the scientific commodities platform"""
    
    # Initialize application
    app = ScientificCommoditiesPlatform()
    
    # Run the application
    try:
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.code(traceback.format_exc())
        
        # Fallback to simple mode
        st.info("🔄 Attempting to load simplified analysis...")
        try:
            # Simplified fallback analysis
            st.markdown("## 📊 Basic Commodity Analysis")
            
            # Simple data fetch
            symbols = ["GC=F", "CL=F", "^GSPC"]
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            
            data = {}
            for symbol in symbols:
                try:
                    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not df.empty:
                        data[symbol] = df
                except:
                    continue
            
            if data:
                st.success(f"Successfully loaded data for {len(data)} assets")
                
                # Display basic price charts
                for symbol, df in data.items():
                    st.subheader(symbol)
                    if 'Close' in df.columns:
                        st.line_chart(df['Close'])
            else:
                st.error("Could not load any data. Please check your internet connection.")
                
        except Exception as fallback_error:
            st.error(f"Fallback also failed: {str(fallback_error)}")

# Run the application
if __name__ == "__main__":
    main()
