"""
🏛️ INSTITUTIONAL COMMODITIES ANALYTICS PLATFORM v7.5
Advanced Quantitative Analytics for Professional Commodity Trading & Risk Management

FEATURES:
• Multi-Asset Portfolio Optimization with Advanced Constraints
• High-Frequency GARCH Volatility Modeling (GARCH, EGARCH, GJRGARCH)
• Machine Learning Regime Detection (HMM, Random Forest, LSTM)
• Advanced Risk Metrics (CVaR, Expected Shortfall, Stress Testing)
• Monte Carlo Simulation with Jump Diffusion & Stochastic Volatility
• Real-time Market Data Integration & News Sentiment Analysis
• Professional Reporting (PDF, Excel, Interactive Dashboards)
• Backtesting Engine with Transaction Costs & Slippage
• Factor Analysis & Smart Beta Strategies
• Alternative Data Integration (Weather, Supply Chain, Macro)

ARCHITECTURE:
• Microservices-ready with async/await support
• GPU acceleration for ML models (CUDA optional)
• Redis caching layer for high-frequency data
• WebSocket real-time updates
• Docker containerization ready
• CI/CD pipeline integration

© 2024 Institutional Trading Analytics. All Rights Reserved.
"""

# =============================================================================
# CONFIGURATION & IMPORTS (COMPREHENSIVE)
# =============================================================================

import os
import sys
import math
import time
import json
import yaml
import pickle
import hashlib
import secrets
import asyncio
import inspect
import textwrap
import warnings
import traceback
import itertools
import subprocess
import threading
import concurrent.futures
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import (
    Dict, Any, Optional, Tuple, List, Union, Callable,
    TypeVar, Generic, Type, NamedTuple, Literal,
    Coroutine, AsyncGenerator, Generator, Set, Protocol,
    runtime_checkable
)
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict, Counter
from functools import (
    lru_cache, wraps, partial, reduce, singledispatch,
    total_ordering, cached_property
)
from pathlib import Path, PurePath
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import chain, cycle, islice, groupby
from statistics import mean, median, mode, stdev, variance
from contextlib import (
    contextmanager, asynccontextmanager,
    ExitStack, AsyncExitStack, suppress
)
from importlib import import_module, reload
from urllib.parse import urlparse, urlencode, quote
from copy import deepcopy, copy
from weakref import WeakValueDictionary, WeakSet
from numbers import Number
from pprint import pprint, pformat

# Environment setup for performance
os.environ.update({
    "NUMEXPR_MAX_THREADS": "8",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "PYTHONWARNINGS": "ignore",
    "TF_CPP_MIN_LOG_LEVEL": "3",  # TensorFlow logging
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "CUDA_VISIBLE_DEVICES": "0",  # GPU management
})

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Streamlit configuration must be first
import streamlit as st
st.set_page_config(
    page_title="Institutional Commodities Platform v7.5",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://institutional-commodities.com/docs',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v7.5
                    Advanced quantitative platform for institutional commodity trading
                    © 2024 Institutional Trading Analytics. All rights reserved."""
    }
)

# =============================================================================
# CORE DEPENDENCIES WITH GRACEFUL FALLBACKS
# =============================================================================

class DependencyManager:
    """Advanced dependency management with lazy loading and fallbacks"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._dependencies = {}
            self._version_cache = {}
            self._load_all_dependencies()
            self._initialized = True
    
    def _load_all_dependencies(self):
        """Load all optional dependencies with comprehensive error handling"""
        
        # Data manipulation and numerical computing
        self._try_import("numpy", "np")
        self._try_import("pandas", "pd")
        self._try_import("scipy", "sp")
        self._try_import("scipy.stats", "stats")
        self._try_import("scipy.optimize", "optimize")
        self._try_import("scipy.signal", "signal")
        self._try_import("scipy.interpolate", "interpolate")
        self._try_import("scipy.spatial", "spatial")
        self._try_import("scipy.sparse", "sparse")
        self._try_import("scipy.linalg", "linalg")
        self._try_import("scipy.integrate", "integrate")
        
        # Visualization
        self._try_import("plotly.graph_objects", "go")
        self._try_import("plotly.express", "px")
        self._try_import("plotly.subplots", "make_subplots")
        self._try_import("plotly.figure_factory", "create_distplot")
        
        # Optional visualization
        try:
            import seaborn as sns
            self._dependencies['seaborn'] = {'module': sns, 'available': True}
        except:
            self._dependencies['seaborn'] = {'available': False}
        
        # Financial data
        self._try_import("yfinance", "yf")
        
        # Advanced statistics and econometrics
        self._try_import_module("statsmodels", [
            ("statsmodels.api", "sm"),
            ("statsmodels.tsa.stattools", "stattools"),
            ("statsmodels.tsa.arima.model", "ARIMA"),
            ("statsmodels.tsa.statespace.sarimax", "SARIMAX"),
            ("statsmodels.tsa.vector_ar.var_model", "VAR"),
            ("statsmodels.regression.linear_model", "OLS"),
            ("statsmodels.regression.rolling", "RollingOLS"),
            ("statsmodels.stats.diagnostic", "diagnostic"),
            ("statsmodels.tsa.api", "tsa"),
            ("statsmodels.tsa.regime_switching.markov_regression", "MarkovRegression"),
            ("statsmodels.tsa.regime_switching.markov_autoregression", "MarkovAutoregression"),
            ("statsmodels.discrete.discrete_model", "discrete"),
            ("statsmodels.genmod.generalized_linear_model", "GLM"),
            ("statsmodels.robust.robust_linear_model", "RLM"),
            ("statsmodels.tsa.holtwinters", "ExponentialSmoothing"),
            ("statsmodels.tsa.filters.filtertools", "filters"),
            ("statsmodels.tsa.x13", "x13"),
        ])
        
        # ARCH/GARCH modeling
        self._try_import_module("arch", [
            ("arch.univariate", "univariate"),
            ("arch.univariate.mean", "mean_models"),
            ("arch.univariate.volatility", "volatility_models"),
            ("arch.univariate.distribution", "distribution_models"),
            ("arch.bootstrap", "bootstrap"),
            ("arch.covariance", "covariance"),
        ])
        
        # Machine Learning
        self._try_import_module("sklearn", [
            ("sklearn.preprocessing", "preprocessing"),
            ("sklearn.decomposition", "decomposition"),
            ("sklearn.cluster", "cluster"),
            ("sklearn.ensemble", "ensemble"),
            ("sklearn.linear_model", "linear_model"),
            ("sklearn.neighbors", "neighbors"),
            ("sklearn.neural_network", "neural_network"),
            ("sklearn.svm", "svm"),
            ("sklearn.tree", "tree"),
            ("sklearn.model_selection", "model_selection"),
            ("sklearn.metrics", "metrics"),
            ("sklearn.feature_selection", "feature_selection"),
            ("sklearn.feature_extraction", "feature_extraction"),
            ("sklearn.pipeline", "pipeline"),
            ("sklearn.impute", "impute"),
            ("sklearn.compose", "compose"),
            ("sklearn.manifold", "manifold"),
            ("sklearn.covariance", "sklearn_covariance"),
            ("sklearn.isotonic", "isotonic"),
            ("sklearn.kernel_ridge", "kernel_ridge"),
            ("sklearn.gaussian_process", "gaussian_process"),
            ("sklearn.cross_decomposition", "cross_decomposition"),
            ("sklearn.discriminant_analysis", "discriminant_analysis"),
        ])
        
        # HMM
        try:
            from hmmlearn import hmm
            self._dependencies['hmmlearn'] = {
                'module': hmm,
                'available': True,
                'GaussianHMM': hmm.GaussianHMM,
                'GMMHMM': hmm.GMMHMM,
                'MultinomialHMM': hmm.MultinomialHMM,
            }
        except:
            self._dependencies['hmmlearn'] = {'available': False}
        
        # Deep Learning (optional)
        self._try_import("tensorflow", "tf")
        self._try_import("torch", "torch")
        
        # Time series specific
        try:
            import properscoring as ps
            self._dependencies['properscoring'] = {'module': ps, 'available': True}
        except:
            self._dependencies['properscoring'] = {'available': False}
        
        # Financial analytics
        self._try_import("quantstats", "qs")
        self._try_import("ta", "ta_lib")
        
        # Alternative data sources
        try:
            import quandl
            self._dependencies['quandl'] = {'module': quandl, 'available': True}
        except:
            self._dependencies['quandl'] = {'available': False}
        
        try:
            import fredapi
            self._dependencies['fredapi'] = {'module': fredapi, 'available': True}
        except:
            self._dependencies['fredapi'] = {'available': False}
        
        # Database and caching
        try:
            import redis
            self._dependencies['redis'] = {'module': redis, 'available': True}
        except:
            self._dependencies['redis'] = {'available': False}
        
        try:
            import sqlalchemy
            self._dependencies['sqlalchemy'] = {'module': sqlalchemy, 'available': True}
        except:
            self._dependencies['sqlalchemy'] = {'available': False}
        
        # Excel and PDF export
        self._try_import("openpyxl", "openpyxl")
        self._try_import("xlsxwriter", "xlsxwriter")
        
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4, landscape
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.graphics.shapes import Drawing, String
            from reportlab.graphics.charts.lineplots import LinePlot
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            from reportlab.graphics import renderPDF
            self._dependencies['reportlab'] = {
                'available': True,
                'colors': colors,
                'pagesizes': {'letter': letter, 'A4': A4, 'landscape': landscape},
                'SimpleDocTemplate': SimpleDocTemplate,
                'Table': Table,
                'TableStyle': TableStyle,
                'Paragraph': Paragraph,
                'Spacer': Spacer,
                'getSampleStyleSheet': getSampleStyleSheet,
                'ParagraphStyle': ParagraphStyle,
                'inch': inch,
                'Drawing': Drawing,
                'String': String,
                'LinePlot': LinePlot,
                'VerticalBarChart': VerticalBarChart,
                'renderPDF': renderPDF,
            }
        except:
            self._dependencies['reportlab'] = {'available': False}
        
        # Web scraping and APIs
        try:
            import requests
            self._dependencies['requests'] = {'module': requests, 'available': True}
        except:
            self._dependencies['requests'] = {'available': False}
        
        try:
            import beautifulsoup4
            from bs4 import BeautifulSoup
            self._dependencies['beautifulsoup4'] = {
                'module': beautifulsoup4,
                'BeautifulSoup': BeautifulSoup,
                'available': True
            }
        except:
            self._dependencies['beautifulsoup4'] = {'available': False}
        
        # Natural Language Processing
        try:
            import nltk
            self._dependencies['nltk'] = {'module': nltk, 'available': True}
        except:
            self._dependencies['nltk'] = {'available': False}
        
        try:
            import textblob
            self._dependencies['textblob'] = {'module': textblob, 'available': True}
        except:
            self._dependencies['textblob'] = {'available': False}
        
        # Geospatial data (for weather, logistics)
        try:
            import geopandas
            self._dependencies['geopandas'] = {'module': geopandas, 'available': True}
        except:
            self._dependencies['geopandas'] = {'available': False}
        
        try:
            import shapely
            self._dependencies['shapely'] = {'module': shapely, 'available': True}
        except:
            self._dependencies['shapely'] = {'available': False}
        
        # Performance monitoring
        try:
            import psutil
            self._dependencies['psutil'] = {'module': psutil, 'available': True}
        except:
            self._dependencies['psutil'] = {'available': False}
        
        # Parallel processing
        try:
            import dask
            import dask.dataframe as dd
            self._dependencies['dask'] = {
                'module': dask,
                'dataframe': dd,
                'available': True
            }
        except:
            self._dependencies['dask'] = {'available': False}
        
        # Date manipulation
        try:
            import pendulum
            self._dependencies['pendulum'] = {'module': pendulum, 'available': True}
        except:
            self._dependencies['pendulum'] = {'available': False}
        
        # Configuration management
        try:
            import hydra
            from omegaconf import OmegaConf
            self._dependencies['hydra'] = {
                'module': hydra,
                'OmegaConf': OmegaConf,
                'available': True
            }
        except:
            self._dependencies['hydra'] = {'available': False}
        
        # Type checking and validation
        try:
            import pydantic
            self._dependencies['pydantic'] = {'module': pydantic, 'available': True}
        except:
            self._dependencies['pydantic'] = {'available': False}
        
        # Logging and monitoring
        try:
            import structlog
            self._dependencies['structlog'] = {'module': structlog, 'available': True}
        except:
            self._dependencies['structlog'] = {'available': False}
    
    def _try_import(self, module_name: str, alias: str = None):
        """Try importing a module with error handling"""
        try:
            module = import_module(module_name)
            self._dependencies[module_name] = {
                'module': module,
                'available': True,
                'alias': alias
            }
            
            # Get version if available
            try:
                version = getattr(module, '__version__', 'Unknown')
                self._version_cache[module_name] = version
            except:
                pass
                
        except ImportError as e:
            self._dependencies[module_name] = {
                'available': False,
                'error': str(e)
            }
    
    def _try_import_module(self, base_module: str, imports: List[Tuple[str, str]]):
        """Try importing multiple submodules from a base module"""
        try:
            base = import_module(base_module)
            self._dependencies[base_module] = {
                'base': base,
                'available': True,
                'submodules': {}
            }
            
            for import_path, alias in imports:
                try:
                    module = import_module(import_path)
                    self._dependencies[base_module]['submodules'][alias] = module
                except ImportError:
                    pass
            
            # Store version
            try:
                version = getattr(base, '__version__', 'Unknown')
                self._version_cache[base_module] = version
            except:
                pass
                
        except ImportError as e:
            self._dependencies[base_module] = {
                'available': False,
                'error': str(e)
            }
    
    def is_available(self, dependency: str) -> bool:
        """Check if dependency is available"""
        return self._dependencies.get(dependency, {}).get('available', False)
    
    def get_module(self, dependency: str):
        """Get dependency module if available"""
        dep = self._dependencies.get(dependency, {})
        return dep.get('module') if dep.get('available') else None
    
    def get_submodule(self, dependency: str, submodule: str):
        """Get specific submodule"""
        dep = self._dependencies.get(dependency, {})
        if dep.get('available'):
            return dep.get('submodules', {}).get(submodule)
        return None
    
    def get_version(self, dependency: str) -> str:
        """Get dependency version"""
        return self._version_cache.get(dependency, 'Unknown')
    
    def check_requirements(self, requirements: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Check if required dependencies and versions are available"""
        results = {}
        for dep, min_version in requirements.items():
            available = self.is_available(dep)
            version = self.get_version(dep)
            
            # Simple version comparison (basic)
            version_ok = True
            if available and min_version != 'any' and version != 'Unknown':
                try:
                    from packaging import version as pkg_version
                    version_ok = pkg_version.parse(version) >= pkg_version.parse(min_version)
                except:
                    version_ok = True  # If we can't parse, assume OK
            
            results[dep] = {
                'available': available,
                'version': version,
                'version_ok': version_ok,
                'meets_requirement': available and version_ok
            }
        
        return results
    
    def install_missing(self, requirements: Dict[str, str]) -> List[str]:
        """Install missing dependencies (for development/debugging)"""
        missing = []
        for dep, version in requirements.items():
            if not self.is_available(dep):
                missing.append(f"{dep}{version if version != 'any' else ''}")
        
        if missing:
            try:
                import pip
                for package in missing:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                # Reload dependencies
                self._load_all_dependencies()
                return missing
            except:
                return missing
        return []
    
    def get_dependency_report(self) -> str:
        """Generate a comprehensive dependency report"""
        report = []
        report.append("=" * 80)
        report.append("DEPENDENCY REPORT")
        report.append("=" * 80)
        
        for dep_name, dep_info in sorted(self._dependencies.items()):
            status = "✓ AVAILABLE" if dep_info.get('available') else "✗ MISSING"
            version = self.get_version(dep_name)
            report.append(f"{dep_name:30} {status:15} v{version}")
            
            if not dep_info.get('available'):
                error = dep_info.get('error', 'Unknown error')
                report.append(f"  Error: {error}")
        
        report.append("=" * 80)
        return "\n".join(report)

# Initialize dependency manager
dep_manager = DependencyManager()

# =============================================================================
# ADVANCED CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class ApplicationConfig:
    """Comprehensive application configuration"""
    
    # Core settings
    app_name: str = "Institutional Commodities Platform"
    app_version: str = "v7.5.0"
    app_build: str = "2024.12.15"
    environment: str = "production"  # development, staging, production
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Data settings
    default_start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365*3))
    default_end_date: datetime = field(default_factory=datetime.now)
    data_cache_ttl: int = 3600  # seconds
    max_data_points: int = 10000
    min_data_points: int = 100
    
    # Financial settings
    risk_free_rate: float = 0.02
    annual_trading_days: int = 252
    monthly_trading_days: int = 21
    quarterly_trading_days: int = 63
    default_currency: str = "USD"
    
    # Analysis settings
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    garch_model_types: Tuple[str, ...] = ("GARCH", "EGARCH", "GJRGARCH")
    default_garch_p: int = 1
    default_garch_q: int = 1
    regime_states: int = 3
    monte_carlo_simulations: int = 10000
    monte_carlo_horizon: int = 252
    
    # Portfolio optimization
    optimization_methods: Tuple[str, ...] = ("sharpe", "min_variance", "max_return", "risk_parity", "max_diversification")
    default_min_weight: float = 0.0
    default_max_weight: float = 1.0
    max_portfolio_assets: int = 50
    min_correlation_for_diversification: float = -0.3
    
    # Risk management
    var_confidences: Tuple[float, ...] = (0.95, 0.99)
    stress_test_scenarios: Tuple[str, ...] = ("2008 Crisis", "COVID-19", "Inflation Shock", "Supply Chain Crisis")
    default_leverage: float = 1.0
    max_leverage: float = 3.0
    
    # Performance metrics
    performance_metrics: Tuple[str, ...] = (
        "total_return", "annual_return", "annual_volatility",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "var_95", "cvar_95", "omega_ratio",
        "gain_loss_ratio", "tail_ratio", "common_sense_ratio",
        "information_ratio", "alpha", "beta", "treynor_ratio",
        "tracking_error", "r_squared"
    )
    
    # UI settings
    theme: str = "default"
    refresh_interval: int = 300  # seconds
    max_chart_points: int = 2000
    animation_speed: int = 500  # ms
    
    # Export settings
    export_formats: Tuple[str, ...] = ("csv", "excel", "json", "pdf")
    default_export_format: str = "excel"
    max_export_rows: int = 100000
    
    # API and external services
    yahoo_finance_timeout: int = 30
    max_api_retries: int = 3
    api_rate_limit: int = 100  # requests per minute
    
    # Machine learning
    ml_train_test_split: float = 0.8
    ml_cv_folds: int = 5
    ml_random_state: int = 42
    ml_feature_scaling: bool = True
    ml_cross_validate: bool = True
    
    # Cache settings
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 86400  # 24 hours
    
    # Security
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 365
    audit_log_enabled: bool = True
    
    # Advanced features
    enable_real_time: bool = False
    enable_alternative_data: bool = True
    enable_sentiment_analysis: bool = True
    enable_nlp_features: bool = False
    enable_deep_learning: bool = False
    enable_gpu_acceleration: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Date validation
        if self.default_start_date >= self.default_end_date:
            errors.append("Start date must be before end date")
        
        # Financial validation
        if not 0 <= self.risk_free_rate <= 1:
            errors.append("Risk-free rate must be between 0 and 1")
        
        if self.annual_trading_days <= 0:
            errors.append("Annual trading days must be positive")
        
        # Portfolio validation
        if self.default_min_weight < 0 or self.default_max_weight > 1:
            errors.append("Portfolio weights must be between 0 and 1")
        
        if self.default_min_weight > self.default_max_weight:
            errors.append("Minimum weight cannot exceed maximum weight")
        
        # Risk management
        if self.default_leverage > self.max_leverage:
            errors.append("Default leverage cannot exceed maximum leverage")
        
        # Performance
        if self.monte_carlo_simulations < 1000:
            errors.append("Monte Carlo simulations should be at least 1000")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # Handle datetime serialization
            if isinstance(value, datetime):
                value = value.isoformat()
            result[field.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationConfig':
        """Create from dictionary"""
        # Handle datetime deserialization
        for field in fields(cls):
            if field.name in data and field.type == datetime:
                data[field.name] = datetime.fromisoformat(data[field.name])
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ApplicationConfig':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

# Default configuration
DEFAULT_CONFIG = ApplicationConfig()

# =============================================================================
# ENHANCED ASSET UNIVERSE WITH COMPREHENSIVE METADATA
# =============================================================================

class AssetType(Enum):
    """Types of financial assets"""
    FUTURE = "Future"
    ETF = "ETF"
    INDEX = "Index"
    STOCK = "Stock"
    BOND = "Bond"
    OPTION = "Option"
    CRYPTO = "Cryptocurrency"
    COMMODITY = "Commodity"
    CURRENCY = "Currency"
    INTEREST_RATE = "Interest Rate"
    VOLATILITY = "Volatility"
    REAL_ESTATE = "Real Estate"
    INFRASTRUCTURE = "Infrastructure"
    ALTERNATIVE = "Alternative"

class AssetRegion(Enum):
    """Geographic regions"""
    NORTH_AMERICA = "North America"
    EUROPE = "Europe"
    ASIA_PACIFIC = "Asia Pacific"
    LATIN_AMERICA = "Latin America"
    MIDDLE_EAST = "Middle East"
    AFRICA = "Africa"
    GLOBAL = "Global"
    EMERGING_MARKETS = "Emerging Markets"
    DEVELOPED_MARKETS = "Developed Markets"

class AssetSector(Enum):
    """Economic sectors"""
    ENERGY = "Energy"
    MATERIALS = "Materials"
    INDUSTRIALS = "Industrials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    COMMUNICATION_SERVICES = "Communication Services"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    TECHNOLOGY = "Technology"
    AGRICULTURE = "Agriculture"
    METALS_MINING = "Metals & Mining"
    OIL_GAS = "Oil & Gas"
    RENEWABLE_ENERGY = "Renewable Energy"
    INFRASTRUCTURE = "Infrastructure"
    TRANSPORTATION = "Transportation"
    LOGISTICS = "Logistics"

@dataclass
class AssetSpecifications:
    """Detailed specifications for derivatives"""
    contract_size: str = "Standard"
    tick_size: float = 0.01
    tick_value: float = 10.0
    contract_unit: str = "USD"
    settlement_type: str = "Financial"
    delivery_months: List[str] = field(default_factory=lambda: ["H", "K", "N", "U", "Z"])
    last_trading_day: str = "Third Friday"
    trading_hours: str = "09:00-16:00 EST"
    exchange: str = "CME"
    clearing_house: str = "CME Clearing"
    margin_initial: float = 0.05
    margin_maintenance: float = 0.04
    position_limits: Optional[int] = None
    reportable_positions: Optional[int] = None

@dataclass
class RiskMetrics:
    """Pre-calculated risk metrics for quick reference"""
    annual_volatility: float = 0.0
    beta: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    correlation_spx: float = 0.0
    correlation_gold: float = 0.0
    correlation_usd: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    liquidity_score: float = 0.0  # 0-1 scale
    volatility_regime: str = "Normal"  # Low, Normal, High, Extreme

@dataclass
class FundamentalData:
    """Fundamental data for commodities"""
    supply_demand_balance: Optional[float] = None  # million metric tons
    production: Optional[float] = None
    consumption: Optional[float] = None
    inventories: Optional[float] = None
    inventory_days: Optional[float] = None
    cost_curve_percentile: Optional[float] = None  # 0-100
    geopolitical_risk: float = 0.0  # 0-10 scale
    weather_risk: float = 0.0  # 0-10 scale
    logistics_risk: float = 0.0  # 0-10 scale
    seasonality_factor: float = 1.0
    contango_backwardation: float = 0.0  # Positive = contango, Negative = backwardation

@dataclass
class ESGScore:
    """Environmental, Social, and Governance scores"""
    environmental_score: float = 0.0  # 0-100
    social_score: float = 0.0
    governance_score: float = 0.0
    overall_score: float = 0.0
    carbon_intensity: Optional[float] = None  # tons CO2 per unit
    water_usage: Optional[float] = None
    land_use: Optional[float] = None
    biodiversity_impact: float = 0.0  # 0-10 scale
    human_rights_score: float = 0.0  # 0-10 scale
    community_impact: float = 0.0  # 0-10 scale

@dataclass
class AdvancedAssetMetadata:
    """Comprehensive asset metadata for institutional analysis"""
    
    # Core identification
    symbol: str
    name: str
    description: str
    asset_type: AssetType
    category: str
    region: AssetRegion
    sector: AssetSector
    
    # Visual and UI
    color: str
    icon: str = "📊"
    display_order: int = 0
    
    # Specifications
    specifications: AssetSpecifications = field(default_factory=AssetSpecifications)
    
    # Risk and performance
    risk_level: str = "Medium"  # Low, Medium, High, Very High
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    
    # Fundamental data
    fundamentals: FundamentalData = field(default_factory=FundamentalData)
    
    # ESG
    esg_score: ESGScore = field(default_factory=ESGScore)
    
    # Trading characteristics
    average_daily_volume: float = 0.0
    average_daily_volume_usd: float = 0.0
    bid_ask_spread: float = 0.0  # percentage
    liquidity_tier: int = 1  # 1=High, 2=Medium, 3=Low
    
    # Market data
    current_price: float = 0.0
    price_currency: str = "USD"
    price_decimals: int = 2
    price_update_frequency: str = "Real-time"
    
    # Relationships
    related_assets: List[str] = field(default_factory=list)
    competing_assets: List[str] = field(default_factory=list)
    complementary_assets: List[str] = field(default_factory=list)
    
    # Regulatory
    is_restricted: bool = False
    restrictions: List[str] = field(default_factory=list)
    compliance_notes: str = ""
    
    # Metadata
    data_source: str = "Bloomberg"
    data_quality: str = "High"  # High, Medium, Low
    last_updated: datetime = field(default_factory=datetime.now)
    update_frequency: str = "Daily"
    
    # Advanced analytics
    alpha_model: Optional[str] = None
    risk_model: Optional[str] = None
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    regime_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Handle enum serialization
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedAssetMetadata':
        """Create from dictionary"""
        # Handle enum deserialization
        if 'asset_type' in data and isinstance(data['asset_type'], str):
            data['asset_type'] = AssetType(data['asset_type'])
        if 'region' in data and isinstance(data['region'], str):
            data['region'] = AssetRegion(data['region'])
        if 'sector' in data and isinstance(data['sector'], str):
            data['sector'] = AssetSector(data['sector'])
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        # Handle nested dataclasses
        if 'specifications' in data and isinstance(data['specifications'], dict):
            data['specifications'] = AssetSpecifications(**data['specifications'])
        if 'risk_metrics' in data and isinstance(data['risk_metrics'], dict):
            data['risk_metrics'] = RiskMetrics(**data['risk_metrics'])
        if 'fundamentals' in data and isinstance(data['fundamentals'], dict):
            data['fundamentals'] = FundamentalData(**data['fundamentals'])
        if 'esg_score' in data and isinstance(data['esg_score'], dict):
            data['esg_score'] = ESGScore(**data['esg_score'])
        
        return cls(**data)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for display"""
        return {
            "risk_level": self.risk_level,
            "annual_volatility": f"{self.risk_metrics.annual_volatility:.1%}",
            "max_drawdown": f"{self.risk_metrics.max_drawdown:.1%}",
            "var_95": f"{self.risk_metrics.var_95:.1%}",
            "sharpe_ratio": f"{self.risk_metrics.sharpe_ratio:.2f}",
            "liquidity_score": f"{self.risk_metrics.liquidity_score:.0%}",
            "volatility_regime": self.risk_metrics.volatility_regime
        }
    
    def get_fundamental_summary(self) -> Dict[str, Any]:
        """Get fundamental summary for display"""
        return {
            "supply_demand": self.fundamentals.supply_demand_balance,
            "inventory_days": self.fundamentals.inventory_days,
            "cost_curve": f"{self.fundamentals.cost_curve_percentile:.0f}%" if self.fundamentals.cost_curve_percentile else "N/A",
            "seasonality": f"{self.fundamentals.seasonality_factor:.2f}x",
            "contango_backwardation": f"{self.fundamentals.contango_backwardation:+.2f}%"
        }
    
    def get_esg_summary(self) -> Dict[str, Any]:
        """Get ESG summary for display"""
        return {
            "overall_score": f"{self.esg_score.overall_score:.1f}/100",
            "environmental": f"{self.esg_score.environmental_score:.1f}",
            "social": f"{self.esg_score.social_score:.1f}",
            "governance": f"{self.esg_score.governance_score:.1f}",
            "carbon_intensity": f"{self.esg_score.carbon_intensity:.1f} t/unit" if self.esg_score.carbon_intensity else "N/A"
        }

# Enhanced commodities universe
COMMODITIES_UNIVERSE = {
    "Precious Metals": {
        "GC=F": AdvancedAssetMetadata(
            symbol="GC=F",
            name="Gold Futures",
            description="COMEX Gold Futures - Benchmark for global gold prices. 100 troy ounce contracts.",
            asset_type=AssetType.FUTURE,
            category="Precious Metals",
            region=AssetRegion.GLOBAL,
            sector=AssetSector.METALS_MINING,
            color="#FFD700",
            icon="🥇",
            display_order=1,
            specifications=AssetSpecifications(
                contract_size="100 troy ounces",
                tick_size=0.10,
                tick_value=10.0,
                contract_unit="USD",
                settlement_type="Physical",
                delivery_months=["G", "J", "M", "Q", "V", "Z"],
                last_trading_day="Third last business day",
                trading_hours="08:20-13:30 ET",
                exchange="COMEX",
                clearing_house="CME Clearing",
                margin_initial=0.045,
                margin_maintenance=0.04,
                position_limits=6000,
                reportable_positions=250
            ),
            risk_metrics=RiskMetrics(
                annual_volatility=0.15,
                beta=0.05,
                sharpe_ratio=0.8,
                max_drawdown=0.30,
                var_95=0.025,
                cvar_95=0.035,
                correlation_spx=-0.15,
                correlation_gold=1.0,
                correlation_usd=-0.40,
                skewness=0.2,
                kurtosis=3.5,
                liquidity_score=0.95,
                volatility_regime="Normal"
            ),
            fundamentals=FundamentalData(
                supply_demand_balance=5000,  # metric tons
                production=3500,
                consumption=4000,
                inventories=35000,
                inventory_days=320,
                cost_curve_percentile=65,
                geopolitical_risk=7.5,
                weather_risk=1.0,
                logistics_risk=2.0,
                seasonality_factor=1.05,
                contango_backwardation=0.5
            ),
            esg_score=ESGScore(
                environmental_score=45.0,
                social_score=60.0,
                governance_score=70.0,
                overall_score=58.3,
                carbon_intensity=12.5,
                water_usage=15000,
                land_use=250,
                biodiversity_impact=6.5,
                human_rights_score=5.0,
                community_impact=4.5
            ),
            average_daily_volume=250000,
            average_daily_volume_usd=45000000000,
            bid_ask_spread=0.02,
            liquidity_tier=1,
            current_price=2150.50,
            price_currency="USD",
            price_decimals=2,
            price_update_frequency="Real-time",
            related_assets=["GLD", "IAU", "GDX", "GDXJ"],
            competing_assets=["SI=F", "PL=F"],
            complementary_assets=["US Treasury", "VIX"],
            is_restricted=False,
            data_source="CME/Reuters",
            data_quality="High",
            update_frequency="Real-time",
            alpha_model="Carry + Momentum",
            risk_model="BARRA Commodity",
            factor_exposures={
                "Inflation": 0.85,
                "Real Rates": -0.75,
                "USD": -0.40,
                "Risk Aversion": 0.60,
                "Carry": 0.25,
                "Momentum": 0.15
            },
            regime_sensitivity={
                "High Inflation": 1.25,
                "Risk-Off": 0.85,
                "Growth": 0.60,
                "Recession": 1.10
            }
        ),
        # Additional precious metals...
    },
    # Additional categories...
}

# Enhanced benchmarks with comprehensive metadata
ENHANCED_BENCHMARKS = {
    "^GSPC": AdvancedAssetMetadata(
        symbol="^GSPC",
        name="S&P 500 Index",
        description="Market-capitalization-weighted index of 500 large-cap US companies.",
        asset_type=AssetType.INDEX,
        category="Equity Index",
        region=AssetRegion.NORTH_AMERICA,
        sector=AssetSector.FINANCIALS,
        color="#1E90FF",
        icon="📈",
        display_order=1,
        specifications=AssetSpecifications(
            contract_size="Index Points",
            tick_size=0.01,
            tick_value=50.0,
            contract_unit="USD",
            settlement_type="Cash",
            trading_hours="09:30-16:00 ET",
            exchange="CME",
            margin_initial=0.05,
            margin_maintenance=0.04
        ),
        risk_metrics=RiskMetrics(
            annual_volatility=0.15,
            beta=1.0,
            sharpe_ratio=0.6,
            max_drawdown=0.34,
            var_95=0.022,
            cvar_95=0.030,
            correlation_spx=1.0,
            skewness=-0.3,
            kurtosis=4.2,
            liquidity_score=1.0
        ),
        current_price=4500.0,
        price_currency="USD",
        price_decimals=2,
        average_daily_volume_usd=500000000000
    ),
    # Additional benchmarks...
}

# =============================================================================
# ADVANCED DATA MANAGER WITH INTELLIGENT CACHING
# =============================================================================

class DataQualityMetrics:
    """Metrics for assessing data quality"""
    
    def __init__(self):
        self.metrics = {
            'completeness': 0.0,  # % of non-NaN values
            'consistency': 0.0,   # logical consistency
            'timeliness': 0.0,    # data freshness
            'validity': 0.0,      # within expected ranges
            'accuracy': 0.0,      # correctness
            'uniqueness': 0.0,    # duplicate records
            'integrity': 0.0      # referential integrity
        }
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        if df.empty:
            return self.metrics
        
        total_cells = df.size
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Completeness
        non_null = df.notna().sum().sum()
        self.metrics['completeness'] = non_null / total_cells if total_cells > 0 else 0.0
        
        # Consistency (check for logical errors)
        consistency_score = 1.0
        if len(numeric_cols) > 0:
            # Check High >= Low
            if 'High' in df.columns and 'Low' in df.columns:
                invalid = (df['High'] < df['Low']).sum()
                consistency_score -= invalid / len(df)
            
            # Check Close within High-Low range
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                invalid = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
                consistency_score -= invalid / len(df)
        
        self.metrics['consistency'] = max(0.0, consistency_score)
        
        # Timeliness (days since last update)
        if not df.empty:
            last_date = df.index[-1]
            days_since = (datetime.now() - last_date).days
            self.metrics['timeliness'] = max(0.0, 1.0 - (days_since / 30))  # 30-day decay
        
        # Validity (check bounds)
        validity_score = 1.0
        for col in numeric_cols:
            if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                negative = (df[col] < 0).sum()
                validity_score -= negative / (len(df) * len(numeric_cols))
        
        self.metrics['validity'] = max(0.0, validity_score)
        
        # Accuracy (hard to measure without truth, use sanity checks)
        self.metrics['accuracy'] = 0.8  # Default
        
        # Uniqueness
        duplicates = df.index.duplicated().sum()
        self.metrics['uniqueness'] = 1.0 - (duplicates / len(df)) if len(df) > 0 else 1.0
        
        # Overall integrity
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15,
            'accuracy': 0.10,
            'uniqueness': 0.10,
            'integrity': 0.05
        }
        
        self.metrics['integrity'] = sum(
            self.metrics[k] * weights[k] for k in weights
        )
        
        return self.metrics

class IntelligentCache:
    """Intelligent caching with adaptive TTL and memory management"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.size_bytes = 0
        
    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True)
        elif isinstance(obj, (dict, list, tuple)):
            return sys.getsizeof(pickle.dumps(obj))
        else:
            return sys.getsizeof(obj)
    
    def _make_room(self, required_bytes: int):
        """Make room in cache by evicting items"""
        while self.size_bytes + required_bytes > self.max_size_mb * 1024 * 1024:
            if not self.cache:
                break
            # Remove oldest item
            key, (value, _, _) = self.cache.popitem(last=False)
            self.size_bytes -= self._get_size(value)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache item"""
        if ttl is None:
            ttl = self.default_ttl
        
        item_size = self._get_size(value)
        self._make_room(item_size)
        
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry, time.time())
        self.size_bytes += item_size
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache item"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, expiry, last_access = self.cache[key]
        
        # Check expiry
        if time.time() > expiry:
            del self.cache[key]
            self.size_bytes -= self._get_size(value)
            self.misses += 1
            return None
        
        # Update access time
        self.cache[key] = (value, expiry, time.time())
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    def clear_expired(self):
        """Clear expired items"""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, expiry, _) in self.cache.items():
            if current_time > expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            value, _, _ = self.cache.pop(key)
            self.size_bytes -= self._get_size(value)
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size_mb': self.size_bytes / (1024 * 1024),
            'items': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'max_size_mb': self.max_size_mb,
            'utilization': (self.size_bytes / (self.max_size_mb * 1024 * 1024)) * 100
        }

class EnhancedDataManager:
    """Advanced data manager with intelligent caching, quality assessment, and multiple sources"""
    
    def __init__(self, config: ApplicationConfig = DEFAULT_CONFIG):
        self.config = config
        self.cache = IntelligentCache(max_size_mb=500)
        self.quality_metrics = DataQualityMetrics()
        self.data_sources = self._initialize_data_sources()
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup structured logging"""
        import logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available data sources"""
        sources = {
            'yfinance': {
                'name': 'Yahoo Finance',
                'priority': 1,
                'enabled': True,
                'rate_limit': 100,
                'timeout': 30,
                'retries': 3
            },
            'quandl': {
                'name': 'Quandl',
                'priority': 2,
                'enabled': dep_manager.is_available('quandl'),
                'api_key': None,  # Should be set via environment variable
                'rate_limit': 50
            },
            'fred': {
                'name': 'FRED',
                'priority': 3,
                'enabled': dep_manager.is_available('fredapi'),
                'api_key': None,
                'rate_limit': 1000
            },
            'bloomberg': {
                'name': 'Bloomberg',
                'priority': 4,
                'enabled': False,  # Requires Bloomberg Terminal
                'requires_terminal': True
            },
            'refinitiv': {
                'name': 'Refinitiv Eikon',
                'priority': 5,
                'enabled': False,  # Requires Eikon access
                'requires_api': True
            }
        }
        
        # Check environment variables for API keys
        if os.getenv('QUANDL_API_KEY'):
            sources['quandl']['api_key'] = os.getenv('QUANDL_API_KEY')
        
        if os.getenv('FRED_API_KEY'):
            sources['fred']['api_key'] = os.getenv('FRED_API_KEY')
        
        return sources
    
    def fetch_asset_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        source: str = "auto",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch asset data from the best available source
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1h, 1m)
            source: Data source ('auto', 'yfinance', 'quandl', 'fred')
            force_refresh: Force refresh from source
        
        Returns:
            DataFrame with asset data
        """
        # Generate cache key
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}_{source}"
        
        # Check cache first
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache hit for {symbol}")
                return cached_data
        
        self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Determine data source
        if source == "auto":
            source = self._select_data_source(symbol)
        
        # Fetch data based on source
        try:
            if source == "yfinance":
                df = self._fetch_yfinance(symbol, start_date, end_date, interval)
            elif source == "quandl":
                df = self._fetch_quandl(symbol, start_date, end_date)
            elif source == "fred":
                df = self._fetch_fred(symbol, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Process and clean data
            df = self._process_dataframe(df, symbol)
            
            # Calculate quality metrics
            quality = self.quality_metrics.calculate(df)
            self.logger.info(f"Data quality for {symbol}: {quality['integrity']:.1%}")
            
            # Store in cache
            ttl = self._calculate_ttl(interval, quality['timeliness'])
            self.cache.set(cache_key, df, ttl=ttl)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            # Try fallback source
            if source != "yfinance":
                self.logger.info(f"Trying fallback source for {symbol}")
                return self.fetch_asset_data(
                    symbol, start_date, end_date, interval,
                    source="yfinance", force_refresh=True
                )
            
            # Return empty dataframe if all sources fail
            return pd.DataFrame()
    
    def _select_data_source(self, symbol: str) -> str:
        """Select the best data source for a symbol"""
        # Check symbol patterns
        if symbol.endswith('=F'):
            return "yfinance"  # Futures
        
        # Check available sources
        for source_name, source_info in sorted(
            self.data_sources.items(),
            key=lambda x: x[1]['priority']
        ):
            if source_info['enabled']:
                return source_name
        
        return "yfinance"  # Default fallback
    
    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            # Configure download parameters
            params = {
                'tickers': symbol,
                'start': start_date,
                'end': end_date,
                'interval': interval,
                'progress': False,
                'auto_adjust': True,
                'threads': True,
                'timeout': self.config.yahoo_finance_timeout
            }
            
            # Handle different yfinance versions
            try:
                df = yf.download(**params)
            except TypeError:
                # Some versions don't accept these parameters
                params.pop('threads', None)
                params.pop('timeout', None)
                df = yf.download(**params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            raise
    
    def _fetch_quandl(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from Quandl"""
        if not dep_manager.is_available('quandl'):
            raise ImportError("Quandl not available")
        
        try:
            quandl = dep_manager.get_module('quandl')
            api_key = self.data_sources['quandl']['api_key']
            
            if api_key:
                quandl.ApiConfig.api_key = api_key
            
            # Different Quandl datasets
            if symbol.startswith('CHRIS/'):
                # Futures data
                df = quandl.get(symbol, start_date=start_date, end_date=end_date)
            else:
                # Try different dataset formats
                for dataset in [f"EOD/{symbol}", f"WIKI/{symbol}"]:
                    try:
                        df = quandl.get(dataset, start_date=start_date, end_date=end_date)
                        break
                    except:
                        continue
                else:
                    raise ValueError(f"Symbol {symbol} not found in Quandl")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Quandl error for {symbol}: {str(e)}")
            raise
    
    def _fetch_fred(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from FRED"""
        if not dep_manager.is_available('fredapi'):
            raise ImportError("FRED API not available")
        
        try:
            from fredapi import Fred
            
            api_key = self.data_sources['fred']['api_key']
            if not api_key:
                raise ValueError("FRED API key not configured")
            
            fred = Fred(api_key=api_key)
            series = fred.get_series(symbol, start_date, end_date)
            
            df = pd.DataFrame({symbol: series})
            df.index.name = 'Date'
            
            return df
            
        except Exception as e:
            self.logger.error(f"FRED error for {symbol}: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process and clean dataframe"""
        df = df.copy()
        
        # Handle MultiIndex columns from yfinance
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
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Clean index
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill then backward fill for small gaps
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Remove rows with too many NaN values
        threshold = len(numeric_cols) * 0.5  # Allow 50% missing
        df = df.dropna(thresh=threshold)
        
        # Calculate returns if we have price data
        if 'Adj_Close' in df.columns:
            df['Returns'] = df['Adj_Close'].pct_change()
            df['Log_Returns'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))
        
        # Add metadata
        df.attrs['symbol'] = symbol
        df.attrs['last_updated'] = datetime.now()
        df.attrs['data_source'] = 'yfinance'  # Would be dynamic in real implementation
        
        return df
    
    def _calculate_ttl(self, interval: str, timeliness: float) -> int:
        """Calculate cache TTL based on interval and data freshness"""
        base_ttl = {
            '1m': 300,      # 5 minutes
            '5m': 900,      # 15 minutes
            '15m': 1800,    # 30 minutes
            '30m': 3600,    # 1 hour
            '1h': 7200,     # 2 hours
            '1d': 86400,    # 24 hours
            '1wk': 604800,  # 1 week
            '1mo': 2592000, # 30 days
        }.get(interval, 3600)
        
        # Adjust based on timeliness
        adjustment = 1.0 + (1.0 - timeliness) * 2  # 1-3x multiplier
        return int(base_ttl * adjustment)
    
    def fetch_multiple_assets(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        max_workers: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple assets in parallel
        
        Args:
            symbols: List of asset symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            max_workers: Maximum parallel workers
            progress_callback: Callback for progress updates
        
        Returns:
            Dictionary mapping symbols to dataframes
        """
        results = {}
        failed = []
        
        total = len(symbols)
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            # Create futures
            future_to_symbol = {}
            for symbol in symbols:
                future = executor.submit(
                    self.fetch_asset_data,
                    symbol,
                    start_date,
                    end_date,
                    interval
                )
                future_to_symbol[future] = symbol
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed / total)
                
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                    else:
                        failed.append(symbol)
                        self.logger.warning(f"No data for {symbol}")
                except Exception as e:
                    failed.append(symbol)
                    self.logger.error(f"Error fetching {symbol}: {str(e)}")
        
        # Log summary
        success_rate = len(results) / total if total > 0 else 0
        self.logger.info(
            f"Fetched {len(results)}/{total} assets ({success_rate:.1%} success)"
        )
        
        if failed:
            self.logger.warning(f"Failed symbols: {failed}")
        
        return results
    
    def calculate_technical_indicators(
        self,
        df: pd.DataFrame,
        include_all: bool = False
    ) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: Input dataframe with price data
            include_all: Include all indicators (can be computationally intensive)
        
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Ensure we have price data
        if 'Adj_Close' not in df.columns:
            if 'Close' in df.columns:
                df['Adj_Close'] = df['Close']
            else:
                raise ValueError("DataFrame must contain price data")
        
        price = df['Adj_Close']
        
        # Basic indicators (always calculated)
        df['Returns'] = price.pct_change()
        df['Log_Returns'] = np.log(price / price.shift(1))
        
        # Moving averages
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            df[f'SMA_{period}'] = price.rolling(window=period).mean()
            df[f'EMA_{period}'] = price.ewm(span=period, adjust=False).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_middle = df['SMA_20']
        bb_std_dev = price.rolling(window=bb_window).std()
        df['BB_Upper'] = bb_middle + (bb_std_dev * bb_std)
        df['BB_Lower'] = bb_middle - (bb_std_dev * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_middle
        df['BB_Position'] = (price - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = price.ewm(span=12, adjust=False).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        stoch_window = 14
        lowest_low = df['Low'].rolling(window=stoch_window).min()
        highest_high = df['High'].rolling(window=stoch_window).max()
        df['Stoch_%K'] = 100 * ((price - lowest_low) / (highest_high - lowest_low))
        df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - price.shift())
        low_close = np.abs(df['Low'] - price.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR'] / price * 100
        
        # On Balance Volume (if volume data available)
        if 'Volume' in df.columns:
            df['OBV'] = (np.sign(df['Returns'].fillna(0)) * df['Volume']).cumsum()
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Additional indicators if requested
        if include_all:
            # Parabolic SAR
            try:
                if dep_manager.is_available('ta'):
                    import ta
                    df['Parabolic_SAR'] = ta.trend.PSARIndicator(
                        high=df['High'],
                        low=df['Low'],
                        close=price,
                        step=0.02,
                        max_step=0.2
                    ).psar()
            except:
                pass
            
            # Ichimoku Cloud
            try:
                # Conversion Line
                period9_high = df['High'].rolling(window=9).max()
                period9_low = df['Low'].rolling(window=9).min()
                df['Ichimoku_Conversion'] = (period9_high + period9_low) / 2
                
                # Base Line
                period26_high = df['High'].rolling(window=26).max()
                period26_low = df['Low'].rolling(window=26).min()
                df['Ichimoku_Base'] = (period26_high + period26_low) / 2
                
                # Leading Span A
                df['Ichimoku_Leading_A'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
                
                # Leading Span B
                period52_high = df['High'].rolling(window=52).max()
                period52_low = df['Low'].rolling(window=52).min()
                df['Ichimoku_Leading_B'] = ((period52_high + period52_low) / 2).shift(26)
            except:
                pass
            
            # Commodity Channel Index
            typical_price = (df['High'] + df['Low'] + price) / 3
            cci_sma = typical_price.rolling(window=20).mean()
            cci_mean_dev = typical_price.rolling(window=20).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            df['CCI'] = (typical_price - cci_sma) / (0.015 * cci_mean_dev)
            
            # Williams %R
            df['Williams_%R'] = ((highest_high - price) / (highest_high - lowest_low)) * -100
            
            # Rate of Change
            df['ROC_10'] = price.pct_change(periods=10) * 100
            df['ROC_20'] = price.pct_change(periods=20) * 100
            
            # Money Flow Index
            if 'Volume' in df.columns:
                typical_price = (df['High'] + df['Low'] + price) / 3
                money_flow = typical_price * df['Volume']
                positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
                negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
                pos_flow_sum = positive_flow.rolling(window=14).sum()
                neg_flow_sum = negative_flow.rolling(window=14).sum()
                money_ratio = pos_flow_sum / neg_flow_sum
                df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # Clean up NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Calculate volatility measures
        df['Volatility_20D'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_60D'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
        df['Volatility_120D'] = df['Returns'].rolling(window=120).std() * np.sqrt(252)
        
        # Momentum indicators
        df['Momentum_1M'] = price.pct_change(periods=21)
        df['Momentum_3M'] = price.pct_change(periods=63)
        df['Momentum_6M'] = price.pct_change(periods=126)
        df['Momentum_12M'] = price.pct_change(periods=252)
        
        # Price position relative to moving averages
        for period in [20, 50, 200]:
            df[f'Price_SMA_{period}_Ratio'] = price / df[f'SMA_{period}']
            df[f'Price_SMA_{period}_Deviation'] = (price / df[f'SMA_{period}'] - 1) * 100
        
        # Trend indicators
        df['Trend_Strength'] = df['Returns'].rolling(window=20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # Market regime classification (simplified)
        df['Market_Regime'] = pd.cut(
            df['Volatility_20D'],
            bins=[0, 0.15, 0.25, 0.35, float('inf')],
            labels=['Low', 'Normal', 'High', 'Extreme']
        )
        
        # Add metadata
        df.attrs['indicators_calculated'] = True
        df.attrs['indicators_count'] = len([col for col in df.columns if col not in [
            'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Returns', 'Log_Returns'
        ]])
        
        return df
    
    def calculate_correlation_matrix(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = 'pearson',
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets
        
        Args:
            data_dict: Dictionary of dataframes
            method: Correlation method ('pearson', 'spearman', 'kendall')
            fill_method: Method for handling missing values
        
        Returns:
            Correlation matrix DataFrame
        """
        # Extract returns
        returns_dict = {}
        for symbol, df in data_dict.items():
            if not df.empty and 'Returns' in df.columns:
                returns_dict[symbol] = df['Returns']
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        # Create returns dataframe
        returns_df = pd.DataFrame(returns_dict)
        
        # Handle missing values
        if fill_method == 'ffill':
            returns_df = returns_df.ffill().bfill()
        elif fill_method == 'drop':
            returns_df = returns_df.dropna()
        else:
            returns_df = returns_df.fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr(method=method)
        
        # Ensure PSD (positive semi-definite) for numerical stability
        corr_matrix = self._make_psd(corr_matrix)
        
        return corr_matrix
    
    def _make_psd(self, matrix: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Make matrix positive semi-definite
        
        Args:
            matrix: Input correlation matrix
            epsilon: Small value for numerical stability
        
        Returns:
            PSD correlation matrix
        """
        matrix = matrix.copy()
        n = matrix.shape[0]
        
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Ensure diagonal is 1
        np.fill_diagonal(matrix.values, 1.0)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)
        
        # Reconstruct matrix
        matrix_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to ensure correlation matrix properties
        d = np.sqrt(np.diag(matrix_psd))
        matrix_psd = matrix_psd / np.outer(d, d)
        
        # Ensure symmetry
        matrix_psd = (matrix_psd + matrix_psd.T) / 2
        
        return pd.DataFrame(matrix_psd, index=matrix.index, columns=matrix.columns)
    
    def calculate_rolling_correlations(
        self,
        asset1_returns: pd.Series,
        asset2_returns: pd.Series,
        window: int = 60,
        min_periods: int = 30
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets
        
        Args:
            asset1_returns: Returns series for asset 1
            asset2_returns: Returns series for asset 2
            window: Rolling window size
            min_periods: Minimum periods required
        
        Returns:
            Rolling correlation series
        """
        # Align series
        aligned = pd.concat([asset1_returns, asset2_returns], axis=1).dropna()
        
        if len(aligned) < min_periods:
            return pd.Series()
        
        # Calculate rolling correlation
        rolling_corr = aligned.iloc[:, 0].rolling(
            window=window,
            min_periods=min_periods
        ).corr(aligned.iloc[:, 1])
        
        return rolling_corr
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate beta relative to market
        
        Args:
            asset_returns: Asset returns series
            market_returns: Market returns series
            window: Rolling window for beta calculation (None for static)
        
        Returns:
            Beta value or series
        """
        # Align returns
        aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 30:
            return np.nan
        
        if window is None:
            # Static beta
            cov_matrix = np.cov(aligned.values.T)
            if cov_matrix[1, 1] == 0:
                return np.nan
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            return beta
        else:
            # Rolling beta
            rolling_cov = aligned.rolling(window=window, min_periods=30).cov(pairwise=True)
            rolling_var = market_returns.rolling(window=window, min_periods=30).var()
            
            # Extract covariance values
            beta_values = []
            for i in range(len(aligned)):
                if i >= window - 1:
                    cov_val = rolling_cov.iloc[i * 2, 1]  # Covariance at position
                    var_val = rolling_var.iloc[i]
                    beta = cov_val / var_val if var_val != 0 else np.nan
                else:
                    beta = np.nan
                beta_values.append(beta)
            
            return pd.Series(beta_values, index=aligned.index)
    
    def get_data_quality_report(
        self,
        df: pd.DataFrame,
        symbol: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            df: Input dataframe
            symbol: Asset symbol for reporting
        
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {
                'symbol': symbol,
                'status': 'Empty',
                'message': 'DataFrame is empty',
                'quality_score': 0.0
            }
        
        # Calculate quality metrics
        quality = self.quality_metrics.calculate(df)
        
        # Additional metrics
        num_rows = len(df)
        num_cols = len(df.columns)
        date_range = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
        
        # Missing values analysis
        missing_by_col = df.isna().sum()
        total_missing = missing_by_col.sum()
        pct_missing = total_missing / df.size * 100
        
        # Outlier detection (simplified)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Limit to first 5 columns for performance
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outlier_count += outliers
        
        # Consistency checks
        consistency_issues = []
        
        if all(col in df.columns for col in ['High', 'Low']):
            high_low_issues = (df['High'] < df['Low']).sum()
            if high_low_issues > 0:
                consistency_issues.append(f"High < Low: {high_low_issues} rows")
        
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            range_issues = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
            if range_issues > 0:
                consistency_issues.append(f"Close outside range: {range_issues} rows")
        
        # Compile report
        report = {
            'symbol': symbol,
            'status': 'Valid' if quality['integrity'] > 0.7 else 'Questionable',
            'quality_score': quality['integrity'],
            'summary': {
                'rows': num_rows,
                'columns': num_cols,
                'date_range_days': date_range,
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None
            },
            'quality_metrics': quality,
            'data_issues': {
                'missing_values_total': int(total_missing),
                'missing_values_pct': float(pct_missing),
                'missing_by_column': missing_by_col.to_dict(),
                'outlier_count': int(outlier_count),
                'consistency_issues': consistency_issues
            },
            'recommendations': self._generate_quality_recommendations(quality)
        }
        
        return report
    
    def _generate_quality_recommendations(
        self,
        quality: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        if quality['completeness'] < 0.95:
            recommendations.append(
                f"Data completeness is low ({quality['completeness']:.1%}). "
                "Consider using multiple data sources or interpolation."
            )
        
        if quality['consistency'] < 0.9:
            recommendations.append(
                f"Data consistency issues detected ({quality['consistency']:.1%}). "
                "Review price ranges and logical constraints."
            )
        
        if quality['timeliness'] < 0.8:
            recommendations.append(
                f"Data may be stale ({quality['timeliness']:.1%}). "
                "Consider refreshing more frequently."
            )
        
        if quality['validity'] < 0.95:
            recommendations.append(
                f"Data validity concerns ({quality['validity']:.1%}). "
                "Check for negative prices or unrealistic values."
            )
        
        if quality['uniqueness'] < 0.99:
            recommendations.append(
                f"Duplicate records detected ({quality['uniqueness']:.1%}). "
                "Remove duplicate dates."
            )
        
        if not recommendations:
            recommendations.append("Data quality is good. No major issues detected.")
        
        return recommendations
    
    def save_data(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = 'parquet',
        compression: str = 'snappy'
    ):
        """
        Save dataframe to disk with metadata
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: File format ('parquet', 'csv', 'feather', 'hdf5')
            compression: Compression algorithm
        """
        filepath = Path(filepath)
        
        # Preserve metadata
        metadata = df.attrs.copy()
        
        try:
            if format.lower() == 'parquet':
                df.to_parquet(filepath, compression=compression)
            elif format.lower() == 'csv':
                df.to_csv(filepath)
            elif format.lower() == 'feather':
                df.to_feather(filepath)
            elif format.lower() == 'hdf5':
                df.to_hdf(filepath, key='data', mode='w')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save metadata separately
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Saved data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_data(
        self,
        filepath: Union[str, Path],
        format: str = 'auto'
    ) -> pd.DataFrame:
        """
        Load dataframe from disk with metadata
        
        Args:
            filepath: Input file path
            format: File format ('auto' for detection)
        
        Returns:
            Loaded DataFrame with restored metadata
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            # Detect format from extension
            if filepath.suffix == '.parquet':
                format = 'parquet'
            elif filepath.suffix == '.csv':
                format = 'csv'
            elif filepath.suffix == '.feather':
                format = 'feather'
            elif filepath.suffix in ['.h5', '.hdf5']:
                format = 'hdf5'
            else:
                raise ValueError(f"Could not detect format for {filepath}")
        
        try:
            if format == 'parquet':
                df = pd.read_parquet(filepath)
            elif format == 'csv':
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == 'feather':
                df = pd.read_feather(filepath)
                if 'index' in df.columns:
                    df.set_index('index', inplace=True)
            elif format == 'hdf5':
                df = pd.read_hdf(filepath, key='data')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Load and restore metadata
            metadata_file = filepath.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                df.attrs.update(metadata)
            
            self.logger.info(f"Loaded data from {filepath}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.logger.info("Cache cleared")

# =============================================================================
# ADVANCED ANALYTICS ENGINE WITH COMPREHENSIVE FEATURES
# =============================================================================

class QuantitativeAnalytics:
    """
    Advanced quantitative analytics engine for institutional commodity trading
    
    Features:
    1. Portfolio Optimization (Multiple Methods)
    2. Risk Analytics (VaR, CVaR, Stress Testing)
    3. Volatility Modeling (GARCH, EGARCH, Stochastic Volatility)
    4. Regime Detection (HMM, Machine Learning)
    5. Factor Analysis & Smart Beta
    6. Monte Carlo Simulations
    7. Backtesting Engine
    8. Alternative Data Integration
    9. Performance Attribution
    10. Transaction Cost Analysis
    """
    
    def __init__(self, config: ApplicationConfig = DEFAULT_CONFIG):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        self.annual_trading_days = config.annual_trading_days
        
        # Initialize components
        self._setup_components()
        
    def _setup_components(self):
        """Setup analytics components"""
        # Performance calculator
        self.performance = PerformanceCalculator(self.config)
        
        # Risk calculator
        self.risk = RiskCalculator(self.config)
        
        # Optimization engine
        self.optimizer = PortfolioOptimizer(self.config)
        
        # Volatility models
        self.volatility = VolatilityModels(self.config)
        
        # Regime detector
        self.regime = RegimeDetector(self.config)
        
        # Factor analyzer
        self.factors = FactorAnalyzer(self.config)
        
        # Monte Carlo simulator
        self.monte_carlo = MonteCarloSimulator(self.config)
        
        # Backtesting engine
        self.backtester = BacktestingEngine(self.config)
        
        # Transaction cost analyzer
        self.transaction = TransactionCostAnalyzer(self.config)
        
        # Performance attributor
        self.attribution = PerformanceAttributor(self.config)

class PerformanceCalculator:
    """Advanced performance calculation engine"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        self.annual_trading_days = config.annual_trading_days
    
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        include_all: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Asset/portfolio returns series
            benchmark_returns: Benchmark returns for relative metrics
            include_all: Include all metrics (computationally intensive)
        
        Returns:
            Dictionary of performance metrics
        """
        returns = returns.dropna()
        
        if len(returns) < 20:
            return {"error": "Insufficient data"}
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Higher moments
        metrics.update(self._calculate_higher_moments(returns))
        
        # Tail risk metrics
        metrics.update(self._calculate_tail_risk_metrics(returns))
        
        # Gain/loss metrics
        metrics.update(self._calculate_gain_loss_metrics(returns))
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            metrics.update(
                self._calculate_benchmark_relative_metrics(returns, benchmark_returns)
            )
        
        # Advanced metrics (if requested)
        if include_all:
            metrics.update(self._calculate_advanced_metrics(returns))
        
        # Quality checks
        metrics.update(self._add_quality_checks(metrics))
        
        return metrics
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate basic return metrics"""
        n_periods = len(returns)
        
        # Cumulative return
        cumulative_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        years = n_periods / self.annual_trading_days
        annualized_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(self.annual_trading_days)
        
        # Sharpe ratio (annualized)
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'n_periods': int(n_periods),
            'years': float(years)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        # Sortino ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_vol = downside_returns.std() * np.sqrt(self.annual_trading_days)
            excess_return = self._calculate_annualized_return(returns) - self.risk_free_rate
            sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino_ratio = np.nan
        
        # Calmar ratio (return / max drawdown)
        drawdown = self._calculate_drawdown_series(returns)
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        annual_return = self._calculate_annualized_return(returns)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Omega ratio
        threshold = self.risk_free_rate / self.annual_trading_days  # Daily threshold
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega_ratio = gains / losses if losses > 0 else float('inf')
        
        # Gain to Pain ratio
        total_gain = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        gain_to_pain = total_gain / total_loss if total_loss > 0 else float('inf')
        
        return {
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'omega_ratio': float(omega_ratio),
            'gain_to_pain_ratio': float(gain_to_pain)
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown-related metrics"""
        # Calculate drawdown series
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / running_max
        
        if len(drawdown_series) == 0:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'recovery_time': 0
            }
        
        # Max drawdown
        max_drawdown = drawdown_series.min()
        
        # Average drawdown (only negative values)
        drawdowns = drawdown_series[drawdown_series < 0]
        avg_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0.0
        
        # Drawdown duration analysis
        durations = self._calculate_drawdown_durations(drawdown_series)
        max_duration = max(durations) if durations else 0
        avg_duration = np.mean(durations) if durations else 0
        
        # Recovery time (time from trough to new high)
        recovery_times = self._calculate_recovery_times(drawdown_series)
        avg_recovery = np.mean(recovery_times) if recovery_times else 0
        
        # Ulcer Index (measure of downside risk)
        ulcer_index = np.sqrt((drawdown_series ** 2).mean())
        
        # Pain index (average drawdown)
        pain_index = abs(drawdown_series[drawdown_series < 0].mean()) if len(drawdown_series[drawdown_series < 0]) > 0 else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'max_drawdown_duration': int(max_duration),
            'avg_drawdown_duration': float(avg_duration),
            'avg_recovery_time': float(avg_recovery),
            'ulcer_index': float(ulcer_index),
            'pain_index': float(pain_index),
            'drawdown_count': len(durations)
        }
    
    def _calculate_higher_moments(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate higher moments of return distribution"""
        if len(returns) < 30:
            return {
                'skewness': np.nan,
                'kurtosis': np.nan,
                'jarque_bera': np.nan,
                'shapiro_wilk': np.nan
            }
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Normality tests
        try:
            jarque_bera = stats.jarque_bera(returns)
            shapiro_wilk = stats.shapiro(returns[:5000])  # Shapiro-Wilk has limit
        except:
            jarque_bera = (np.nan, np.nan)
            shapiro_wilk = (np.nan, np.nan)
        
        return {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'jarque_bera_statistic': float(jarque_bera[0]),
            'jarque_bera_pvalue': float(jarque_bera[1]),
            'shapiro_wilk_statistic': float(shapiro_wilk[0]),
            'shapiro_wilk_pvalue': float(shapiro_wilk[1])
        }
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate tail risk metrics"""
        if len(returns) < 100:
            return {
                'var_95': np.nan,
                'var_99': np.nan,
                'cvar_95': np.nan,
                'cvar_99': np.nan,
                'expected_shortfall': np.nan,
                'tail_ratio': np.nan,
                'max_consecutive_losses': 0
            }
        
        # Value at Risk (Historical)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Tail ratio (95th percentile gain / 5th percentile loss)
        tail_gain = np.percentile(returns, 95)
        tail_loss = abs(np.percentile(returns, 5))
        tail_ratio = tail_gain / tail_loss if tail_loss > 0 else float('inf')
        
        # Maximum consecutive losses
        consecutive_losses = self._calculate_max_consecutive_losses(returns)
        
        # Semi-deviation (downside deviation)
        downside_returns = returns[returns < 0]
        semi_deviation = downside_returns.std() if len(downside_returns) > 1 else 0
        
        return {
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'tail_ratio': float(tail_ratio),
            'max_consecutive_losses': int(consecutive_losses),
            'semi_deviation': float(semi_deviation)
        }
    
    def _calculate_gain_loss_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate gain/loss metrics"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        n_positive = len(positive_returns)
        n_negative = len(negative_returns)
        total_trades = n_positive + n_negative
        
        win_rate = n_positive / total_trades if total_trades > 0 else 0
        loss_rate = n_negative / total_trades if total_trades > 0 else 0
        
        avg_gain = positive_returns.mean() if n_positive > 0 else 0
        avg_loss = negative_returns.mean() if n_negative > 0 else 0
        
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        
        total_gain = positive_returns.sum()
        total_loss = abs(negative_returns.sum())
        profit_factor = total_gain / total_loss if total_loss > 0 else float('inf')
        
        # Payoff ratio (average win / average loss)
        payoff_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        
        # Expectancy (expected value per trade)
        expectancy = (win_rate * avg_gain) + (loss_rate * avg_loss)
        
        return {
            'win_rate': float(win_rate),
            'loss_rate': float(loss_rate),
            'avg_gain': float(avg_gain),
            'avg_loss': float(avg_loss),
            'gain_loss_ratio': float(gain_loss_ratio),
            'profit_factor': float(profit_factor),
            'payoff_ratio': float(payoff_ratio),
            'expectancy': float(expectancy),
            'total_gain': float(total_gain),
            'total_loss': float(total_loss),
            'n_winning_trades': int(n_positive),
            'n_losing_trades': int(n_negative)
        }
    
    def _calculate_benchmark_relative_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate benchmark-relative metrics"""
        # Align returns
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return {}
        
        asset_returns = aligned.iloc[:, 0]
        bench_returns = aligned.iloc[:, 1]
        
        # Beta calculation
        cov_matrix = np.cov(asset_returns, bench_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else np.nan
        
        # Alpha calculation (Jensen's Alpha)
        asset_annual_return = self._calculate_annualized_return(asset_returns)
        bench_annual_return = self._calculate_annualized_return(bench_returns)
        alpha = asset_annual_return - (self.risk_free_rate + beta * (bench_annual_return - self.risk_free_rate))
        
        # Tracking error
        tracking_error = (asset_returns - bench_returns).std() * np.sqrt(self.annual_trading_days)
        
        # Information ratio
        excess_return = asset_annual_return - bench_annual_return
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio
        treynor_ratio = (asset_annual_return - self.risk_free_rate) / beta if beta != 0 else 0
        
        # R-squared (coefficient of determination)
        correlation = asset_returns.corr(bench_returns)
        r_squared = correlation ** 2
        
        # Up/Down capture ratios
        up_market = bench_returns > 0
        down_market = bench_returns < 0
        
        if up_market.any():
            up_capture = asset_returns[up_market].mean() / bench_returns[up_market].mean()
        else:
            up_capture = np.nan
        
        if down_market.any():
            down_capture = asset_returns[down_market].mean() / bench_returns[down_market].mean()
        else:
            down_capture = np.nan
        
        capture_ratio = up_capture / down_capture if down_capture != 0 else float('inf')
        
        return {
            'beta': float(beta),
            'alpha': float(alpha),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'treynor_ratio': float(treynor_ratio),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'up_capture': float(up_capture),
            'down_capture': float(down_capture),
            'capture_ratio': float(capture_ratio)
        }
    
    def _calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        if len(returns) < 100:
            return {}
        
        # Modigliani Risk-Adjusted Performance (M2)
        volatility = returns.std() * np.sqrt(self.annual_trading_days)
        sharpe_ratio = (self._calculate_annualized_return(returns) - self.risk_free_rate) / volatility
        m2 = self.risk_free_rate + sharpe_ratio * volatility
        
        # Sterling ratio (return / average drawdown)
        drawdown_series = self._calculate_drawdown_series(returns)
        avg_drawdown = abs(drawdown_series.mean()) if len(drawdown_series) > 0 else 0
        sterling_ratio = self._calculate_annualized_return(returns) / avg_drawdown if avg_drawdown > 0 else 0
        
        # Burke ratio (return / square root of sum of squared drawdowns)
        drawdown_squared_sum = (drawdown_series ** 2).sum()
        burke_ratio = self._calculate_annualized_return(returns) / np.sqrt(drawdown_squared_sum) if drawdown_squared_sum > 0 else 0
        
        # Martin ratio (Ulcer Performance Index)
        ulcer_index = np.sqrt((drawdown_series ** 2).mean())
        martin_ratio = self._calculate_annualized_return(returns) / ulcer_index if ulcer_index > 0 else 0
        
        # Pain ratio (return / pain index)
        pain_index = abs(drawdown_series[drawdown_series < 0].mean()) if len(drawdown_series[drawdown_series < 0]) > 0 else 0
        pain_ratio = self._calculate_annualized_return(returns) / pain_index if pain_index > 0 else 0
        
        # Return on VaR
        var_95 = np.percentile(returns, 5)
        return_on_var = self._calculate_annualized_return(returns) / abs(var_95) if var_95 != 0 else 0
        
        # Conditional Sharpe ratio
        cvar_95 = returns[returns <= var_95].mean()
        conditional_sharpe = (self._calculate_annualized_return(returns) - self.risk_free_rate) / abs(cvar_95) if cvar_95 != 0 else 0
        
        # Skewness/Kurtosis adjusted Sharpe
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        adjusted_sharpe = sharpe_ratio * (1 + (skewness / 6) * sharpe_ratio - ((kurtosis - 3) / 24) * sharpe_ratio ** 2)
        
        return {
            'm2_ratio': float(m2),
            'sterling_ratio': float(sterling_ratio),
            'burke_ratio': float(burke_ratio),
            'martin_ratio': float(martin_ratio),
            'pain_ratio': float(pain_ratio),
            'return_on_var': float(return_on_var),
            'conditional_sharpe': float(conditional_sharpe),
            'adjusted_sharpe': float(adjusted_sharpe)
        }
    
    def _add_quality_checks(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Add data quality checks to metrics"""
        quality = {
            'data_quality': 'Good',
            'data_warnings': []
        }
        
        # Check for sufficient data
        if metrics.get('n_periods', 0) < 100:
            quality['data_quality'] = 'Limited'
            quality['data_warnings'].append('Less than 100 data points')
        
        # Check for extreme values
        if abs(metrics.get('annualized_return', 0)) > 1.0:  # >100% return
            quality['data_warnings'].append('Extreme annualized return')
        
        if metrics.get('annualized_volatility', 0) > 1.0:  # >100% volatility
            quality['data_warnings'].append('Extreme volatility')
        
        # Check for unrealistic Sharpe ratio
        if abs(metrics.get('sharpe_ratio', 0)) > 10:
            quality['data_warnings'].append('Unusually high Sharpe ratio')
        
        # Check for data errors
        if np.isnan(metrics.get('sharpe_ratio', np.nan)):
            quality['data_quality'] = 'Questionable'
            quality['data_warnings'].append('NaN values in calculations')
        
        metrics.update(quality)
        return metrics
    
    # Helper methods
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).prod() - 1
        years = len(returns) / self.annual_trading_days
        return (1 + cumulative) ** (1 / years) - 1 if years > 0 else 0
    
    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        if len(returns) == 0:
            return pd.Series()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max
    
    def _calculate_drawdown_durations(self, drawdown_series: pd.Series) -> List[int]:
        """Calculate durations of drawdown periods"""
        durations = []
        current_duration = 0
        
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations
    
    def _calculate_recovery_times(self, drawdown_series: pd.Series) -> List[int]:
        """Calculate recovery times from drawdown troughs"""
        recovery_times = []
        in_drawdown = False
        trough_idx = None
        
        for i, dd in enumerate(drawdown_series):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                trough_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                if trough_idx is not None:
                    recovery_times.append(i - trough_idx)
                trough_idx = None
        
        return recovery_times
    
    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0
        
        for ret in returns:
            if ret < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses

class RiskCalculator:
    """Advanced risk calculation engine"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical',
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk using multiple methods
        
        Args:
            returns: Returns series
            confidence_level: Confidence level (0.95, 0.99, etc.)
            method: VaR method ('historical', 'parametric', 'monte_carlo', 'cornish_fisher')
            horizon: Time horizon in days
        
        Returns:
            Dictionary with VaR results
        """
        returns = returns.dropna()
        
        if len(returns) < 100:
            return {"error": "Insufficient data for VaR calculation"}
        
        results = {}
        
        if method == 'historical':
            results = self._calculate_historical_var(returns, confidence_level, horizon)
        elif method == 'parametric':
            results = self._calculate_parametric_var(returns, confidence_level, horizon)
        elif method == 'monte_carlo':
            results = self._calculate_monte_carlo_var(returns, confidence_level, horizon)
        elif method == 'cornish_fisher':
            results = self._calculate_cornish_fisher_var(returns, confidence_level, horizon)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Calculate expected shortfall (CVaR)
        if 'var' in results:
            var_value = results['var']
            cvar = returns[returns <= var_value].mean()
            results['cvar'] = float(cvar)
            results['cvar_ratio'] = float(cvar / var_value) if var_value != 0 else 0
        
        # Add metadata
        results.update({
            'method': method,
            'confidence_level': confidence_level,
            'horizon_days': horizon,
            'n_observations': len(returns)
        })
        
        return results
    
    def _calculate_historical_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        horizon: int
    ) -> Dict[str, Any]:
        """Calculate historical VaR"""
        # Scale returns for horizon
        if horizon > 1:
            # Simulate horizon returns using rolling windows
            horizon_returns = returns.rolling(window=horizon).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            ).dropna()
        else:
            horizon_returns = returns
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(horizon_returns, var_percentile)
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(horizon_returns, size=len(horizon_returns), replace=True)
            bootstrap_vars.append(np.percentile(sample, var_percentile))
        
        ci_lower = np.percentile(bootstrap_vars, 2.5)
        ci_upper = np.percentile(bootstrap_vars, 97.5)
        
        return {
            'var': float(var),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'method': 'historical',
            'bootstrap_samples': n_bootstrap
        }
    
    def _calculate_parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        horizon: int
    ) -> Dict[str, Any]:
        """Calculate parametric (Gaussian) VaR"""
        mean = returns.mean()
        std = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR formula: μ - z * σ * √h
        var = mean - z_score * std * np.sqrt(horizon)
        
        return {
            'var': float(var),
            'mean': float(mean),
            'std': float(std),
            'z_score': float(z_score),
            'method': 'parametric_gaussian'
        }
    
    def _calculate_cornish_fisher_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        horizon: int
    ) -> Dict[str, Any]:
        """Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis)"""
        mean = returns.mean()
        std = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        z_cf = (z +
                (z ** 2 - 1) * skew / 6 +
                (z ** 3 - 3 * z) * kurt / 24 -
                (2 * z ** 3 - 5 * z) * skew ** 2 / 36)
        
        # VaR formula
        var = mean - z_cf * std * np.sqrt(horizon)
        
        return {
            'var': float(var),
            'mean': float(mean),
            'std': float(std),
            'skewness': float(skew),
            'kurtosis': float(kurt),
            'z_score': float(z),
            'z_cornish_fisher': float(z_cf),
            'method': 'cornish_fisher'
        }
    
    def _calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        horizon: int,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Calculate VaR using Monte Carlo simulation"""
        mean = returns.mean()
        std = returns.std()
        
        # Simulate returns
        simulated_returns = np.random.normal(mean, std, (n_simulations, horizon))
        
        # Calculate horizon returns
        horizon_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(horizon_returns, var_percentile)
        
        return {
            'var': float(var),
            'mean': float(mean),
            'std': float(std),
            'n_simulations': n_simulations,
            'method': 'monte_carlo'
        }
    
    def calculate_risk_decomposition(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """
        Decompose portfolio risk into component contributions
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Portfolio weights
        
        Returns:
            Dictionary with risk decomposition
        """
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * self.config.annual_trading_days
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_vol == 0:
            return {"error": "Portfolio has zero volatility"}
        
        # Marginal contributions to risk
        marginal_contributions = (cov_matrix @ weights) / portfolio_vol
        
        # Risk contributions
        risk_contributions = marginal_contributions * weights
        
        # Percentage contributions
        pct_contributions = risk_contributions / portfolio_vol
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.sum(np.abs(weights) * asset_vols)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Concentration measures
        herfindahl = np.sum(weights ** 2)
        gini = self._calculate_gini_coefficient(np.abs(weights))
        
        return {
            'portfolio_volatility': float(portfolio_vol),
            'marginal_contributions': dict(zip(returns_df.columns, marginal_contributions)),
            'risk_contributions': dict(zip(returns_df.columns, risk_contributions)),
            'percentage_contributions': dict(zip(returns_df.columns, pct_contributions)),
            'diversification_ratio': float(diversification_ratio),
            'herfindahl_index': float(herfindahl),
            'gini_coefficient': float(gini),
            'effective_number': float(1 / herfindahl) if herfindahl > 0 else len(weights)
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for concentration"""
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        
        return gini
    
    def stress_test(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        scenarios: List[Dict[str, float]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform stress testing for different market scenarios
        
        Args:
            returns_df: Historical returns
            weights: Portfolio weights
            scenarios: List of scenario definitions
            confidence_level: Confidence level for VaR
        
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        # Base case
        portfolio_returns = returns_df @ weights
        base_var = self.calculate_var(portfolio_returns, confidence_level)
        
        results['base_case'] = {
            'portfolio_return': float(portfolio_returns.mean() * self.config.annual_trading_days),
            'portfolio_volatility': float(portfolio_returns.std() * np.sqrt(self.config.annual_trading_days)),
            'var_95': base_var.get('var', 0),
            'cvar_95': base_var.get('cvar', 0)
        }
        
        # Scenario analysis
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            # Apply scenario shocks (simplified - in practice would use more sophisticated models)
            shocked_returns = returns_df.copy()
            
            for asset, shock in scenario.items():
                if asset in shocked_returns.columns:
                    # Apply shock to returns
                    shocked_returns[asset] = shocked_returns[asset] * (1 + shock)
            
            # Calculate shocked portfolio
            shocked_portfolio = shocked_returns @ weights
            shocked_var = self.calculate_var(shocked_portfolio, confidence_level)
            
            scenario_result = {
                'scenario_id': i + 1,
                'scenario_name': scenario.get('name', f'Scenario {i + 1}'),
                'shocks': {k: v for k, v in scenario.items() if k != 'name'},
                'portfolio_return': float(shocked_portfolio.mean() * self.config.annual_trading_days),
                'portfolio_volatility': float(shocked_portfolio.std() * np.sqrt(self.config.annual_trading_days)),
                'var_95': shocked_var.get('var', 0),
                'cvar_95': shocked_var.get('cvar', 0),
                'return_change_pct': (shocked_portfolio.mean() - portfolio_returns.mean()) / abs(portfolio_returns.mean()) * 100 if portfolio_returns.mean() != 0 else 0,
                'var_change_pct': (shocked_var.get('var', 0) - base_var.get('var', 0)) / abs(base_var.get('var', 0)) * 100 if base_var.get('var', 0) != 0 else 0
            }
            
            scenario_results.append(scenario_result)
        
        results['scenarios'] = scenario_results
        
        # Worst-case analysis
        if scenario_results:
            worst_var = max(scenario_results, key=lambda x: abs(x['var_95']))
            worst_return = min(scenario_results, key=lambda x: x['portfolio_return'])
            
            results['worst_case'] = {
                'worst_var': worst_var,
                'worst_return': worst_return,
                'max_var_increase_pct': worst_var['var_change_pct'],
                'max_return_decrease_pct': worst_return['return_change_pct']
            }
        
        return results
    
    def calculate_liquidity_metrics(
        self,
        volumes: pd.Series,
        prices: pd.Series,
        position_size: float = 1000000  # $1M position
    ) -> Dict[str, Any]:
        """
        Calculate liquidity risk metrics
        
        Args:
            volumes: Trading volumes
            prices: Asset prices
            position_size: Position size in dollars
        
        Returns:
            Dictionary with liquidity metrics
        """
        if len(volumes) < 20 or len(prices) < 20:
            return {"error": "Insufficient data"}
        
        # Calculate average metrics
        avg_volume = volumes.mean()
        avg_price = prices.mean()
        avg_dollar_volume = avg_volume * avg_price
        
        # Market impact estimates (simplified)
        # Days to liquidate based on average volume
        days_to_liquidate = position_size / avg_dollar_volume
        
        # Price impact (simplified Kyle's lambda)
        returns = prices.pct_change().dropna()
        volumes_clean = volumes.reindex(returns.index).fillna(avg_volume)
        
        # Estimate price impact coefficient
        try:
            # Simple regression of absolute returns on volume
            X = np.log(volumes_clean.values).reshape(-1, 1)
            y = np.abs(returns.values)
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            lambda_coef = model.coef_[0]
        except:
            lambda_coef = 0.0001  # Default
        
        # Estimated price impact for position
        estimated_impact = lambda_coef * np.log(position_size / avg_dollar_volume)
        
        # Liquidity score (0-100, higher is more liquid)
        liquidity_score = min(100, max(0, 100 - days_to_liquidate * 10 - estimated_impact * 1000))
        
        return {
            'avg_daily_volume': float(avg_volume),
            'avg_dollar_volume': float(avg_dollar_volume),
            'days_to_liquidate': float(days_to_liquidate),
            'price_impact_coefficient': float(lambda_coef),
            'estimated_price_impact_pct': float(estimated_impact * 100),
            'liquidity_score': float(liquidity_score),
            'liquidity_tier': self._classify_liquidity_tier(liquidity_score)
        }
    
    def _classify_liquidity_tier(self, score: float) -> str:
        """Classify liquidity based on score"""
        if score >= 80:
            return "High"
        elif score >= 60:
            return "Medium-High"
        elif score >= 40:
            return "Medium"
        elif score >= 20:
            return "Low-Medium"
        else:
            return "Low"

class PortfolioOptimizer:
    """Advanced portfolio optimization engine with multiple methods"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        self.annual_trading_days = config.annual_trading_days
    
    def optimize(
        self,
        returns_df: pd.DataFrame,
        method: str = 'sharpe',
        constraints: Optional[Dict[str, Any]] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method
        
        Args:
            returns_df: DataFrame of asset returns
            method: Optimization method
            constraints: Optimization constraints
            target_return: Target return (for mean-variance)
            target_risk: Target risk (for risk targeting)
        
        Returns:
            Dictionary with optimization results
        """
        # Validate input
        if returns_df.empty or len(returns_df) < 60:
            return {"success": False, "error": "Insufficient data"}
        
        # Clean and prepare data
        returns_df = self._prepare_returns_data(returns_df)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': self.config.default_min_weight,
                'max_weight': self.config.default_max_weight,
                'sum_to_one': True,
                'allow_short': False
            }
        
        # Select optimization method
        method = method.lower()
        
        try:
            if method == 'sharpe':
                result = self._optimize_sharpe(returns_df, constraints)
            elif method == 'min_variance':
                result = self._optimize_min_variance(returns_df, constraints)
            elif method == 'max_return':
                result = self._optimize_max_return(returns_df, constraints)
            elif method == 'risk_parity':
                result = self._optimize_risk_parity(returns_df, constraints)
            elif method == 'max_diversification':
                result = self._optimize_max_diversification(returns_df, constraints)
            elif method == 'mean_variance':
                result = self._optimize_mean_variance(returns_df, constraints, target_return)
            elif method == 'risk_target':
                result = self._optimize_risk_target(returns_df, constraints, target_risk)
            elif method == 'equal_weight':
                result = self._optimize_equal_weight(returns_df)
            elif method == 'inverse_volatility':
                result = self._optimize_inverse_volatility(returns_df, constraints)
            elif method == 'equal_risk_contribution':
                result = self._optimize_equal_risk_contribution(returns_df, constraints)
            else:
                return {"success": False, "error": f"Unknown optimization method: {method}"}
            
            # Add common metrics
            if result['success']:
                result = self._add_portfolio_metrics(returns_df, result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Optimization failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _prepare_returns_data(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare returns data for optimization"""
        # Remove assets with insufficient data
        min_obs = max(60, returns_df.shape[1] * 2)
        returns_df = returns_df.dropna(thresh=min_obs, axis=1)
        
        # Remove assets with zero variance
        valid_cols = []
        for col in returns_df.columns:
            if returns_df[col].std() > 1e-8:
                valid_cols.append(col)
        
        returns_df = returns_df[valid_cols]
        
        # Forward fill then backward fill small gaps
        returns_df = returns_df.ffill().bfill()
        
        # Drop any remaining NaN
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _optimize_sharpe(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maximize Sharpe ratio"""
        n_assets = returns_df.shape[1]
        
        # Calculate inputs
        mean_returns = returns_df.mean() * self.annual_trading_days
        cov_matrix = returns_df.cov() * self.annual_trading_days
        
        # Ensure PSD
        cov_matrix = self._make_psd(cov_matrix)
        
        # Objective function: minimize negative Sharpe
        def objective(weights):
            port_return = np.sum(mean_returns * weights)
            port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            # Add small epsilon to avoid division by zero
            risk = max(port_risk, 1e-12)
            return -(port_return - self.risk_free_rate) / risk
        
        # Constraints and bounds
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Initial guess (equal weight)
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": -result.fun,
                "iterations": result.nit,
                "method": "sharpe_maximization"
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "sharpe_maximization"
            }
    
    def _optimize_min_variance(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Minimize portfolio variance"""
        n_assets = returns_df.shape[1]
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        
        # Objective function: minimize variance
        def objective(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints and bounds
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": result.fun,
                "iterations": result.nit,
                "method": "min_variance"
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "min_variance"
            }
    
    def _optimize_max_return(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maximize portfolio return"""
        n_assets = returns_df.shape[1]
        
        # Calculate mean returns
        mean_returns = returns_df.mean() * self.annual_trading_days
        
        # Objective function: maximize return (minimize negative return)
        def objective(weights):
            return -np.sum(mean_returns * weights)
        
        # Constraints and bounds
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": -result.fun,
                "iterations": result.nit,
                "method": "max_return"
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "max_return"
            }
    
    def _optimize_risk_parity(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Risk parity optimization"""
        n_assets = returns_df.shape[1]
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        
        # Objective: minimize deviation from equal risk contribution
        def objective(weights):
            # Portfolio volatility
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if port_vol < 1e-12:
                return 1e12
            
            # Marginal risk contributions
            mrc = (cov_matrix @ weights) / port_vol
            
            # Risk contributions
            rc = mrc * weights
            
            # Target equal risk contribution
            target_rc = port_vol / n_assets
            
            # Sum of squared deviations
            deviations = (rc - target_rc) ** 2
            
            return np.sum(deviations)
        
        # Constraints and bounds
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Initial guess (equal weight)
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": result.fun,
                "iterations": result.nit,
                "method": "risk_parity"
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "risk_parity"
            }
    
    def _optimize_max_diversification(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Maximize diversification ratio"""
        n_assets = returns_df.shape[1]
        
        # Calculate covariance matrix and asset volatilities
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        asset_vols = np.sqrt(np.diag(cov_matrix))
        
        # Objective: maximize diversification ratio
        def objective(weights):
            # Portfolio volatility
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if port_vol < 1e-12:
                return 1e12
            
            # Weighted average volatility
            weighted_vol = np.sum(np.abs(weights) * asset_vols)
            
            # Diversification ratio (maximize = minimize negative)
            diversification = weighted_vol / port_vol
            
            return -diversification
        
        # Constraints and bounds
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Initial guess (equal weight)
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": -result.fun,
                "iterations": result.nit,
                "method": "max_diversification"
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "max_diversification"
            }
    
    def _optimize_mean_variance(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any],
        target_return: Optional[float]
    ) -> Dict[str, Any]:
        """Mean-variance optimization with target return"""
        if target_return is None:
            return self._optimize_sharpe(returns_df, constraints)
        
        n_assets = returns_df.shape[1]
        
        # Calculate inputs
        mean_returns = returns_df.mean() * self.annual_trading_days
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        
        # Objective: minimize variance
        def objective(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Add target return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(mean_returns * w) - target_return
        }
        constraints_list.append(return_constraint)
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": result.fun,
                "iterations": result.nit,
                "method": "mean_variance",
                "target_return": target_return
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "mean_variance"
            }
    
    def _optimize_risk_target(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any],
        target_risk: Optional[float]
    ) -> Dict[str, Any]:
        """Optimize for target risk level"""
        if target_risk is None:
            return self._optimize_sharpe(returns_df, constraints)
        
        n_assets = returns_df.shape[1]
        
        # Calculate inputs
        mean_returns = returns_df.mean() * self.annual_trading_days
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        
        # Objective: maximize return
        def objective(weights):
            return -np.sum(mean_returns * weights)
        
        # Constraints
        bounds, constraints_list = self._build_constraints(n_assets, constraints)
        
        # Add target risk constraint
        risk_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sqrt(w.T @ cov_matrix @ w) - target_risk
        }
        constraints_list.append(risk_constraint)
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            weights = self._normalize_weights(weights, constraints)
            
            return {
                "success": True,
                "weights": dict(zip(returns_df.columns, weights)),
                "objective_value": -result.fun,
                "iterations": result.nit,
                "method": "risk_target",
                "target_risk": target_risk
            }
        else:
            return {
                "success": False,
                "error": f"Optimization failed: {result.message}",
                "method": "risk_target"
            }
    
    def _optimize_equal_weight(
        self,
        returns_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Equal weight portfolio"""
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        return {
            "success": True,
            "weights": dict(zip(returns_df.columns, weights)),
            "method": "equal_weight"
        }
    
    def _optimize_inverse_volatility(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inverse volatility weighting"""
        # Calculate asset volatilities
        volatilities = returns_df.std() * np.sqrt(self.annual_trading_days)
        
        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights, constraints)
        weights = self._normalize_weights(weights, constraints)
        
        return {
            "success": True,
            "weights": dict(zip(returns_df.columns, weights)),
            "method": "inverse_volatility"
        }
    
    def _optimize_equal_risk_contribution(
        self,
        returns_df: pd.DataFrame,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Equal risk contribution (similar to risk parity)"""
        return self._optimize_risk_parity(returns_df, constraints)
    
    def _build_constraints(
        self,
        n_assets: int,
        constraints: Dict[str, Any]
    ) -> Tuple[List[Tuple[float, float]], List[Dict]]:
        """Build optimization constraints and bounds"""
        # Bounds
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        allow_short = constraints.get('allow_short', False)
        
        if allow_short:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        else:
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Constraints list
        constraints_list = []
        
        # Sum to one constraint
        if constraints.get('sum_to_one', True):
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            })
        
        # Sector constraints (if provided)
        if 'sector_constraints' in constraints:
            sector_constraints = constraints['sector_constraints']
            for sector, (min_pct, max_pct) in sector_constraints.items():
                # This would need sector mapping - simplified for now
                pass
        
        # Turnover constraint
        if 'max_turnover' in constraints:
            max_turnover = constraints['max_turnover']
            if 'current_weights' in constraints:
                current_weights = constraints['current_weights']
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: max_turnover - np.sum(np.abs(w - current_weights))
                })
        
        return bounds, constraints_list
    
    def _apply_weight_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Apply weight constraints"""
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        # Clip weights
        weights = np.clip(weights, min_weight, max_weight)
        
        return weights
    
    def _normalize_weights(
        self,
        weights: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Normalize weights to sum to 1 (if required)"""
        if constraints.get('sum_to_one', True):
            total = np.sum(weights)
            if total != 0:
                weights = weights / total
        
        return weights
    
    def _make_psd(self, matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Make matrix positive semi-definite"""
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)
        
        # Reconstruct
        matrix_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return matrix_psd
    
    def _add_portfolio_metrics(
        self,
        returns_df: pd.DataFrame,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add portfolio metrics to optimization result"""
        if not result['success']:
            return result
        
        # Extract weights
        weights_dict = result['weights']
        assets = list(weights_dict.keys())
        weights = np.array([weights_dict[asset] for asset in assets])
        
        # Portfolio returns
        portfolio_returns = returns_df[assets] @ weights
        
        # Calculate performance metrics
        perf_calc = PerformanceCalculator(self.config)
        metrics = perf_calc.calculate_performance_metrics(portfolio_returns)
        
        # Risk decomposition
        risk_calc = RiskCalculator(self.config)
        risk_decomp = risk_calc.calculate_risk_decomposition(returns_df[assets], weights)
        
        # Add to result
        result['performance_metrics'] = metrics
        result['risk_decomposition'] = risk_decomp
        
        # Additional portfolio metrics
        result['portfolio_characteristics'] = {
            'number_of_assets': len(assets),
            'effective_number': risk_decomp.get('effective_number', len(assets)),
            'concentration_herfindahl': risk_decomp.get('herfindahl_index', 0),
            'concentration_gini': risk_decomp.get('gini_coefficient', 0),
            'diversification_ratio': risk_decomp.get('diversification_ratio', 1.0)
        }
        
        return result
    
    def efficient_frontier(
        self,
        returns_df: pd.DataFrame,
        n_points: int = 20,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate efficient frontier
        
        Args:
            returns_df: DataFrame of asset returns
            n_points: Number of points on frontier
            constraints: Optimization constraints
        
        Returns:
            Dictionary with efficient frontier data
        """
        # Prepare data
        returns_df = self._prepare_returns_data(returns_df)
        
        if returns_df.empty:
            return {"success": False, "error": "Insufficient data"}
        
        # Calculate inputs
        mean_returns = returns_df.mean() * self.annual_trading_days
        cov_matrix = returns_df.cov() * self.annual_trading_days
        cov_matrix = self._make_psd(cov_matrix)
        
        # Get min and max return portfolios
        min_var_result = self._optimize_min_variance(returns_df, constraints or {})
        max_return_result = self._optimize_max_return(returns_df, constraints or {})
        
        if not min_var_result['success'] or not max_return_result['success']:
            return {"success": False, "error": "Could not compute frontier endpoints"}
        
        # Extract min and max returns
        min_var_weights = np.array(list(min_var_result['weights'].values()))
        max_return_weights = np.array(list(max_return_result['weights'].values()))
        
        min_return = np.sum(mean_returns * min_var_weights)
        max_return = np.sum(mean_returns * max_return_weights)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate efficient frontier points
        frontier_points = []
        
        for target in target_returns:
            result = self._optimize_mean_variance(
                returns_df,
                constraints or {},
                target
            )
            
            if result['success']:
                weights = np.array(list(result['weights'].values()))
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                
                frontier_points.append({
                    'target_return': float(target),
                    'actual_return': float(portfolio_return),
                    'risk': float(portfolio_risk),
                    'sharpe': float((portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0),
                    'weights': result['weights']
                })
        
        # Calculate market portfolio (tangency portfolio)
        market_result = self._optimize_sharpe(returns_df, constraints or {})
        
        if market_result['success']:
            market_weights = np.array(list(market_result['weights'].values()))
            market_return = np.sum(mean_returns * market_weights)
            market_risk = np.sqrt(market_weights.T @ cov_matrix @ market_weights)
            market_sharpe = (market_return - self.risk_free_rate) / market_risk if market_risk > 0 else 0
            
            market_portfolio = {
                'return': float(market_return),
                'risk': float(market_risk),
                'sharpe': float(market_sharpe),
                'weights': market_result['weights']
            }
        else:
            market_portfolio = None
        
        # Capital Market Line
        cml_points = []
        if market_portfolio:
            # Generate CML from risk-free to market portfolio and beyond
            max_cml_risk = market_risk * 2
            cml_risks = np.linspace(0, max_cml_risk, n_points)
            
            for risk in cml_risks:
                if risk <= market_risk:
                    # On CML segment
                    weight_market = risk / market_risk if market_risk > 0 else 0
                    cml_return = self.risk_free_rate + weight_market * (market_return - self.risk_free_rate)
                else:
                    # Extension beyond market portfolio (leveraged)
                    weight_market = risk / market_risk
                    cml_return = self.risk_free_rate + weight_market * (market_return - self.risk_free_rate)
                
                cml_points.append({
                    'risk': float(risk),
                    'return': float(cml_return),
                    'sharpe': float(market_sharpe)  # Constant along CML
                })
        
        return {
            "success": True,
            "frontier_points": frontier_points,
            "market_portfolio": market_portfolio,
            "capital_market_line": cml_points,
            "risk_free_rate": float(self.risk_free_rate),
            "min_variance_portfolio": min_var_result,
            "max_return_portfolio": max_return_result,
            "n_points": len(frontier_points)
        }
    
    def calculate_turnover(
        self,
        weights_current: Dict[str, float],
        weights_previous: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate portfolio turnover
        
        Args:
            weights_current: Current portfolio weights
            weights_previous: Previous portfolio weights
        
        Returns:
            Dictionary with turnover metrics
        """
        # Get common assets
        assets = set(weights_current.keys()) | set(weights_previous.keys())
        
        # Initialize arrays
        current_array = np.zeros(len(assets))
        previous_array = np.zeros(len(assets))
        
        for i, asset in enumerate(assets):
            current_array[i] = weights_current.get(asset, 0)
            previous_array[i] = weights_previous.get(asset, 0)
        
        # Calculate turnover metrics
        absolute_turnover = np.sum(np.abs(current_array - previous_array))
        one_way_turnover = absolute_turnover / 2
        
        # Buy and sell breakdown
        buys = np.sum(np.maximum(current_array - previous_array, 0))
        sells = np.sum(np.maximum(previous_array - current_array, 0))
        
        # Tracking error (if both portfolios have same assets)
        if set(weights_current.keys()) == set(weights_previous.keys()):
            tracking_error = np.sqrt(np.sum((current_array - previous_array) ** 2))
        else:
            tracking_error = None
        
        return {
            'absolute_turnover': float(absolute_turnover),
            'one_way_turnover': float(one_way_turnover),
            'buy_turnover': float(buys),
            'sell_turnover': float(sells),
            'turnover_ratio': float(absolute_turnover / 2),  # Common definition
            'tracking_error': float(tracking_error) if tracking_error is not None else None,
            'n_assets': len(assets),
            'assets_added': list(set(weights_current.keys()) - set(weights_previous.keys())),
            'assets_removed': list(set(weights_previous.keys()) - set(weights_current.keys()))
        }

# =============================================================================
# ADVANCED VISUALIZATION ENGINE
# =============================================================================

class VisualizationEngine:
    """Advanced visualization engine for institutional analytics"""
    
    def __init__(self, config: ApplicationConfig = DEFAULT_CONFIG):
        self.config = config
        self.colors = self._get_color_palette()
        
    def _get_color_palette(self) -> Dict[str, List[str]]:
        """Get comprehensive color palettes"""
        return {
            'sequential': [
                '#003f5c', '#2f4b7c', '#665191', '#a05195',
                '#d45087', '#f95d6a', '#ff7c43', '#ffa600'
            ],
            'diverging': [
                '#8e0152', '#c51b7d', '#de77ae', '#f1b6da',
                '#fde0ef', '#e6f5d0', '#b8e186', '#7fbc41',
                '#4d9221', '#276419'
            ],
            'qualitative': [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ],
            'commodities': {
                'gold': '#FFD700',
                'silver': '#C0C0C0',
                'copper': '#B87333',
                'crude_oil': '#000000',
                'natural_gas': '#4169E1',
                'corn': '#FFD700',
                'wheat': '#F5DEB3',
                'soybeans': '#8B4513'
            }
        }
    
    def create_performance_dashboard(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Performance Analysis"
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard
        
        Args:
            returns: Asset/portfolio returns
            benchmark_returns: Benchmark returns (optional)
            title: Dashboard title
        
        Returns:
            Plotly figure object
        """
        # Calculate metrics
        calculator = PerformanceCalculator(self.config)
        metrics = calculator.calculate_performance_metrics(returns, benchmark_returns)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "Cumulative Returns", "Drawdown Analysis",
                "Rolling Returns (12M)", "Return Distribution",
                "Risk Metrics", "Performance Metrics",
                "Monthly Returns Heatmap", "Return Quantiles", "Underwater Plot"
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}, {"type": "domain"}],
                [{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        # 1. Cumulative returns
        cumulative = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['sequential'][0], width=2)
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            bench_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=bench_cumulative.index,
                    y=bench_cumulative.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['sequential'][3], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Drawdown analysis
        drawdown_series = calculator._calculate_drawdown_series(returns)
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1),
                name='Drawdown'
            ),
            row=1, col=2
        )
        
        # 3. Rolling returns
        rolling_window = min(252, len(returns))
        if rolling_window > 0:
            rolling_returns = returns.rolling(window=rolling_window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            ) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns.values,
                    mode='lines',
                    line=dict(color=self.colors['sequential'][2], width=2),
                    name=f'Rolling {rolling_window}D'
                ),
                row=1, col=3
            )
        
        # 4. Return distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                marker_color=self.colors['sequential'][0],
                opacity=0.7,
                name='Return Distribution'
            ),
            row=2, col=1
        )
        
        # Add normal distribution overlay
        if len(returns) > 30:
            x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            y_norm = stats.norm.pdf(x_norm, returns.mean() * 100, returns.std() * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Normal Distribution'
                ),
                row=2, col=1
            )
        
        # 5. Risk metrics gauge (simplified)
        risk_metrics = [
            ("Volatility", metrics.get('annualized_volatility', 0) * 100),
            ("Max DD", abs(metrics.get('max_drawdown', 0)) * 100),
            ("VaR 95%", abs(metrics.get('var_95', 0)) * 100),
            ("CVaR 95%", abs(metrics.get('cvar_95', 0)) * 100)
        ]
        
        # Create custom visualization for risk metrics
        for i, (metric_name, value) in enumerate(risk_metrics):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={"text": metric_name},
                    domain={"row": 1, "column": 2},  # Adjust based on layout
                    gauge={
                        "axis": {"range": [0, max(50, value * 2)]},
                        "bar": {"color": self.colors['sequential'][i]},
                        "steps": [
                            {"range": [0, value * 0.5], "color": "lightgreen"},
                            {"range": [value * 0.5, value * 0.8], "color": "yellow"},
                            {"range": [value * 0.8, value * 2], "color": "red"}
                        ]
                    }
                ),
                row=2, col=2
            )
        
        # 6. Performance metrics table
        key_metrics = {
            "Annual Return": f"{metrics.get('annualized_return', 0) * 100:.2f}%",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}" if not np.isnan(metrics.get('sortino_ratio', np.nan)) else "N/A",
            "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.2f}",
            "Win Rate": f"{metrics.get('win_rate', 0) * 100:.1f}%",
            "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
            "Skewness": f"{metrics.get('skewness', 0):.2f}",
            "Kurtosis": f"{metrics.get('kurtosis', 0):.2f}"
        }
        
        # Create table visualization
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color=self.colors['sequential'][0],
                    align="left",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=[list(key_metrics.keys()), list(key_metrics.values())],
                    fill_color=[["white", "lightgray"] * 4],
                    align="left",
                    font=dict(size=11)
                )
            ),
            row=2, col=3
        )
        
        # 7. Monthly returns heatmap
        if len(returns) > 252:  # At least 1 year of data
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_matrix = self._create_monthly_returns_matrix(monthly_returns)
            
            fig.add_trace(
                go.Heatmap(
                    z=monthly_matrix.values,
                    x=monthly_matrix.columns,  # Years
                    y=monthly_matrix.index,   # Months
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Return %"),
                    name='Monthly Returns'
                ),
                row=3, col=1
            )
        
        # 8. Return quantiles
        quantiles = np.percentile(returns * 100, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        fig.add_trace(
            go.Bar(
                x=[f"{q}%" for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]],
                y=quantiles,
                marker_color=self.colors['sequential'],
                name='Return Quantiles'
            ),
            row=3, col=2
        )
        
        # 9. Underwater plot (enhanced drawdown)
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.5)',
                line=dict(color='darkred', width=2),
                name='Underwater Plot'
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            height=1200,
            showlegend=True,
            template="plotly_white",
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=1, col=3)
        fig.update_yaxes(title_text="Rolling Return %", row=1, col=3)
        
        fig.update_xaxes(title_text="Return %", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return fig
    
    def _create_monthly_returns_matrix(self, monthly_returns: pd.Series) -> pd.DataFrame:
        """Create matrix for monthly returns heatmap"""
        monthly_returns = monthly_returns * 100
        
        # Extract year and month
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create matrix
        matrix = pd.DataFrame(index=month_names, columns=years, dtype=float)
        
        for date, ret in monthly_returns.items():
            matrix.loc[month_names[date.month - 1], date.year] = ret
        
        return matrix
    
    def create_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Create interactive correlation matrix visualization
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Add annotations for significant correlations
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if i > j:  # Lower triangle
                    value = corr_matrix.iloc[i, j]
                    if abs(value) > 0.7:
                        color = "white" if abs(value) > 0.8 else "black"
                        annotations.append(
                            dict(
                                x=col,
                                y=row,
                                text=f"{value:.2f}",
                                showarrow=False,
                                font=dict(color=color, size=9)
                            )
                        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0),
            width=800,
            height=800,
            annotations=annotations,
            template="plotly_white"
        )
        
        return fig
    
    def create_efficient_frontier(
        self,
        frontier_data: Dict[str, Any],
        title: str = "Efficient Frontier"
    ) -> go.Figure:
        """
        Create efficient frontier visualization
        
        Args:
            frontier_data: Output from PortfolioOptimizer.efficient_frontier
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        if not frontier_data.get('success', False):
            return go.Figure()
        
        fig = go.Figure()
        
        # Plot efficient frontier
        frontier_points = frontier_data['frontier_points']
        risks = [p['risk'] for p in frontier_points]
        returns = [p['actual_return'] for p in frontier_points]
        
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Efficient Frontier'
            )
        )
        
        # Plot individual assets
        if 'assets' in frontier_data:
            asset_risks = frontier_data['assets']['risks']
            asset_returns = frontier_data['assets']['returns']
            asset_names = frontier_data['assets']['names']
            
            fig.add_trace(
                go.Scatter(
                    x=asset_risks,
                    y=asset_returns,
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=asset_names,
                    textposition="top center",
                    name='Assets'
                )
            )
        
        # Plot market portfolio
        market_portfolio = frontier_data.get('market_portfolio')
        if market_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[market_portfolio['risk']],
                    y=[market_portfolio['return']],
                    mode='markers',
                    marker=dict(size=20, color='gold', symbol='star'),
                    name='Market Portfolio'
                )
            )
        
        # Plot capital market line
        cml_points = frontier_data.get('capital_market_line', [])
        if cml_points:
            cml_risks = [p['risk'] for p in cml_points]
            cml_returns = [p['return'] for p in cml_points]
            
            fig.add_trace(
                go.Scatter(
                    x=cml_risks,
                    y=cml_returns,
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Capital Market Line'
                )
            )
        
        # Plot risk-free rate
        risk_free_rate = frontier_data.get('risk_free_rate', 0.02)
        fig.add_trace(
            go.Scatter(
                x=[0, max(risks) * 0.8],
                y=[risk_free_rate, risk_free_rate],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name=f'Risk-Free Rate ({risk_free_rate:.1%})'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(title="Annual Risk (Volatility)"),
            yaxis=dict(title="Annual Return"),
            hovermode='closest',
            showlegend=True,
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_risk_decomposition(
        self,
        risk_data: Dict[str, Any],
        title: str = "Risk Decomposition"
    ) -> go.Figure:
        """
        Create risk decomposition visualization
        
        Args:
            risk_data: Output from RiskCalculator.calculate_risk_decomposition
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        if 'risk_contributions' not in risk_data:
            return go.Figure()
        
        risk_contributions = risk_data['risk_contributions']
        marginal_contributions = risk_data.get('marginal_contributions', {})
        pct_contributions = risk_data.get('percentage_contributions', {})
        
        # Sort by contribution
        assets = sorted(risk_contributions.keys(),
                       key=lambda x: abs(risk_contributions[x]), reverse=True)
        
        # Create stacked bar chart
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Risk Contributions", "Marginal Contributions", "% Contributions"),
            horizontal_spacing=0.1
        )
        
        # Risk contributions
        fig.add_trace(
            go.Bar(
                x=assets,
                y=[risk_contributions[asset] for asset in assets],
                name='Risk Contribution',
                marker_color=self.colors['sequential'][0]
            ),
            row=1, col=1
        )
        
        # Marginal contributions
        if marginal_contributions:
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=[marginal_contributions[asset] for asset in assets],
                    name='Marginal Contribution',
                    marker_color=self.colors['sequential'][2]
                ),
                row=1, col=2
            )
        
        # Percentage contributions
        if pct_contributions:
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=[pct_contributions[asset] * 100 for asset in assets],
                    name='% Contribution',
                    marker_color=self.colors['sequential'][4]
                ),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            template="plotly_white",
            height=500
        )
        
        fig.update_yaxes(title_text="Risk Contribution", row=1, col=1)
        fig.update_yaxes(title_text="Marginal Contribution", row=1, col=2)
        fig.update_yaxes(title_text="% Contribution", row=1, col=3)
        
        return fig
    
    def create_monte_carlo_simulation(
        self,
        simulation_data: Dict[str, Any],
        title: str = "Monte Carlo Simulation"
    ) -> go.Figure:
        """
        Create Monte Carlo simulation visualization
        
        Args:
            simulation_data: Output from Monte Carlo simulation
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        if not simulation_data.get('success', False):
            return go.Figure()
        
        simulations = simulation_data.get('simulations', np.array([]))
        final_prices = simulation_data.get('final_prices', np.array([]))
        initial_price = simulation_data.get('initial_price', 0)
        
        if simulations.size == 0 or final_prices.size == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Simulation Paths",
                "Final Price Distribution",
                "Cumulative Probability",
                "Confidence Intervals"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Simulation paths (sample)
        n_paths = min(100, simulations.shape[0])
        sample_indices = np.random.choice(simulations.shape[0], n_paths, replace=False)
        
        for idx in sample_indices:
            fig.add_trace(
                go.Scatter(
                    x=list(range(simulations.shape[1])),
                    y=simulations[idx],
                    mode='lines',
                    line=dict(width=1, color='rgba(0,100,255,0.1)'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add mean path
        mean_path = simulations.mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(range(simulations.shape[1])),
                y=mean_path,
                mode='lines',
                line=dict(color='red', width=3),
                name='Mean Path'
            ),
            row=1, col=1
        )
        
        # 2. Final price distribution
        fig.add_trace(
            go.Histogram(
                x=final_prices,
                nbinsx=50,
                marker_color=self.colors['sequential'][0],
                opacity=0.7,
                name='Price Distribution'
            ),
            row=1, col=2
        )
        
        # Add vertical lines
        fig.add_vline(
            x=initial_price,
            line_dash="dash",
            line_color="green",
            annotation_text="Initial Price",
            row=1, col=2
        )
        
        fig.add_vline(
            x=np.mean(final_prices),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean",
            row=1, col=2
        )
        
        # 3. Cumulative probability
        sorted_prices = np.sort(final_prices)
        cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_prices,
                y=cdf * 100,
                mode='lines',
                line=dict(color='blue', width=2),
                name='CDF'
            ),
            row=2, col=1
        )
        
        # Add probability lines
        for confidence in [0.05, 0.25, 0.5, 0.75, 0.95]:
            price = np.percentile(final_prices, confidence * 100)
            fig.add_vline(
                x=price,
                line_dash="dot",
                line_color="gray",
                row=2, col=1
            )
        
        # 4. Confidence intervals over time
        time_points = list(range(simulations.shape[1]))
        percentiles = [5, 25, 50, 75, 95]
        
        for i, p in enumerate(percentiles):
            ci_lower = np.percentile(simulations, p, axis=0)
            ci_upper = np.percentile(simulations, 100 - p, axis=0)
            
            # Only plot outer CIs
            if p in [5, 95]:
                fig.add_trace(
                    go.Scatter(
                        x=time_points + time_points[::-1],
                        y=list(ci_upper) + list(ci_lower[::-1]),
                        fill='toself',
                        fillcolor=f'rgba(0,100,255,{0.3/(i+1)})',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{100-2*p}% CI',
                        showlegend=True
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=True,
            template="plotly_white",
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Days", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        fig.update_xaxes(title_text="Final Price", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Probability %", row=2, col=1)
        
        fig.update_xaxes(title_text="Days", row=2, col=2)
        fig.update_yaxes(title_text="Price", row=2, col=2)
        
        return fig
    
    def create_regime_detection(
        self,
        prices: pd.Series,
        states: np.ndarray,
        title: str = "Market Regime Detection"
    ) -> go.Figure:
        """
        Create regime detection visualization
        
        Args:
            prices: Price series
            states: Regime states array
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price with Regimes", "Regime States"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True
        )
        
        # Plot price
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode='lines',
                line=dict(color='blue', width=1),
                name='Price'
            ),
            row=1, col=1
        )
        
        # Color background by regime
        unique_states = np.unique(states)
        colors = ['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.1)', 'rgba(0,0,255,0.1)',
                 'rgba(255,255,0,0.1)', 'rgba(255,0,255,0.1)']
        
        for i, state in enumerate(unique_states):
            mask = states == state
            if mask.any():
                # Add shaded regions
                fig.add_trace(
                    go.Scatter(
                        x=prices.index[mask],
                        y=prices.values[mask],
                        mode='markers',
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=8,
                            symbol='square'
                        ),
                        name=f'Regime {state}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Plot regime states
        fig.add_trace(
            go.Scatter(
                x=prices.index[:len(states)],
                y=states,
                mode='lines',
                line=dict(color='purple', width=2),
                name='Regime State'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=True,
            template="plotly_white",
            height=600
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1)
        
        return fig

# =============================================================================
# PROFESSIONAL REPORT GENERATOR
# =============================================================================

class ProfessionalReportGenerator:
    """Generate professional PDF and Excel reports"""
    
    def __init__(self, config: ApplicationConfig = DEFAULT_CONFIG):
        self.config = config
        
    def generate_pdf_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Union[str, Path],
        title: str = "Commodities Analysis Report"
    ) -> bool:
        """
        Generate professional PDF report
        
        Args:
            analysis_results: Dictionary with analysis results
            output_path: Output file path
            title: Report title
        
        Returns:
            Success status
        """
        if not dep_manager.is_available('reportlab'):
            st.warning("ReportLab not available - PDF export disabled")
            return False
        
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.graphics.shapes import Drawing
            from reportlab.graphics.charts.lineplots import LinePlot
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            
            # Create document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'Heading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            )
            
            normal_style = styles['Normal']
            
            # Story elements
            story = []
            
            # Title
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Metadata
            metadata = [
                ["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Analysis Period", f"{analysis_results.get('start_date', 'N/A')} to {analysis_results.get('end_date', 'N/A')}"],
                ["Assets Analyzed", str(len(analysis_results.get('assets', [])))],
                ["Report Version", self.config.app_version]
            ]
            
            metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 24))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            summary_text = analysis_results.get('summary', 'No summary available.')
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 12))
            
            # Performance Metrics
            if 'performance_metrics' in analysis_results:
                story.append(Paragraph("Performance Metrics", heading_style))
                
                perf_data = analysis_results['performance_metrics']
                perf_table_data = [["Metric", "Value"]]
                
                key_metrics = [
                    ("Annual Return", f"{perf_data.get('annualized_return', 0)*100:.2f}%"),
                    ("Annual Volatility", f"{perf_data.get('annualized_volatility', 0)*100:.2f}%"),
                    ("Sharpe Ratio", f"{perf_data.get('sharpe_ratio', 0):.2f}"),
                    ("Max Drawdown", f"{abs(perf_data.get('max_drawdown', 0))*100:.2f}%"),
                    ("Win Rate", f"{perf_data.get('win_rate', 0)*100:.1f}%"),
                    ("Profit Factor", f"{perf_data.get('profit_factor', 0):.2f}")
                ]
                
                for metric, value in key_metrics:
                    perf_table_data.append([metric, value])
                
                perf_table = Table(perf_table_data, colWidths=[2*inch, 1.5*inch])
                perf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(perf_table)
                story.append(Spacer(1, 12))
            
            # Portfolio Allocation
            if 'portfolio_allocation' in analysis_results:
                story.append(Paragraph("Portfolio Allocation", heading_style))
                
                alloc_data = analysis_results['portfolio_allocation']
                alloc_table_data = [["Asset", "Weight", "Contribution"]]
                
                for asset, weight in alloc_data.items():
                    alloc_table_data.append([
                        asset,
                        f"{weight*100:.1f}%",
                        f"{weight*100:.1f}%"  # Simplified - would use risk contribution in real implementation
                    ])
                
                alloc_table = Table(alloc_table_data, colWidths=[1.5*inch, 1*inch, 1.5*inch])
                alloc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(alloc_table)
                story.append(Spacer(1, 12))
            
            # Risk Metrics
            if 'risk_metrics' in analysis_results:
                story.append(Paragraph("Risk Analysis", heading_style))
                
                risk_data = analysis_results['risk_metrics']
                risk_table_data = [["Risk Metric", "Value"]]
                
                risk_metrics = [
                    ("VaR (95%)", f"{abs(risk_data.get('var_95', 0))*100:.2f}%"),
                    ("CVaR (95%)", f"{abs(risk_data.get('cvar_95', 0))*100:.2f}%"),
                    ("Expected Shortfall", f"{abs(risk_data.get('cvar_95', 0))*100:.2f}%"),
                    ("Tail Ratio", f"{risk_data.get('tail_ratio', 0):.2f}"),
                    ("Max Consecutive Losses", str(risk_data.get('max_consecutive_losses', 0)))
                ]
                
                for metric, value in risk_metrics:
                    risk_table_data.append([metric, value])
                
                risk_table = Table(risk_table_data, colWidths=[2*inch, 1.5*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(risk_table)
                story.append(Spacer(1, 12))
            
            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            
            recommendations = analysis_results.get('recommendations', [
                "Maintain current portfolio allocation.",
                "Monitor volatility and adjust risk exposure if necessary.",
                "Consider diversification into uncorrelated assets.",
                "Review position sizing based on current market conditions."
            ])
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", normal_style))
            
            story.append(Spacer(1, 12))
            
            # Disclaimer
            disclaimer = Paragraph(
                "<b>Disclaimer:</b> This report is for informational purposes only. "
                "Past performance is not indicative of future results. "
                "Investing involves risks including possible loss of principal.",
                styles['Italic']
            )
            
            story.append(disclaimer)
            
            # Build PDF
            doc.build(story)
            
            return True
            
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            return False
    
    def generate_excel_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Generate comprehensive Excel report
        
        Args:
            analysis_results: Dictionary with analysis results
            output_path: Output file path
        
        Returns:
            Success status
        """
        buffer = BytesIO()
        writer, engine = icd_safe_excel_writer(buffer)
        
        if writer is None:
            st.warning("Excel export not available - no engine found")
            return False
        
        try:
            # Summary sheet
            summary_data = {
                "Report Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Analysis Period": [f"{analysis_results.get('start_date', 'N/A')} to {analysis_results.get('end_date', 'N/A')}"],
                "Assets Analyzed": [len(analysis_results.get('assets', []))],
                "Report Version": [self.config.app_version]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Performance metrics sheet
            if 'performance_metrics' in analysis_results:
                perf_df = pd.DataFrame([analysis_results['performance_metrics']])
                perf_df.to_excel(writer, sheet_name="Performance")
            
            # Portfolio allocation sheet
            if 'portfolio_allocation' in analysis_results:
                alloc_df = pd.DataFrame(
                    list(analysis_results['portfolio_allocation'].items()),
                    columns=["Asset", "Weight"]
                )
                alloc_df.to_excel(writer, sheet_name="Allocation", index=False)
            
            # Risk metrics sheet
            if 'risk_metrics' in analysis_results:
                risk_df = pd.DataFrame([analysis_results['risk_metrics']])
                risk_df.to_excel(writer, sheet_name="Risk")
            
            # Correlation matrix sheet
            if 'correlation_matrix' in analysis_results:
                corr_df = analysis_results['correlation_matrix']
                if isinstance(corr_df, pd.DataFrame):
                    corr_df.to_excel(writer, sheet_name="Correlations")
            
            # Efficient frontier sheet
            if 'efficient_frontier' in analysis_results:
                frontier_data = analysis_results['efficient_frontier']
                if frontier_data.get('success', False):
                    frontier_points = frontier_data.get('frontier_points', [])
                    if frontier_points:
                        frontier_df = pd.DataFrame(frontier_points)
                        frontier_df.to_excel(writer, sheet_name="Efficient Frontier")
            
            # Monte Carlo simulation sheet
            if 'monte_carlo' in analysis_results:
                mc_data = analysis_results['monte_carlo']
                if mc_data.get('success', False):
                    mc_summary = {
                        "Initial Price": [mc_data.get('initial_price', 0)],
                        "Expected Price": [mc_data.get('expected_price', 0)],
                        "Probability of Profit": [f"{mc_data.get('prob_profit', 0):.1f}%"],
                        "VaR 95%": [mc_data.get('var_95', 0)],
                        "CVaR 95%": [mc_data.get('cvar_95', 0)]
                    }
                    mc_summary_df = pd.DataFrame(mc_summary)
                    mc_summary_df.to_excel(writer, sheet_name="Monte Carlo", index=False)
            
            # Regime detection sheet
            if 'regime_detection' in analysis_results:
                regime_data = analysis_results['regime_detection']
                if regime_data.get('success', False):
                    regime_df = pd.DataFrame({
                        "State": regime_data.get('states', []),
                        "Probability": regime_data.get('state_probabilities', [])
                    })
                    regime_df.to_excel(writer, sheet_name="Regimes", index=False)
            
            # Save workbook
            writer.close()
            buffer.seek(0)
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(buffer.read())
            
            return True
            
        except Exception as e:
            st.error(f"Error generating Excel report: {str(e)}")
            return False

# =============================================================================
# MAIN APPLICATION WITH ADVANCED UI
# =============================================================================

class InstitutionalCommoditiesApp:
    """Main application class"""
    
    def __init__(self):
        # Initialize configuration
        self.config = DEFAULT_CONFIG
        
        # Initialize components
        self.data_manager = EnhancedDataManager(self.config)
        self.analytics = QuantitativeAnalytics(self.config)
        self.visualizer = VisualizationEngine(self.config)
        self.report_generator = ProfessionalReportGenerator(self.config)
        
        # Session state initialization
        self._init_session_state()
        
        # UI theming
        self._apply_styles()
    
    def _init_session_state(self):
        """Initialize session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data_cache = {}
            st.session_state.analysis_cache = {}
            st.session_state.current_page = "dashboard"
            st.session_state.selected_assets = []
            st.session_state.analysis_results = {}
            st.session_state.export_data = {}
    
    def _apply_styles(self):
        """Apply custom CSS styles"""
        st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    color: white;
                    margin-bottom: 2rem;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }
                
                .metric-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;
                    border-left: 5px solid #667eea;
                }
                
                .status-badge {
                    display: inline-block;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: bold;
                    margin: 0.2rem;
                }
                
                .status-success {
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                }
                
                .status-warning {
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    color: white;
                }
                
                .status-danger {
                    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render application header"""
        st.markdown(f"""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.8rem;">🏛️ Institutional Commodities Analytics</h1>
                <p style="font-size: 1.2rem; opacity: 0.9; margin: 0.5rem 0 0 0;">
                    Advanced Portfolio Analytics • GARCH Volatility Modeling • Machine Learning Regime Detection
                </p>
                <div style="margin-top: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span class="status-badge status-success">Live Market Data</span>
                    <span class="status-badge status-warning">Real-time Analytics</span>
                    <span class="status-badge status-danger">Institutional Grade</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with configuration"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=self.config.default_start_date,
                    max_value=datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=self.config.default_end_date,
                    max_value=datetime.now()
                )
            
            # Asset selection
            st.subheader("📊 Asset Selection")
            
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
            
            st.session_state.selected_assets = selected_assets
            
            # Benchmark selection
            st.subheader("📈 Benchmarks")
            benchmark_options = {k: v['name'] for k, v in BENCHMARKS.items()}
            selected_benchmarks = st.multiselect(
                "Select Benchmarks",
                options=list(benchmark_options.keys()),
                default=["^GSPC", "GLD"],
                format_func=lambda x: benchmark_options[x]
            )
            
            # Analysis parameters
            st.subheader("🔧 Analysis Parameters")
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=self.config.risk_free_rate * 100,
                step=0.1
            ) / 100
            
            self.config.risk_free_rate = risk_free_rate
            
            # Advanced options
            with st.expander("Advanced Options"):
                garch_p = st.slider("GARCH p parameter", 1, 5, self.config.default_garch_p)
                garch_q = st.slider("GARCH q parameter", 1, 5, self.config.default_garch_q)
                
                monte_carlo_sims = st.slider(
                    "Monte Carlo Simulations",
                    1000, 50000, self.config.monte_carlo_simulations, 1000
                )
                
                confidence_level = st.select_slider(
                    "Confidence Level",
                    options=[0.90, 0.95, 0.99],
                    value=0.95
                )
            
            # Action buttons
            st.subheader("🚀 Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Run Analysis", type="primary", use_container_width=True):
                    st.session_state.run_analysis = True
            
            with col2:
                if st.button("📊 Export Results", use_container_width=True):
                    st.session_state.export_results = True
            
            # System info
            st.divider()
            st.caption(f"Build: {self.config.app_version}")
            st.caption(f"Environment: {self.config.environment}")
            
            # Cache info
            if st.button("Clear Cache"):
                self.data_manager.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "risk_free_rate": risk_free_rate,
                "garch_p": garch_p,
                "garch_q": garch_q,
                "monte_carlo_sims": monte_carlo_sims,
                "confidence_level": confidence_level
            }
    
    def run_analysis(self, config: Dict[str, Any]):
        """Run comprehensive analysis"""
        with st.spinner("🔄 Running analysis..."):
            try:
                # Fetch data
                all_symbols = config["selected_assets"] + config["selected_benchmarks"]
                
                progress_bar = st.progress(0, text="Fetching market data...")
                
                def update_progress(progress):
                    progress_bar.progress(progress, text="Fetching market data...")
                
                data = self.data_manager.fetch_multiple_assets(
                    all_symbols,
                    config["start_date"],
                    config["end_date"],
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0, text="Data fetched successfully!")
                
                if not data:
                    st.error("Failed to fetch data. Please check your selections.")
                    return
                
                # Calculate returns
                returns_data = {}
                prices_data = {}
                
                for symbol, df in data.items():
                    if not df.empty and 'Adj_Close' in df.columns:
                        returns_data[symbol] = df['Adj_Close'].pct_change().dropna()
                        prices_data[symbol] = df['Adj_Close']
                
                # Store in session state
                st.session_state.data_cache = {
                    'prices': prices_data,
                    'returns': returns_data,
                    'raw_data': data
                }
                
                # Run analyses
                st.info("📈 Running quantitative analyses...")
                
                analysis_results = {}
                
                # Portfolio optimization
                if len(config["selected_assets"]) >= 2:
                    returns_df = pd.DataFrame({
                        symbol: returns_data[symbol] 
                        for symbol in config["selected_assets"] 
                        if symbol in returns_data
                    }).dropna()
                    
                    if not returns_df.empty:
                        optimization_result = self.analytics.optimizer.optimize(
                            returns_df,
                            method='sharpe'
                        )
                        
                        if optimization_result['success']:
                            analysis_results['optimization'] = optimization_result
                            
                            # Calculate efficient frontier
                            frontier_result = self.analytics.optimizer.efficient_frontier(
                                returns_df,
                                n_points=20
                            )
                            analysis_results['efficient_frontier'] = frontier_result
                
                # GARCH analysis for selected assets
                garch_results = {}
                for symbol in config["selected_assets"][:3]:  # Limit to 3 for performance
                    if symbol in returns_data:
                        garch_result = self.analytics.volatility.garch_analysis(
                            returns_data[symbol],
                            p=config["garch_p"],
                            q=config["garch_q"]
                        )
                        garch_results[symbol] = garch_result
                
                analysis_results['garch'] = garch_results
                
                # Monte Carlo simulation
                if config["selected_assets"]:
                    symbol = config["selected_assets"][0]
                    if symbol in prices_data and symbol in returns_data:
                        current_price = prices_data[symbol].iloc[-1]
                        mc_result = self.analytics.monte_carlo.simulate(
                            returns_data[symbol],
                            current_price,
                            n_simulations=config["monte_carlo_sims"]
                        )
                        analysis_results['monte_carlo'] = mc_result
                
                # Store results
                st.session_state.analysis_results = analysis_results
                st.session_state.last_analysis_config = config
                
                st.success("✅ Analysis completed successfully!")
                
                # Display results
                self.display_results(analysis_results, config)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.error(traceback.format_exc())
    
    def display_results(self, analysis_results: Dict[str, Any], config: Dict[str, Any]):
        """Display analysis results"""
        # Create tabs for different sections
        tabs = st.tabs([
            "📊 Portfolio Analysis",
            "📈 Volatility Modeling",
            "🎯 Monte Carlo Simulation",
            "📋 Reports",
            "📊 System Diagnostics"
        ])
        
        with tabs[0]:
            self._display_portfolio_analysis(analysis_results)
        
        with tabs[1]:
            self._display_volatility_analysis(analysis_results)
        
        with tabs[2]:
            self._display_monte_carlo_analysis(analysis_results)
        
        with tabs[3]:
            self._display_reports(analysis_results, config)
        
        with tabs[4]:
            self._display_system_diagnostics()
    
    def _display_portfolio_analysis(self, analysis_results: Dict[str, Any]):
        """Display portfolio analysis results"""
        st.header("📊 Portfolio Analysis")
        
        if 'optimization' not in analysis_results:
            st.info("Run portfolio optimization to see results.")
            return
        
        opt_result = analysis_results['optimization']
        
        if not opt_result['success']:
            st.error(f"Optimization failed: {opt_result.get('error', 'Unknown error')}")
            return
        
        # Display optimized weights
        st.subheader("Optimized Portfolio Allocation")
        
        weights_df = pd.DataFrame(
            list(opt_result['weights'].items()),
            columns=["Asset", "Weight"]
        )
        weights_df["Weight %"] = weights_df["Weight"] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=weights_df["Asset"],
                values=weights_df["Weight %"],
                hole=0.3,
                textinfo='label+percent',
                marker=dict(colors=self.visualizer.colors['sequential'])
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Metrics table
            metrics = opt_result.get('performance_metrics', {})
            
            metric_table = pd.DataFrame({
                "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", 
                          "Max Drawdown", "Win Rate"],
                "Value": [
                    f"{metrics.get('annualized_return', 0)*100:.2f}%",
                    f"{metrics.get('annualized_volatility', 0)*100:.2f}%",
                    f"{metrics.get('sharpe_ratio', 0):.2f}",
                    f"{abs(metrics.get('max_drawdown', 0))*100:.2f}%",
                    f"{metrics.get('win_rate', 0)*100:.1f}%"
                ]
            })
            
            st.dataframe(
                metric_table,
                use_container_width=True,
                hide_index=True
            )
        
        # Efficient frontier
        if 'efficient_frontier' in analysis_results:
            st.subheader("Efficient Frontier")
            
            frontier_data = analysis_results['efficient_frontier']
            if frontier_data['success']:
                fig = self.visualizer.create_efficient_frontier(frontier_data)
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        if 'data_cache' in st.session_state and 'returns' in st.session_state.data_cache:
            returns_data = st.session_state.data_cache['returns']
            assets = [a for a in config["selected_assets"] if a in returns_data]
            
            if len(assets) >= 2:
                st.subheader("Correlation Matrix")
                
                returns_df = pd.DataFrame({
                    asset: returns_data[asset] for asset in assets
                }).dropna()
                
                corr_matrix = returns_df.corr()
                fig = self.visualizer.create_correlation_matrix(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_volatility_analysis(self, analysis_results: Dict[str, Any]):
        """Display volatility analysis results"""
        st.header("📈 Volatility Modeling")
        
        if 'garch' not in analysis_results:
            st.info("Run GARCH analysis to see results.")
            return
        
        garch_results = analysis_results['garch']
        
        # Select asset to display
        assets = list(garch_results.keys())
        selected_asset = st.selectbox(
            "Select Asset",
            assets,
            format_func=lambda x: COMMODITIES_UNIVERSE[
                [cat for cat, assets in COMMODITIES_UNIVERSE.items() if x in assets][0]
            ][x].name
        )
        
        if selected_asset not in garch_results:
            return
        
        garch_result = garch_results[selected_asset]
        
        if not garch_result.get('success', False):
            st.error(f"GARCH analysis failed: {garch_result.get('message', 'Unknown error')}")
            return
        
        # Display GARCH results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Parameters")
            
            if 'best_model' in garch_result:
                best_model = garch_result['best_model']
                
                params_table = pd.DataFrame({
                    "Parameter": ["p", "q", "Distribution", "BIC", "AIC", "Log-Likelihood"],
                    "Value": [
                        best_model.get('p', 'N/A'),
                        best_model.get('q', 'N/A'),
                        best_model.get('distribution', 'N/A'),
                        f"{best_model.get('bic', 0):.2f}",
                        f"{best_model.get('aic', 0):.2f}",
                        f"{best_model.get('log_likelihood', 0):.2f}"
                    ]
                })
                
                st.dataframe(params_table, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Model Comparison")
            
            if 'all_models' in garch_result:
                models_df = pd.DataFrame(garch_result['all_models'])
                if not models_df.empty:
                    st.dataframe(
                        models_df.sort_values('bic').style.highlight_min(
                            subset=['bic', 'aic'],
                            color='lightgreen'
                        ),
                        use_container_width=True,
                        height=300
                    )
        
        # Volatility plot
        st.subheader("Volatility Analysis")
        
        if 'conditional_volatility' in garch_result:
            conditional_vol = garch_result['conditional_volatility']
            returns = garch_result['returns']
            
            # Calculate realized volatility for comparison
            realized_vol = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Create comparison plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=conditional_vol.index,
                y=conditional_vol.values * 100,
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
                title=f"Volatility Comparison: {selected_asset}",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=500,
                hovermode='x unified',
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_monte_carlo_analysis(self, analysis_results: Dict[str, Any]):
        """Display Monte Carlo simulation results"""
        st.header("🎯 Monte Carlo Simulation")
        
        if 'monte_carlo' not in analysis_results:
            st.info("Run Monte Carlo simulation to see results.")
            return
        
        mc_result = analysis_results['monte_carlo']
        
        if not mc_result.get('success', False):
            st.error(f"Monte Carlo simulation failed: {mc_result.get('message', 'Unknown error')}")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Price",
                f"${mc_result.get('expected_price', 0):.2f}",
                f"{((mc_result.get('expected_price', 0) / mc_result.get('initial_price', 1) - 1) * 100):.1f}%"
            )
        
        with col2:
            st.metric(
                "Probability of Profit",
                f"{mc_result.get('prob_profit', 0):.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "95% VaR",
                f"${mc_result.get('var_95', 0):.2f}",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "95% CVaR",
                f"${mc_result.get('cvar_95', 0):.2f}",
                delta_color="inverse"
            )
        
        # Create visualization
        fig = self.visualizer.create_monte_carlo_simulation(mc_result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional statistics
        with st.expander("📊 Detailed Statistics"):
            stats_data = {
                "Initial Price": mc_result.get('initial_price', 0),
                "Median Price": mc_result.get('median_price', 0),
                "Standard Deviation": np.std(mc_result.get('final_prices', [0])),
                "Skewness": stats.skew(mc_result.get('final_prices', [0])),
                "Kurtosis": stats.kurtosis(mc_result.get('final_prices', [0])),
                "Minimum Price": np.min(mc_result.get('final_prices', [0])),
                "Maximum Price": np.max(mc_result.get('final_prices', [0])),
                "Interquartile Range": np.percentile(mc_result.get('final_prices', [0]), 75) - 
                                      np.percentile(mc_result.get('final_prices', [0]), 25)
            }
            
            stats_df = pd.DataFrame(
                list(stats_data.items()),
                columns=["Statistic", "Value"]
            )
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def _display_reports(self, analysis_results: Dict[str, Any], config: Dict[str, Any]):
        """Display reports and export options"""
        st.header("📋 Reports & Export")
        
        # Compile report data
        report_data = {
            "summary": f"Commodities Analysis Report for {len(config['selected_assets'])} assets",
            "start_date": config["start_date"].isoformat(),
            "end_date": config["end_date"].isoformat(),
            "assets": config["selected_assets"],
            "performance_metrics": analysis_results.get('optimization', {}).get('performance_metrics', {}),
            "portfolio_allocation": analysis_results.get('optimization', {}).get('weights', {}),
            "risk_metrics": analysis_results.get('monte_carlo', {}),
            "correlation_matrix": None,  # Would be added in real implementation
            "efficient_frontier": analysis_results.get('efficient_frontier', {}),
            "monte_carlo": analysis_results.get('monte_carlo', {}),
            "garch_analysis": analysis_results.get('garch', {}),
            "recommendations": [
                "Maintain diversified portfolio allocation",
                "Monitor volatility and adjust risk exposure",
                "Consider tactical allocations based on regime signals",
                "Review position sizing and risk limits regularly"
            ]
        }
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    # Create temporary file
                    temp_file = Path("temp_report.pdf")
                    success = self.report_generator.generate_pdf_report(
                        report_data,
                        temp_file,
                        title="Commodities Analysis Report"
                    )
                    
                    if success:
                        with open(temp_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=f,
                                file_name=f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        # Clean up
                        temp_file.unlink(missing_ok=True)
        
        with col2:
            if st.button("📊 Generate Excel Report", use_container_width=True):
                with st.spinner("Generating Excel report..."):
                    temp_file = Path("temp_report.xlsx")
                    success = self.report_generator.generate_excel_report(
                        report_data,
                        temp_file
                    )
                    
                    if success:
                        with open(temp_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Excel",
                                data=f,
                                file_name=f"commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        temp_file.unlink(missing_ok=True)
        
        with col3:
            if st.button("📋 Export JSON Data", use_container_width=True):
                # Export as JSON
                json_data = json.dumps(report_data, indent=2, default=str)
                
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name=f"commodities_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Display report preview
        st.subheader("Report Preview")
        
        with st.expander("📊 Performance Summary", expanded=True):
            if 'performance_metrics' in report_data:
                perf_data = report_data['performance_metrics']
                
                cols = st.columns(4)
                metrics_display = [
                    ("Annual Return", f"{perf_data.get('annualized_return', 0)*100:.2f}%"),
                    ("Annual Volatility", f"{perf_data.get('annualized_volatility', 0)*100:.2f}%"),
                    ("Sharpe Ratio", f"{perf_data.get('sharpe_ratio', 0):.2f}"),
                    ("Max Drawdown", f"{abs(perf_data.get('max_drawdown', 0))*100:.2f}%")
                ]
                
                for i, (label, value) in enumerate(metrics_display):
                    with cols[i]:
                        st.metric(label, value)
        
        with st.expander("📈 Portfolio Allocation"):
            if 'portfolio_allocation' in report_data:
                alloc_data = report_data['portfolio_allocation']
                
                if alloc_data:
                    alloc_df = pd.DataFrame(
                        list(alloc_data.items()),
                        columns=["Asset", "Weight"]
                    )
                    alloc_df["Weight %"] = alloc_df["Weight"] * 100
                    
                    st.dataframe(
                        alloc_df.sort_values("Weight %", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
        
        with st.expander("⚠️ Risk Metrics"):
            if 'risk_metrics' in report_data:
                risk_data = report_data['risk_metrics']
                
                if risk_data and risk_data.get('success', False):
                    cols = st.columns(3)
                    
                    risk_metrics = [
                        ("VaR 95%", f"${risk_data.get('var_95', 0):.2f}"),
                        ("CVaR 95%", f"${risk_data.get('cvar_95', 0):.2f}"),
                        ("Probability of Profit", f"{risk_data.get('prob_profit', 0):.1f}%")
                    ]
                    
                    for i, (label, value) in enumerate(risk_metrics):
                        with cols[i]:
                            st.metric(label, value)
    
    def _display_system_diagnostics(self):
        """Display system diagnostics"""
        st.header("📊 System Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Information")
            
            sys_info = {
                "Platform": sys.platform,
                "Python Version": sys.version.split()[0],
                "Streamlit Version": st.__version__,
                "Pandas Version": pd.__version__,
                "NumPy Version": np.__version__,
                "App Version": self.config.app_version,
                "Environment": self.config.environment
            }
            
            for key, value in sys_info.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.subheader("Dependencies")
            
            deps_report = dep_manager.get_dependency_report()
            st.text(deps_report)
        
        # Cache statistics
        st.subheader("Cache Statistics")
        
        cache_stats = self.data_manager.get_cache_stats()
        
        if cache_stats:
            cols = st.columns(4)
            
            cache_metrics = [
                ("Cache Size", f"{cache_stats.get('size_mb', 0):.1f} MB"),
                ("Items", str(cache_stats.get('items', 0))),
                ("Hit Rate", f"{cache_stats.get('hit_rate', 0)*100:.1f}%"),
                ("Utilization", f"{cache_stats.get('utilization', 0):.1f}%")
            ]
            
            for i, (label, value) in enumerate(cache_metrics):
                with cols[i]:
                    st.metric(label, value)
        
        # Memory usage
        st.subheader("Memory Usage")
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_metrics = {
                "RSS": f"{memory_info.rss / 1024 / 1024:.1f} MB",
                "VMS": f"{memory_info.vms / 1024 / 1024:.1f} MB",
                "Available RAM": f"{psutil.virtual_memory().available / 1024 / 1024:.0f} MB",
                "CPU Usage": f"{psutil.cpu_percent()}%"
            }
            
            cols = st.columns(4)
            for i, (label, value) in enumerate(memory_metrics.items()):
                with cols[i]:
                    st.metric(label, value)
                    
        except ImportError:
            st.info("Install psutil for memory monitoring")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        if 'analysis_results' in st.session_state:
            analysis_time = st.session_state.get('last_analysis_time', 0)
            st.metric("Last Analysis Time", f"{analysis_time:.1f} seconds")
        
        # Reset button
        if st.button("🔄 Reset Application", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    def run(self):
        """Run the application"""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Check if we should run analysis
        if (hasattr(st.session_state, 'run_analysis') and 
            st.session_state.run_analysis and 
            config["selected_assets"]):
            
            # Reset flag
            st.session_state.run_analysis = False
            
            # Run analysis
            self.run_analysis(config)
        
        # Display welcome message if no analysis has been run
        elif not st.session_state.get('analysis_results'):
            self._display_welcome()
    
    def _display_welcome(self):
        """Display welcome message"""
        st.markdown("""
            ## 🏛️ Welcome to Institutional Commodities Analytics
        
            This platform provides institutional-grade analytics for commodity trading and risk management.
            
            ### Key Features:
            
            **📊 Portfolio Analysis**
            - Multi-asset portfolio optimization
            - Efficient frontier calculation
            - Risk decomposition and attribution
            
            **📈 Volatility Modeling**
            - GARCH/EGARCH volatility forecasting
            - Regime detection using machine learning
            - Stress testing and scenario analysis
            
            **🎯 Monte Carlo Simulation**
            - Price path simulation
            - Value at Risk (VaR) calculation
            - Probability analysis
            
            **📋 Professional Reporting**
            - PDF and Excel report generation
            - Interactive visualizations
            - Comprehensive risk metrics
            
            ### Getting Started:
            
            1. Select assets and benchmarks in the sidebar
            2. Configure analysis parameters
            3. Click "Run Analysis" to generate insights
            4. Export results for professional reporting
            
            ### System Requirements:
            
            - Modern web browser with JavaScript enabled
            - Internet connection for market data
            - Recommended: 8GB+ RAM for complex analyses
            
            ---
            
            *For institutional use only. Past performance is not indicative of future results.*
        """)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main application entry point"""
    try:
        # Initialize application
        app = InstitutionalCommoditiesApp()
        
        # Run application
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        # Provide recovery options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📋 Show Error Details"):
                st.code(traceback.format_exc())

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
