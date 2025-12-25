"""
🏛️ Institutional Commodities Analytics Platform v8.0 (Regime Science Edition)
Enhanced Scientific Analytics • Gaussian HMM • Advanced Correlation • Risk Metrics
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
    page_title="Institutional Commodities Platform v8.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/institutional-commodities',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v8.0
                    Scientific Computing Division • Regime Switching & HMM Analytics
                    © 2024 Institutional Trading Analytics"""
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
        "GC=F": ScientificAssetMetadata("GC=F", "Gold Futures", AssetCategory.PRECIOUS_METALS, "#FFD700", "COMEX Gold Futures", "COMEX", "100 oz", 0.045, 0.10, True, "Low", 0.15, 0.95),
        "SI=F": ScientificAssetMetadata("SI=F", "Silver Futures", AssetCategory.PRECIOUS_METALS, "#C0C0C0", "COMEX Silver Futures", "COMEX", "5000 oz", 0.065, 0.005, True, "Medium", 0.25, 0.85),
        "PL=F": ScientificAssetMetadata("PL=F", "Platinum Futures", AssetCategory.PRECIOUS_METALS, "#E5E4E2", "NYMEX Platinum Futures", "NYMEX", "50 oz", 0.075, 0.10, True, "High", 0.35, 0.70),
    },
    AssetCategory.ENERGY.value: {
        "CL=F": ScientificAssetMetadata("CL=F", "Crude Oil WTI", AssetCategory.ENERGY, "#000000", "NYMEX WTI Crude", "NYMEX", "1000 bbl", 0.085, 0.01, True, "High", 0.60, 0.98),
        "NG=F": ScientificAssetMetadata("NG=F", "Natural Gas", AssetCategory.ENERGY, "#4169E1", "NYMEX Natural Gas", "NYMEX", "10000 MMBtu", 0.095, 0.001, True, "High", 0.30, 0.88),
    },
    AssetCategory.INDUSTRIAL_METALS.value: {
        "HG=F": ScientificAssetMetadata("HG=F", "Copper Futures", AssetCategory.INDUSTRIAL_METALS, "#B87333", "COMEX Copper", "COMEX", "25000 lbs", 0.085, 0.0005, True, "Medium", 0.45, 0.90),
    },
    AssetCategory.AGRICULTURE.value: {
        "ZC=F": ScientificAssetMetadata("ZC=F", "Corn Futures", AssetCategory.AGRICULTURE, "#FFD700", "CBOT Corn", "CBOT", "5000 bu", 0.065, 0.25, True, "Medium", 0.20, 0.82),
        "ZW=F": ScientificAssetMetadata("ZW=F", "Wheat Futures", AssetCategory.AGRICULTURE, "#F5DEB3", "CBOT Wheat", "CBOT", "5000 bu", 0.075, 0.25, True, "Medium", 0.18, 0.80),
    }
}

BENCHMARKS = {
    "^GSPC": {"name": "S&P 500 Index", "type": "equity", "color": "#1E90FF", "description": "US Large Cap"},
    "DX-Y.NYB": {"name": "US Dollar Index", "type": "currency", "color": "#32CD32", "description": "USD Strength"},
    "TLT": {"name": "20+ Year Treasury", "type": "fixed_income", "color": "#8A2BE2", "description": "Long-term Bonds"}
}

# =============================================================================
# SCIENTIFIC THEMING & INSTITUTIONAL STYLING
# =============================================================================

class ScientificThemeManager:
    """Institutional scientific theming"""
    THEMES = {
        "institutional": {
            "primary": "#1a237e", "secondary": "#283593", "accent": "#3949ab", 
            "success": "#2e7d32", "warning": "#f57c00", "danger": "#c62828", 
            "dark": "#0d1b2a", "light": "#e0e1dd", "gray": "#415a77", 
            "background": "#ffffff", "grid": "#e8eaf6", "border": "#c5cae9"
        }
    }
    
    @staticmethod
    def get_styles(theme: str = "institutional") -> str:
        colors = ScientificThemeManager.THEMES.get(theme, ScientificThemeManager.THEMES["institutional"])
        return f"""
        <style>
            :root {{ --primary: {colors['primary']}; --secondary: {colors['secondary']}; --accent: {colors['accent']}; --success: {colors['success']}; --warning: {colors['warning']}; --danger: {colors['danger']}; --dark: {colors['dark']}; --light: {colors['light']}; --gray: {colors['gray']}; --background: {colors['background']}; --grid: {colors['grid']}; --border: {colors['border']}; }}
            .scientific-header {{ background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 100%); padding: 2rem; border-radius: 8px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid var(--border); }}
            .institutional-card {{ background: var(--background); padding: 1.5rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid var(--border); margin-bottom: 1rem; transition: all 0.2s ease-in-out; }}
            .institutional-card:hover {{ box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-color: var(--primary); }}
            .metric-value {{ font-family: 'SF Mono', 'Roboto Mono', monospace; font-size: 1.8rem; font-weight: 700; color: var(--dark); }}
            .scientific-badge {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; border: 1px solid transparent; }}
            .scientific-badge.info {{ background: rgba(41, 98, 255, 0.1); color: var(--primary); border-color: rgba(41, 98, 255, 0.2); }}
            .scientific-badge.warning {{ background: rgba(245, 124, 0, 0.1); color: var(--warning); border-color: rgba(245, 124, 0, 0.2); }}
            .scientific-badge.danger {{ background: rgba(198, 40, 40, 0.1); color: var(--danger); border-color: rgba(198, 40, 40, 0.2); }}
            .section-header {{ display: flex; align-items: center; justify-content: space-between; margin: 2rem 0 1rem; padding-bottom: 0.75rem; border-bottom: 2px solid var(--border); }}
            .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: var(--light); padding: 8px; border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid var(--border); }}
            .stTabs [data-baseweb="tab"] {{ border-radius: 6px; padding: 8px 16px; background-color: var(--background); border: 1px solid var(--border); font-weight: 600; }}
            .stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: white; border-color: var(--primary); }}
        </style>
        """

st.markdown(ScientificThemeManager.get_styles("institutional"), unsafe_allow_html=True)

# =============================================================================
# ADVANCED DEPENDENCY MANAGEMENT WITH VALIDATION
# =============================================================================

class ScientificDependencyManager:
    """Scientific dependency management with validation"""
    
    def __init__(self):
        self.dependencies = {}
        self._scientific_imports()
    
    def _scientific_imports(self):
        """Load scientific dependencies with validation"""
        # statsmodels
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
            self.dependencies['statsmodels'] = {'available': True, 'version': sm.__version__, 'module': sm}
        except ImportError as e:
            self.dependencies['statsmodels'] = {'available': False, 'error': str(e)}
        
        # arch
        try:
            import arch
            from arch import arch_model
            self.dependencies['arch'] = {'available': True, 'version': arch.__version__, 'module': arch}
        except ImportError as e:
            self.dependencies['arch'] = {'available': False, 'error': str(e)}
        
        # hmmlearn & sklearn
        try:
            import sklearn
            import hmmlearn
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
            self.dependencies['hmmlearn'] = {'available': True, 'version': sklearn.__version__, 'module': hmmlearn}
        except ImportError as e:
            self.dependencies['hmmlearn'] = {'available': False, 'error': str(e)}
            
    def display_status(self):
        status_html = '<div class="institutional-card"><div class="metric-title">🧪 Scientific Dependencies</div><div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 0.5rem;">'
        for dep, info in self.dependencies.items():
            status = "🟢" if info.get('available') else "🔴"
            version = info.get('version', 'N/A') if info.get('available') else 'Missing'
            status_html += f'<div style="display: flex; justify-content: space-between;"><span style="font-size:0.85rem;color:#415a77;">{dep}</span><div><span style="font-size:0.75rem;color:#415a77;margin-right:0.5rem;">{version}</span><span>{status}</span></div></div>'
        status_html += "</div></div>"
        return status_html

sci_dep_manager = ScientificDependencyManager()

# =============================================================================
# SCIENTIFIC CACHING SYSTEM WITH VALIDATION
# =============================================================================

class ScientificCache:
    @staticmethod
    def generate_scientific_key(*args, **kwargs) -> str:
        key_string = str(args) + str(kwargs)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def cache_scientific_data(ttl: int = 7200, max_entries: int = 100):
        return st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=False)

# =============================================================================
# SCIENTIFIC DATA MANAGER
# =============================================================================

class ScientificDataManager:
    def __init__(self):
        self.cache = ScientificCache()
    
    @ScientificCache.cache_scientific_data(ttl=10800, max_entries=150)
    def fetch_multiple_assets_scientific(self, symbols: List[str], start_date: datetime, end_date: datetime, max_workers: int = 6) -> Dict[str, pd.DataFrame]:
        results = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_single_asset, symbol, start_date, end_date): symbol 
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty and len(df) > 20:
                        results[symbol] = df
                except Exception as e:
                    print(f"Failed to fetch {symbol}: {e}")
        return results

    def _fetch_single_asset(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        # Force single thread download inside the worker to prevent rate limits
        df = yf.download(symbol, start=start, end=end + timedelta(days=1), interval="1d", progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        if 'Close' in df.columns and 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        return df.dropna()

    def calculate_scientific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Close' not in df.columns: return df
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20D'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        return df.dropna()

# =============================================================================
# SCIENTIFIC CORRELATION ENGINE (Higham & EWMA)
# =============================================================================

class ScientificCorrelationEngine:
    """Advanced scientific correlation analysis with Higham's Algorithm"""
    
    def __init__(self, config: ScientificAnalysisConfiguration):
        self.config = config
        
    def calculate_correlation_matrix(self, returns_dict: Dict[str, pd.Series], method: str = "pearson", min_common_periods: int = 50) -> Dict[str, Any]:
        df = pd.DataFrame(returns_dict)
        df = df.dropna(thresh=min_common_periods, axis=1).dropna()
        if df.empty: return {'correlation_matrix': pd.DataFrame()}
        
        # 

[Image of covariance matrix heatmap]

        if method == "ewma":
            corr_matrix = self._calculate_ewma_correlation(df)
        else:
            corr_matrix = df.corr(method=method)
            
        corr_matrix = self._ensure_valid_correlation_matrix(corr_matrix)
        
        # Calculate significance (p-values) - Fallback safe implementation
        p_values = pd.DataFrame(np.ones(corr_matrix.shape), index=corr_matrix.index, columns=corr_matrix.columns)
        for c1 in df.columns:
            for c2 in df.columns:
                if c1 != c2:
                    try:
                        _, p = stats.pearsonr(df[c1], df[c2])
                        p_values.loc[c1, c2] = p
                    except: pass
        
        return {
            'correlation_matrix': corr_matrix,
            'significance_matrix': p_values,
            'summary_stats': {
                'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()),
                'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max())
            }
        }

    def _calculate_ewma_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        decay = self.config.ewma_lambda
        weights = np.array([(1 - decay) * (decay ** i) for i in range(len(df))])[::-1]
        weights /= weights.sum()
        centered = df - df.mean()
        cov = centered.T @ (centered.mul(weights, axis=0))
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        corr = cov / outer_v
        return pd.DataFrame(corr, index=df.columns, columns=df.columns)

    def _ensure_valid_correlation_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        # Higham's Algorithm (Simplified)
        A = (matrix.values + matrix.values.T) / 2
        vals, vecs = np.linalg.eigh(A)
        vals[vals < 1e-8] = 1e-8
        B = vecs @ np.diag(vals) @ vecs.T
        D = np.diag(1 / np.sqrt(np.diag(B)))
        B = D @ B @ D
        return pd.DataFrame(B, index=matrix.index, columns=matrix.columns)

    def calculate_rolling_correlation(self, s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
        return s1.rolling(window).corr(s2).dropna()

# =============================================================================
# ENHANCED ANALYTICS ENGINE (HMM REGIME SWITCHING)
# =============================================================================

class ScientificAnalyticsEngine:
    """Enhanced scientific analytics with HMM and Regime Detection"""
    
    def __init__(self, config: ScientificAnalysisConfiguration):
        self.config = config
        self.correlation_engine = ScientificCorrelationEngine(config)
        
    def calculate_scientific_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        if len(returns) < 20: return {}
        # Winsorize for robustness
        clean_ret = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99))
        
        ann_factor = np.sqrt(self.config.annual_trading_days)
        vol = clean_ret.std() * ann_factor
        ret = clean_ret.mean() * self.config.annual_trading_days
        sharpe = (ret - self.config.risk_free_rate) / vol if vol > 0 else 0
        
        cum_ret = (1 + clean_ret).cumprod()
        drawdown = (cum_ret - cum_ret.expanding().max()) / cum_ret.expanding().max()
        max_dd = drawdown.min()
        
        var_95 = np.percentile(clean_ret, 5)
        cvar_95 = clean_ret[clean_ret <= var_95].mean()
        
        # Calculate additional advanced metrics
        downside = clean_ret[clean_ret < 0]
        sortino = (ret - self.config.risk_free_rate) / (downside.std() * ann_factor + 1e-9)
        calmar = (ret - self.config.risk_free_rate) / abs(max_dd) if max_dd != 0 else 0
        
        metrics = {
            'Annualized_Return': ret,
            'Annualized_Volatility': vol,
            'Sharpe_Ratio': sharpe,
            'Maximum_Drawdown': max_dd * 100,
            'VaR_95_Historical': abs(var_95) * 100,
            'CVaR_95': abs(cvar_95) * 100,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar
        }
        return {'metrics': metrics, 'validation': {'valid': True, 'n_observations': len(clean_ret)}}

    # -------------------------------------------------------------------------
    # HMM REGIME SWITCHING IMPLEMENTATION
    # -------------------------------------------------------------------------
    
    def detect_market_regimes(self, returns: pd.Series, n_states: int = 2, covariance_type: str = "full", n_iter: int = 100) -> Dict[str, Any]:
        """Fits a Gaussian HMM to detect market regimes. """
        if len(returns) < 100: return {'valid': False, 'error': "Insufficient data for HMM (min 100 points)"}
            
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError: return {'valid': False, 'error': "hmmlearn/sklearn not installed"}
            
        X = returns.values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=42, verbose=False)
        try: model.fit(X_scaled)
        except Exception as e: return {'valid': False, 'error': f"HMM Fit Failed: {str(e)}"}
            
        hidden_states = model.predict(X_scaled)
        posterior_probs = model.predict_proba(X_scaled)
        
        # STATE SORTING (0=Low Vol, 1=High Vol)
        if covariance_type == "full":
            state_vars = np.array([cov[0][0] for cov in model.covars_])
        elif covariance_type == "diag":
            state_vars = np.array([cov[0] for cov in model.covars_])
        else:
            state_vars = model.covars_.flatten()
            
        sorted_indices = np.argsort(state_vars)
        state_map = {old: new for new, old in enumerate(sorted_indices)}
        sorted_hidden_states = np.array([state_map[s] for s in hidden_states])
        sorted_posterior_probs = posterior_probs[:, sorted_indices]
        
        # Regime Statistics
        regime_stats = {}
        df_analysis = pd.DataFrame({'Return': returns.values, 'State': sorted_hidden_states}, index=returns.index)
        for state in range(n_states):
            state_data = df_analysis[df_analysis['State'] == state]['Return']
            if len(state_data) > 0:
                ann_vol = state_data.std() * np.sqrt(252)
                ann_ret = state_data.mean() * 252
                regime_stats[state] = {'count': len(state_data), 'frequency': len(state_data)/len(returns), 'volatility': ann_vol, 'sharpe': (ann_ret - self.config.risk_free_rate)/(ann_vol if ann_vol > 0 else 1.0)}
            else:
                regime_stats[state] = {'count': 0, 'frequency': 0.0, 'volatility': 0.0, 'sharpe': 0.0}
                
        return {'valid': True, 'hidden_states': pd.Series(sorted_hidden_states, index=returns.index), 'regime_stats': regime_stats}

    def calculate_conditional_correlations(self, returns_dict: Dict[str, pd.Series], benchmark_symbol: str, n_states: int = 2) -> Dict[str, Any]:
        """Calculate correlation matrices conditional on the benchmark's regime"""
        if benchmark_symbol not in returns_dict: return {'valid': False, 'error': "Benchmark symbol not found"}
            
        hmm_result = self.detect_market_regimes(returns_dict[benchmark_symbol], n_states=n_states)
        if not hmm_result['valid']: return hmm_result
            
        regimes = hmm_result['hidden_states']
        df_returns = pd.DataFrame(returns_dict).dropna()
        common_idx = df_returns.index.intersection(regimes.index)
        df_returns = df_returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        conditional_matrices = {}
        for state in range(n_states):
            state_returns = df_returns.loc[regimes == state]
            if len(state_returns) > 30:
                res = self.correlation_engine.calculate_correlation_matrix(state_returns.to_dict('series'), method="pearson")
                conditional_matrices[state] = res['correlation_matrix']
            else:
                conditional_matrices[state] = pd.DataFrame()
                
        return {'valid': True, 'regimes_used': regimes, 'conditional_matrices': conditional_matrices, 'regime_stats': hmm_result['regime_stats']}

# =============================================================================
# SCIENTIFIC VISUALIZATION ENGINE
# =============================================================================

class ScientificVisualizationEngine:
    """Institutional scientific visualization engine"""
    
    def __init__(self):
        self.theme = ScientificThemeManager()
        
    def create_scientific_correlation_matrix(self, corr_data: Dict[str, Any], title: str) -> go.Figure:
        if not corr_data or 'correlation_matrix' not in corr_data: return go.Figure()
        matrix = corr_data['correlation_matrix']
        fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix.round(2).values, texttemplate='%{text}'))
        fig.update_layout(title=title, height=600, template="plotly_white")
        return fig

    def create_regime_chart(self, price_series: pd.Series, regimes: pd.Series, stats: Dict[int, Any]) -> go.Figure:
        """Create institutional regime chart with colored background regions"""
        common_idx = price_series.index.intersection(regimes.index)
        price = price_series.loc[common_idx]
        regime = regimes.loc[common_idx]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=price.index, y=price.values, name="Price", line=dict(color='black', width=1)), row=1, col=1)
        
        colors = ['rgba(46, 125, 50, 0.2)', 'rgba(198, 40, 40, 0.2)', 'rgba(255, 143, 0, 0.2)']
        state_names = ['Low Volatility', 'High Volatility', 'Transition']
        
        df_regime = pd.DataFrame({'state': regime})
        df_regime['group'] = (df_regime['state'] != df_regime['state'].shift()).cumsum()
        
        for _, block in df_regime.groupby('group'):
            start, end = block.index[0], block.index[-1]
            state = block['state'].iloc[0]
            if state < len(colors):
                fig.add_vrect(x0=start, x1=end, fillcolor=colors[state], opacity=1, layer="below", line_width=0, row=1, col=1)
                
        fig.add_trace(go.Scatter(x=regime.index, y=regime.values, mode='lines', name="Regime", line=dict(shape='hv', width=2, color='#1a237e'), fill='tozeroy', fillcolor='rgba(26, 35, 126, 0.1)'), row=2, col=1)
        
        stats_text = []
        for state, metrics in stats.items():
            s_name = state_names[state] if state < len(state_names) else f"State {state}"
            stats_text.append(f"<b>{s_name}</b>: Vol {metrics['volatility']:.1%} | Freq {metrics['frequency']:.1%}")
            
        fig.add_annotation(text="<br>".join(stats_text), xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, bgcolor="rgba(255,255,255,0.9)", bordercolor="gray")
        fig.update_layout(title="Scientific Regime Detection (Gaussian HMM)", height=600, template="plotly_white", yaxis2=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=state_names))
        return fig

    def create_conditional_matrix_comparison(self, conditional_matrices: Dict[int, pd.DataFrame], state_names: List[str]) -> go.Figure:
        n_states = len(conditional_matrices)
        fig = make_subplots(rows=1, cols=n_states, subplot_titles=[f"{state_names[i] if i < len(state_names) else f'State {i}'}" for i in range(n_states)], horizontal_spacing=0.05)
        
        for i, (state, matrix) in enumerate(conditional_matrices.items()):
            if matrix.empty: continue
            fig.add_trace(go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale='RdBu', zmin=-1, zmax=1, showscale=(i==n_states-1), text=matrix.round(2).values, texttemplate='%{text}', textfont={"size": 10}), row=1, col=i+1)
        fig.update_layout(title="Regime-Conditional Correlation Structure", height=500, template="plotly_white")
        return fig

    def create_rolling_correlation_chart(self, rolling_corr: pd.Series, asset1: str, asset2: str, title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, name="Correlation", line=dict(color='#1a237e', width=2)))
        fig.update_layout(title=title, height=400, template="plotly_white", yaxis=dict(range=[-1, 1]))
        return fig
    
    def create_comparison_chart(self, comparison_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        metrics = ['Sharpe_Ratio', 'Annualized_Volatility', 'Maximum_Drawdown']
        colors = ['#1a237e', '#283593', '#3949ab']
        for metric, color in zip(metrics, colors):
            if metric in comparison_df.columns:
                fig.add_trace(go.Bar(x=comparison_df['Asset'], y=comparison_df[metric], name=metric, marker_color=color, opacity=0.7))
        fig.update_layout(title="Risk Metrics Comparison", barmode='group', template="plotly_white", height=500)
        return fig

    def _create_empty_plot(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

# =============================================================================
# MAIN SCIENTIFIC APPLICATION
# =============================================================================

class ScientificCommoditiesPlatform:
    """Main scientific Streamlit application"""
    
    def __init__(self):
        self.data_manager = ScientificDataManager()
        self.visualization = ScientificVisualizationEngine()
        self.config = None
        self.analytics_engine = None
        
        if 'scientific_analysis_results' not in st.session_state: st.session_state.scientific_analysis_results = {}
        if 'selected_scientific_assets' not in st.session_state: st.session_state.selected_scientific_assets = []
        if 'selected_benchmarks' not in st.session_state: st.session_state.selected_benchmarks = []
        if 'correlation_methods' not in st.session_state: st.session_state.correlation_methods = ['pearson', 'ewma']

    def render_scientific_header(self):
        st.markdown("""
        <div class="scientific-header">
            <h1>📈 Institutional Commodities Platform v8.0</h1>
            <p>Scientific Computing Division • Regime Analytics • HMM • Risk Management</p>
            <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                <span class="scientific-badge info">🔬 Scientific Validation</span>
                <span class="scientific-badge warning">🧬 Regime Switching</span>
                <span class="scientific-badge success">⚡ Higham Algo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(sci_dep_manager.display_status(), unsafe_allow_html=True)

    def render_scientific_sidebar(self):
        with st.sidebar:
            st.header("🔬 Configuration")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
            end_date = st.date_input("End Date", datetime.now())
            
            st.subheader("📊 Asset Universe")
            selected_assets = []
            for category, assets in COMMODITIES_UNIVERSE.items():
                with st.expander(f"{category}", expanded=False):
                    for symbol, meta in assets.items():
                        if st.checkbox(f"{meta.name}", value=symbol in ["GC=F", "CL=F"], key=f"sel_{symbol}"):
                            selected_assets.append(symbol)
            st.session_state.selected_scientific_assets = selected_assets
            
            st.subheader("🎯 Benchmarks")
            benchmarks = []
            for symbol, meta in BENCHMARKS.items():
                if st.checkbox(meta['name'], value=symbol=="^GSPC", key=f"bench_{symbol}"):
                    benchmarks.append(symbol)
            st.session_state.selected_benchmarks = benchmarks
            
            st.subheader("⚙️ Parameters")
            corr_methods = st.multiselect("Correlation Methods", ["pearson", "spearman", "ewma"], default=["pearson", "ewma"])
            st.session_state.correlation_methods = corr_methods
            ewma_lambda = st.slider("EWMA Lambda", 0.90, 0.99, 0.94) if "ewma" in corr_methods else 0.94
            
            self.config = ScientificAnalysisConfiguration(start_date=datetime.combine(start_date, datetime.min.time()), end_date=datetime.combine(end_date, datetime.min.time()), confidence_levels=(0.95, 0.99), ewma_lambda=ewma_lambda)
            
            if st.button("🚀 Run Scientific Analysis", type="primary", use_container_width=True):
                st.session_state.run_scientific_analysis = True
                st.rerun()

    def run_scientific_analysis(self):
        with st.spinner("🔄 Executing Institutional Analytics Pipeline..."):
            all_symbols = st.session_state.selected_scientific_assets + st.session_state.selected_benchmarks
            if not all_symbols:
                st.error("No assets selected")
                return

            data_results = self.data_manager.fetch_multiple_assets_scientific(all_symbols, self.config.start_date, self.config.end_date)
            if not data_results:
                st.error("No data fetched. Check date range.")
                return

            features = {s: self.data_manager.calculate_scientific_features(df) for s, df in data_results.items()}
            returns = {s: df['Returns'].dropna() for s, df in features.items() if 'Returns' in df.columns}
            
            self.analytics_engine = ScientificAnalyticsEngine(self.config)
            risk_metrics = {s: self.analytics_engine.calculate_scientific_risk_metrics(r) for s, r in returns.items()}
            
            corr_results = {}
            for method in st.session_state.correlation_methods:
                corr_results[method] = self.analytics_engine.correlation_engine.calculate_correlation_matrix(returns, method=method)
            
            st.session_state.scientific_analysis_results = {'data': data_results, 'returns': returns, 'features': features, 'risk_metrics': risk_metrics, 'correlation_results': corr_results, 'analytics_engine': self.analytics_engine, 'config': self.config, 'timestamp': datetime.now()}
        st.success(f"Analysis Complete! Processed {len(returns)} assets.")

    def render_scientific_dashboard(self):
        if st.session_state.get('run_scientific_analysis'):
            self.run_scientific_analysis()
            st.session_state.run_scientific_analysis = False
            
        results = st.session_state.scientific_analysis_results
        if not results:
            st.info("👈 Configure and Run Analysis from the Sidebar")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Risk Analytics", "🔗 Correlation", "📉 Portfolio", "🧬 Regime Science"])
        
        with tab1: # OVERVIEW
            st.subheader("Asset Performance Overview")
            risk_metrics = results['risk_metrics']
            cols = st.columns(4)
            for i, (sym, data) in enumerate(list(risk_metrics.items())[:4]):
                metrics = data.get('metrics', {})
                with cols[i]:
                    st.metric(sym, f"{metrics.get('Annualized_Return', 0):.1%}", f"Sharpe: {metrics.get('Sharpe_Ratio', 0):.2f}")

        with tab2: # RISK ANALYTICS
            st.subheader("Comprehensive Risk Metrics")
            risk_data = []
            for sym, data in results['risk_metrics'].items():
                m = data.get('metrics', {})
                risk_data.append({'Asset': sym, 'Volatility': m.get('Annualized_Volatility'), 'Max Drawdown': m.get('Maximum_Drawdown'), 'VaR (95%)': m.get('VaR_95_Historical'), 'CVaR (95%)': m.get('CVaR_95'), 'Sortino': m.get('Sortino_Ratio')})
            st.dataframe(pd.DataFrame(risk_data).style.format({'Volatility': '{:.2%}', 'Max Drawdown': '{:.2f}%', 'VaR (95%)': '{:.2f}%', 'CVaR (95%)': '{:.2f}%', 'Sortino': '{:.2f}'}), use_container_width=True)
            if len(risk_data) > 0:
                fig = self.visualization.create_comparison_chart(pd.DataFrame(risk_data))
                st.plotly_chart(fig, use_container_width=True)

        with tab3: # CORRELATION
            st.subheader("Scientific Correlation Analysis")
            method = st.selectbox("Method", list(results['correlation_results'].keys()), key='corr_select')
            corr_data = results['correlation_results'][method]
            fig = self.visualization.create_scientific_correlation_matrix(corr_data, f"{method.upper()} Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Rolling Pairwise Correlation")
            col1, col2 = st.columns(2)
            assets = list(results['returns'].keys())
            if len(assets) >= 2:
                a1 = col1.selectbox("Asset 1", assets, 0)
                a2 = col2.selectbox("Asset 2", assets, 1)
                if a1 != a2:
                    rolling = self.analytics_engine.correlation_engine.calculate_rolling_correlation(results['returns'][a1], results['returns'][a2], window=60)
                    fig_roll = self.visualization.create_rolling_correlation_chart(rolling, a1, a2, f"60-Day Rolling Correlation: {a1} vs {a2}")
                    st.plotly_chart(fig_roll, use_container_width=True)

        with tab4: # PORTFOLIO
            # 
            st.info("Portfolio Optimization Module (Mean-Variance, Risk Parity) ready for v8.1 activation")
            
        with tab5: # REGIME SCIENCE
            st.markdown("""<div class="section-header"><h2>🧬 Regime Switching Analysis (HMM)</h2><div class="section-actions"><span class="scientific-badge high-risk">Gaussian HMM</span><span class="scientific-badge info">Conditional Stats</span></div></div>""", unsafe_allow_html=True)
            if 'analytics_engine' in results:
                col1, col2 = st.columns(2)
                with col1:
                    bench_assets = list(results['returns'].keys())
                    def_idx = 0
                    if "^GSPC" in bench_assets: def_idx = bench_assets.index("^GSPC")
                    elif "CL=F" in bench_assets: def_idx = bench_assets.index("CL=F")
                    hmm_asset = st.selectbox("Regime Indicator Asset", bench_assets, index=def_idx)
                with col2:
                    n_states = st.radio("Number of Regimes", [2, 3], index=0, horizontal=True)
                
                if st.button("🧬 Run HMM Analysis"):
                    with st.spinner(f"Fitting Gaussian HMM to {hmm_asset}..."):
                        hmm_output = results['analytics_engine'].calculate_conditional_correlations(results['returns'], benchmark_symbol=hmm_asset, n_states=n_states)
                    if hmm_output['valid']:
                        st.markdown("#### 🌊 Temporal Regime Map")
                        price_series = results['data'][hmm_asset]['Close']
                        fig_hmm = self.visualization.create_regime_chart(price_series, hmm_output['regimes_used'], hmm_outpu['regime_stats'])
                        st.plotly_chart(fig_hmm, use_container_width=True)
                        
                        st.markdown("#### 🧩 Regime-Conditional Correlations")
                        st.info("Correlations often converge to 1.0 (red) during High Volatility regimes.")
                        state_labels = ["🟢 Low Volatility", "🔴 High Volatility", "🟠 Transition"]
                        fig_cond = self.visualization.create_conditional_matrix_comparison(hmm_output['conditional_matrices'], state_names=state_labels)
                        st.plotly_chart(fig_cond, use_container_width=True)
                        
                        st.markdown("#### 📊 Regime Statistics")
                        stats_list = []
                        for s, stats in hmm_output['regime_stats'].items():
                            stats_list.append({"Regime": f"State {s}", "Frequency": f"{stats['frequency']:.1%}", "Volatility": f"{stats['volatility']:.1%}", "Sharpe": f"{stats['sharpe']:.2f}"})
                        st.table(pd.DataFrame(stats_list))
                    else:
                        st.error(f"HMM Analysis Failed: {hmm_output.get('error')}")

    def run(self):
        try:
            self.render_scientific_header()
            self.render_scientific_sidebar()
            self.render_scientific_dashboard()
            self.render_scientific_footer()
        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.code(traceback.format_exc())

    def render_scientific_footer(self):
        st.markdown("---")
        st.markdown("""<div style="text-align: center; color: #415a77; font-size: 0.8rem;">Institutional Commodities Analytics Platform v8.0 • Scientific Computing Division</div>""", unsafe_allow_html=True)

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    app = ScientificCommoditiesPlatform()
    app.run()
