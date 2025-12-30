"""
🏛️ Institutional Commodities Analytics Platform v7.3.7 (Refactored)
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • Machine Learning • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance

PATCH NOTES v7.3.8:
- FIXED: VaR/CVaR calculations now use robust data cleaning to prevent NaNs.
- FIXED: Horizon scaling now correctly handles multi-day lookaheads using log-aggregation where data permits.
- FIXED: Added safeguards against zero-volatility and insufficient data scenarios.
- UPDATED: Risk Analytics display logic to gracefully handle calculation failures.
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
# EXCEL EXPORT (CLOUD-SAFE ENGINE FALLBACK)
# =============================================================================
def icd_safe_excel_writer(buffer_obj):
    """
    Create a pandas ExcelWriter with a robust engine fallback.
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
try:
    st.set_page_config(
        page_title="Institutional Commodities Platform v7.3",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/institutional-commodities',
            'Report a bug': "https://github.com/institutional-commodities/issues",
            'About': """🏛️ Institutional Commodities Analytics v7.3"""
        }
    )
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

# Enhanced commodities universe
COMMODITIES_UNIVERSE = {
    AssetCategory.PRECIOUS_METALS.value: {
        "GC=F": AssetMetadata("GC=F", "Gold Futures", AssetCategory.PRECIOUS_METALS, "#FFD700", "COMEX Gold", "COMEX"),
        "SI=F": AssetMetadata("SI=F", "Silver Futures", AssetCategory.PRECIOUS_METALS, "#C0C0C0", "COMEX Silver", "COMEX"),
        "PL=F": AssetMetadata("PL=F", "Platinum Futures", AssetCategory.PRECIOUS_METALS, "#E5E4E2", "NYMEX Platinum", "NYMEX"),
    },
    AssetCategory.INDUSTRIAL_METALS.value: {
        "HG=F": AssetMetadata("HG=F", "Copper Futures", AssetCategory.INDUSTRIAL_METALS, "#B87333", "COMEX Copper", "COMEX"),
        "ALI=F": AssetMetadata("ALI=F", "Aluminum Futures", AssetCategory.INDUSTRIAL_METALS, "#848482", "COMEX Aluminum", "COMEX"),
    },
    AssetCategory.ENERGY.value: {
        "CL=F": AssetMetadata("CL=F", "Crude Oil WTI", AssetCategory.ENERGY, "#000000", "NYMEX Crude", "NYMEX"),
        "NG=F": AssetMetadata("NG=F", "Natural Gas", AssetCategory.ENERGY, "#4169E1", "NYMEX Nat Gas", "NYMEX"),
    },
    AssetCategory.AGRICULTURE.value: {
        "ZC=F": AssetMetadata("ZC=F", "Corn Futures", AssetCategory.AGRICULTURE, "#FFD700", "CBOT Corn", "CBOT"),
        "ZW=F": AssetMetadata("ZW=F", "Wheat Futures", AssetCategory.AGRICULTURE, "#F5DEB3", "CBOT Wheat", "CBOT"),
    }
}

BENCHMARKS = {
    "^GSPC": {"name": "S&P 500 Index", "type": "equity", "color": "#1E90FF"},
    "DX-Y.NYB": {"name": "US Dollar Index", "type": "currency", "color": "#32CD32"},
    "TLT": {"name": "20+ Year Treasury ETF", "type": "fixed_income", "color": "#8A2BE2"},
    "GLD": {"name": "SPDR Gold Shares", "type": "commodity", "color": "#FFD700"},
    "DBC": {"name": "Invesco DB Commodity Index", "type": "commodity", "color": "#FF6347"}
}

# =============================================================================
# STYLES
# =============================================================================

class ThemeManager:
    """Manage application theming and styling"""
    THEMES = {
        "default": {
            "primary": "#1a2980", "secondary": "#26d0ce", "accent": "#7c3aed",
            "success": "#10b981", "warning": "#f59e0b", "danger": "#ef4444",
            "dark": "#1f2937", "light": "#f3f4f6", "gray": "#6b7280", "background": "#ffffff"
        }
    }
    
    @staticmethod
    def get_styles(theme: str = "default") -> str:
        colors = ThemeManager.THEMES.get(theme, ThemeManager.THEMES["default"])
        return f"""
        <style>
            :root {{
                --primary: {colors['primary']}; --secondary: {colors['secondary']};
                --accent: {colors['accent']}; --success: {colors['success']};
                --warning: {colors['warning']}; --danger: {colors['danger']};
                --dark: {colors['dark']}; --light: {colors['light']};
            }}
            .metric-card {{
                background: var(--background); padding: 1.5rem; border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid var(--primary);
                margin-bottom: 1rem;
            }}
            .metric-value {{
                font-size: 2rem; font-weight: 800; color: var(--dark);
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }}
            .metric-label {{ font-size: 0.85rem; color: var(--gray); text-transform: uppercase; font-weight: 600; }}
        </style>
        """

st.markdown(ThemeManager.get_styles("default"), unsafe_allow_html=True)

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

class DependencyManager:
    def __init__(self):
        self.dependencies = {}
        self._load_dependencies()
    
    def _load_dependencies(self):
        try:
            import statsmodels.api as sm
            self.dependencies['statsmodels'] = {'available': True, 'module': sm}
        except ImportError:
            self.dependencies['statsmodels'] = {'available': False}
        
        try:
            from arch import arch_model
            self.dependencies['arch'] = {'available': True, 'arch_model': arch_model}
        except ImportError:
            self.dependencies['arch'] = {'available': False}
        
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
            self.dependencies['hmmlearn'] = {'available': True, 'GaussianHMM': GaussianHMM, 'StandardScaler': StandardScaler}
        except ImportError:
            self.dependencies['hmmlearn'] = {'available': False}

    def is_available(self, dependency: str) -> bool:
        return self.dependencies.get(dependency, {}).get('available', False)

dep_manager = DependencyManager()

# =============================================================================
# CACHING & DATA MANAGER
# =============================================================================

class SmartCache:
    @staticmethod
    def cache_data(ttl: int = 3600, max_entries: int = 50):
        def decorator(func):
            @wraps(func)
            @st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=False)
            def wrapper(_arg0, *args, **kwargs):
                return func(_arg0, *args, **kwargs)
            return wrapper
        return decorator

class EnhancedDataManager:
    def __init__(self):
        pass
    
    @SmartCache.cache_data(ttl=3600)
    def fetch_multiple_assets(self, symbols: List[str], start_date: datetime, end_date: datetime, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not df.empty:
                    # Flat column rename
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    # Ensure close
                    if 'Close' not in df.columns and 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    if 'Close' in df.columns:
                         results[symbol] = df
            except Exception:
                pass
        return results
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Close' not in df.columns: return df
        df['Returns'] = df['Close'].pct_change()
        # Simple Vol
        df['Volatility_20D'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

# =============================================================================
# ANALYTICS ENGINE (PATCHED)
# =============================================================================

class InstitutionalAnalytics:
    """Institutional-grade analytics engine with advanced methods"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.annual_trading_days = 252

    def _clean_returns_series(self, returns: pd.Series) -> pd.Series:
        """Robust cleaning of returns series."""
        if returns is None or len(returns) == 0:
            return pd.Series(dtype=float)
        
        try:
            # Convert to numeric
            rr = pd.to_numeric(returns, errors='coerce')
            # Remove infinities
            rr = rr.replace([np.inf, -np.inf], np.nan)
            # Drop NaN values
            rr = rr.dropna()
            return rr
        except Exception:
            return pd.Series(dtype=float)

    def calculate_performance_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Basic performance metrics."""
        returns = self._clean_returns_series(returns)
        if len(returns) < 20: return {}
        
        total_ret = (1 + returns).prod() - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - self.risk_free_rate) / vol if vol > 0 else 0
        
        return {
            'annual_return': returns.mean() * 252,
            'annual_volatility': vol,
            'sharpe_ratio': sharpe
        }

    # -------------------------------------------------------------------------
    # FIXED VaR CALCULATION
    # -------------------------------------------------------------------------
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
        horizon: int = 1,
        use_log_aggregation: bool = True
    ) -> Dict[str, Any]:
        """
        Robust VaR / CVaR(ES) / ES engine (NaN-proof, horizon-aware).
        """
        # 1. Robust data cleaning
        try:
            rr = self._clean_returns_series(returns)
            
            # Check minimum data requirements
            if len(rr) < 10:  # Minimum reasonable sample
                return {
                    "success": False,
                    "message": f"Insufficient data points ({len(rr)} < 10)",
                    "n_obs": len(rr),
                    "horizon": horizon
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Data cleaning failed: {str(e)[:100]}",
                "n_obs": 0,
                "horizon": horizon
            }
        
        # 2. Horizon aggregation with robust handling
        try:
            h = int(max(1, horizon))
            
            if h == 1:
                # Single day - use returns as is
                rr_h = rr.copy()
            elif h > 1 and len(rr) >= h:
                if use_log_aggregation:
                    # Log returns aggregation (more stable)
                    log_returns = np.log1p(rr)  # log(1 + r)
                    aggregated_log_returns = log_returns.rolling(window=h).sum()
                    rr_h = np.expm1(aggregated_log_returns).dropna()  # exp(sum) - 1
                else:
                    # Simple sum aggregation
                    rr_h = rr.rolling(window=h).sum().dropna()
            else:
                # Not enough data for horizon, use sqrt scaling on results later or approx here
                # We will use sqrt scaling on the series for approximation if rolling not possible
                if h > 1:
                    rr_h = rr * np.sqrt(h)
                else:
                    rr_h = rr
                    
            # Final cleaning after aggregation
            rr_h = rr_h.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(rr_h) < 5:
                return {
                    "success": False,
                    "message": f"Insufficient data after horizon aggregation ({len(rr_h)} points)",
                    "n_obs": len(rr_h),
                    "horizon": h
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Horizon aggregation failed: {str(e)[:100]}",
                "n_obs": len(rr),
                "horizon": horizon
            }
        
        # 3. Calculate statistics
        n = len(rr_h)
        mu = float(rr_h.mean())
        sigma = float(rr_h.std(ddof=1)) if n > 1 else 0.0
        
        # Clamp confidence level
        cl = float(max(0.5, min(0.999, confidence_level)))
        alpha = 1.0 - cl
        
        # 4. VaR calculation based on method
        var_value = 0.0
        cvar_value = 0.0
        
        try:
            if method.lower() == "historical":
                # Historical VaR (non-parametric)
                if n >= 10:
                    quantile = float(np.nanpercentile(rr_h, alpha * 100))
                    var_value = max(0.0, -quantile)  # VaR is positive loss
                    
                    # Calculate CVaR/ES
                    tail_returns = rr_h[rr_h <= quantile]
                    if len(tail_returns) > 0:
                        cvar_value = max(0.0, -float(tail_returns.mean()))
                    else:
                        cvar_value = var_value
                else:
                    return {
                        "success": False,
                        "message": f"Insufficient data for historical VaR (n={n})",
                        "n_obs": n,
                        "horizon": h
                    }
                    
            elif method.lower() == "parametric":
                # Parametric (Normal) VaR
                if sigma > 1e-12:  # Avoid division by zero
                    z_score = float(stats.norm.ppf(alpha))
                    var_value = max(0.0, -(mu + z_score * sigma))
                    
                    # Calculate CVaR/ES for normal distribution
                    if alpha > 1e-12:
                        pdf_z = float(stats.norm.pdf(z_score))
                        cvar_value = max(0.0, -mu + sigma * (pdf_z / alpha))
                    else:
                        cvar_value = var_value
                else:
                    # Zero volatility case
                    var_value = max(0.0, -mu)
                    cvar_value = var_value
                    
            elif method.lower() == "modified":
                # Cornish-Fisher VaR (accounts for skewness and kurtosis)
                if sigma > 1e-12 and n >= 30:
                    z = float(stats.norm.ppf(alpha))
                    
                    try:
                        skew = float(rr_h.skew())
                        kurt_excess = float(rr_h.kurtosis())
                    except:
                        skew = 0.0
                        kurt_excess = 0.0
                    
                    # Cornish-Fisher expansion
                    z_cf = (z + 
                           (1/6) * (z**2 - 1) * skew +
                           (1/24) * (z**3 - 3*z) * kurt_excess -
                           (1/36) * (2*z**3 - 5*z) * skew**2)
                    
                    var_value = max(0.0, -(mu + z_cf * sigma))
                    
                    # For modified VaR, use historical CVaR on the tail defined by CF quantile
                    quantile_cf = mu + z_cf * sigma
                    tail_returns = rr_h[rr_h <= quantile_cf]
                    if len(tail_returns) > 0:
                        cvar_value = max(0.0, -float(tail_returns.mean()))
                    else:
                        cvar_value = var_value
                else:
                    # Fall back to parametric if insufficient data
                    return self.calculate_var(
                        returns, confidence_level, "parametric", horizon, use_log_aggregation
                    )
            else:
                # Default to historical
                return self.calculate_var(
                    returns, confidence_level, "historical", horizon, use_log_aggregation
                )
                
        except Exception as e:
            return {
                "success": False,
                "message": f"VaR calculation error: {str(e)[:100]}",
                "n_obs": n,
                "horizon": h
            }
        
        # 5. Sanity checks and final formatting
        if not np.isfinite(var_value):
            var_value = 0.0
        if not np.isfinite(cvar_value):
            cvar_value = var_value
        
        # Convert to percentages for display
        var_pct = var_value * 100
        cvar_pct = cvar_value * 100
        
        return {
            "success": True,
            "VaR": var_value,
            "CVaR": cvar_value,
            "ES": cvar_value,  # ES is same as CVaR
            "VaR_pct": var_pct,
            "CVaR_pct": cvar_pct,
            "confidence_level": cl,
            "method": method.lower(),
            "n_obs": n,
            "horizon": h,
            "mu": mu,
            "sigma": sigma,
            "warning": f"Based on {n} observations" if n < 100 else ""
        }

    # -------------------------------------------------------------------------
    # OTHER ANALYTICS METHODS
    # -------------------------------------------------------------------------
    def garch_analysis(self, returns: pd.Series, p=1, q=1, **kwargs) -> Dict[str, Any]:
        if not dep_manager.is_available('arch'):
            return {'success': False, 'message': 'Arch package not available'}
        try:
            returns = self._clean_returns_series(returns)
            am = dep_manager.dependencies['arch']['arch_model'](returns * 100, p=p, q=q, rescale=False)
            res = am.fit(disp='off', show_warning=False)
            return {
                'success': True,
                'conditional_volatility': res.conditional_volatility / 100 * np.sqrt(252),
                'aic': res.aic,
                'bic': res.bic
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def detect_regimes(self, returns: pd.Series, n_states: int = 3) -> Dict[str, Any]:
        if not dep_manager.is_available('hmmlearn'):
            return {'success': False, 'message': 'HMM not available'}
        try:
            returns = self._clean_returns_series(returns)
            X = returns.values.reshape(-1, 1)
            model = dep_manager.dependencies['hmmlearn']['GaussianHMM'](n_components=n_states, covariance_type="full", n_iter=100)
            model.fit(X)
            states = model.predict(X)
            return {'success': True, 'states': pd.Series(states, index=returns.index)}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def compute_ewma_volatility_ratio(self, returns: pd.Series, span_fast=22, span_mid=33, span_slow=99, annualize=False):
        returns = self._clean_returns_series(returns)
        if returns.empty: return pd.DataFrame()
        
        v_fast = returns.ewm(span=span_fast).std()
        v_mid = returns.ewm(span=span_mid).std()
        v_slow = returns.ewm(span=span_slow).std()
        
        ratio = v_fast / (v_mid + v_slow)
        df = pd.DataFrame({'EWMA_RATIO': ratio})
        return df

    def optimize_portfolio(self, returns_df: pd.DataFrame, method: str = 'sharpe', target_return=None):
        if returns_df.empty: return {'success': False, 'message': 'Empty data'}
        n_assets = len(returns_df.columns)
        mean_ret = returns_df.mean().values * 252
        cov_mat = returns_df.cov().values * 252
        
        def fun(w):
            p_ret = np.sum(mean_ret * w)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            if method == 'sharpe': return -(p_ret - self.risk_free_rate) / p_vol
            if method == 'min_var': return p_vol
            if method == 'max_ret': return -p_ret
            return 0
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(n_assets))
        init_guess = [1./n_assets] * n_assets
        
        try:
            res = optimize.minimize(fun, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
            if res.success:
                return {'success': True, 'weights': dict(zip(returns_df.columns, res.x))}
            else:
                return {'success': False, 'message': res.message}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def stress_test(self, returns: pd.Series, shock: float = -0.10, duration: int = 5):
        returns = self._clean_returns_series(returns)
        if returns.empty: return {'success': False}
        
        # Create shock path
        shock_daily = shock / duration
        path = (1 + returns).cumprod()
        sim_path = path.copy()
        
        # Apply shock to last N days conceptually or future
        # Here we just return a simulated path starting from 1
        future_rets = [shock_daily] * duration
        future_path = [1.0]
        for r in future_rets:
            future_path.append(future_path[-1] * (1 + r))
            
        return {'success': True, 'path': pd.Series(future_path)}

    def rolling_beta(self, returns: pd.Series, benchmark: pd.Series, window: int = 60):
        returns = self._clean_returns_series(returns)
        benchmark = self._clean_returns_series(benchmark)
        
        df = pd.DataFrame({'a': returns, 'b': benchmark}).dropna()
        cov = df['a'].rolling(window).cov(df['b'])
        var = df['b'].rolling(window).var()
        return cov / var

    def rolling_tracking_error(self, returns: pd.Series, benchmark: pd.Series, window: int = 60):
        returns = self._clean_returns_series(returns)
        benchmark = self._clean_returns_series(benchmark)
        diff = returns - benchmark
        return diff.rolling(window).std() * np.sqrt(252)

# =============================================================================
# VISUALIZER
# =============================================================================

class InstitutionalVisualizer:
    def __init__(self):
        self.template = "plotly_white"
        
    def create_performance_chart(self, df: pd.DataFrame, title: str):
        fig = go.Figure()
        for col in df.columns:
            cum_ret = (1 + df[col]).cumprod()
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, name=col))
        fig.update_layout(title=title, template=self.template, height=500)
        return fig
        
    def create_garch_volatility(self, returns, conditional_vol, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=returns.index, y=conditional_vol, name="GARCH Vol"))
        fig.update_layout(title=title, template=self.template)
        return fig
        
    def create_regime_chart(self, states, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=states.index, y=states.values, mode='markers', name="Regime"))
        fig.update_layout(title=title, template=self.template)
        return fig
    
    def create_correlation_matrix(self, corr_df, title):
        fig = px.imshow(corr_df, title=title, text_auto=True, aspect="auto", color_continuous_scale='RdBu', zmin=-1, zmax=1)
        return fig
        
    def create_ewma_ratio_signal_chart(self, df, title, bb_window, bb_k, green_max, red_min, show_bollinger, show_threshold_lines):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['EWMA_RATIO'], name="Ratio"))
        fig.add_hline(y=green_max, line_color="green", line_dash="dash")
        fig.add_hline(y=red_min, line_color="red", line_dash="dash")
        fig.update_layout(title=title, template=self.template)
        return fig
        
    def create_rolling_beta_chart(self, beta, symbol, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=beta.index, y=beta.values, name="Beta"))
        fig.update_layout(title=title, template=self.template)
        return fig
        
    def create_tracking_error_chart(self, te, symbol, green, orange, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=te.index, y=te.values, name="TE"))
        fig.update_layout(title=title, template=self.template)
        return fig
        
    def create_relative_risk_chart(self, df, symbol, green, orange, title):
        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
        fig.update_layout(title=title, template=self.template)
        return fig

# =============================================================================
# SCIENTIFIC ANALYTICS (v7.2 Module)
# =============================================================================

class ScientificAnalyticsEngine:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.risk_free_rate = 0.02
        self.annual_trading_days = 252

    def _clean_returns_series(self, returns: pd.Series) -> pd.Series:
        """Robust cleaning of returns series."""
        if returns is None or len(returns) == 0:
            return pd.Series(dtype=float)
        
        try:
            rr = pd.to_numeric(returns, errors='coerce')
            rr = rr.replace([np.inf, -np.inf], np.nan)
            rr = rr.dropna()
            return rr
        except Exception:
            return pd.Series(dtype=float)

    def _historical_var_cvar(self, r: pd.Series, confidence: float, horizon: int) -> Tuple[float, float]:
        """Robust historical VaR and CVaR calculation."""
        
        # 1. Robust data cleaning
        try:
            rr = self._clean_returns_series(r)
            if len(rr) < 20: return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
        
        # 2. Horizon scaling
        h = max(1, int(horizon))
        
        if h == 1:
            scaled_returns = rr.copy()
        elif h > 1:
            # Use rolling window for actual horizon returns if enough data
            try:
                if len(rr) > h * 2:
                    # Calculate h-period returns
                    if h <= 10:  # For short horizons, use product
                        h_period_returns = (1 + rr).rolling(window=h).apply(
                            lambda x: x.prod() - 1, raw=False
                        ).dropna()
                    else:  # For longer horizons, use sqrt approximation
                        h_period_returns = rr * np.sqrt(h)
                    scaled_returns = h_period_returns
                else:
                    scaled_returns = rr * np.sqrt(h)
            except:
                scaled_returns = rr * np.sqrt(h)
        else:
            scaled_returns = rr * np.sqrt(h)
        
        # 3. Calculate VaR and CVaR
        try:
            if len(scaled_returns) < 10: return 0.0, 0.0
            
            alpha = 1.0 - max(0.5, min(0.999, confidence))
            quantile = float(np.nanpercentile(scaled_returns, alpha * 100))
            
            # VaR is positive loss
            var = max(0.0, -quantile)
            
            # Calculate CVaR/ES
            tail_returns = scaled_returns[scaled_returns <= quantile]
            if len(tail_returns) > 0:
                cvar = max(0.0, -float(tail_returns.mean()))
            else:
                cvar = var
                
            return var, cvar
            
        except Exception:
            return 0.0, 0.0

    def _parametric_var_cvar(self, r: pd.Series, confidence: float, horizon: int, use_t: bool = True) -> Tuple[float, float]:
        """Robust parametric VaR and CVaR calculation."""
        
        # 1. Robust data cleaning
        try:
            rr = self._clean_returns_series(r)
            if len(rr) < 20: return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
        
        # 2. Calculate statistics
        n = len(rr)
        mu = float(rr.mean()) if n > 0 else 0.0
        sigma = float(rr.std(ddof=1)) if n > 1 else 0.0
        
        if sigma < 1e-12:
            var = max(0.0, -mu)
            return var, var
        
        # 3. Horizon scaling
        h = max(1, int(horizon))
        mu_h = mu * h
        sigma_h = sigma * np.sqrt(h)
        
        # 4. Calculate VaR and CVaR
        alpha = 1.0 - max(0.5, min(0.999, confidence))
        
        try:
            if use_t and n >= 30:
                # Student's t
                try:
                    df, loc, scale = stats.t.fit(rr.values)
                    df = max(2.0, float(df))
                    
                    t_quantile = float(stats.t.ppf(alpha, df))
                    var = max(0.0, -(loc + scale * t_quantile))
                    
                    if df > 2 and alpha > 1e-12:
                        cvar_factor = (stats.t.pdf(t_quantile, df) / alpha) * \
                                     ((df + t_quantile**2) / (df - 1))
                        cvar = max(0.0, -(loc + scale * cvar_factor))
                    else:
                        cvar = var
                except:
                    # Fallback Normal
                    z = float(stats.norm.ppf(alpha))
                    var = max(0.0, -(mu_h + z * sigma_h))
                    if alpha > 1e-12:
                        pdf_z = float(stats.norm.pdf(z))
                        cvar = max(0.0, -mu_h + sigma_h * (pdf_z / alpha))
                    else:
                        cvar = var
            else:
                # Normal
                z = float(stats.norm.ppf(alpha))
                var = max(0.0, -(mu_h + z * sigma_h))
                if alpha > 1e-12:
                    pdf_z = float(stats.norm.pdf(z))
                    cvar = max(0.0, -mu_h + sigma_h * (pdf_z / alpha))
                else:
                    cvar = var
                    
            return var, cvar
            
        except Exception:
            return 0.0, 0.0
    
    # Placeholder for other methods referenced in the merged file
    def rolling_beta(self, returns, benchmark, window): return pd.Series()
    def rolling_tracking_error(self, returns, benchmark, window): return pd.Series()
    def calculate_scientific_risk_metrics(self, returns, benchmark_returns): return {}

# =============================================================================
# DASHBOARD LOGIC (FALLBACKS & PATCHES)
# =============================================================================

def _icd__get_returns_df(self):
    return pd.DataFrame(st.session_state.get('returns_data', {}))

def _icd__get_benchmark_df(self):
    return pd.DataFrame(st.session_state.get('benchmark_data', {}))

def _icd__equal_weight_portfolio(df, cols):
    return df[cols].mean(axis=1)

def _icd_display_risk_analytics_fallback(self, cfg):
    """Refactored display method with robust NaN handling."""
    st.markdown("### 🧮 Risk Analytics (Institutional)")

    returns_df = _icd__get_returns_df(self)
    if returns_df.empty:
        st.info("Load data from the sidebar to begin.")
        return

    scope = st.radio("Scope", ["Equal-Weight Portfolio", "Single Asset"], horizontal=True, key="risk_scope")

    if scope.startswith("Equal"):
        assets = list(returns_df.columns)
        default_assets = assets[: min(6, len(assets))]
        sel = st.multiselect("Assets", assets, default=default_assets, key="risk_assets")
        series = _icd__equal_weight_portfolio(returns_df, sel)
    else:
        sym = st.selectbox("Select Asset", options=list(returns_df.columns), key="risk_asset")
        series = pd.to_numeric(returns_df[sym], errors="coerce").dropna()

    if series.empty or len(series) < 60:
        st.warning("Insufficient data for VaR (need ~60+ observations).")
        return

    c1, c2, c3 = st.columns(3)
    with c1: cl = st.select_slider("Confidence", [0.90, 0.95, 0.99], value=0.95, key="risk_cl")
    with c2: method = st.selectbox("Method", ["historical", "parametric", "modified"], key="risk_var_method")
    with c3: horizon = st.select_slider("Horizon (days)", [1, 5, 10, 20], value=1, key="risk_horizon")

    try:
        # Use the robust calculate_var from InstitutionalAnalytics
        out = self.analytics.calculate_var(series, confidence_level=float(cl), method=method, horizon=int(horizon))
    except Exception as e:
        out = {"success": False, "message": str(e)}

    if out and out.get("success", False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_pct = out.get("VaR_pct", 0.0)
            if np.isfinite(var_pct):
                st.metric(f"VaR {int(cl*100)}%", f"{var_pct:.2f}%")
            else:
                st.metric(f"VaR {int(cl*100)}%", "N/A")
                
        with col2:
            cvar_pct = out.get("CVaR_pct", 0.0)
            if np.isfinite(cvar_pct):
                st.metric(f"CVaR/ES {int(cl*100)}%", f"{cvar_pct:.2f}%")
            else:
                st.metric(f"CVaR/ES {int(cl*100)}%", "N/A")
                
        with col3:
            st.metric("Method", method)
            st.caption(f"{out.get('n_obs')} obs")
            
        st.json({k: v for k, v in out.items() if k not in ("returns", "VaR", "CVaR", "ES", "mu", "sigma")}, expanded=False)
    else:
        st.warning(out.get("message", "VaR engine returned no result."))


# =============================================================================
# MAIN DASHBOARD CLASS
# =============================================================================

class InstitutionalCommoditiesDashboard:
    def __init__(self):
        self.data_manager = EnhancedDataManager()
        self.analytics = InstitutionalAnalytics()
        self.visualizer = InstitutionalVisualizer()
        
        # Patch display methods
        self._display_risk_analytics = _icd_display_risk_analytics_fallback.__get__(self, InstitutionalCommoditiesDashboard)

    def _render_sidebar_controls(self):
        st.sidebar.header("Controls")
        assets = st.sidebar.multiselect("Assets", ["GC=F", "CL=F", "SI=F", "HG=F", "^GSPC"], default=["GC=F", "CL=F"])
        start = st.sidebar.date_input("Start", datetime.now() - timedelta(days=365*2))
        end = st.sidebar.date_input("End", datetime.now())
        if st.sidebar.button("Load Data"):
            data = self.data_manager.fetch_multiple_assets(assets, start, end)
            returns = pd.DataFrame({k: v['Close'].pct_change() for k, v in data.items()})
            st.session_state['returns_data'] = returns
            st.session_state['data_loaded'] = True

    def run(self):
        self._render_sidebar_controls()
        
        if not st.session_state.get('data_loaded'):
            st.info("Please load data.")
            return
            
        tabs = st.tabs(["Risk Analytics", "Portfolio", "Settings"])
        
        with tabs[0]:
            self._display_risk_analytics(None)
        with tabs[1]:
            st.write("Portfolio Placeholder")
        with tabs[2]:
            st.write("Settings Placeholder")

# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    app = InstitutionalCommoditiesDashboard()
    app.run()
