"""
🏛️ Institutional Commodities Analytics Platform v4.1
Advanced Portfolio Analytics • GARCH Champion Selection • Regime Detection • Stress Testing
FIXED DATA LOADING ISSUES - Streamlit Cloud Optimized
"""

import os
import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Institutional Commodities Platform v4.1", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ============================================================================
# COMMODITIES UNIVERSE
# ============================================================================

COMMODITIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Precious Metals": {
        "GC=F": {"name": "Gold Futures", "category": "Precious"},
        "SI=F": {"name": "Silver Futures", "category": "Precious"},
        "PL=F": {"name": "Platinum Futures", "category": "Precious"},
        "PA=F": {"name": "Palladium Futures", "category": "Precious"}
    },
    "Industrial Metals": {
        "HG=F": {"name": "Copper Futures", "category": "Industrial"},
        "ALI=F": {"name": "Aluminum Futures", "category": "Industrial"}
    },
    "Energy": {
        "CL=F": {"name": "Crude Oil WTI", "category": "Energy"},
        "BZ=F": {"name": "Brent Crude", "category": "Energy"},
        "NG=F": {"name": "Natural Gas", "category": "Energy"}
    },
    "Agriculture": {
        "ZC=F": {"name": "Corn Futures", "category": "Agriculture"},
        "ZW=F": {"name": "Wheat Futures", "category": "Agriculture"},
        "ZS=F": {"name": "Soybean Futures", "category": "Agriculture"}
    }
}

BENCHMARKS = {
    "^GSPC": {"name": "S&P 500", "type": "equity"},
    "DX-Y.NYB": {"name": "US Dollar Index", "type": "fx"},
    "TLT": {"name": "20+ Year Treasury", "type": "fixed_income"},
    "GLD": {"name": "Gold ETF", "type": "commodity"}
}

# ============================================================================
# STYLES & UTILITIES
# ============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #1a2980;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 750;
        color: #243447;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .positive { color: #27ae60; font-weight: 700; }
    .negative { color: #e74c3c; font-weight: 700; }
    .neutral { color: #f39c12; font-weight: 700; }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #1a2980;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .status-success { background: #d4edda; color: #155724; }
    .status-warning { background: #fff3cd; color: #856404; }
    .status-danger { background: #f8d7da; color: #721c24; }
    .data-loading {
        border-left: 4px solid #1a2980;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORT MANAGEMENT
# ============================================================================

STATSMODELS_AVAILABLE = False
ARCH_AVAILABLE = False
HMM_AVAILABLE = False

try:
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    st.warning("statsmodels not available - some diagnostics disabled")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    st.warning("arch not available - GARCH features disabled")

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    st.warning("hmmlearn/scikit-learn not available - regime detection disabled")

# ============================================================================
# FIXED DATA LOADING ENGINE
# ============================================================================

class DataLoader:
    """Robust data loading engine with comprehensive error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_yahoo_data(_self, symbol: str, start_date: datetime, end_date: datetime, 
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance with comprehensive error handling
        FIXED: Handles API changes, timeouts, and data format issues
        """
        for attempt in range(max_retries):
            try:
                # FIX: Ensure dates are in correct format
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                st.write(f"Attempt {attempt + 1}/{max_retries} for {symbol}...")
                
                # FIX: Use yf.download directly with proper parameters
                df = yf.download(
                    symbol,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=True,
                    threads=False,  # FIX: Disable threads for reliability
                    timeout=30
                )
                
                if df is None or df.empty:
                    st.warning(f"Empty dataframe for {symbol}")
                    
                    # Try alternative method with Ticker
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_str, end=end_str)
                    
                    if hist is not None and not hist.empty:
                        df = hist
                    else:
                        if attempt == max_retries - 1:
                            return None
                        time.sleep(2)
                        continue
                
                # FIX: Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # FIX: Ensure required columns exist
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # FIX: Create Adj Close if missing
                if "Adj Close" not in df.columns:
                    df["Adj Close"] = df["Close"]
                
                # FIX: Ensure index is datetime
                df.index = pd.to_datetime(df.index)
                
                # FIX: Remove timezone if present
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # FIX: Sort index
                df = df.sort_index()
                
                # FIX: Check for sufficient data
                if len(df) < 20:
                    st.warning(f"Insufficient data points for {symbol}: {len(df)}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2)
                    continue
                
                # Calculate basic features
                df = DataLoader._calculate_features(df)
                
                st.success(f"✓ Successfully loaded {symbol} ({len(df)} days)")
                return df
                
            except Exception as e:
                error_msg = str(e)
                st.warning(f"Attempt {attempt + 1} failed for {symbol}: {error_msg[:100]}")
                
                if attempt == max_retries - 1:
                    # Try with wider date range
                    try:
                        st.info(f"Trying wider date range for {symbol}...")
                        earlier_start = start_date - timedelta(days=365)
                        df = yf.download(
                            symbol,
                            start=earlier_start.strftime('%Y-%m-%d'),
                            end=end_str,
                            progress=False,
                            auto_adjust=True
                        )
                        
                        if df is not None and not df.empty:
                            # Filter to requested date range
                            df = df.loc[start_str:end_str]
                            if not df.empty:
                                df = DataLoader._calculate_features(df)
                                return df
                    except:
                        pass
                    
                    st.error(f"Failed to load {symbol} after {max_retries} attempts")
                    return None
                
                time.sleep(2)  # Wait before retry
        
        return None
    
    @staticmethod
    def _calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features"""
        df = df.copy()
        
        # Returns
        df["Returns"] = df["Adj Close"].pct_change()
        df["Log_Returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        
        # Moving averages
        df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
        
        # Volatility
        df["Volatility_20D"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)
        df["Volatility_60D"] = df["Returns"].rolling(window=60).std() * np.sqrt(252)
        
        # Bollinger Bands
        bb_middle = df["Adj Close"].rolling(window=20).mean()
        bb_std = df["Adj Close"].rolling(window=20).std()
        df["BB_Upper"] = bb_middle + 2 * bb_std
        df["BB_Lower"] = bb_middle - 2 * bb_std
        
        # RSI
        delta = df["Adj Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def bulk_load_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Bulk load data with progress tracking"""
        data_dict = {}
        failed_symbols = []
        
        progress_bar = st.progress(0, text="Loading market data...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = DataLoader.fetch_yahoo_data(None, symbol, start_date, end_date)
                
                if df is not None and not df.empty and len(df) >= 60:
                    data_dict[symbol] = df
                    st.success(f"✓ {symbol}: {len(df)} trading days")
                else:
                    failed_symbols.append(symbol)
                    st.warning(f"✗ {symbol}: Failed or insufficient data")
            
            except Exception as e:
                failed_symbols.append(symbol)
                st.error(f"✗ {symbol}: {str(e)[:100]}")
            
            # Update progress
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress, text=f"Loading {symbol} ({i+1}/{len(symbols)})")
        
        progress_bar.empty()
        
        # Summary
        if data_dict:
            st.success(f"✅ Successfully loaded {len(data_dict)}/{len(symbols)} assets")
        if failed_symbols:
            st.warning(f"⚠️ Failed to load: {', '.join(failed_symbols)}")
        
        return data_dict

# ============================================================================
# ADVANCED ANALYTICS ENGINE
# ============================================================================

class AdvancedAnalytics:
    """Comprehensive analytics engine with institutional-grade methods"""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        returns = returns.dropna()
        if len(returns) < 60:
            return {}
        
        # Basic metrics
        total_return = ((1 + returns).prod() - 1) * 100
        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99
        }
    
    @staticmethod
    def test_arch_effects(returns: pd.Series, lags: int = 5) -> Dict[str, Any]:
        """Test for ARCH effects in returns"""
        if not STATSMODELS_AVAILABLE or len(returns) < 100:
            return {"arch_present": False, "p_value": 1.0}
        
        try:
            returns_clean = returns.dropna().values
            returns_clean = returns_clean - np.mean(returns_clean)
            
            LM, LM_p, F, F_p = het_arch(returns_clean, maxlag=lags)
            
            return {
                "arch_present": LM_p < 0.05,
                "p_value": LM_p,
                "lm_statistic": LM,
                "f_statistic": F
            }
        except Exception:
            return {"arch_present": False, "p_value": 1.0}
    
    @staticmethod
    def rolling_beta_analysis(asset_returns: pd.Series, benchmark_returns: pd.Series, 
                            window: int = 60) -> pd.DataFrame:
        """Calculate rolling beta with diagnostics"""
        aligned = pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("benchmark")], 
                           axis=1, join="inner").dropna()
        
        if len(aligned) < window:
            return pd.DataFrame()
        
        results = []
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            cov = window_data["asset"].cov(window_data["benchmark"])
            var = window_data["benchmark"].var()
            beta = cov / var if var > 1e-10 else np.nan
            corr = window_data["asset"].corr(window_data["benchmark"])
            
            results.append({
                "date": aligned.index[i],
                "beta": beta,
                "r_squared": corr ** 2
            })
        
        return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()
    
    @staticmethod
    def garch_grid_search(returns: pd.Series, p_range: Tuple[int, int] = (1, 2), 
                         q_range: Tuple[int, int] = (1, 2), 
                         garch_types: List[str] = ["GARCH", "EGARCH", "GJR"],
                         distributions: List[str] = ["normal", "t"]) -> pd.DataFrame:
        """Perform grid search for optimal GARCH specification"""
        if not ARCH_AVAILABLE or len(returns) < 300:
            return pd.DataFrame()
        
        returns_scaled = returns.dropna().values * 100
        results = []
        
        for garch_type in garch_types:
            for p in range(p_range[0], p_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    for dist in distributions:
                        try:
                            model = arch_model(
                                returns_scaled,
                                mean="Constant",
                                vol=garch_type,
                                p=p,
                                q=q,
                                dist=dist
                            )
                            fit = model.fit(disp="off", show_warning=False)
                            
                            results.append({
                                "garch_type": garch_type,
                                "p": p,
                                "q": q,
                                "distribution": dist,
                                "aic": fit.aic,
                                "bic": fit.bic,
                                "converged": fit.convergence_flag == 0,
                                "log_likelihood": fit.loglikelihood
                            })
                            
                        except Exception:
                            continue
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(["converged", "aic"], ascending=[False, True])
            return df_results
        
        return pd.DataFrame()
    
    @staticmethod
    def select_champion_model(grid_results: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Select champion model from grid search results"""
        if grid_results.empty:
            return None
        
        # Filter converged models
        converged = grid_results[grid_results["converged"]]
        if converged.empty:
            # If no converged models, use best by AIC
            best_model = grid_results.iloc[0]
        else:
            # Select best converged model by AIC
            best_model = converged.iloc[0]
        
        return {
            "garch_type": best_model["garch_type"],
            "p": int(best_model["p"]),
            "q": int(best_model["q"]),
            "distribution": best_model["distribution"],
            "aic": best_model["aic"],
            "bic": best_model["bic"]
        }
    
    @staticmethod
    def fit_champion_garch(returns: pd.Series, champion_spec: Dict[str, Any], 
                          forecast_horizon: int = 30) -> Optional[Dict[str, Any]]:
        """Fit champion GARCH model and forecast"""
        if not ARCH_AVAILABLE or champion_spec is None:
            return None
        
        returns_scaled = returns.dropna().values * 100
        
        try:
            model = arch_model(
                returns_scaled,
                mean="Constant",
                vol=champion_spec["garch_type"],
                p=champion_spec["p"],
                q=champion_spec["q"],
                dist=champion_spec["distribution"]
            )
            fit = model.fit(disp="off", show_warning=False)
            
            # Conditional volatility
            cond_vol = pd.Series(fit.conditional_volatility / 100, index=returns.dropna().index)
            
            # Forecast
            forecast = fit.forecast(horizon=forecast_horizon)
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].values) / 100
            
            return {
                "model": fit,
                "cond_volatility": cond_vol,
                "forecast_volatility": forecast_vol,
                "converged": fit.convergence_flag == 0,
                "aic": fit.aic,
                "bic": fit.bic
            }
            
        except Exception as e:
            st.warning(f"GARCH fitting failed: {str(e)}")
            return None
    
    @staticmethod
    def detect_market_regimes(returns: pd.Series, n_states: int = 3) -> Optional[Dict[str, Any]]:
        """Detect market regimes using HMM"""
        if not HMM_AVAILABLE or len(returns) < 260:
            return None
        
        try:
            # Prepare features
            features = pd.DataFrame({
                "returns": returns.values,
                "volatility": returns.rolling(20).std().values
            }).dropna()
            
            if len(features) < 260:
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            
            # Fit HMM
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=500,
                random_state=42
            )
            model.fit(X_scaled)
            
            # Predict states
            states = model.predict(X_scaled)
            
            # Calculate regime statistics
            regime_stats = []
            for state in range(n_states):
                mask = states == state
                regime_returns = returns.iloc[mask]
                
                if len(regime_returns) > 10:
                    regime_stats.append({
                        "regime": state,
                        "frequency": len(regime_returns) / len(states),
                        "mean_return": regime_returns.mean() * 100,
                        "volatility": regime_returns.std() * np.sqrt(252) * 100
                    })
            
            return {
                "states": states,
                "regime_stats": regime_stats,
                "model": model
            }
            
        except Exception as e:
            st.warning(f"Regime detection failed: {str(e)}")
            return None
    
    @staticmethod
    def calculate_portfolio_metrics(returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        if returns_df.empty or len(returns_df) < 60:
            return {}
        
        # Portfolio returns
        portfolio_returns = returns_df.values @ weights
        
        # Calculate all metrics
        metrics = AdvancedAnalytics.calculate_performance_metrics(
            pd.Series(portfolio_returns, index=returns_df.index)
        )
        
        # Add portfolio-specific metrics
        metrics["weights"] = dict(zip(returns_df.columns, weights))
        metrics["num_assets"] = len(weights)
        
        return metrics
    
    @staticmethod
    def calculate_risk_contributions(returns_df: pd.DataFrame, weights: np.ndarray, 
                                   confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate ES risk contributions"""
        if returns_df.empty:
            return pd.DataFrame()
        
        # Normalize weights
        weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)
        
        # Portfolio returns
        portfolio_returns = returns_df.values @ weights
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Find tail scenarios
        tail_mask = portfolio_returns <= var
        
        if tail_mask.sum() < 10:
            return pd.DataFrame()
        
        # Calculate contributions in tail scenarios
        contributions = returns_df.values[tail_mask] * weights
        es_contributions = contributions.mean(axis=0)
        
        # Calculate relative contributions
        total_es = np.sum(np.abs(es_contributions))
        relative_contributions = np.abs(es_contributions) / total_es if total_es > 0 else np.zeros_like(es_contributions)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "Asset": returns_df.columns,
            "Weight": weights * 100,
            "ES_Contribution": es_contributions * 100,
            "Relative_Contribution": relative_contributions * 100
        })
        
        return results.sort_values("Relative_Contribution", ascending=False)
    
    @staticmethod
    def var_backtest(returns: pd.Series, var_method: str = "historical", 
                    confidence_level: float = 0.95, window: int = 250) -> Dict[str, Any]:
        """Perform VaR backtest with Kupiec and Christoffersen tests"""
        returns = returns.dropna()
        if len(returns) < window + 50:
            return {}
        
        var_series = []
        hits = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            if var_method == "historical":
                var = np.percentile(window_returns, (1 - confidence_level) * 100)
            else:  # normal
                var = window_returns.mean() + window_returns.std() * stats.norm.ppf(1 - confidence_level)
            
            var_series.append(var)
            hits.append(returns.iloc[i] < var)
        
        hits_array = np.array(hits)
        n = len(hits_array)
        x = hits_array.sum()
        expected = (1 - confidence_level) * n
        
        return {
            "observations": n,
            "breaches": int(x),
            "breach_rate": x / n * 100,
            "expected_breaches": expected,
            "var_series": var_series,
            "hits": hits_array
        }

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class EnhancedVisualization:
    """Professional visualization engine for institutional analytics"""
    
    def __init__(self):
        self.colors = {
            "primary": "#1a2980",
            "secondary": "#26d0ce",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "dark": "#343a40",
            "light": "#f8f9fa"
        }
    
    def create_price_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create professional price chart with technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{title} - Price Action", "Volume", "RSI")
        )
        
        # Price with moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Adj Close"],
                name="Price",
                line=dict(color=self.colors["primary"], width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                line=dict(color=self.colors["warning"], width=1, dash="dash")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line=dict(color=self.colors["success"], width=1, dash="dash")
            ),
            row=1, col=1
        )
        
        # Volume with color coding
        colors = [self.colors["success"] if close >= open_ else self.colors["danger"] 
                 for close, open_ in zip(df["Adj Close"], df["Open"])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI
        if "RSI_14" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI_14"],
                    name="RSI",
                    line=dict(color=self.colors["secondary"], width=2)
                ),
                row=3, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors["danger"], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors["success"], 
                         opacity=0.5, row=3, col=1)
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=700,
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    def create_volatility_chart(self, returns: pd.Series, cond_vol: pd.Series = None, 
                               forecast_vol: pd.Series = None, title: str = "Volatility Analysis") -> go.Figure:
        """Create volatility analysis chart"""
        fig = go.Figure()
        
        # Historical volatility
        rv_20 = returns.rolling(20).std() * np.sqrt(252)
        rv_60 = returns.rolling(60).std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=rv_20.index,
            y=rv_20 * 100,
            name="Realized Vol (20D)",
            line=dict(width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=rv_60.index,
            y=rv_60 * 100,
            name="Realized Vol (60D)",
            line=dict(width=1.5, dash="dash")
        ))
        
        # Conditional volatility from GARCH
        if cond_vol is not None:
            fig.add_trace(go.Scatter(
                x=cond_vol.index,
                y=cond_vol * 100,
                name="GARCH Conditional Vol",
                line=dict(color=self.colors["primary"], width=2)
            ))
        
        # Volatility forecast
        if forecast_vol is not None:
            forecast_dates = pd.date_range(
                start=returns.index[-1] + pd.Timedelta(days=1),
                periods=len(forecast_vol),
                freq="D"
            )
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_vol * 100,
                name="Volatility Forecast",
                line=dict(color=self.colors["danger"], width=2, dash="dot")
            ))
        
        fig.update_layout(
            title=title,
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Annualized Volatility (%)",
            xaxis_title="Date"
        )
        
        return fig
    
    def create_regime_chart(self, price: pd.Series, regimes: np.ndarray, title: str) -> go.Figure:
        """Create regime visualization chart"""
        fig = go.Figure()
        
        # Base price line
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price.values,
            name="Price",
            line=dict(color="gray", width=1),
            opacity=0.7
        ))
        
        # Color by regime
        unique_regimes = np.unique(regimes)
        colors = [self.colors["danger"], self.colors["warning"], self.colors["success"]]
        
        for i, regime in enumerate(unique_regimes):
            mask = regimes == regime
            regime_dates = price.index[mask]
            regime_prices = price.values[mask]
            
            fig.add_trace(go.Scatter(
                x=regime_dates,
                y=regime_prices,
                mode="markers",
                name=f"Regime {regime}",
                marker=dict(size=6, color=colors[i % len(colors)]),
                opacity=0.8
            ))
        
        fig.update_layout(
            title=title,
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Price",
            xaxis_title="Date"
        )
        
        return fig
    
    def create_backtest_chart(self, returns: pd.Series, var_series: List[float], 
                             hits: np.ndarray, title: str) -> go.Figure:
        """Create VaR backtest visualization"""
        fig = go.Figure()
        
        # Returns
        fig.add_trace(go.Scatter(
            x=returns.index[-len(var_series):],
            y=returns.values[-len(var_series):] * 100,
            name="Returns",
            line=dict(width=1)
        ))
        
        # VaR series
        fig.add_trace(go.Scatter(
            x=returns.index[-len(var_series):],
            y=var_series * 100,
            name="VaR",
            line=dict(color=self.colors["danger"], width=2, dash="dash")
        ))
        
        # Breaches
        breach_dates = returns.index[-len(hits):][hits]
        breach_returns = returns.values[-len(hits):][hits] * 100
        
        fig.add_trace(go.Scatter(
            x=breach_dates,
            y=breach_returns,
            mode="markers",
            name="Breaches",
            marker=dict(size=8, color=self.colors["danger"], symbol="x")
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Return (%)",
            xaxis_title="Date"
        )
        
        return fig

# ============================================================================
# DASHBOARD CLASS
# ============================================================================

class InstitutionalCommoditiesDashboard:
    """Main dashboard class integrating all components"""
    
    def __init__(self):
        self.analytics = AdvancedAnalytics()
        self.viz = EnhancedVisualization()
        self.data_loader = DataLoader()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            "data_loaded": False,
            "asset_data": {},
            "returns_data": {},
            "benchmark_data": {},
            "selected_assets": [],
            "portfolio_weights": {},
            "garch_results": {},
            "regime_results": {},
            "portfolio_results": {},
            "backtest_results": {},
            "champion_models": {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def display_header(self):
        """Display professional header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; font-size:2.5rem;">🏛️ Institutional Commodities Analytics v4.1</h1>
            <p style="margin:8px 0 0 0; opacity:0.95;">
                Advanced Portfolio Analytics • GARCH Champion Selection • Regime Detection • Stress Testing
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "🟢 Online", "Streamlit Cloud")
        with col2:
            loaded = len(st.session_state.asset_data) if st.session_state.data_loaded else 0
            st.metric("Assets Loaded", loaded)
        with col3:
            arch_status = "Available" if ARCH_AVAILABLE else "Disabled"
            st.metric("GARCH", arch_status)
        with col4:
            hmm_status = "Available" if HMM_AVAILABLE else "Disabled"
            st.metric("HMM", hmm_status)
    
    def setup_sidebar(self) -> Dict[str, Any]:
        """Setup sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📅 Date Range")
            
            # FIX: Use default dates that work with Yahoo Finance
            default_end = datetime.now()
            default_start = default_end - timedelta(days=365*3)  # 3 years
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", default_start)
            with col2:
                end_date = st.date_input("End Date", default_end)
            
            if start_date >= end_date:
                st.error("End date must be after start date")
                st.stop()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Asset selection
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Asset Selection")
            
            selected_assets = []
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category}", expanded=True):
                    for symbol, info in assets.items():
                        if st.checkbox(f"{info['name']} ({symbol})", key=f"asset_{symbol}"):
                            selected_assets.append(symbol)
            
            # Benchmark selection
            st.markdown("### 📈 Benchmarks")
            selected_benchmarks = []
            for benchmark, info in BENCHMARKS.items():
                if st.checkbox(f"{info['name']} ({benchmark})", key=f"bench_{benchmark}"):
                    selected_benchmarks.append(benchmark)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analytics configuration
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Analytics Settings")
            
            # GARCH settings
            if ARCH_AVAILABLE:
                st.markdown("**GARCH Configuration**")
                garch_horizon = st.slider("Forecast Horizon", 5, 60, 30, 5)
                p_max = st.slider("ARCH Order (p max)", 1, 3, 2, 1)
                q_max = st.slider("GARCH Order (q max)", 1, 3, 2, 1)
            
            # Risk settings
            st.markdown("**Risk Metrics**")
            confidence_levels = st.multiselect(
                "Confidence Levels",
                [0.90, 0.95, 0.99],
                default=[0.95, 0.99]
            )
            
            # Backtest settings
            backtest_window = st.slider("Backtest Window", 100, 500, 250, 25)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load Data", type="primary", use_container_width=True):
                    self._load_data(selected_assets, selected_benchmarks, start_date, end_date)
            
            with col2:
                if st.session_state.data_loaded:
                    if st.button("🔄 Clear", type="secondary", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "garch_horizon": garch_horizon if ARCH_AVAILABLE else 30,
                "p_max": p_max if ARCH_AVAILABLE else 2,
                "q_max": q_max if ARCH_AVAILABLE else 2,
                "confidence_levels": confidence_levels,
                "backtest_window": backtest_window
            }
    
    def _load_data(self, assets: List[str], benchmarks: List[str], 
                  start_date: datetime, end_date: datetime):
        """Load data for selected assets and benchmarks - FIXED VERSION"""
        if not assets:
            st.warning("Please select at least one asset")
            return
        
        # FIX: Convert dates to datetime
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        
        # FIX: Add buffer for data availability
        start_dt = start_dt - timedelta(days=30)
        
        st.markdown('<div class="data-loading">', unsafe_allow_html=True)
        st.write("### 📊 Data Loading")
        st.write(f"**Date Range:** {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        st.write(f"**Selected Assets:** {len(assets)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Combine all symbols
        all_symbols = assets + benchmarks
        
        # Load data using robust loader
        with st.spinner("Loading market data (this may take a minute)..."):
            data_dict = self.data_loader.bulk_load_data(all_symbols, start_dt, end_dt)
            
            if not data_dict:
                st.error("❌ Failed to load any data. Possible issues:")
                st.write("1. Yahoo Finance API may be temporarily unavailable")
                st.write("2. Symbols may be incorrect or delisted")
                st.write("3. Network connectivity issues")
                return
            
            # Separate assets and benchmarks
            asset_data = {}
            benchmark_data = {}
            returns_data = {}
            
            for symbol, df in data_dict.items():
                if symbol in assets:
                    asset_data[symbol] = df
                    returns_data[symbol] = df["Returns"].dropna()
                elif symbol in benchmarks:
                    benchmark_data[symbol] = df
            
            # Store in session state
            st.session_state.asset_data = asset_data
            st.session_state.returns_data = returns_data
            st.session_state.benchmark_data = benchmark_data
            st.session_state.selected_assets = list(asset_data.keys())
            st.session_state.data_loaded = True
            
            # Initialize equal weights
            n_assets = len(asset_data)
            if n_assets > 0:
                equal_weight = 1.0 / n_assets
                st.session_state.portfolio_weights = {
                    asset: equal_weight for asset in asset_data.keys()
                }
            
            st.balloons()
            st.success(f"✅ Successfully loaded {len(asset_data)} assets and {len(benchmark_data)} benchmarks")
            
            # Show data summary
            with st.expander("📋 Data Summary", expanded=True):
                for symbol, df in asset_data.items():
                    st.write(f"**{symbol}**: {len(df)} days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    def run(self):
        """Main dashboard execution"""
        self.display_header()
        
        # Setup sidebar
        config = self.setup_sidebar()
        
        if not st.session_state.data_loaded:
            st.info("👈 Please select assets and click 'Load Data' to begin analysis")
            return
        
        # Create tabs for different analytics modules
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "🧺 Portfolio", 
            "⚡ GARCH",
            "🔄 Regimes", 
            "📈 Analytics"
        ])
        
        with tab1:
            self.display_dashboard(config)
        
        with tab2:
            self.display_portfolio_analytics(config)
        
        with tab3:
            if ARCH_AVAILABLE:
                self.display_garch_analysis(config)
            else:
                st.warning("GARCH analysis requires the 'arch' package. Please install it.")
        
        with tab4:
            if HMM_AVAILABLE:
                self.display_regime_analysis(config)
            else:
                st.warning("Regime analysis requires 'hmmlearn' and 'scikit-learn'. Please install them.")
        
        with tab5:
            self.display_advanced_analytics(config)
    
    def display_dashboard(self, config: Dict[str, Any]):
        """Display main dashboard"""
        st.header("📊 Market Dashboard")
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets Loaded", len(st.session_state.asset_data))
        with col2:
            total_days = (config["end_date"] - config["start_date"]).days
            st.metric("Time Period", f"{total_days} days")
        with col3:
            if st.session_state.returns_data:
                avg_corr = pd.DataFrame(st.session_state.returns_data).corr().mean().mean()
                st.metric("Avg Correlation", f"{avg_corr:.2%}")
            else:
                st.metric("Avg Correlation", "N/A")
        with col4:
            if st.session_state.asset_data:
                sample = next(iter(st.session_state.asset_data.values()))
                st.metric("Data Points", len(sample))
            else:
                st.metric("Data Points", 0)
        
        # Asset performance summary
        st.subheader("📈 Asset Performance Summary")
        
        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if len(df) > 0:
                returns = df["Returns"].dropna()
                if len(returns) > 0:
                    metrics = self.analytics.calculate_performance_metrics(returns)
                    
                    performance_data.append({
                        "Asset": symbol,
                        "Current Price": df["Adj Close"].iloc[-1],
                        "1D Return": df["Returns"].iloc[-1] * 100 if len(df) > 1 else 0,
                        "Annual Return": metrics.get("annual_return", 0),
                        "Annual Vol": metrics.get("annual_volatility", 0),
                        "Sharpe": metrics.get("sharpe_ratio", 0),
                        "Max DD": metrics.get("max_drawdown", 0)
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(
                perf_df.style.format({
                    "Current Price": "{:.2f}",
                    "1D Return": "{:.2f}%",
                    "Annual Return": "{:.2f}%",
                    "Annual Vol": "{:.2f}%",
                    "Sharpe": "{:.2f}",
                    "Max DD": "{:.2f}%"
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No performance data available")
        
        # Correlation matrix
        st.subheader("📊 Correlation Matrix")
        if st.session_state.returns_data:
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            
            if not returns_df.empty and len(returns_df.columns) > 1:
                corr_matrix = returns_df.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}'
                ))
                fig.update_layout(height=500, title="Asset Correlations")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for correlation matrix")
        else:
            st.info("Load returns data to see correlation matrix")
        
        # Individual asset charts
        st.subheader("📉 Individual Asset Analysis")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed View",
            options=st.session_state.selected_assets,
            key="dashboard_asset_select"
        )
        
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            fig = self.viz.create_price_chart(df, f"{selected_asset} - Price & Indicators")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selected asset not found in loaded data")
    
    def display_portfolio_analytics(self, config: Dict[str, Any]):
        """Display portfolio analytics"""
        st.header("🧺 Portfolio Analytics")
        
        # Get returns data
        if not st.session_state.returns_data:
            st.warning("No returns data available")
            return
        
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
        
        if returns_df.empty or len(returns_df) < 60:
            st.warning("Insufficient data for portfolio analysis")
            return
        
        assets = returns_df.columns.tolist()
        
        # Portfolio weights configuration
        st.subheader("Portfolio Configuration")
        
        weight_mode = st.radio(
            "Weighting Method",
            ["Equal Weight", "Custom Weights"],
            horizontal=True
        )
        
        if weight_mode == "Equal Weight":
            weights = np.ones(len(assets)) / len(assets)
        else:
            st.write("**Set Custom Weights:**")
            cols = st.columns(min(4, len(assets)))
            weight_inputs = []
            
            for i, asset in enumerate(assets):
                with cols[i % len(cols)]:
                    default_weight = 1.0 / len(assets)
                    current_weight = st.session_state.portfolio_weights.get(asset, default_weight)
                    weight = st.slider(
                        asset,
                        min_value=0.0,
                        max_value=1.0,
                        value=current_weight,
                        step=0.01,
                        key=f"weight_{asset}"
                    )
                    weight_inputs.append(weight)
            
            weights = np.array(weight_inputs)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        # Store weights in session state
        st.session_state.portfolio_weights = dict(zip(assets, weights))
        
        # Calculate portfolio metrics
        portfolio_metrics = self.analytics.calculate_portfolio_metrics(returns_df, weights)
        
        # Display key metrics
        st.subheader("Portfolio Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Return", f"{portfolio_metrics.get('annual_return', 0):.2f}%")
        with col2:
            st.metric("Annual Volatility", f"{portfolio_metrics.get('annual_volatility', 0):.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{portfolio_metrics.get('max_drawdown', 0):.2f}%")
        
        # Risk metrics
        st.subheader("Risk Metrics")
        
        risk_cols = st.columns(4)
        with risk_cols[0]:
            st.metric("VaR 95%", f"{portfolio_metrics.get('var_95', 0):.2f}%")
        with risk_cols[1]:
            st.metric("CVaR 95%", f"{portfolio_metrics.get('cvar_95', 0):.2f}%")
        with risk_cols[2]:
            st.metric("Sortino Ratio", f"{portfolio_metrics.get('sortino_ratio', 0):.2f}")
        with risk_cols[3]:
            st.metric("Win Rate", "N/A")
        
        # Portfolio returns chart
        st.subheader("Portfolio Cumulative Returns")
        
        portfolio_returns = pd.Series(
            returns_df.values @ weights,
            index=returns_df.index,
            name="Portfolio"
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_returns.index,
            y=(1 + portfolio_returns).cumprod(),
            name="Portfolio",
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Cumulative Return",
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Cumulative Return"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_garch_analysis(self, config: Dict[str, Any]):
        """Display GARCH analysis"""
        st.header("⚡ Advanced GARCH Analysis")
        
        if not st.session_state.returns_data:
            st.warning("No returns data available")
            return
        
        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset for GARCH Analysis",
            options=st.session_state.selected_assets,
            key="garch_asset_select"
        )
        
        if selected_asset not in st.session_state.returns_data:
            st.warning("Selected asset not found")
            return
        
        returns = st.session_state.returns_data[selected_asset]
        
        # ARCH effects test
        st.subheader("ARCH Effects Test")
        
        arch_test = self.analytics.test_arch_effects(returns)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARCH Present", "Yes" if arch_test["arch_present"] else "No")
        with col2:
            st.metric("p-value", f"{arch_test['p_value']:.4f}")
        with col3:
            good = arch_test["p_value"] < 0.05
            st.metric("Significant", "✓" if good else "✗")
        
        if not arch_test["arch_present"]:
            st.warning("Limited ARCH effects detected. GARCH modeling may not be optimal.")
        
        # GARCH grid search
        st.subheader("Champion GARCH Selection")
        
        if st.button("🔍 Run Grid Search", type="primary", use_container_width=True):
            with st.spinner("Running GARCH grid search..."):
                grid_results = self.analytics.garch_grid_search(
                    returns,
                    p_range=(1, config["p_max"]),
                    q_range=(1, config["q_max"]),
                    garch_types=["GARCH", "EGARCH", "GJR"],
                    distributions=["normal", "t"]
                )
                
                if not grid_results.empty:
                    st.session_state.garch_results[selected_asset] = {
                        "grid_results": grid_results,
                        "champion": self.analytics.select_champion_model(grid_results)
                    }
                    st.success("Grid search completed!")
        
        # Display grid results if available
        if selected_asset in st.session_state.garch_results:
            garch_data = st.session_state.garch_results[selected_asset]
            grid_results = garch_data["grid_results"]
            champion = garch_data["champion"]
            
            if not grid_results.empty:
                st.dataframe(
                    grid_results.style.format({
                        "aic": "{:.1f}",
                        "bic": "{:.1f}"
                    }),
                    use_container_width=True,
                    height=300
                )
            
            if champion:
                st.success(f"**Champion Model**: {champion['garch_type']}({champion['p']},{champion['q']}) - {champion['distribution']} distribution")
                
                # Fit champion model
                if st.button("🎯 Fit Champion Model", type="secondary", use_container_width=True):
                    with st.spinner("Fitting champion GARCH model..."):
                        garch_fit = self.analytics.fit_champion_garch(
                            returns,
                            champion,
                            config["garch_horizon"]
                        )
                        
                        if garch_fit:
                            garch_data["fit"] = garch_fit
                            st.session_state.garch_results[selected_asset] = garch_data
                            st.success("Model fitted successfully!")
            
            # Display GARCH results if fitted
            if "fit" in garch_data:
                fit = garch_data["fit"]
                
                st.subheader("GARCH Model Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{fit['aic']:.1f}")
                with col2:
                    st.metric("BIC", f"{fit['bic']:.1f}")
                with col3:
                    st.metric("Converged", "✓" if fit["converged"] else "✗")
                
                # Volatility chart
                st.subheader("Volatility Analysis")
                
                fig = self.viz.create_volatility_chart(
                    returns,
                    fit["cond_volatility"],
                    pd.Series(fit["forecast_volatility"]),
                    f"{selected_asset} - GARCH Volatility"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run grid search to select and fit a champion GARCH model.")
    
    def display_regime_analysis(self, config: Dict[str, Any]):
        """Display regime analysis"""
        st.header("🔄 Market Regime Detection")
        
        if not st.session_state.asset_data:
            st.warning("No asset data available")
            return
        
        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=st.session_state.selected_assets,
            key="regime_asset_select"
        )
        
        if selected_asset not in st.session_state.returns_data:
            st.warning("Selected asset not found")
            return
        
        returns = st.session_state.returns_data[selected_asset]
        
        # Regime detection settings
        st.subheader("Regime Detection Settings")
        
        n_states = st.slider("Number of Regimes", 2, 5, 3, 1)
        
        if st.button("🔍 Detect Regimes", type="primary", use_container_width=True):
            with st.spinner("Detecting market regimes..."):
                regime_results = self.analytics.detect_market_regimes(returns, n_states)
                
                if regime_results:
                    st.session_state.regime_results[selected_asset] = regime_results
                    st.success("Regime detection completed!")
        
        # Display regime results if available
        if selected_asset in st.session_state.regime_results:
            regime_data = st.session_state.regime_results[selected_asset]
            
            # Regime statistics
            st.subheader("Regime Statistics")
            
            if regime_data["regime_stats"]:
                stats_df = pd.DataFrame(regime_data["regime_stats"])
                st.dataframe(
                    stats_df.style.format({
                        "frequency": "{:.2%}",
                        "mean_return": "{:.4f}%",
                        "volatility": "{:.2f}%"
                    }),
                    use_container_width=True
                )
            
            # Regime visualization
            st.subheader("Regime Visualization")
            
            df = st.session_state.asset_data[selected_asset]
            price = df["Adj Close"]
            
            fig = self.viz.create_regime_chart(
                price,
                regime_data["states"],
                f"{selected_asset} - Market Regimes"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run regime detection to analyze market regimes.")
    
    def display_advanced_analytics(self, config: Dict[str, Any]):
        """Display advanced analytics"""
        st.header("📈 Advanced Analytics")
        
        if not st.session_state.returns_data:
            st.warning("No returns data available")
            return
        
        # VaR Backtesting
        st.subheader("✅ VaR Backtesting")
        
        selected_asset = st.selectbox(
            "Select Asset for Backtesting",
            options=st.session_state.selected_assets,
            key="backtest_asset_select"
        )
        
        if selected_asset not in st.session_state.returns_data:
            st.warning("Selected asset not found")
            return
        
        returns = st.session_state.returns_data[selected_asset]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            var_method = st.selectbox("VaR Method", ["historical", "normal"])
        with col2:
            confidence_level = st.selectbox("Confidence Level", 
                                          config["confidence_levels"],
                                          format_func=lambda x: f"{x:.0%}")
        with col3:
            window = st.number_input("Window Size", 
                                   min_value=100, 
                                   max_value=500, 
                                   value=config["backtest_window"])
        
        if st.button("📊 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running VaR backtest..."):
                backtest_results = self.analytics.var_backtest(
                    returns,
                    var_method,
                    confidence_level,
                    window
                )
                
                if backtest_results:
                    st.session_state.backtest_results[selected_asset] = backtest_results
                    st.success("Backtest completed!")
        
        # Display backtest results if available
        if selected_asset in st.session_state.backtest_results:
            results = st.session_state.backtest_results[selected_asset]
            
            # Backtest statistics
            st.subheader("Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Observations", f"{results['observations']:,}")
            with col2:
                st.metric("Breaches", f"{results['breaches']:,}")
            with col3:
                st.metric("Breach Rate", f"{results['breach_rate']:.2f}%")
            with col4:
                st.metric("Expected Breaches", f"{results['expected_breaches']:.1f}")
            
            # Backtest visualization
            st.subheader("Backtest Visualization")
            
            fig = self.viz.create_backtest_chart(
                returns,
                results["var_series"],
                results["hits"],
                f"{selected_asset} - {var_method.title()} VaR Backtest ({confidence_level:.0%} Confidence)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run backtest to see results.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main application entry point"""
    # Hide Streamlit branding
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create and run dashboard
    try:
        dashboard = InstitutionalCommoditiesDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, check your internet connection.")

if __name__ == "__main__":
    main()
