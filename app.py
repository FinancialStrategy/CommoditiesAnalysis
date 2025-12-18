"""
🏛️ Institutional Commodities Analytics Platform v4.0
Advanced Portfolio Analytics • GARCH Champion Selection • Regime Detection • Stress Testing
Streamlit Cloud Optimized for Institutional Use
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

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Institutional Commodities Platform v4.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COMMODITIES UNIVERSE (FIXED: Defined BEFORE any usage)
# =============================================================================

COMMODITIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Precious Metals": {
        "GC=F": {"name": "Gold Futures", "category": "Precious"},
        "SI=F": {"name": "Silver Futures", "category": "Precious"},
        "PL=F": {"name": "Platinum Futures", "category": "Precious"},
        "PA=F": {"name": "Palladium Futures", "category": "Precious"}
    },
    "Industrial Metals": {
        "HG=F": {"name": "Copper Futures", "category": "Industrial"},
        # NOTE: Some Yahoo symbols vary by venue; keep as-is but app will warn on failures.
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

# -----------------------------------------------------------------------------
# GOLD ETF BENCHMARK (Most Active US Gold ETF Selector)
# -----------------------------------------------------------------------------
GOLD_ETF_CANDIDATES = {
    "GLD": {"name": "SPDR Gold Shares", "type": "commodity"},
    "IAU": {"name": "iShares Gold Trust", "type": "commodity"},
    "GLDM": {"name": "SPDR Gold MiniShares Trust", "type": "commodity"},
    "SGOL": {"name": "abrdn Physical Gold Shares ETF", "type": "commodity"},
    "BAR": {"name": "GraniteShares Gold Trust", "type": "commodity"}
}

@st.cache_data(ttl=1800, show_spinner=False)
def select_most_active_gold_etf(start_date: datetime, end_date: datetime,
                                candidates: Optional[List[str]] = None) -> Tuple[str, pd.DataFrame]:
    """Select the most active US-listed Gold ETF (by average dollar volume) for the period.
    Returns (symbol, dataframe). Falls back to GLD if selection fails.
    """
    symbols = candidates if candidates else list(GOLD_ETF_CANDIDATES.keys())
    best_sym = "GLD"
    best_score = -1.0
    best_df = pd.DataFrame()

    for sym in symbols:
        try:
            df = fetch_asset_data(sym, start_date, end_date)
            if df is None or df.empty:
                continue
            if "Volume" not in df.columns or ("Close" not in df.columns and "Adj Close" not in df.columns):
                continue
            px_col = "Close" if "Close" in df.columns else "Adj Close"
            tail = df.tail(80).dropna(subset=[px_col, "Volume"])
            if tail.empty:
                continue
            dollar_vol = (tail[px_col].astype(float) * tail["Volume"].astype(float)).mean()
            if (dollar_vol is not None) and (float(dollar_vol) == float(dollar_vol)) and float(dollar_vol) > best_score:
                best_score = float(dollar_vol)
                best_sym = sym
                best_df = df
        except Exception:
            continue

    if best_df is None or best_df.empty:
        try:
            best_df = fetch_asset_data(best_sym, start_date, end_date)
        except Exception:
            best_df = pd.DataFrame()

    return best_sym, best_df

# =============================================================================
# STYLES & UTILITIES
# =============================================================================

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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# IMPORT MANAGEMENT
# =============================================================================

STATSMODELS_AVAILABLE = False
ARCH_AVAILABLE = False
HMM_AVAILABLE = False
QUANTSTATS_AVAILABLE = False

# IMPORTANT FIX: statsmodels.api as sm was missing, causing NameError in rolling beta.
try:
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    st.warning("statsmodels not available - some diagnostics/rolling beta disabled")

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

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================

def safe_float(x: Any, default: float = np.nan) -> float:
    """Safely convert to float with fallback"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def format_number(x: Any, decimals: int = 3) -> str:
    """Format number with specified decimals"""
    v = safe_float(x, np.nan)
    return "—" if not np.isfinite(v) else f"{v:.{decimals}f}"

def format_percentage(x: Any, decimals: int = 2) -> str:
    """Format percentage with sign"""
    v = safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def badge(text: str, badge_type: str = "success") -> str:
    """Create HTML badge"""
    types = {"success": "status-success", "warning": "status-warning", "danger": "status-danger"}
    cls = types.get(badge_type, "status-success")
    return f'<span class="status-badge {cls}">{text}</span>'

def annualize_vol(series: pd.Series, annualization: int = 252) -> pd.Series:
    """Annualize volatility series"""
    return series * math.sqrt(annualization)

# =============================================================================
# ENHANCED DATA MANAGER (CRITICAL FIXES FOR STREAMLIT CLOUD)
# =============================================================================

def _yf_download_robust(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Robust yfinance download wrapper:
    - Handles different yfinance versions (threads param may not exist)
    - Provides fallback auto_adjust behavior
    - Avoids raising and returns DataFrame()
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # First attempt: auto_adjust=True (fast, adjusted OHLC), with threads if supported
    try:
        try:
            df = yf.download(
                symbol,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=True,
                threads=True
            )
        except TypeError:
            df = yf.download(
                symbol,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=True
            )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    # Second attempt: auto_adjust=False (keeps 'Adj Close' for older patterns)
    try:
        try:
            df = yf.download(
                symbol,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=False,
                threads=True
            )
        except TypeError:
            df = yf.download(
                symbol,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=False
            )
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_asset_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch and enhance asset data with comprehensive features (cloud-safe)."""
    try:
        df = _yf_download_robust(symbol, start_date, end_date)

        if df is None or df.empty or len(df) < 60:
            return pd.DataFrame()

        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]

        # ---------------------------------------------------------------------
        # CRITICAL FIX:
        # With auto_adjust=True, yfinance often DOES NOT provide "Adj Close".
        # Your old logic created NaN for "Adj Close" then dropna -> EMPTY DF.
        # Solution: if Adj Close missing, map it to Close (already adjusted).
        # ---------------------------------------------------------------------
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        # Ensure required columns (do NOT introduce NaN that kills the dataset)
        if "Close" not in df.columns:
            return pd.DataFrame()

        # If OHLC missing, approximate using Close to keep indicators functional
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]

        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        # Basic cleaning
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.dropna(subset=["Adj Close"])

        # Guard against pathological series
        if df["Adj Close"].dropna().shape[0] < 60:
            return pd.DataFrame()

        # Calculate enhanced features
        df = calculate_enhanced_features(df)

        return df

    except Exception:
        # Avoid streamlit calls inside cached functions (can error on Cloud)
        return pd.DataFrame()

def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical features"""
    df = df.copy()

    # Returns
    df["Returns"] = df["Adj Close"].pct_change()
    df["Log_Returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

    # Moving averages
    df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Adj Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Adj Close"].ewm(span=26).mean()

    # Bollinger Bands
    bb_middle = df["Adj Close"].rolling(window=20).mean()
    bb_std = df["Adj Close"].rolling(window=20).std()
    df["BB_Upper"] = bb_middle + 2 * bb_std
    df["BB_Lower"] = bb_middle - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_middle.replace(0, np.nan)

    # Volatility measures
    df["Volatility_20D"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)
    df["Volatility_60D"] = df["Returns"].rolling(window=60).std() * np.sqrt(252)

    # RSI
    delta = df["Adj Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Adj Close"].shift())
    low_close = np.abs(df["Low"] - df["Adj Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    df["ATR_Pct"] = df["ATR"] / df["Adj Close"].replace(0, np.nan) * 100

    # Volume indicators
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"].replace(0, np.nan)

    return df

# =============================================================================
# ADVANCED ANALYTICS ENGINE
# =============================================================================

class AdvancedAnalytics:
    """Comprehensive analytics engine with institutional-grade methods"""

    def __init__(self):
        pass

    @staticmethod
    def calculate_performance_metrics(returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        returns = returns.dropna()
        if len(returns) < 60:
            return {}

        # Basic metrics (improved annualization: geometric CAGR)
        cumulative = (1 + returns).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        years = max(len(returns) / 252.0, 1e-9)
        annual_return = ((cumulative.iloc[-1]) ** (1 / years) - 1) * 100

        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if downside_returns.std() > 0 else 0

        # Maximum drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Gain/Loss metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        avg_gain = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')

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
            "win_rate": win_rate,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
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
        """Calculate rolling beta with diagnostics (fixed: statsmodels gating + alignment)."""
        if not STATSMODELS_AVAILABLE:
            return pd.DataFrame()

        aligned = pd.concat(
            [asset_returns.rename("asset"), benchmark_returns.rename("benchmark")],
            axis=1,
            join="inner"
        ).dropna()

        if len(aligned) < window:
            return pd.DataFrame()

        results = []
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i - window:i]

            # OLS regression for beta
            X = sm.add_constant(window_data["benchmark"].values)
            y = window_data["asset"].values

            try:
                model = sm.OLS(y, X).fit()
                beta = model.params[1]
                r_squared = model.rsquared
                beta_se = model.bse[1] if len(model.bse) > 1 else np.nan
                beta_t = model.tvalues[1] if len(model.tvalues) > 1 else np.nan

                results.append({
                    "date": aligned.index[i],
                    "beta": beta,
                    "r_squared": r_squared,
                    "beta_se": beta_se,
                    "beta_t": beta_t,
                    "p_value": model.pvalues[1] if len(model.pvalues) > 1 else np.nan
                })
            except Exception:
                continue

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

                            # Calculate diagnostics
                            std_resid = fit.resid / fit.conditional_volatility

                            # Ljung-Box test on squared residuals
                            try:
                                if STATSMODELS_AVAILABLE:
                                    lb_test = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
                                    lb_pval = float(lb_test["lb_pvalue"].iloc[0])
                                else:
                                    lb_pval = 1.0
                            except Exception:
                                lb_pval = 1.0

                            # Calculate model score
                            score = (
                                fit.aic * 0.3 +
                                fit.bic * 0.3 +
                                (1 - lb_pval) * 100 * 0.2 +
                                (fit.convergence_flag == 0) * 20
                            )

                            results.append({
                                "garch_type": garch_type,
                                "p": p,
                                "q": q,
                                "distribution": dist,
                                "aic": fit.aic,
                                "bic": fit.bic,
                                "converged": fit.convergence_flag == 0,
                                "score": score,
                                "lb_pval": lb_pval,
                                "log_likelihood": fit.loglikelihood
                            })

                        except Exception:
                            continue

        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(["converged", "score"], ascending=[False, True])
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
            best_model = grid_results.iloc[0]
        else:
            best_model = converged.iloc[0]

        return {
            "garch_type": best_model["garch_type"],
            "p": int(best_model["p"]),
            "q": int(best_model["q"]),
            "distribution": best_model["distribution"],
            "aic": float(best_model["aic"]),
            "bic": float(best_model["bic"]),
            "score": float(best_model["score"])
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

            # Diagnostics
            std_resid = fit.resid / fit.conditional_volatility

            return {
                "model": fit,
                "cond_volatility": cond_vol,
                "forecast_volatility": forecast_vol,
                "std_residuals": std_resid,
                "converged": fit.convergence_flag == 0,
                "aic": fit.aic,
                "bic": fit.bic,
                "params": dict(fit.params),
                "specification": champion_spec
            }

        except Exception as e:
            st.warning(f"GARCH fitting failed: {str(e)[:120]}")
            return None

    @staticmethod
    def detect_market_regimes(returns: pd.Series, volatility: pd.Series,
                              n_states: int = 3) -> Optional[Dict[str, Any]]:
        """Detect market regimes using HMM (fixed alignment)."""
        if not HMM_AVAILABLE or len(returns) < 260:
            return None

        try:
            # FIX: preserve index alignment (do NOT use .values with independent indices)
            features = pd.concat(
                [returns.rename("returns"), volatility.rename("volatility")],
                axis=1,
                join="inner"
            ).dropna()

            if len(features) < 260:
                return None

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features.values)

            # Fit HMM
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=500,
                random_state=42
            )
            model.fit(X_scaled)

            # Predict states as a Series aligned to features index
            states = pd.Series(model.predict(X_scaled), index=features.index, name="state")

            # Calculate regime statistics
            regime_stats = []
            for state in range(n_states):
                mask = (states == state)
                regime_returns = returns.reindex(states.index)[mask].dropna()

                if len(regime_returns) > 10:
                    regime_stats.append({
                        "regime": state,
                        "frequency": len(regime_returns) / len(states),
                        "mean_return": regime_returns.mean() * 100,
                        "volatility": regime_returns.std() * np.sqrt(252) * 100,
                        "sharpe": (regime_returns.mean() / regime_returns.std()) * np.sqrt(252)
                                  if regime_returns.std() > 0 else 0,
                        "var_95": np.percentile(regime_returns, 5) * 100
                    })

            # Label regimes
            if regime_stats:
                stats_df = pd.DataFrame(regime_stats).sort_values("mean_return")
                labels = {}
                for i, (_, row) in enumerate(stats_df.iterrows()):
                    if i == 0:
                        labels[int(row["regime"])] = "Risk-Off"
                    elif i == len(stats_df) - 1:
                        labels[int(row["regime"])] = "Risk-On"
                    else:
                        labels[int(row["regime"])] = "Neutral"
            else:
                labels = {}

            return {
                "states": states,  # pd.Series
                "regime_stats": regime_stats,
                "regime_labels": labels,
                "model": model,
                "features": features
            }

        except Exception as e:
            st.warning(f"Regime detection failed: {str(e)[:120]}")
            return None

    @staticmethod
    def calculate_portfolio_metrics(returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        if returns_df.empty or len(returns_df) < 60:
            return {}

        portfolio_returns = returns_df.values @ weights
        metrics = AdvancedAnalytics.calculate_performance_metrics(pd.Series(portfolio_returns, index=returns_df.index))

        metrics["weights"] = dict(zip(returns_df.columns, weights))
        metrics["num_assets"] = len(weights)

        return metrics

    @staticmethod
    def calculate_risk_contributions(returns_df: pd.DataFrame, weights: np.ndarray,
                                     confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate ES risk contributions"""
        if returns_df.empty:
            return pd.DataFrame()

        weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)
        portfolio_returns = returns_df.values @ weights

        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        tail_mask = portfolio_returns <= var

        if tail_mask.sum() < 10:
            return pd.DataFrame()

        contributions = returns_df.values[tail_mask] * weights
        es_contributions = contributions.mean(axis=0)

        total_es = np.sum(np.abs(es_contributions))
        relative_contributions = np.abs(es_contributions) / total_es if total_es > 0 else np.zeros_like(es_contributions)

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
        idx = returns.index[window:]

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window:i]

            if var_method == "historical":
                var = np.percentile(window_returns, (1 - confidence_level) * 100)
            else:  # normal
                var = window_returns.mean() + window_returns.std() * stats.norm.ppf(1 - confidence_level)

            var_series.append(var)
            hits.append(returns.iloc[i] < var)

        hits_array = np.array(hits, dtype=bool)
        n = len(hits_array)
        x = hits_array.sum()
        expected = (1 - confidence_level) * n

        # Kupiec POF test
        if x > 0 and n - x > 0:
            p_hat = x / n
            p = 1 - confidence_level

            L0 = (1 - p) ** (n - x) * p ** x
            L1 = (1 - p_hat) ** (n - x) * p_hat ** x

            LR = -2 * np.log(L0 / L1)
            kupiec_pval = 1 - stats.chi2.cdf(LR, 1)
        else:
            LR = 0
            kupiec_pval = 1.0

        # Christoffersen independence test
        if n >= 2:
            transitions = np.zeros((2, 2))
            for i in range(1, n):
                prev = int(hits_array[i - 1])
                curr = int(hits_array[i])
                transitions[prev, curr] += 1

            n00, n01 = transitions[0, 0], transitions[0, 1]
            n10, n11 = transitions[1, 0], transitions[1, 1]

            if n00 + n01 > 0 and n10 + n11 > 0:
                p01 = n01 / (n00 + n01)
                p11 = n11 / (n10 + n11)
                p1 = (n01 + n11) / n

                L0 = (1 - p1) ** (n00 + n10) * p1 ** (n01 + n11)
                L1 = (1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11

                LR_ind = -2 * np.log(L0 / L1)
                christoffersen_pval = 1 - stats.chi2.cdf(LR_ind, 1)
            else:
                LR_ind = 0
                christoffersen_pval = 1.0
        else:
            LR_ind = 0
            christoffersen_pval = 1.0

        # Return as Series for safe plotting
        var_s = pd.Series(var_series, index=idx, name="VaR")
        hits_s = pd.Series(hits_array, index=idx, name="Breach")

        return {
            "observations": n,
            "breaches": int(x),
            "breach_rate": x / n * 100 if n > 0 else np.nan,
            "expected_breaches": expected,
            "kupiec_stat": LR,
            "kupiec_pval": kupiec_pval,
            "christoffersen_stat": LR_ind,
            "christoffersen_pval": christoffersen_pval,
            "var_series": var_s,
            "hits": hits_s
        }


    # -------------------------------------------------------------------------
    # ADVANCED VaR / CVaR / ES (Absolute + Relative) - Institutional Extensions
    # -------------------------------------------------------------------------
    @staticmethod
    def _compound_horizon_returns(returns: pd.Series, horizon: int = 1) -> pd.Series:
        """Convert daily returns into horizon (h-day) compounded returns."""
        r = returns.dropna()
        if horizon <= 1:
            return r
        hr = (1.0 + r).rolling(horizon).apply(np.prod, raw=True) - 1.0
        return hr.dropna()

    @staticmethod
    def _var_es_gaussian(mu: float, sigma: float, p: float) -> Tuple[float, float]:
        """Gaussian VaR/ES for left tail probability p."""
        z = stats.norm.ppf(p)
        var = mu + sigma * z
        es = mu - sigma * stats.norm.pdf(z) / max(p, 1e-12)
        return float(var), float(es)

    @staticmethod
    def _var_es_student_t(df: float, loc: float, scale: float, p: float) -> Tuple[float, float]:
        """Student-t VaR/ES for left tail probability p (df>1)."""
        q = stats.t.ppf(p, df, loc=loc, scale=scale)
        t_std = stats.t.ppf(p, df)
        if df <= 1:
            return float(q), float(q)
        es_std = -((df + t_std ** 2) / (df - 1.0)) * (stats.t.pdf(t_std, df) / max(p, 1e-12))
        es = loc + scale * es_std
        return float(q), float(es)

    @staticmethod
    def compute_var_cvar_es(returns: pd.Series,
                            confidence_level: float = 0.95,
                            method: str = "historical",
                            horizon: int = 1) -> Dict[str, float]:
        """Compute VaR / CVaR / ES for a return series.
        - VaR: quantile of returns (left tail)
        - CVaR/ES: conditional tail expectation (mean of returns below VaR threshold)

        Methods:
          - historical
          - gaussian (parametric normal)
          - student_t (parametric Student-t)
          - cornish_fisher (non-normal adjustment; ES is an approximation)
        """
        r = AdvancedAnalytics._compound_horizon_returns(returns, horizon=horizon)
        if r.empty or len(r) < 60:
            return {"var": np.nan, "cvar": np.nan, "es": np.nan}

        alpha = float(confidence_level)
        p = 1.0 - alpha
        p = min(max(p, 1e-6), 0.5)

        method = (method or "historical").lower().strip()
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if float(r.std(ddof=1)) > 0 else 1e-12

        if method == "historical":
            var = float(np.percentile(r.values, p * 100.0))
            tail = r[r <= var]
            es = float(tail.mean()) if len(tail) > 0 else float(var)
            cvar = es

        elif method in ("normal", "gaussian", "parametric"):
            var, es = AdvancedAnalytics._var_es_gaussian(mu, sigma, p)
            cvar = es

        elif method in ("t", "student_t", "student-t"):
            try:
                df_hat, loc_hat, scale_hat = stats.t.fit(r.values)
                df_hat = float(max(df_hat, 2.05))
                var, es = AdvancedAnalytics._var_es_student_t(df_hat, float(loc_hat), float(scale_hat), p)
                cvar = es
            except Exception:
                var, es = AdvancedAnalytics._var_es_gaussian(mu, sigma, p)
                cvar = es

        elif method in ("cornish_fisher", "cornish-fisher", "cf"):
            s = float(stats.skew(r.values, bias=False))
            k = float(stats.kurtosis(r.values, fisher=True, bias=False))
            z = float(stats.norm.ppf(p))
            z_cf = (z
                    + (1.0 / 6.0) * (z ** 2 - 1.0) * s
                    + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * k
                    - (1.0 / 36.0) * (2.0 * z ** 3 - 5.0 * z) * (s ** 2))
            var = mu + sigma * z_cf
            es = mu - sigma * stats.norm.pdf(z_cf) / max(p, 1e-12)  # approximation
            cvar = es

        else:
            var = float(np.percentile(r.values, p * 100.0))
            tail = r[r <= var]
            es = float(tail.mean()) if len(tail) > 0 else float(var)
            cvar = es

        return {"var": float(var), "cvar": float(cvar), "es": float(es)}

    @staticmethod
    def build_var_cvar_es_table(asset_returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                methods: Optional[List[str]] = None,
                                confidence_levels: Optional[List[float]] = None,
                                horizons: Optional[List[int]] = None) -> pd.DataFrame:
        """Create an institutional table for Absolute and Relative VaR/CVaR/ES."""
        methods = methods or ["historical", "gaussian", "student_t", "cornish_fisher"]
        confidence_levels = confidence_levels or [0.95, 0.99]
        horizons = horizons or [1]

        asset_r = asset_returns.dropna()
        bench_r = benchmark_returns.dropna() if isinstance(benchmark_returns, pd.Series) else None

        rel_r = None
        if bench_r is not None and not bench_r.empty:
            aligned = pd.concat([asset_r.rename("asset"), bench_r.rename("bench")], axis=1, join="inner").dropna()
            if not aligned.empty:
                asset_r = aligned["asset"]
                rel_r = aligned["asset"] - aligned["bench"]

        rows = []
        for h in horizons:
            for conf in confidence_levels:
                for mth in methods:
                    out_abs = AdvancedAnalytics.compute_var_cvar_es(asset_r, conf, mth, horizon=h)
                    rows.append({
                        "Type": "Absolute",
                        "Method": mth,
                        "Confidence": conf,
                        "Horizon_Days": h,
                        "VaR": out_abs["var"] * 100.0,
                        "CVaR": out_abs["cvar"] * 100.0,
                        "ES": out_abs["es"] * 100.0
                    })

                    if rel_r is not None and (not rel_r.empty):
                        out_rel = AdvancedAnalytics.compute_var_cvar_es(rel_r, conf, mth, horizon=h)
                        rows.append({
                            "Type": "Relative (Asset - Gold ETF)",
                            "Method": mth,
                            "Confidence": conf,
                            "Horizon_Days": h,
                            "VaR": out_rel["var"] * 100.0,
                            "CVaR": out_rel["cvar"] * 100.0,
                            "ES": out_rel["es"] * 100.0
                        })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["Type", "Horizon_Days", "Confidence", "Method"]).reset_index(drop=True)
        return df

# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

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

        colors = [self.colors["success"] if close >= open_ else self.colors["danger"]
                  for close, open_ in zip(df["Adj Close"].fillna(method="ffill"), df["Open"].fillna(method="ffill"))]

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
                                forecast_vol: Union[pd.Series, np.ndarray, None] = None,
                                title: str = "Volatility Analysis") -> go.Figure:
        """Create volatility analysis chart"""
        fig = go.Figure()

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

        if cond_vol is not None and isinstance(cond_vol, pd.Series) and not cond_vol.empty:
            fig.add_trace(go.Scatter(
                x=cond_vol.index,
                y=cond_vol * 100,
                name="GARCH Conditional Vol",
                line=dict(color=self.colors["primary"], width=2)
            ))

        if forecast_vol is not None:
            if isinstance(forecast_vol, pd.Series):
                fv = forecast_vol.values
                horizon = len(forecast_vol)
            else:
                fv = np.array(forecast_vol)
                horizon = len(fv)

            forecast_dates = pd.date_range(
                start=returns.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=fv * 100,
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

    def create_regime_chart(self, price: pd.Series, regimes: Union[np.ndarray, pd.Series],
                            regime_labels: Dict[int, str], title: str) -> go.Figure:
        """Create regime visualization chart (supports Series-aligned regimes)."""
        fig = go.Figure()

        # Align regimes to price index if possible
        if isinstance(regimes, pd.Series):
            common_idx = price.index.intersection(regimes.index)
            price_aligned = price.reindex(common_idx)
            regimes_aligned = regimes.reindex(common_idx)
        else:
            price_aligned = price
            regimes_aligned = pd.Series(regimes, index=price.index[:len(regimes)])

        fig.add_trace(go.Scatter(
            x=price_aligned.index,
            y=price_aligned.values,
            name="Price",
            line=dict(color="gray", width=1),
            opacity=0.7
        ))

        unique_regimes = np.unique(regimes_aligned.dropna().values.astype(int))
        colors = [self.colors["danger"], self.colors["warning"], self.colors["success"]]

        for i, regime in enumerate(unique_regimes):
            mask = regimes_aligned == regime
            regime_dates = price_aligned.index[mask]
            regime_prices = price_aligned.values[mask]

            label = regime_labels.get(int(regime), f"Regime {int(regime)}")
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=regime_dates,
                y=regime_prices,
                mode="markers",
                name=label,
                marker=dict(size=6, color=color),
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

    def create_backtest_chart(self, returns: pd.Series, var_series: Union[List[float], pd.Series],
                              hits: Union[np.ndarray, pd.Series], title: str) -> go.Figure:
        """Create VaR backtest visualization (FIX: safe Series/array handling)."""
        fig = go.Figure()

        if isinstance(var_series, pd.Series):
            var_s = var_series.dropna()
            ret_s = returns.reindex(var_s.index).dropna()
            x = var_s.index
            y_ret = ret_s.values * 100
            y_var = var_s.values * 100
            hits_s = hits.reindex(var_s.index) if isinstance(hits, pd.Series) else pd.Series(hits, index=x)
            hits_s = hits_s.astype(bool).reindex(x).fillna(False)
        else:
            var_arr = np.array(var_series, dtype=float)
            x = returns.index[-len(var_arr):]
            y_ret = returns.values[-len(var_arr):] * 100
            y_var = var_arr * 100
            hits_s = pd.Series(hits, index=x).astype(bool)

        fig.add_trace(go.Scatter(
            x=x,
            y=y_ret,
            name="Returns",
            line=dict(width=1)
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y_var,
            name="VaR",
            line=dict(color=self.colors["danger"], width=2, dash="dash")
        ))

        breach_dates = hits_s.index[hits_s.values]
        breach_returns = pd.Series(y_ret, index=x).reindex(breach_dates)

        fig.add_trace(go.Scatter(
            x=breach_dates,
            y=breach_returns.values,
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


    def create_advanced_backtest_chart(self,
                                       returns: pd.Series,
                                       var_series: Union[List[float], pd.Series],
                                       hits: Union[np.ndarray, pd.Series],
                                       title: str,
                                       expected_rate: Optional[float] = None) -> go.Figure:
        """Advanced backtesting visualization:
        1) Returns with VaR and breach markers
        2) Hit sequence (0/1) bars
        3) Rolling breach rate vs expected
        """
        if isinstance(var_series, pd.Series):
            var_s = var_series.dropna()
            ret_s = returns.reindex(var_s.index).dropna()
            x = var_s.index
            hits_s = hits.reindex(x) if isinstance(hits, pd.Series) else pd.Series(hits, index=x)
            hits_s = hits_s.astype(bool).reindex(x).fillna(False)
        else:
            var_arr = np.array(var_series, dtype=float)
            start_idx = len(returns) - len(var_arr)
            start_idx = max(start_idx, 0)
            x = returns.index[start_idx:]
            var_s = pd.Series(var_arr, index=x).dropna()
            ret_s = returns.reindex(var_s.index).dropna()
            hits_s = hits.reindex(var_s.index) if isinstance(hits, pd.Series) else pd.Series(hits, index=var_s.index)
            hits_s = hits_s.astype(bool).reindex(var_s.index).fillna(False)

        df = pd.DataFrame({"ret": ret_s, "var": var_s, "hit": hits_s}).dropna()
        if df.empty:
            return go.Figure()

        if expected_rate is None:
            expected_rate = max(min(float(df["hit"].mean()), 0.5), 1e-6)

        roll_win = max(min(60, len(df)), 20)
        df["roll_hit_rate"] = df["hit"].rolling(roll_win).mean() * 100.0

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.20, 0.25],
            subplot_titles=(
                "Returns & VaR (Breaches highlighted)",
                "Hit Sequence (1 = breach)",
                f"Rolling Breach Rate ({roll_win}D) vs Expected"
            )
        )

        fig.add_trace(go.Scatter(
            x=df.index, y=df["ret"] * 100.0, name="Return (%)",
            mode="lines", line=dict(width=1.2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["var"] * 100.0, name="VaR Threshold (%)",
            mode="lines", line=dict(width=1.5, dash="dash")
        ), row=1, col=1)

        breaches = df[df["hit"]]
        if not breaches.empty:
            fig.add_trace(go.Scatter(
                x=breaches.index, y=breaches["ret"] * 100.0,
                name="Breaches", mode="markers",
                marker=dict(size=6, symbol="x")
            ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=df.index,
            y=df["hit"].astype(int),
            name="Hit (0/1)",
            opacity=0.8
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["roll_hit_rate"],
            name="Rolling Breach Rate (%)",
            mode="lines",
            line=dict(width=1.8)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=np.full(len(df), expected_rate * 100.0),
            name="Expected Rate (%)",
            mode="lines",
            line=dict(width=1.2, dash="dot")
        ), row=3, col=1)

        fig.update_layout(
            title=title,
            height=820,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Hit", row=2, col=1, range=[-0.05, 1.25])
        fig.update_yaxes(title_text="Breach Rate (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        return fig

    def create_var_cvar_es_comparison_chart(self, risk_table: pd.DataFrame, title: str) -> go.Figure:
        """Create comparative charts for VaR/CVaR/ES across methods and types."""
        if risk_table is None or risk_table.empty:
            return go.Figure()

        df = risk_table.copy()
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{float(x):.0%}")
        df["Horizon_Days"] = df["Horizon_Days"].astype(int)

        long_df = df.melt(
            id_vars=["Type", "Method", "Confidence", "Horizon_Days"],
            value_vars=["VaR", "CVaR", "ES"],
            var_name="Metric",
            value_name="Value"
        )

        if long_df[["Confidence", "Horizon_Days"]].drop_duplicates().shape[0] > 1:
            first = long_df[["Confidence", "Horizon_Days"]].drop_duplicates().iloc[0]
            long_df = long_df[(long_df["Confidence"] == first["Confidence"]) &
                              (long_df["Horizon_Days"] == first["Horizon_Days"])]

        fig = make_subplots(rows=1, cols=3, shared_yaxes=False,
                            subplot_titles=("VaR (%)", "CVaR (%)", "ES (%)"))

        for j, metric in enumerate(["VaR", "CVaR", "ES"], start=1):
            part = long_df[long_df["Metric"] == metric]
            for t in part["Type"].unique():
                p2 = part[part["Type"] == t]
                fig.add_trace(go.Bar(
                    x=p2["Method"],
                    y=p2["Value"],
                    name=t,
                    showlegend=(j == 1),
                    opacity=0.9
                ), row=1, col=j)
            fig.update_xaxes(title_text="Method", row=1, col=j)

        fig.update_layout(
            title=title,
            height=480,
            template="plotly_white",
            barmode="group",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1)
        )
        return fig

    def create_conditional_vol_band_chart(self,
                                          cond_vol: pd.Series,
                                          band_window: int = 60,
                                          band_level: float = 0.90,
                                          title: str = "Conditional Volatility Bands") -> go.Figure:
        """Non-line visualization of conditional volatility with upper/lower bounds.
        Uses rolling quantile bands and displays conditional volatility as bars (not a line).
        """
        if cond_vol is None:
            return go.Figure()

        v = pd.Series(cond_vol).dropna()
        if v.empty:
            return go.Figure()

        w = int(max(10, min(band_window, len(v))))
        level = float(min(max(band_level, 0.60), 0.99))
        lo_q = (1.0 - level) / 2.0
        hi_q = 1.0 - lo_q

        lower = v.rolling(w).quantile(lo_q)
        upper = v.rolling(w).quantile(hi_q)

        df2 = pd.DataFrame({"vol": v, "lower": lower, "upper": upper}).dropna()
        if df2.empty:
            return go.Figure()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df2.index, y=df2["upper"] * 100.0,
            name="Upper Bound",
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df2.index, y=df2["lower"] * 100.0,
            name="Lower Bound",
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False
        ))

        fig.add_trace(go.Bar(
            x=df2.index,
            y=df2["vol"] * 100.0,
            name="Conditional Vol (%)",
            opacity=0.85
        ))

        fig.update_layout(
            title=title,
            height=520,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Volatility (%)",
            xaxis_title="Date"
        )
        return fig

    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str) -> go.Figure:
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=title,
            height=500,
            template="plotly_white"
        )

        return fig

# =============================================================================
# DASHBOARD CLASS
# =============================================================================

class InstitutionalCommoditiesDashboard:
    """Main dashboard class integrating all components"""

    def __init__(self):
        self.analytics = AdvancedAnalytics()
        self.viz = EnhancedVisualization()

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
            "champion_models": {},
            "relative_backtest_results": {},
            "gold_benchmark_symbol": "GLD"
                }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def display_header(self):
        """Display professional header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; font-size:2.5rem;">🏛️ Institutional Commodities Analytics v4.0</h1>
            <p style="margin:8px 0 0 0; opacity:0.95;">
                Advanced Portfolio Analytics • GARCH Champion Selection • Regime Detection • Stress Testing
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "🟢 Online", "Streamlit Cloud")
        with col2:
            st.metric("Modules", "6", "Active")
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

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date",
                                           datetime.now().date() - timedelta(days=1095))
            with col2:
                end_date = st.date_input("End Date", datetime.now().date())

            if start_date >= end_date:
                st.error("End date must be after start date")
                st.stop()

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Asset Selection")

            selected_assets = []
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category}", expanded=True):
                    for symbol, info in assets.items():
                        if st.checkbox(f"{info['name']} ({symbol})", key=f"asset_{symbol}"):
                            selected_assets.append(symbol)

            st.markdown("### 📈 Benchmarks")
            selected_benchmarks = []
            for benchmark, info in BENCHMARKS.items():
                if st.checkbox(f"{info['name']} ({benchmark})", key=f"bench_{benchmark}"):
                    selected_benchmarks.append(benchmark)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Analytics Settings")

            if ARCH_AVAILABLE:
                st.markdown("**GARCH Configuration**")
                garch_horizon = st.slider("Forecast Horizon", 5, 60, 30, 5)
                p_max = st.slider("ARCH Order (p max)", 1, 3, 2, 1)
                q_max = st.slider("GARCH Order (q max)", 1, 3, 2, 1)

            st.markdown("**Risk Metrics**")
            confidence_levels = st.multiselect(
                "Confidence Levels",
                [0.90, 0.95, 0.99],
                default=[0.95, 0.99]
            )

            backtest_window = st.slider("Backtest Window", 100, 500, 250, 25)

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("📥 Load Market Data", type="primary", use_container_width=True):
                self._load_data(selected_assets, selected_benchmarks, start_date, end_date)

            if st.session_state.data_loaded:
                if st.button("🔄 Clear Cache", type="secondary", use_container_width=True):
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
        """Load data for selected assets and benchmarks (adds diagnostics for failures)."""
        if not assets:
            st.warning("Please select at least one asset")
            return

        with st.spinner("Loading market data..."):
            asset_data = {}
            returns_data = {}
            failed_assets = []
            short_assets = []

            # Progress indicator (institutional UX)
            progress = st.progress(0)
            total = max(len(assets) + len(benchmarks), 1)
            done = 0

            for symbol in assets:
                df = fetch_asset_data(symbol, start_date, end_date)
                if not df.empty and len(df) >= 60 and "Returns" in df.columns:
                    asset_data[symbol] = df
                    returns_data[symbol] = df["Returns"].dropna()
                else:
                    if df.empty:
                        failed_assets.append(symbol)
                    else:
                        short_assets.append(symbol)
                done += 1
                progress.progress(min(done / total, 1.0))

            benchmark_data = {}
            failed_bench = []
            for benchmark in benchmarks:
                df = fetch_asset_data(benchmark, start_date, end_date)
                if not df.empty and len(df) >= 60 and "Returns" in df.columns:
                    benchmark_data[benchmark] = df
                else:
                    failed_bench.append(benchmark)
                done += 1
                progress.progress(min(done / total, 1.0))

            progress.empty()

            if not asset_data:
                st.error("Failed to load any asset data. Try a longer date range or different tickers.")
                if failed_assets:
                    st.warning(f"Failed tickers: {', '.join(failed_assets)}")
                return

            st.session_state.asset_data = asset_data
            st.session_state.returns_data = returns_data
            st.session_state.benchmark_data = benchmark_data
            st.session_state.selected_assets = list(asset_data.keys())
            st.session_state.data_loaded = True

            n_assets = len(asset_data)
            equal_weight = 1.0 / n_assets
            st.session_state.portfolio_weights = {
                asset: equal_weight for asset in asset_data.keys()
            }

            st.success(f"✓ Loaded {len(asset_data)} assets and {len(benchmark_data)} benchmarks")

            # Extra diagnostics (kept lightweight)
            if failed_assets:
                st.warning(f"Assets failed to load (Yahoo symbol/history issue): {', '.join(failed_assets)}")
            if short_assets:
                st.info(f"Assets loaded but too short (<60 obs) for analytics: {', '.join(short_assets)}")
            if failed_bench:
                st.warning(f"Benchmarks failed to load: {', '.join(failed_bench)}")

    def run(self):
        """Main dashboard execution"""
        self.display_header()

        config = self.setup_sidebar()

        if not st.session_state.data_loaded:
            st.info("👈 Please select assets and load data from the sidebar to begin analysis")
            return

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard",
            "🧺 Portfolio",
            "⚡ GARCH",
            "🔄 Regimes",
            "📈 Analytics",
            "📋 Reports"
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

        with tab6:
            self.display_reports()

    def display_dashboard(self, config: Dict[str, Any]):
        """Display main dashboard"""
        st.header("📊 Market Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets Loaded", len(st.session_state.asset_data))
        with col2:
            total_days = (config["end_date"] - config["start_date"]).days
            st.metric("Time Period", f"{total_days} days")
        with col3:
            ret_df_for_corr = pd.DataFrame(st.session_state.returns_data).dropna()
            avg_corr = ret_df_for_corr.corr().mean().mean() if (not ret_df_for_corr.empty and ret_df_for_corr.shape[1] > 1) else 1.0
            st.metric("Avg Correlation", f"{avg_corr:.2%}")
        with col4:
            st.metric("Data Points", len(next(iter(st.session_state.returns_data.values()))))

        st.subheader("📈 Asset Performance Summary")

        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if len(df) > 0 and "Returns" in df.columns:
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
                use_container_width=True
            )

        st.subheader("📊 Correlation Matrix")
        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()

        if not returns_df.empty and len(returns_df.columns) > 1:
            corr_matrix = returns_df.corr()
            fig = self.viz.create_correlation_heatmap(corr_matrix, "Asset Correlations")
            st.plotly_chart(fig, use_container_width=True)

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

    def display_portfolio_analytics(self, config: Dict[str, Any]):
        """Display portfolio analytics"""
        st.header("🧺 Portfolio Analytics")

        returns_df = pd.DataFrame(st.session_state.returns_data).dropna()

        if returns_df.empty or len(returns_df) < 60:
            st.warning("Insufficient data for portfolio analysis")
            return

        assets = returns_df.columns.tolist()

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
                    weight = st.slider(
                        asset,
                        min_value=0.0,
                        max_value=1.0,
                        value=default_weight,
                        step=0.01,
                        key=f"weight_{asset}"
                    )
                    weight_inputs.append(weight)

            weights = np.array(weight_inputs)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

        st.session_state.portfolio_weights = dict(zip(assets, weights))

        portfolio_metrics = self.analytics.calculate_portfolio_metrics(returns_df, weights)

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

        st.subheader("Risk Metrics")

        risk_cols = st.columns(4)
        with risk_cols[0]:
            st.metric("VaR 95%", f"{portfolio_metrics.get('var_95', 0):.2f}%")
        with risk_cols[1]:
            st.metric("CVaR 95%", f"{portfolio_metrics.get('cvar_95', 0):.2f}%")
        with risk_cols[2]:
            st.metric("Sortino Ratio", f"{portfolio_metrics.get('sortino_ratio', 0):.2f}")
        with risk_cols[3]:
            st.metric("Win Rate", f"{portfolio_metrics.get('win_rate', 0):.1f}%")

        st.subheader("Expected Shortfall Risk Contributions")

        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.90, 0.95, 0.99],
            value=0.95
        )

        risk_contributions = self.analytics.calculate_risk_contributions(
            returns_df, weights, confidence_level
        )

        if not risk_contributions.empty:
            st.dataframe(
                risk_contributions.style.format({
                    "Weight": "{:.1f}%",
                    "ES_Contribution": "{:.2f}%",
                    "Relative_Contribution": "{:.1f}%"
                }),
                use_container_width=True
            )

            fig = go.Figure(data=[go.Pie(
                labels=risk_contributions["Asset"],
                values=risk_contributions["Relative_Contribution"],
                hole=0.4
            )])
            fig.update_layout(title="Risk Contribution Breakdown")
            st.plotly_chart(fig, use_container_width=True)

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

        # FIX/ENHANCEMENT: Benchmarks were never in returns_df columns; use benchmark_data Returns.
        for benchmark, bdf in st.session_state.benchmark_data.items():
            if "Returns" in bdf.columns:
                bret = bdf["Returns"].dropna()
                aligned = pd.concat([portfolio_returns.rename("p"), bret.rename("b")], axis=1, join="inner").dropna()
                if not aligned.empty:
                    fig.add_trace(go.Scatter(
                        x=aligned.index,
                        y=(1 + aligned["b"]).cumprod(),
                        name=f"Benchmark: {benchmark}",
                        line=dict(dash="dash")
                    ))

        fig.update_layout(
            title="Portfolio vs Benchmarks",
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Cumulative Return"
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_garch_analysis(self, config: Dict[str, Any]):
        """Display GARCH analysis"""
        st.header("⚡ Advanced GARCH Analysis")

        selected_asset = st.selectbox(
            "Select Asset for GARCH Analysis",
            options=st.session_state.selected_assets,
            key="garch_asset_select"
        )

        if selected_asset not in st.session_state.returns_data:
            return

        returns = st.session_state.returns_data[selected_asset]

        st.subheader("ARCH Effects Test")

        arch_test = self.analytics.test_arch_effects(returns)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARCH Present", "Yes" if arch_test["arch_present"] else "No")
        with col2:
            st.metric("p-value", f"{arch_test['p_value']:.4f}")
        with col3:
            good = arch_test["p_value"] < 0.05
            st.metric("Significant", "✓" if good else "✗",
                      "Suitable for GARCH" if good else "Limited ARCH effects")

        if not arch_test["arch_present"]:
            st.warning("Limited ARCH effects detected. GARCH modeling may not be optimal.")

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

        if selected_asset in st.session_state.garch_results:
            garch_data = st.session_state.garch_results[selected_asset]
            grid_results = garch_data["grid_results"]
            champion = garch_data["champion"]

            if not grid_results.empty:
                st.dataframe(
                    grid_results.style.format({
                        "aic": "{:.1f}",
                        "bic": "{:.1f}",
                        "score": "{:.1f}",
                        "lb_pval": "{:.4f}"
                    }),
                    use_container_width=True
                )

            if champion:
                st.success(f"**Champion Model**: {champion['garch_type']}({champion['p']},{champion['q']}) - {champion['distribution']} distribution")

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

                st.write("**Model Parameters:**")
                params_df = pd.DataFrame.from_dict(fit["params"], orient="index", columns=["Value"])
                st.dataframe(params_df.style.format({"Value": "{:.6f}"}), use_container_width=True)

                st.subheader("Volatility Analysis")

                fig = self.viz.create_volatility_chart(
                    returns,
                    fit["cond_volatility"],
                    pd.Series(fit["forecast_volatility"]),
                    f"{selected_asset} - GARCH Volatility"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Additional (non-line) visualization: Conditional Volatility with Upper/Lower Bounds
                st.subheader("Conditional Volatility Bands (Non-Line View)")

                b1, b2 = st.columns(2)
                with b1:
                    band_window = st.slider("Band Window (days)", 20, 252, 60, 10, key=f"band_window_{selected_asset}")
                with b2:
                    band_level = st.select_slider(
                        "Band Level",
                        options=[0.80, 0.90, 0.95, 0.99],
                        value=0.90,
                        format_func=lambda x: f"{x:.0%}",
                        key=f"band_level_{selected_asset}"
                    )

                fig_band = self.viz.create_conditional_vol_band_chart(
                    pd.Series(fit["cond_volatility"]),
                    band_window=int(band_window),
                    band_level=float(band_level),
                    title=f"{selected_asset} - Conditional Volatility Bands"
                )
                st.plotly_chart(fig_band, use_container_width=True)



        else:
            st.info("Run grid search to select and fit a champion GARCH model.")

    def display_regime_analysis(self, config: Dict[str, Any]):
        """Display regime analysis"""
        st.header("🔄 Market Regime Detection")

        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=st.session_state.selected_assets,
            key="regime_asset_select"
        )

        if selected_asset not in st.session_state.asset_data:
            return

        df = st.session_state.asset_data[selected_asset]
        returns = df["Returns"].dropna()
        volatility = df["Volatility_20D"].dropna()

        st.subheader("Regime Detection Settings")

        col1, col2 = st.columns(2)
        with col1:
            n_states = st.slider("Number of Regimes", 2, 5, 3, 1)
        with col2:
            use_volatility = st.checkbox("Include Volatility Feature", value=True)

        if st.button("🔍 Detect Regimes", type="primary", use_container_width=True):
            with st.spinner("Detecting market regimes..."):
                regime_results = self.analytics.detect_market_regimes(
                    returns,
                    volatility if use_volatility else returns.rolling(20).std() * np.sqrt(252),
                    n_states
                )

                if regime_results:
                    st.session_state.regime_results[selected_asset] = regime_results
                    st.success("Regime detection completed!")

        if selected_asset in st.session_state.regime_results:
            regime_data = st.session_state.regime_results[selected_asset]

            st.subheader("Regime Statistics")

            if regime_data["regime_stats"]:
                stats_df = pd.DataFrame(regime_data["regime_stats"])
                st.dataframe(
                    stats_df.style.format({
                        "frequency": "{:.2%}",
                        "mean_return": "{:.4f}%",
                        "volatility": "{:.2f}%",
                        "sharpe": "{:.2f}",
                        "var_95": "{:.2f}%"
                    }),
                    use_container_width=True
                )

            st.subheader("Regime Visualization")

            # FIX: use aligned index from regimes/features
            states_s = regime_data["states"] if isinstance(regime_data["states"], pd.Series) else pd.Series(regime_data["states"])
            price = df["Adj Close"].reindex(states_s.index).dropna()

            fig = self.viz.create_regime_chart(
                price,
                states_s.reindex(price.index),
                regime_data["regime_labels"],
                f"{selected_asset} - Market Regimes"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Regime-Conditioned Risk Metrics")

            states_s = states_s.reindex(returns.index).dropna()
            aligned_rets = returns.reindex(states_s.index).dropna()
            states_s = states_s.reindex(aligned_rets.index).dropna()

            regime_df = pd.DataFrame({
                "returns": aligned_rets.values,
                "regime": [regime_data["regime_labels"].get(int(s), f"Regime {int(s)}")
                           for s in states_s.values]
            }, index=aligned_rets.index)

            risk_data = []
            for regime in regime_df["regime"].unique():
                regime_returns = regime_df[regime_df["regime"] == regime]["returns"]

                if len(regime_returns) > 20:
                    for conf in config["confidence_levels"]:
                        var = np.percentile(regime_returns, (1 - conf) * 100) * 100
                        cvar = regime_returns[regime_returns <= np.percentile(regime_returns,
                                                                             (1 - conf) * 100)].mean() * 100

                        risk_data.append({
                            "Regime": regime,
                            "Confidence": f"{conf:.0%}",
                            "Observations": len(regime_returns),
                            "VaR": var,
                            "CVaR": cvar
                        })

            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(
                    risk_df.style.format({
                        "VaR": "{:.2f}%",
                        "CVaR": "{:.2f}%"
                    }),
                    use_container_width=True
                )

        else:
            st.info("Run regime detection to analyze market regimes.")

    def display_advanced_analytics(self, config: Dict[str, Any]):
        """Display advanced analytics"""
        st.header("📈 Advanced Analytics")

        st.subheader("✅ VaR Backtesting")

        selected_asset = st.selectbox(
            "Select Asset for Backtesting",
            options=st.session_state.selected_assets,
            key="backtest_asset_select"
        )

        if selected_asset not in st.session_state.returns_data:
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

        if selected_asset in st.session_state.backtest_results:
            results = st.session_state.backtest_results[selected_asset]

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

            st.subheader("Statistical Tests")

            test_cols = st.columns(2)
            with test_cols[0]:
                st.metric("Kupiec POF Test",
                          f"p = {results['kupiec_pval']:.4f}",
                          "Pass" if results['kupiec_pval'] > 0.05 else "Fail")
            with test_cols[1]:
                st.metric("Christoffersen Test",
                          f"p = {results['christoffersen_pval']:.4f}",
                          "Pass" if results['christoffersen_pval'] > 0.05 else "Fail")

            st.subheader("Backtest Visualization")

            fig = self.viz.create_backtest_chart(
                returns,
                results["var_series"],
                results["hits"],
                f"{selected_asset} - {var_method.title()} VaR Backtest ({confidence_level:.0%} Confidence)"
            )

            st.plotly_chart(fig, use_container_width=True)


        # ---------------------------------------------------------------------
        # ADVANCED VaR / CVaR / ES + RELATIVE RISK vs Most Active US Gold ETF
        # ---------------------------------------------------------------------
        st.subheader("🧮 Advanced VaR / CVaR / ES (Absolute + Relative vs Gold ETF)")

        # Select the most active Gold ETF benchmark (by avg $ volume) for the period
        try:
            gold_sym, gold_df = select_most_active_gold_etf(config["start_date"], config["end_date"])
        except Exception:
            gold_sym, gold_df = "GLD", pd.DataFrame()

        gold_returns = pd.Series(dtype=float)
        if isinstance(gold_df, pd.DataFrame) and (not gold_df.empty) and ("Returns" in gold_df.columns):
            gold_returns = gold_df["Returns"].dropna()
        elif "GLD" in st.session_state.benchmark_data and "Returns" in st.session_state.benchmark_data["GLD"].columns:
            gold_sym = "GLD"
            gold_returns = st.session_state.benchmark_data["GLD"]["Returns"].dropna()

        st.session_state.gold_benchmark_symbol = gold_sym
        if not gold_returns.empty:
            st.caption(f"Gold ETF benchmark selected (most active): **{gold_sym}**")
        else:
            st.caption("Gold ETF benchmark unavailable for the selected date range (try extending the range).")

        # Controls
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            adv_methods = st.multiselect(
                "Methods",
                ["historical", "gaussian", "student_t", "cornish_fisher"],
                default=["historical", "gaussian", "cornish_fisher"]
            )
        with c2:
            adv_conf = st.multiselect(
                "Confidence",
                config["confidence_levels"] if config.get("confidence_levels") else [0.95, 0.99],
                default=config["confidence_levels"] if config.get("confidence_levels") else [0.95, 0.99],
                format_func=lambda x: f"{x:.0%}"
            )
        with c3:
            adv_horizon = st.selectbox("Horizon (days)", options=[1, 5, 10, 20], index=0)
        with c4:
            show_table = st.checkbox("Show Risk Table", value=True)

        aligned = None
        if not gold_returns.empty:
            aligned = pd.concat([returns.rename("asset"), gold_returns.rename("gold")], axis=1, join="inner").dropna()

        if show_table:
            if aligned is not None and len(aligned) >= 100:
                risk_df = self.analytics.build_var_cvar_es_table(
                    aligned["asset"],
                    aligned["gold"],
                    methods=adv_methods,
                    confidence_levels=adv_conf,
                    horizons=[int(adv_horizon)]
                )
                if not risk_df.empty:
                    st.dataframe(
                        risk_df.style.format({
                            "Confidence": "{:.0%}",
                            "VaR": "{:.2f}%",
                            "CVaR": "{:.2f}%",
                            "ES": "{:.2f}%"
                        }),
                        use_container_width=True
                    )

                    fig_risk = self.viz.create_var_cvar_es_comparison_chart(
                        risk_df,
                        title=f"{selected_asset}: VaR/CVaR/ES Comparison (H={adv_horizon}D)"
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("Risk table could not be computed (insufficient data after alignment).")
            else:
                st.info("Relative risk requires sufficient overlapping data between the asset and the Gold ETF benchmark.")

        # ---------------------------------------------------------------------
        # ADVANCED BACKTESTING CHART (Absolute) + Optional Relative Backtesting
        # ---------------------------------------------------------------------
        if selected_asset in st.session_state.backtest_results:
            results = st.session_state.backtest_results[selected_asset]
            if results and ("var_series" in results) and ("hits" in results):
                expected_rate = (results["expected_breaches"] / max(results["observations"], 1)) if results.get("observations") else None

                st.subheader("🧪 Advanced Backtesting Visualization (Absolute)")
                fig_adv_bt = self.viz.create_advanced_backtest_chart(
                    returns,
                    results["var_series"],
                    results["hits"],
                    title=f"{selected_asset} - Advanced VaR Backtest",
                    expected_rate=expected_rate
                )
                st.plotly_chart(fig_adv_bt, use_container_width=True)

                # Basel-style traffic light (generic binomial quantiles)
                try:
                    n = int(results.get("observations", 0))
                    x = int(results.get("breaches", 0))
                    p = float(expected_rate) if expected_rate is not None else 0.01
                    green_max = int(stats.binom.ppf(0.95, n, p))
                    yellow_max = int(stats.binom.ppf(0.99, n, p))
                    zone = "🟢 Green" if x <= green_max else ("🟠 Yellow" if x <= yellow_max else "🔴 Red")
                    st.info(f"Backtest Traffic Light (binomial quantiles): **{zone}**  | breaches={x}, green≤{green_max}, yellow≤{yellow_max}")
                except Exception:
                    pass

        st.subheader("🪙 Relative VaR Backtesting vs Gold ETF (Asset - Gold ETF)")

        if aligned is not None and len(aligned) >= max(250, int(config.get("backtest_window", 250)) + 50):
            rel_series = (aligned["asset"] - aligned["gold"]).dropna()

            r1, r2, r3 = st.columns(3)
            with r1:
                rel_method = st.selectbox("Relative VaR Method", ["historical", "normal"], key="rel_var_method")
            with r2:
                rel_conf = st.selectbox(
                    "Relative Confidence",
                    config["confidence_levels"],
                    index=0,
                    key="rel_conf_level",
                    format_func=lambda x: f"{x:.0%}"
                )
            with r3:
                rel_window = st.number_input(
                    "Relative Window Size",
                    min_value=100,
                    max_value=500,
                    value=int(config.get("backtest_window", 250)),
                    key="rel_backtest_window"
                )

            if st.button("📊 Run Relative Backtest", type="secondary", use_container_width=True):
                with st.spinner("Running relative VaR backtest..."):
                    rel_bt = self.analytics.var_backtest(rel_series, rel_method, rel_conf, int(rel_window))
                    if rel_bt:
                        st.session_state.relative_backtest_results[selected_asset] = rel_bt
                        st.success("Relative backtest completed!")

            if selected_asset in st.session_state.relative_backtest_results:
                rel_res = st.session_state.relative_backtest_results[selected_asset]
                if rel_res:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Observations", f"{rel_res['observations']:,}")
                    with c2:
                        st.metric("Breaches", f"{rel_res['breaches']:,}")
                    with c3:
                        st.metric("Breach Rate", f"{rel_res['breach_rate']:.2f}%")
                    with c4:
                        st.metric("Expected Breaches", f"{rel_res['expected_breaches']:.1f}")

                    fig_rel_adv = self.viz.create_advanced_backtest_chart(
                        rel_series,
                        rel_res["var_series"],
                        rel_res["hits"],
                        title=f"{selected_asset} - Relative VaR Backtest vs {gold_sym}",
                        expected_rate=(rel_res["expected_breaches"] / max(rel_res["observations"], 1))
                    )
                    st.plotly_chart(fig_rel_adv, use_container_width=True)
        else:
            st.info("Relative backtesting needs a longer overlapping history with the Gold ETF benchmark.")

        st.subheader("📉 Rolling Beta Analysis")

        if st.session_state.benchmark_data:
            benchmark = st.selectbox(
                "Select Benchmark",
                options=list(st.session_state.benchmark_data.keys()),
                key="beta_benchmark_select"
            )

            if benchmark in st.session_state.benchmark_data:
                benchmark_returns = st.session_state.benchmark_data[benchmark]["Returns"].dropna()

                rolling_beta = self.analytics.rolling_beta_analysis(
                    returns,
                    benchmark_returns,
                    window=60
                )

                if not rolling_beta.empty:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=rolling_beta.index,
                        y=rolling_beta["beta"],
                        name="Beta",
                        line=dict(width=2)
                    ))

                    fig.add_hline(y=1, line_dash="dash", line_color="gray")
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")

                    fig.update_layout(
                        title=f"{selected_asset} vs {benchmark} - Rolling Beta",
                        height=400,
                        template="plotly_white",
                        hovermode="x unified",
                        yaxis_title="Beta",
                        xaxis_title="Date"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Load benchmark data to enable beta analysis.")

    def display_reports(self):
        """Display reporting interface"""
        st.header("📋 Institutional Reports")

        st.subheader("Snapshot Report")

        if st.button("📄 Generate Snapshot", use_container_width=True):
            snapshot = self._generate_snapshot()

            st.json(snapshot)

            st.download_button(
                label="📥 Download JSON Snapshot",
                data=pd.Series(snapshot).to_json(indent=2),
                file_name=f"commodities_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        st.subheader("Portfolio Tear Sheet")

        if st.button("📊 Generate Tear Sheet", use_container_width=True):
            tear_sheet = self._generate_tear_sheet()
            st.components.v1.html(tear_sheet, height=600, scrolling=True)

            st.download_button(
                label="📥 Download HTML Report",
                data=tear_sheet,
                file_name=f"portfolio_tear_sheet_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )

    def _generate_snapshot(self) -> Dict[str, Any]:
        """Generate snapshot report of current analysis"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "assets_loaded": list(st.session_state.asset_data.keys()),
            "benchmarks_loaded": list(st.session_state.benchmark_data.keys()),
            "data_points": len(next(iter(st.session_state.returns_data.values())))
            if st.session_state.returns_data else 0,
            "portfolio_weights": st.session_state.portfolio_weights,
            "champion_models": st.session_state.get("champion_models", {}),
            "system_info": {
                "arch_available": ARCH_AVAILABLE,
                "hmm_available": HMM_AVAILABLE,
                "statsmodels_available": STATSMODELS_AVAILABLE
            }
        }

        return snapshot

    def _generate_tear_sheet(self) -> str:
        """Generate HTML tear sheet"""
        portfolio_metrics = {}
        if st.session_state.portfolio_weights and st.session_state.returns_data:
            returns_df = pd.DataFrame(st.session_state.returns_data).dropna()
            weights = np.array(list(st.session_state.portfolio_weights.values()))
            portfolio_metrics = self.analytics.calculate_portfolio_metrics(returns_df, weights)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Tear Sheet</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
                         color: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; }}
                .section {{ margin: 25px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                              gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 6px;
                              box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Institutional Commodities Portfolio Tear Sheet</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>Portfolio Overview</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Assets</div>
                        <div class="metric-value">{len(st.session_state.asset_data)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if portfolio_metrics.get('total_return', 0) > 0 else 'negative'}">
                            {portfolio_metrics.get('total_return', 0):.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{portfolio_metrics.get('sharpe_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{portfolio_metrics.get('max_drawdown', 0):.2f}%</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Risk Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Annual Volatility</div>
                        <div class="metric-value">{portfolio_metrics.get('annual_volatility', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">VaR 95%</div>
                        <div class="metric-value negative">{portfolio_metrics.get('var_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">CVaR 95%</div>
                        <div class="metric-value negative">{portfolio_metrics.get('cvar_95', 0):.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{portfolio_metrics.get('sortino_ratio', 0):.2f}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Disclaimer</h2>
                <p><em>This report is generated for informational purposes only.
                Past performance is not indicative of future results.
                Consult with a qualified financial advisor before making investment decisions.
                Data source: Yahoo Finance. Generated by Institutional Commodities Analytics Platform v4.0.</em></p>
            </div>
        </body>
        </html>
        """

        return html_content

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main application entry point"""
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    dashboard = InstitutionalCommoditiesDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
