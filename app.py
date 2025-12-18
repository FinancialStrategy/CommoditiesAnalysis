"""
🏛️ Institutional Commodities Analytics Platform v5.0
Superior Advanced GARCH Analysis with Comprehensive Volatility Charts
Historical Realized Volatility + GARCH Conditional Volatility + Prediction Bands
20-day & 60-day Volatility Comparisons
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
import plotly.express as px
from scipy import stats
import time
import json

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Institutional Commodities Platform v5.0", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ============================================================================
# COMMODITIES UNIVERSE
# ============================================================================

COMMODITIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Precious Metals": {
        "GC=F": {"name": "Gold Futures", "category": "Precious", "exchange": "COMEX"},
        "SI=F": {"name": "Silver Futures", "category": "Precious", "exchange": "COMEX"},
        "PL=F": {"name": "Platinum Futures", "category": "Precious", "exchange": "NYMEX"},
        "PA=F": {"name": "Palladium Futures", "category": "Precious", "exchange": "NYMEX"}
    },
    "Industrial Metals": {
        "HG=F": {"name": "Copper Futures", "category": "Industrial", "exchange": "COMEX"},
        "ALI=F": {"name": "Aluminum Futures", "category": "Industrial", "exchange": "COMEX"}
    },
    "Energy": {
        "CL=F": {"name": "Crude Oil WTI", "category": "Energy", "exchange": "NYMEX"},
        "BZ=F": {"name": "Brent Crude", "category": "Energy", "exchange": "ICE"},
        "NG=F": {"name": "Natural Gas", "category": "Energy", "exchange": "NYMEX"}
    },
    "Agriculture": {
        "ZC=F": {"name": "Corn Futures", "category": "Agriculture", "exchange": "CBOT"},
        "ZW=F": {"name": "Wheat Futures", "category": "Agriculture", "exchange": "CBOT"},
        "ZS=F": {"name": "Soybean Futures", "category": "Agriculture", "exchange": "CBOT"}
    }
}

BENCHMARKS = {
    "^GSPC": {"name": "S&P 500 Index", "type": "equity", "description": "US Large Cap"},
    "DX-Y.NYB": {"name": "US Dollar Index", "type": "fx", "description": "USD vs Basket"},
    "TLT": {"name": "20+ Year Treasury", "type": "fixed_income", "description": "Long Bonds"},
    "GLD": {"name": "Gold ETF", "type": "commodity", "description": "Gold Proxy"}
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
    .volatility-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
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
    .garch-params {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #26d0ce;
        margin: 1rem 0;
    }
    .volatility-compare {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
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
PYMC_AVAILABLE = False

try:
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    import statsmodels.api as sm
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
# ENHANCED DATA LOADING ENGINE WITH VOLATILITY PREPARATION
# ============================================================================

class InstitutionalDataLoader:
    """Professional data loading engine with comprehensive volatility preparation"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_institutional_data(_self, symbol: str, start_date: datetime, end_date: datetime, 
                                max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch institutional-grade data with comprehensive volatility metrics
        """
        for attempt in range(max_retries):
            try:
                # Format dates for Yahoo Finance
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Download with institutional parameters
                df = yf.download(
                    symbol,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    timeout=45,
                    interval="1d"
                )
                
                if df is None or df.empty:
                    # Try alternative method
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_str, end=end_str, interval="1d")
                    if hist is not None and not hist.empty:
                        df = hist
                    else:
                        if attempt == max_retries - 1:
                            return None
                        time.sleep(3)
                        continue
                
                # Standardize column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Ensure required columns
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Use Close if Adj Close missing
                if "Adj Close" not in df.columns:
                    df["Adj Close"] = df["Close"]
                
                # Process dates
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Sort and clean
                df = df.sort_index()
                df = df.dropna(subset=["Adj Close"])
                
                if len(df) < 30:
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2)
                    continue
                
                # Calculate comprehensive features including volatility
                df = InstitutionalDataLoader._calculate_comprehensive_features(df)
                
                return df
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to load {symbol}: {str(e)[:100]}")
                    return None
                time.sleep(2)
        
        return None
    
    @staticmethod
    def _calculate_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical and volatility features"""
        df = df.copy()
        
        # Basic price features
        df["Returns"] = df["Adj Close"].pct_change()
        df["Log_Returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f"SMA_{window}"] = df["Adj Close"].rolling(window=window).mean()
            df[f"EMA_{window}"] = df["Adj Close"].ewm(span=window, adjust=False).mean()
        
        # ==================== VOLATILITY METRICS ====================
        # Historical realized volatility (annualized)
        for window in [5, 10, 20, 60, 120, 252]:
            df[f"RV_{window}D"] = df["Returns"].rolling(window=window).std() * np.sqrt(252) * 100
        
        # Parkinson volatility (high-low based)
        df["HL_Ratio"] = np.log(df["High"] / df["Low"])
        df["Parkinson_Vol"] = df["HL_Ratio"].rolling(20).std() * np.sqrt(252 / (4 * np.log(2))) * 100
        
        # Garman-Klass volatility
        df["GK_Numerator"] = 0.5 * (np.log(df["High"] / df["Low"])) ** 2
        df["GK_Denominator"] = (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"])) ** 2
        df["Garman_Klass_Vol"] = np.sqrt(df["GK_Numerator"] - df["GK_Denominator"]) * np.sqrt(252) * 100
        
        # Rogers-Satchell volatility
        df["RS_HC"] = np.log(df["High"] / df["Close"])
        df["RS_HO"] = np.log(df["High"] / df["Open"])
        df["RS_LC"] = np.log(df["Low"] / df["Close"])
        df["RS_LO"] = np.log(df["Low"] / df["Open"])
        df["Rogers_Satchell_Vol"] = np.sqrt((df["RS_HC"] * df["RS_HO"] + df["RS_LC"] * df["RS_LO"]).rolling(20).mean()) * np.sqrt(252) * 100
        
        # Yang-Zhang volatility (accounts for opening jumps)
        df["YZ_Overnight"] = np.log(df["Open"] / df["Close"].shift(1))
        df["YZ_Intraday"] = np.log(df["Close"] / df["Open"])
        df["Yang_Zhang_Vol"] = np.sqrt(
            df["YZ_Overnight"].rolling(20).var() + 
            0.5 * df["YZ_Intraday"].rolling(20).var() +
            0.5 * df["HL_Ratio"].rolling(20).var()
        ) * np.sqrt(252) * 100
        
        # ==================== TECHNICAL INDICATORS ====================
        # Bollinger Bands
        bb_middle = df["Adj Close"].rolling(window=20).mean()
        bb_std = df["Adj Close"].rolling(window=20).std()
        df["BB_Upper"] = bb_middle + 2 * bb_std
        df["BB_Lower"] = bb_middle - 2 * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_middle * 100
        df["BB_Position"] = (df["Adj Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]) * 100
        
        # RSI
        delta = df["Adj Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df["Adj Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Adj Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
        
        # ATR
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14).mean()
        df["ATR_Pct"] = df["ATR"] / df["Adj Close"] * 100
        
        # Volume indicators
        df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
        df["OBV"] = (np.sign(df["Returns"].fillna(0)) * df["Volume"]).cumsum()
        
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            df[f"Momentum_{period}D"] = df["Adj Close"].pct_change(period) * 100
            df[f"ROC_{period}D"] = ((df["Adj Close"] - df["Adj Close"].shift(period)) / df["Adj Close"].shift(period)) * 100
        
        # Volatility ratio (20D/60D)
        df["Vol_Ratio_20_60"] = df["RV_20D"] / df["RV_60D"].replace(0, np.nan)
        
        # Volatility regimes
        df["Vol_Regime"] = pd.cut(
            df["RV_20D"],
            bins=[0, 15, 30, 50, 100, np.inf],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )
        
        return df
    
    @staticmethod
    def bulk_load_with_volatility(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Bulk load data with comprehensive volatility preparation"""
        data_dict = {}
        failed_symbols = []
        
        progress_placeholder = st.empty()
        progress_bar = st.progress(0, text="Loading institutional data...")
        
        for i, symbol in enumerate(symbols):
            try:
                progress_placeholder.markdown(f'<div class="data-loading">📊 Loading {symbol}...</div>', unsafe_allow_html=True)
                
                df = InstitutionalDataLoader.fetch_institutional_data(None, symbol, start_date, end_date)
                
                if df is not None and not df.empty and len(df) >= 100:
                    data_dict[symbol] = df
                else:
                    failed_symbols.append(symbol)
                
            except Exception as e:
                failed_symbols.append(symbol)
            
            # Update progress
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress, text=f"Loading {symbol} ({i+1}/{len(symbols)})")
        
        progress_bar.empty()
        progress_placeholder.empty()
        
        # Summary
        if data_dict:
            st.success(f"✅ Successfully loaded {len(data_dict)}/{len(symbols)} assets with comprehensive volatility metrics")
        if failed_symbols:
            st.warning(f"⚠️ Failed to load: {', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''}")
        
        return data_dict

# ============================================================================
# SUPERIOR ADVANCED GARCH ENGINE WITH COMPREHENSIVE VOLATILITY ANALYSIS
# ============================================================================

class SuperiorGARCHAnalyzer:
    """Superior GARCH analysis engine with comprehensive volatility modeling"""
    
    def __init__(self):
        self.supported_garch_models = ["GARCH", "EGARCH", "GJR-GARCH", "TARCH", "APARCH"]
        self.supported_distributions = ["normal", "t", "skewt", "ged", "studentst"]
    
    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics"""
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return {}
        
        # Basic volatility metrics
        daily_vol = returns_clean.std() * 100
        annual_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatilities
        rv_20 = returns_clean.rolling(20).std() * np.sqrt(252) * 100
        rv_60 = returns_clean.rolling(60).std() * np.sqrt(252) * 100
        rv_252 = returns_clean.rolling(252).std() * np.sqrt(252) * 100
        
        # Volatility ratios
        vol_ratio_20_60 = rv_20.mean() / rv_60.mean() if rv_60.mean() > 0 else np.nan
        vol_ratio_20_252 = rv_20.mean() / rv_252.mean() if rv_252.mean() > 0 else np.nan
        
        # Volatility persistence
        if len(rv_20) > 50:
            vol_autocorr = rv_20.autocorr(lag=1)
        else:
            vol_autocorr = np.nan
        
        # Volatility clusters
        vol_clusters = self._detect_volatility_clusters(returns_clean)
        
        # Volatility regimes
        vol_regimes = self._identify_volatility_regimes(rv_20)
        
        # Extreme volatility days
        extreme_up = (returns_clean > returns_clean.quantile(0.95)).sum()
        extreme_down = (returns_clean < returns_clean.quantile(0.05)).sum()
        
        return {
            "daily_volatility": daily_vol,
            "annualized_volatility": annual_vol,
            "rv_20_mean": rv_20.mean(),
            "rv_60_mean": rv_60.mean(),
            "rv_252_mean": rv_252.mean(),
            "vol_ratio_20_60": vol_ratio_20_60,
            "vol_ratio_20_252": vol_ratio_20_252,
            "volatility_persistence": vol_autocorr,
            "volatility_clusters": vol_clusters,
            "volatility_regimes": vol_regimes,
            "extreme_up_days": extreme_up,
            "extreme_down_days": extreme_down,
            "volatility_skew": rv_20.skew(),
            "volatility_kurtosis": rv_20.kurtosis()
        }
    
    def _detect_volatility_clusters(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect volatility clustering"""
        squared_returns = returns ** 2
        clusters = []
        
        high_vol_threshold = squared_returns.quantile(0.75)
        in_cluster = False
        cluster_start = None
        
        for i, val in enumerate(squared_returns):
            if val > high_vol_threshold and not in_cluster:
                in_cluster = True
                cluster_start = squared_returns.index[i]
            elif val <= high_vol_threshold and in_cluster:
                in_cluster = False
                clusters.append({
                    "start": cluster_start,
                    "end": squared_returns.index[i-1],
                    "duration": (squared_returns.index[i-1] - cluster_start).days
                })
        
        if clusters:
            avg_duration = np.mean([c["duration"] for c in clusters])
            total_days = sum([c["duration"] for c in clusters])
        else:
            avg_duration = 0
            total_days = 0
        
        return {
            "num_clusters": len(clusters),
            "avg_duration_days": avg_duration,
            "total_cluster_days": total_days,
            "clusters": clusters[:5]  # Return first 5 clusters
        }
    
    def _identify_volatility_regimes(self, volatility_series: pd.Series) -> Dict[str, Any]:
        """Identify volatility regimes"""
        if len(volatility_series) < 100:
            return {}
        
        regimes = []
        thresholds = [
            volatility_series.quantile(0.25),
            volatility_series.quantile(0.5),
            volatility_series.quantile(0.75)
        ]
        
        current_regime = None
        regime_start = volatility_series.index[0]
        
        for i, vol in enumerate(volatility_series):
            if vol < thresholds[0]:
                regime = "Very Low"
            elif vol < thresholds[1]:
                regime = "Low"
            elif vol < thresholds[2]:
                regime = "Medium"
            else:
                regime = "High"
            
            if regime != current_regime:
                if current_regime is not None:
                    regimes.append({
                        "regime": current_regime,
                        "start": regime_start,
                        "end": volatility_series.index[i-1],
                        "duration": (volatility_series.index[i-1] - regime_start).days,
                        "avg_vol": volatility_series.loc[regime_start:volatility_series.index[i-1]].mean()
                    })
                current_regime = regime
                regime_start = volatility_series.index[i]
        
        # Add last regime
        if current_regime is not None:
            regimes.append({
                "regime": current_regime,
                "start": regime_start,
                "end": volatility_series.index[-1],
                "duration": (volatility_series.index[-1] - regime_start).days,
                "avg_vol": volatility_series.loc[regime_start:].mean()
            })
        
        return {
            "regimes": regimes,
            "current_regime": current_regime,
            "current_vol_level": volatility_series.iloc[-1]
        }
    
    def perform_comprehensive_garch_analysis(self, returns: pd.Series, 
                                           garch_type: str = "GARCH",
                                           p: int = 1, q: int = 1,
                                           distribution: str = "t",
                                           forecast_horizon: int = 60) -> Dict[str, Any]:
        """Perform comprehensive GARCH analysis with superior diagnostics"""
        if not ARCH_AVAILABLE or len(returns) < 300:
            return {"error": "Insufficient data or ARCH not available"}
        
        try:
            returns_clean = returns.dropna()
            returns_scaled = returns_clean.values * 100
            
            # Fit GARCH model
            model = arch_model(
                returns_scaled,
                mean="Constant",
                vol=garch_type,
                p=p,
                q=q,
                dist=distribution,
                rescale=False
            )
            
            fit = model.fit(disp="off", show_warning=False, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
            
            # ==================== MODEL RESULTS ====================
            cond_vol = pd.Series(fit.conditional_volatility / 100, index=returns_clean.index)
            std_resid = pd.Series(fit.resid / fit.conditional_volatility, index=returns_clean.index)
            
            # ==================== VOLATILITY FORECAST ====================
            forecast = fit.forecast(horizon=forecast_horizon, reindex=False)
            forecast_variance = forecast.variance.iloc[-1].values
            forecast_vol = np.sqrt(np.maximum(forecast_variance, 0)) / 100
            
            # Generate forecast dates
            forecast_dates = pd.date_range(
                start=returns_clean.index[-1] + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq="B"
            )
            
            # Calculate confidence intervals for forecast
            forecast_upper = forecast_vol * 1.96  # 95% confidence assuming normality
            forecast_lower = forecast_vol / 1.96
            
            # ==================== COMPREHENSIVE DIAGNOSTICS ====================
            diagnostics = self._calculate_garch_diagnostics(fit, std_resid)
            
            # ==================== VOLATILITY DECOMPOSITION ====================
            vol_decomposition = self._decompose_volatility(returns_clean, cond_vol)
            
            # ==================== MODEL COMPARISON METRICS ====================
            comparison_metrics = self._calculate_model_comparison_metrics(
                returns_clean, cond_vol, forecast_vol[:20]
            )
            
            # ==================== PARAMETER STABILITY ====================
            param_stability = self._assess_parameter_stability(fit, returns_scaled)
            
            return {
                "success": True,
                "model_type": garch_type,
                "distribution": distribution,
                "p": p,
                "q": q,
                
                # Model results
                "cond_volatility": cond_vol,
                "std_residuals": std_resid,
                "params": dict(fit.params),
                "pvalues": dict(fit.pvalues),
                "tvalues": dict(fit.tvalues),
                
                # Forecast
                "forecast_volatility": pd.Series(forecast_vol, index=forecast_dates),
                "forecast_upper": pd.Series(forecast_upper, index=forecast_dates),
                "forecast_lower": pd.Series(forecast_lower, index=forecast_dates),
                "forecast_horizon": forecast_horizon,
                
                # Diagnostics
                "converged": fit.convergence_flag == 0,
                "stationary": self._check_stationarity(fit.params),
                "aic": fit.aic,
                "bic": fit.bic,
                "log_likelihood": fit.loglikelihood,
                "diagnostics": diagnostics,
                
                # Volatility analysis
                "volatility_decomposition": vol_decomposition,
                "model_comparison": comparison_metrics,
                "parameter_stability": param_stability,
                
                # Quality scores
                "model_quality_score": self._calculate_model_quality_score(fit, diagnostics),
                "forecast_accuracy_score": self._calculate_forecast_accuracy(
                    returns_clean, cond_vol, forecast_vol[:20]
                )
            }
            
        except Exception as e:
            return {"error": f"GARCH analysis failed: {str(e)}", "success": False}
    
    def _calculate_garch_diagnostics(self, fit, std_resid: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive GARCH model diagnostics"""
        diagnostics = {}
        
        try:
            # Ljung-Box tests
            lags = [5, 10, 20]
            lb_results = []
            for lag in lags:
                try:
                    lb_test = acorr_ljungbox(std_resid, lags=[lag], return_df=True)
                    lb_results.append({
                        "lag": lag,
                        "statistic": lb_test["lb_stat"].iloc[0],
                        "pvalue": lb_test["lb_pvalue"].iloc[0]
                    })
                except:
                    lb_results.append({"lag": lag, "statistic": np.nan, "pvalue": np.nan})
            diagnostics["ljung_box_std_resid"] = lb_results
            
            # Ljung-Box on squared residuals
            lb_squared = []
            for lag in lags:
                try:
                    lb_test = acorr_ljungbox(std_resid ** 2, lags=[lag], return_df=True)
                    lb_squared.append({
                        "lag": lag,
                        "statistic": lb_test["lb_stat"].iloc[0],
                        "pvalue": lb_test["lb_pvalue"].iloc[0]
                    })
                except:
                    lb_squared.append({"lag": lag, "statistic": np.nan, "pvalue": np.nan})
            diagnostics["ljung_box_squared"] = lb_squared
            
            # ARCH-LM test
            try:
                LM, LM_p, F, F_p = het_arch(std_resid, maxlag=10)
                diagnostics["arch_lm"] = {
                    "lm_statistic": LM,
                    "lm_pvalue": LM_p,
                    "f_statistic": F,
                    "f_pvalue": F_p
                }
            except:
                diagnostics["arch_lm"] = {"lm_statistic": np.nan, "lm_pvalue": np.nan}
            
            # Normality tests
            try:
                jb_stat, jb_pval = stats.jarque_bera(std_resid.dropna())
                shapiro_stat, shapiro_pval = stats.shapiro(std_resid.dropna().sample(min(5000, len(std_resid))))
                diagnostics["normality_tests"] = {
                    "jarque_bera": {"statistic": jb_stat, "pvalue": jb_pval},
                    "shapiro_wilk": {"statistic": shapiro_stat, "pvalue": shapiro_pval}
                }
            except:
                diagnostics["normality_tests"] = {}
            
            # Information criteria
            diagnostics["information_criteria"] = {
                "aic": fit.aic,
                "bic": fit.bic,
                "hqic": fit.hqic if hasattr(fit, 'hqic') else np.nan
            }
            
        except Exception as e:
            diagnostics["error"] = f"Diagnostics calculation failed: {str(e)}"
        
        return diagnostics
    
    def _decompose_volatility(self, returns: pd.Series, cond_vol: pd.Series) -> Dict[str, Any]:
        """Decompose volatility into persistent and transitory components"""
        try:
            # Fit AR(1) to log volatility to estimate persistence
            log_vol = np.log(cond_vol ** 2).dropna()
            if len(log_vol) > 100:
                X = sm.add_constant(log_vol.shift(1).dropna())
                y = log_vol.iloc[1:]
                X = X.iloc[:len(y)]
                
                model = sm.OLS(y, X).fit()
                persistence = model.params.iloc[1] if len(model.params) > 1 else np.nan
            else:
                persistence = np.nan
            
            # Calculate volatility regimes
            vol_levels = pd.cut(
                cond_vol * 100,
                bins=[0, 20, 40, 60, 100, np.inf],
                labels=["Very Low", "Low", "Medium", "High", "Extreme"]
            )
            regime_counts = vol_levels.value_counts().to_dict()
            
            # Volatility clustering
            squared_returns = returns ** 2
            vol_clustering = squared_returns.autocorr(lag=1)
            
            return {
                "volatility_persistence": persistence,
                "regime_distribution": regime_counts,
                "volatility_clustering": vol_clustering,
                "avg_conditional_vol": cond_vol.mean() * 100,
                "std_conditional_vol": cond_vol.std() * 100
            }
        except:
            return {}
    
    def _calculate_model_comparison_metrics(self, returns: pd.Series, 
                                          cond_vol: pd.Series, 
                                          forecast_vol: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics for model comparison"""
        try:
            # Calculate realized volatility
            rv_20 = returns.rolling(20).std() * np.sqrt(252) * 100
            rv_60 = returns.rolling(60).std() * np.sqrt(252) * 100
            
            # Align series
            common_idx = cond_vol.index.intersection(rv_20.index).intersection(rv_60.index)
            cond_vol_aligned = cond_vol.loc[common_idx] * 100
            rv_20_aligned = rv_20.loc[common_idx]
            rv_60_aligned = rv_60.loc[common_idx]
            
            # Calculate correlations
            corr_20 = cond_vol_aligned.corr(rv_20_aligned)
            corr_60 = cond_vol_aligned.corr(rv_60_aligned)
            
            # Calculate forecast errors (if we have future realized vol)
            if len(forecast_vol) > 0 and len(returns) > 20:
                future_returns = returns.iloc[-20:]
                future_rv = future_returns.std() * np.sqrt(252) * 100
                forecast_error = np.abs(forecast_vol[0] * 100 - future_rv)
            else:
                forecast_error = np.nan
            
            return {
                "correlation_rv_20": corr_20,
                "correlation_rv_60": corr_60,
                "forecast_error": forecast_error,
                "mae_20d": np.mean(np.abs(cond_vol_aligned - rv_20_aligned)),
                "mae_60d": np.mean(np.abs(cond_vol_aligned - rv_60_aligned)),
                "rmse_20d": np.sqrt(np.mean((cond_vol_aligned - rv_20_aligned) ** 2)),
                "rmse_60d": np.sqrt(np.mean((cond_vol_aligned - rv_60_aligned) ** 2))
            }
        except:
            return {}
    
    def _assess_parameter_stability(self, fit, returns_scaled: np.ndarray) -> Dict[str, Any]:
        """Assess parameter stability through rolling estimation"""
        try:
            n = len(returns_scaled)
            if n < 500:
                return {"insufficient_data": True}
            
            # Perform rolling estimation
            window = min(300, n // 2)
            steps = 10
            step_size = max(1, (n - window) // steps)
            
            alpha_values = []
            beta_values = []
            
            for i in range(0, n - window, step_size):
                try:
                    window_data = returns_scaled[i:i+window]
                    model = arch_model(
                        window_data,
                        mean="Constant",
                        vol=fit.model.volatility.__class__.__name__,
                        p=fit.model.volatility.p,
                        q=fit.model.volatility.q,
                        dist=fit.model.distribution.name
                    )
                    window_fit = model.fit(disp="off", show_warning=False)
                    
                    if 'alpha[1]' in window_fit.params:
                        alpha_values.append(window_fit.params['alpha[1]'])
                    if 'beta[1]' in window_fit.params:
                        beta_values.append(window_fit.params['beta[1]'])
                except:
                    continue
            
            if alpha_values and beta_values:
                return {
                    "alpha_mean": np.mean(alpha_values),
                    "alpha_std": np.std(alpha_values),
                    "beta_mean": np.mean(beta_values),
                    "beta_std": np.std(beta_values),
                    "parameter_variation": np.std(alpha_values + beta_values),
                    "stability_score": 1 - min(np.std(alpha_values + beta_values), 0.5)
                }
            else:
                return {}
        except:
            return {}
    
    def _check_stationarity(self, params: pd.Series) -> bool:
        """Check if GARCH model is stationary"""
        try:
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            return alpha + beta < 1
        except:
            return False
    
    def _calculate_model_quality_score(self, fit, diagnostics: Dict[str, Any]) -> float:
        """Calculate overall model quality score (0-100)"""
        score = 0
        
        # Convergence (30 points)
        if fit.convergence_flag == 0:
            score += 30
        
        # Stationarity (20 points)
        if self._check_stationarity(fit.params):
            score += 20
        
        # Residual diagnostics (20 points)
        if "ljung_box_squared" in diagnostics:
            pvalues = [d["pvalue"] for d in diagnostics["ljung_box_squared"] if not np.isnan(d["pvalue"])]
            if pvalues:
                avg_pval = np.mean(pvalues)
                score += min(20, avg_pval * 20)  # Higher p-value is better
        
        # Parameter significance (15 points)
        pvals = [fit.pvalues.get(p, 1) for p in fit.params.index if p not in ['mu', 'omega']]
        if pvals:
            sig_params = sum(1 for p in pvals if p < 0.05)
            score += min(15, (sig_params / len(pvals)) * 15)
        
        # Information criteria (15 points)
        aic_score = max(0, 1 - (fit.aic / 10000))
        score += min(15, aic_score * 15)
        
        return min(100, score)
    
    def _calculate_forecast_accuracy(self, returns: pd.Series, 
                                   cond_vol: pd.Series, 
                                   forecast: np.ndarray) -> float:
        """Calculate forecast accuracy score"""
        try:
            if len(forecast) < 5:
                return 0
            
            # Use last 20 days for validation
            val_returns = returns.iloc[-20:]
            val_rv = val_returns.std() * np.sqrt(252) * 100
            
            # Calculate forecast error
            forecast_avg = np.mean(forecast[:5]) * 100
            error = np.abs(forecast_avg - val_rv)
            
            # Convert to score (0-100, lower error = higher score)
            accuracy = max(0, 100 - error * 10)
            return accuracy
        except:
            return 0
    
    def perform_garch_grid_search(self, returns: pd.Series,
                                p_values: List[int] = [1, 2, 3],
                                q_values: List[int] = [1, 2, 3],
                                garch_types: List[str] = ["GARCH", "EGARCH", "GJR-GARCH"],
                                distributions: List[str] = ["normal", "t", "skewt"],
                                max_models: int = 50) -> pd.DataFrame:
        """Perform comprehensive GARCH grid search"""
        if not ARCH_AVAILABLE or len(returns) < 400:
            return pd.DataFrame()
        
        results = []
        model_count = 0
        
        progress_bar = st.progress(0, text="Running GARCH grid search...")
        
        for garch_type in garch_types:
            for p in p_values:
                for q in q_values:
                    for dist in distributions:
                        if model_count >= max_models:
                            break
                        
                        try:
                            model_count += 1
                            progress_bar.progress(model_count / max_models, 
                                                text=f"Testing {garch_type}({p},{q}) - {dist}")
                            
                            # Fit model
                            garch_result = self.perform_comprehensive_garch_analysis(
                                returns, garch_type, p, q, dist, forecast_horizon=30
                            )
                            
                            if garch_result.get("success", False):
                                score = garch_result.get("model_quality_score", 0)
                                forecast_score = garch_result.get("forecast_accuracy_score", 0)
                                
                                results.append({
                                    "garch_type": garch_type,
                                    "p": p,
                                    "q": q,
                                    "distribution": dist,
                                    "aic": garch_result.get("aic", np.nan),
                                    "bic": garch_result.get("bic", np.nan),
                                    "log_likelihood": garch_result.get("log_likelihood", np.nan),
                                    "quality_score": score,
                                    "forecast_score": forecast_score,
                                    "converged": garch_result.get("converged", False),
                                    "stationary": garch_result.get("stationary", False),
                                    "alpha": garch_result.get("params", {}).get("alpha[1]", np.nan),
                                    "beta": garch_result.get("params", {}).get("beta[1]", np.nan),
                                    "omega": garch_result.get("params", {}).get("omega", np.nan),
                                    "persistence": garch_result.get("params", {}).get("alpha[1]", 0) + 
                                                 garch_result.get("params", {}).get("beta[1]", 0)
                                })
                        
                        except Exception as e:
                            continue
        
        progress_bar.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            # Calculate combined score
            df_results["combined_score"] = (
                df_results["quality_score"] * 0.6 +
                df_results["forecast_score"] * 0.4
            )
            df_results = df_results.sort_values(["converged", "combined_score"], 
                                               ascending=[False, False])
            return df_results
        
        return pd.DataFrame()
    
    def select_champion_model(self, grid_results: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Select champion model from grid search results"""
        if grid_results.empty:
            return None
        
        # Filter converged models
        converged = grid_results[grid_results["converged"]]
        if converged.empty:
            return None
        
        # Select model with highest combined score
        champion = converged.iloc[0]
        
        return {
            "garch_type": champion["garch_type"],
            "p": int(champion["p"]),
            "q": int(champion["q"]),
            "distribution": champion["distribution"],
            "quality_score": champion["quality_score"],
            "forecast_score": champion["forecast_score"],
            "combined_score": champion["combined_score"],
            "aic": champion["aic"],
            "bic": champion["bic"],
            "persistence": champion["persistence"]
        }

# ============================================================================
# SUPERIOR VOLATILITY VISUALIZATION ENGINE
# ============================================================================

class SuperiorVolatilityVisualization:
    """Superior visualization engine for comprehensive volatility analysis"""
    
    def __init__(self):
        self.colors = {
            "primary": "#1a2980",
            "secondary": "#26d0ce",
            "tertiary": "#764ba2",
            "quaternary": "#667eea",
            "success": "#27ae60",
            "warning": "#f39c12",
            "danger": "#e74c3c",
            "neutral": "#95a5a6",
            "dark": "#2c3e50",
            "light": "#ecf0f1"
        }
        
        self.volatility_palette = {
            "rv_20": "#3498db",  # Blue
            "rv_60": "#9b59b6",  # Purple
            "garch_cond": "#e74c3c",  # Red
            "forecast": "#2ecc71",  # Green
            "confidence_band": "rgba(52, 152, 219, 0.2)",
            "historical_range": "rgba(231, 76, 60, 0.1)"
        }
    
    def create_comprehensive_volatility_chart(self, 
                                            returns: pd.Series,
                                            cond_vol: Optional[pd.Series] = None,
                                            forecast_vol: Optional[pd.Series] = None,
                                            forecast_upper: Optional[pd.Series] = None,
                                            forecast_lower: Optional[pd.Series] = None,
                                            title: str = "Comprehensive Volatility Analysis",
                                            show_20d_rv: bool = True,
                                            show_60d_rv: bool = True,
                                            show_garch: bool = True,
                                            show_forecast: bool = True,
                                            show_confidence: bool = True) -> go.Figure:
        """
        Create comprehensive volatility chart with all components
        
        Parameters:
        -----------
        returns: pd.Series
            Return series for calculating realized volatility
        cond_vol: pd.Series, optional
            GARCH conditional volatility
        forecast_vol: pd.Series, optional
            GARCH volatility forecast
        forecast_upper: pd.Series, optional
            Upper confidence bound for forecast
        forecast_lower: pd.Series, optional
            Lower confidence bound for forecast
        title: str
            Chart title
        show_20d_rv: bool
            Show 20-day realized volatility
        show_60d_rv: bool
            Show 60-day realized volatility
        show_garch: bool
            Show GARCH conditional volatility
        show_forecast: bool
            Show volatility forecast
        show_confidence: bool
            Show forecast confidence bands
        
        Returns:
        --------
        go.Figure
            Comprehensive volatility chart
        """
        # Create figure with secondary y-axis for returns
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{title} - Volatility Analysis", "Returns")
        )
        
        # ==================== VOLATILITY PLOT (Row 1) ====================
        
        # 1. Historical Realized Volatility (20-day)
        if show_20d_rv:
            rv_20 = returns.rolling(20).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=rv_20.index,
                    y=rv_20.values,
                    name="RV 20D",
                    line=dict(color=self.volatility_palette["rv_20"], width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # 2. Historical Realized Volatility (60-day)
        if show_60d_rv:
            rv_60 = returns.rolling(60).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=rv_60.index,
                    y=rv_60.values,
                    name="RV 60D",
                    line=dict(color=self.volatility_palette["rv_60"], width=2, dash="dash"),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # 3. GARCH Conditional Volatility
        if show_garch and cond_vol is not None:
            garch_vol = cond_vol * 100  # Convert to percentage
            fig.add_trace(
                go.Scatter(
                    x=garch_vol.index,
                    y=garch_vol.values,
                    name="GARCH Cond Vol",
                    line=dict(color=self.volatility_palette["garch_cond"], width=3),
                    opacity=0.9
                ),
                row=1, col=1
            )
        
        # 4. Volatility Forecast with Confidence Bands
        if show_forecast and forecast_vol is not None:
            forecast_vol_pct = forecast_vol * 100
            
            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_vol_pct.index,
                    y=forecast_vol_pct.values,
                    name="Vol Forecast",
                    line=dict(color=self.volatility_palette["forecast"], width=3, dash="dot"),
                    opacity=0.9
                ),
                row=1, col=1
            )
            
            # Confidence bands if available
            if show_confidence and forecast_upper is not None and forecast_lower is not None:
                forecast_upper_pct = forecast_upper * 100
                forecast_lower_pct = forecast_lower * 100
                
                # Upper bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast_upper_pct.index,
                        y=forecast_upper_pct.values,
                        name="Upper 95%",
                        line=dict(color=self.volatility_palette["forecast"], width=1, dash="dash"),
                        opacity=0.5,
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Lower bound
                fig.add_trace(
                    go.Scatter(
                        x=forecast_lower_pct.index,
                        y=forecast_lower_pct.values,
                        name="Lower 95%",
                        line=dict(color=self.volatility_palette["forecast"], width=1, dash="dash"),
                        opacity=0.5,
                        fill="tonexty",
                        fillcolor=self.volatility_palette["confidence_band"],
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Add historical volatility range shading
        if show_20d_rv:
            rv_20 = returns.rolling(20).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([rv_20.index, rv_20.index[::-1]]),
                    y=pd.concat([rv_20 * 1.5, (rv_20 * 0.5)[::-1]]),
                    fill="toself",
                    fillcolor=self.volatility_palette["historical_range"],
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Historical Range",
                    showlegend=True,
                    opacity=0.2
                ),
                row=1, col=1
            )
        
        # ==================== RETURNS PLOT (Row 2) ====================
        
        # Color returns by positive/negative
        colors = [self.colors["success"] if x >= 0 else self.colors["danger"] 
                 for x in returns.values]
        
        fig.add_trace(
            go.Bar(
                x=returns.index,
                y=returns.values * 100,
                name="Daily Returns",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # ==================== LAYOUT CONFIGURATION ====================
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=800,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Annualized Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
        
        # Update x-axis label
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_volatility_comparison_chart(self, 
                                         rv_20: pd.Series,
                                         rv_60: pd.Series,
                                         garch_vol: Optional[pd.Series] = None,
                                         title: str = "Volatility Comparison") -> go.Figure:
        """Create side-by-side volatility comparison chart"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("20D Realized Vol", "60D Realized Vol", "GARCH vs Realized"),
            horizontal_spacing=0.1
        )
        
        # 1. 20-day RV distribution
        fig.add_trace(
            go.Histogram(
                x=rv_20.dropna(),
                nbinsx=50,
                name="20D RV",
                marker_color=self.volatility_palette["rv_20"],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add mean line for 20D
        mean_20 = rv_20.mean()
        fig.add_vline(x=mean_20, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_20:.1f}%", row=1, col=1)
        
        # 2. 60-day RV distribution
        fig.add_trace(
            go.Histogram(
                x=rv_60.dropna(),
                nbinsx=50,
                name="60D RV",
                marker_color=self.volatility_palette["rv_60"],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add mean line for 60D
        mean_60 = rv_60.mean()
        fig.add_vline(x=mean_60, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_60:.1f}%", row=1, col=2)
        
        # 3. GARCH vs Realized comparison
        if garch_vol is not None and len(garch_vol) > 0:
            # Align series
            common_idx = rv_20.index.intersection(garch_vol.index)
            if len(common_idx) > 0:
                rv_20_aligned = rv_20.loc[common_idx]
                garch_aligned = garch_vol.loc[common_idx] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=rv_20_aligned.values,
                        y=garch_aligned.values,
                        mode="markers",
                        name="GARCH vs RV",
                        marker=dict(
                            color=self.volatility_palette["garch_cond"],
                            size=8,
                            opacity=0.6
                        )
                    ),
                    row=1, col=3
                )
                
                # Add 45-degree line
                max_val = max(rv_20_aligned.max(), garch_aligned.max())
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode="lines",
                        name="Perfect Fit",
                        line=dict(color="red", dash="dash", width=2),
                        opacity=0.5
                    ),
                    row=1, col=3
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_xaxes(title_text="20D RV (%)", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="GARCH Vol (%)", row=1, col=3)
        
        return fig
    
    def create_volatility_forecast_chart(self,
                                       historical_vol: pd.Series,
                                       forecast_vol: pd.Series,
                                       forecast_upper: pd.Series,
                                       forecast_lower: pd.Series,
                                       title: str = "Volatility Forecast") -> go.Figure:
        """Create detailed volatility forecast chart"""
        fig = go.Figure()
        
        # Historical volatility
        fig.add_trace(
            go.Scatter(
                x=historical_vol.index,
                y=historical_vol.values,
                name="Historical Vol",
                line=dict(color=self.colors["primary"], width=2),
                opacity=0.8
            )
        )
        
        # Forecast with confidence bands
        x_forecast = forecast_vol.index
        y_forecast = forecast_vol.values
        
        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([x_forecast, x_forecast[::-1]]),
                y=pd.concat([forecast_upper.values, forecast_lower.values[::-1]]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence",
                showlegend=True
            )
        )
        
        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=y_forecast,
                name="Forecast",
                line=dict(color=self.colors["danger"], width=3, dash="dash")
            )
        )
        
        # Add vertical line separating history and forecast
        fig.add_vline(
            x=historical_vol.index[-1],
            line_dash="dash",
            line_color="gray",
            opacity=0.7,
            annotation_text="Forecast Start"
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=600,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Annualized Volatility (%)",
            xaxis_title="Date",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_volatility_regime_chart(self, volatility_series: pd.Series,
                                     regimes: Optional[Dict[str, Any]] = None,
                                     title: str = "Volatility Regimes") -> go.Figure:
        """Create volatility regime chart"""
        fig = go.Figure()
        
        # Base volatility line
        fig.add_trace(
            go.Scatter(
                x=volatility_series.index,
                y=volatility_series.values,
                name="Volatility",
                line=dict(color=self.colors["neutral"], width=2),
                opacity=0.7
            )
        )
        
        # Add regime shading if available
        if regimes and "regimes" in regimes:
            for regime in regimes["regimes"]:
                fig.add_vrect(
                    x0=regime["start"],
                    x1=regime["end"],
                    fillcolor=self._get_regime_color(regime["regime"]),
                    opacity=0.2,
                    line_width=0,
                    annotation_text=regime["regime"],
                    annotation_position="top left"
                )
        
        # Add horizontal lines for volatility levels
        if len(volatility_series) > 0:
            mean_vol = volatility_series.mean()
            std_vol = volatility_series.std()
            
            fig.add_hline(
                y=mean_vol,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                annotation_text=f"Mean: {mean_vol:.1f}%"
            )
            
            fig.add_hline(
                y=mean_vol + std_vol,
                line_dash="dot",
                line_color="orange",
                opacity=0.5,
                annotation_text=f"+1σ: {mean_vol + std_vol:.1f}%"
            )
            
            fig.add_hline(
                y=mean_vol - std_vol,
                line_dash="dot",
                line_color="orange",
                opacity=0.5,
                annotation_text=f"-1σ: {mean_vol - std_vol:.1f}%"
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Volatility (%)",
            xaxis_title="Date"
        )
        
        return fig
    
    def _get_regime_color(self, regime: str) -> str:
        """Get color for volatility regime"""
        regime_colors = {
            "Very Low": "rgba(46, 204, 113, 0.3)",  # Green
            "Low": "rgba(52, 152, 219, 0.3)",      # Blue
            "Medium": "rgba(155, 89, 182, 0.3)",   # Purple
            "High": "rgba(241, 196, 15, 0.3)",     # Yellow
            "Very High": "rgba(231, 76, 60, 0.3)", # Red
            "Extreme": "rgba(192, 57, 43, 0.3)"    # Dark Red
        }
        return regime_colors.get(regime, "rgba(149, 165, 166, 0.3)")

# ============================================================================
# SUPERIOR INSTITUTIONAL DASHBOARD
# ============================================================================

class SuperiorInstitutionalDashboard:
    """Superior institutional dashboard with comprehensive GARCH analysis"""
    
    def __init__(self):
        self.data_loader = InstitutionalDataLoader()
        self.garch_analyzer = SuperiorGARCHAnalyzer()
        self.vol_viz = SuperiorVolatilityVisualization()
        
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
            "garch_grid_results": {},
            "champion_models": {},
            "volatility_metrics": {},
            "regime_results": {},
            "current_garch_analysis": None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def display_header(self):
        """Display professional header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; font-size:2.5rem;">🏛️ Institutional Commodities Analytics v5.0</h1>
            <p style="margin:8px 0 0 0; opacity:0.95;">
                Superior Advanced GARCH Analysis • Comprehensive Volatility Charts • Institutional Grade
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "🟢 Online" if st.session_state.data_loaded else "🟡 Ready"
            st.metric("Status", status, "Streamlit Cloud")
        with col2:
            loaded = len(st.session_state.asset_data) if st.session_state.data_loaded else 0
            st.metric("Assets Loaded", loaded)
        with col3:
            arch_status = "🟢 Available" if ARCH_AVAILABLE else "🔴 Disabled"
            st.metric("GARCH Engine", arch_status)
        with col4:
            models = len(st.session_state.champion_models)
            st.metric("Champion Models", models)
    
    def setup_sidebar(self) -> Dict[str, Any]:
        """Setup comprehensive sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📅 Date Range Configuration")
            
            # Date range with sensible defaults
            default_end = datetime.now()
            default_start = default_end - timedelta(days=365*5)  # 5 years for robust GARCH
            
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
            
            # GARCH Configuration
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### ⚙️ GARCH Configuration")
            
            if ARCH_AVAILABLE:
                # Model type selection
                garch_types = st.multiselect(
                    "GARCH Models",
                    ["GARCH", "EGARCH", "GJR-GARCH", "TARCH", "APARCH"],
                    default=["GARCH", "EGARCH", "GJR-GARCH"]
                )
                
                # Order selection
                col1, col2 = st.columns(2)
                with col1:
                    p_max = st.slider("ARCH Order (p max)", 1, 5, 2, 1)
                with col2:
                    q_max = st.slider("GARCH Order (q max)", 1, 5, 2, 1)
                
                # Distribution selection
                distributions = st.multiselect(
                    "Distributions",
                    ["normal", "t", "skewt", "ged"],
                    default=["t", "normal"]
                )
                
                # Forecast settings
                forecast_horizon = st.slider("Forecast Horizon (days)", 10, 120, 60, 5)
                
                # Grid search settings
                max_models = st.slider("Max Models in Grid", 20, 100, 50, 5)
            else:
                garch_types = ["GARCH"]
                p_max = 1
                q_max = 1
                distributions = ["normal"]
                forecast_horizon = 30
                max_models = 20
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Volatility Chart Settings
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📈 Chart Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                show_20d_rv = st.checkbox("20D RV", value=True)
                show_60d_rv = st.checkbox("60D RV", value=True)
            with col2:
                show_garch = st.checkbox("GARCH Vol", value=True)
                show_confidence = st.checkbox("Confidence Bands", value=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                load_clicked = st.button("📥 Load Data", type="primary", use_container_width=True)
            with col2:
                if st.session_state.data_loaded:
                    if st.button("🔄 Clear", type="secondary", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
            
            if load_clicked:
                self._load_data(selected_assets, selected_benchmarks, start_date, end_date)
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "selected_assets": selected_assets,
                "selected_benchmarks": selected_benchmarks,
                "garch_types": garch_types,
                "p_max": p_max,
                "q_max": q_max,
                "distributions": distributions,
                "forecast_horizon": forecast_horizon,
                "max_models": max_models,
                "chart_settings": {
                    "show_20d_rv": show_20d_rv,
                    "show_60d_rv": show_60d_rv,
                    "show_garch": show_garch,
                    "show_confidence": show_confidence
                }
            }
    
    def _load_data(self, assets: List[str], benchmarks: List[str], 
                  start_date: datetime, end_date: datetime):
        """Load data with comprehensive volatility preparation"""
        if not assets:
            st.warning("Please select at least one asset")
            return
        
        # Convert dates
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        
        # Add buffer for volatility calculations
        start_dt = start_dt - timedelta(days=60)
        
        with st.spinner("Loading institutional data with comprehensive volatility metrics..."):
            # Load data
            data_dict = self.data_loader.bulk_load_with_volatility(
                assets + benchmarks, start_dt, end_dt
            )
            
            if not data_dict:
                st.error("Failed to load data. Please check your internet connection and try again.")
                return
            
            # Separate assets and benchmarks
            asset_data = {}
            benchmark_data = {}
            returns_data = {}
            volatility_metrics = {}
            
            for symbol, df in data_dict.items():
                if symbol in assets:
                    asset_data[symbol] = df
                    returns_data[symbol] = df["Returns"].dropna()
                    
                    # Calculate volatility metrics
                    if len(df) > 100:
                        volatility_metrics[symbol] = self.garch_analyzer.calculate_volatility_metrics(
                            df["Returns"].dropna()
                        )
                elif symbol in benchmarks:
                    benchmark_data[symbol] = df
            
            # Store in session state
            st.session_state.asset_data = asset_data
            st.session_state.returns_data = returns_data
            st.session_state.benchmark_data = benchmark_data
            st.session_state.volatility_metrics = volatility_metrics
            st.session_state.selected_assets = list(asset_data.keys())
            st.session_state.data_loaded = True
            
            # Initialize portfolio weights
            if asset_data:
                n_assets = len(asset_data)
                equal_weight = 1.0 / n_assets
                st.session_state.portfolio_weights = {
                    asset: equal_weight for asset in asset_data.keys()
                }
            
            st.success(f"✅ Successfully loaded {len(asset_data)} assets with comprehensive volatility metrics")
            
            # Show data summary
            with st.expander("📋 Data Summary", expanded=True):
                for symbol, df in asset_data.items():
                    vol_metrics = volatility_metrics.get(symbol, {})
                    st.write(f"**{symbol}**: {len(df)} days | "
                            f"RV20: {vol_metrics.get('rv_20_mean', 'N/A'):.1f}% | "
                            f"RV60: {vol_metrics.get('rv_60_mean', 'N/A'):.1f}%")
    
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
            "⚡ Superior GARCH", 
            "📈 Volatility Analysis",
            "🔄 Regimes", 
            "📋 Reports"
        ])
        
        with tab1:
            self.display_dashboard(config)
        
        with tab2:
            self.display_superior_garch_analysis(config)
        
        with tab3:
            self.display_volatility_analysis(config)
        
        with tab4:
            if HMM_AVAILABLE:
                self.display_regime_analysis(config)
            else:
                st.warning("Regime analysis requires 'hmmlearn' and 'scikit-learn'.")
        
        with tab5:
            self.display_reports(config)
    
    def display_dashboard(self, config: Dict[str, Any]):
        """Display main dashboard"""
        st.header("📊 Institutional Dashboard")
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets Loaded", len(st.session_state.asset_data))
        with col2:
            days = (config["end_date"] - config["start_date"]).days
            st.metric("Time Period", f"{days} days")
        with col3:
            if st.session_state.volatility_metrics:
                avg_rv20 = np.mean([m.get("rv_20_mean", 0) for m in st.session_state.volatility_metrics.values()])
                st.metric("Avg RV20", f"{avg_rv20:.1f}%")
        with col4:
            champion_count = len(st.session_state.champion_models)
            st.metric("Champion Models", champion_count)
        
        # Volatility overview
        st.subheader("📉 Volatility Overview")
        
        if st.session_state.volatility_metrics:
            vol_data = []
            for symbol, metrics in st.session_state.volatility_metrics.items():
                vol_data.append({
                    "Asset": symbol,
                    "RV20": metrics.get("rv_20_mean", 0),
                    "RV60": metrics.get("rv_60_mean", 0),
                    "Vol Ratio": metrics.get("vol_ratio_20_60", 0),
                    "Persistence": metrics.get("volatility_persistence", 0),
                    "Clusters": metrics.get("volatility_clusters", {}).get("num_clusters", 0)
                })
            
            vol_df = pd.DataFrame(vol_data)
            st.dataframe(
                vol_df.style.format({
                    "RV20": "{:.1f}%",
                    "RV60": "{:.1f}%",
                    "Vol Ratio": "{:.2f}",
                    "Persistence": "{:.2f}"
                }),
                use_container_width=True,
                height=300
            )
        
        # Asset performance
        st.subheader("📈 Asset Performance")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed View",
            options=st.session_state.selected_assets,
            key="dashboard_asset_select"
        )
        
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            
            # Create comprehensive price chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{selected_asset} - Price Action", "Volume")
            )
            
            # Price with moving averages
            fig.add_trace(
                go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(dash="dash")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(dash="dot")),
                row=1, col=1
            )
            
            # Volume
            colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(df["Adj Close"], df["Open"])]
            fig.add_trace(
                go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.7),
                row=2, col=1
            )
            
            fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_superior_garch_analysis(self, config: Dict[str, Any]):
        """Display superior GARCH analysis with comprehensive volatility charts"""
        st.header("⚡ Superior Advanced GARCH Analysis")
        
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
        
        # ==================== GARCH GRID SEARCH ====================
        st.subheader("🔍 GARCH Grid Search & Champion Selection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Run Comprehensive Grid Search", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive GARCH grid search..."):
                    grid_results = self.garch_analyzer.perform_garch_grid_search(
                        returns,
                        p_values=list(range(1, config["p_max"] + 1)),
                        q_values=list(range(1, config["q_max"] + 1)),
                        garch_types=config["garch_types"],
                        distributions=config["distributions"],
                        max_models=config["max_models"]
                    )
                    
                    if not grid_results.empty:
                        st.session_state.garch_grid_results[selected_asset] = grid_results
                        
                        # Select champion
                        champion = self.garch_analyzer.select_champion_model(grid_results)
                        if champion:
                            st.session_state.champion_models[selected_asset] = champion
                            st.success(f"🎯 Champion Model Selected: {champion['garch_type']}({champion['p']},{champion['q']}) - {champion['distribution']}")
        
        with col2:
            if st.session_state.champion_models.get(selected_asset):
                champion = st.session_state.champion_models[selected_asset]
                st.markdown('<div class="volatility-compare">', unsafe_allow_html=True)
                st.write(f"**Champion**")
                st.write(f"{champion['garch_type']}({champion['p']},{champion['q']})")
                st.write(f"Score: {champion['combined_score']:.1f}/100")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display grid results if available
        if selected_asset in st.session_state.garch_grid_results:
            grid_results = st.session_state.garch_grid_results[selected_asset]
            
            with st.expander("📋 Grid Search Results", expanded=True):
                # Top 10 models
                top_models = grid_results.head(10)
                st.dataframe(
                    top_models.style.format({
                        "aic": "{:.1f}",
                        "bic": "{:.1f}",
                        "quality_score": "{:.1f}",
                        "forecast_score": "{:.1f}",
                        "combined_score": "{:.1f}",
                        "alpha": "{:.4f}",
                        "beta": "{:.4f}",
                        "persistence": "{:.3f}"
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Visualization of model scores
                fig = px.scatter(
                    top_models,
                    x="quality_score",
                    y="forecast_score",
                    color="garch_type",
                    size="combined_score",
                    hover_data=["p", "q", "distribution", "persistence"],
                    title="Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ==================== COMPREHENSIVE GARCH ANALYSIS ====================
        st.subheader("📊 Comprehensive GARCH Analysis")
        
        if selected_asset in st.session_state.champion_models:
            champion = st.session_state.champion_models[selected_asset]
            
            # Run comprehensive analysis
            if st.button("🎯 Run Comprehensive GARCH Analysis", type="secondary", use_container_width=True):
                with st.spinner("Running comprehensive GARCH analysis..."):
                    garch_result = self.garch_analyzer.perform_comprehensive_garch_analysis(
                        returns,
                        champion["garch_type"],
                        champion["p"],
                        champion["q"],
                        champion["distribution"],
                        config["forecast_horizon"]
                    )
                    
                    if garch_result.get("success", False):
                        st.session_state.current_garch_analysis = garch_result
                        st.success("✅ Comprehensive GARCH analysis completed!")
                    else:
                        st.error(f"GARCH analysis failed: {garch_result.get('error', 'Unknown error')}")
            
            # Display analysis results if available
            if st.session_state.current_garch_analysis:
                garch_result = st.session_state.current_garch_analysis
                
                # Display model parameters
                st.markdown('<div class="garch-params">', unsafe_allow_html=True)
                st.markdown("### 📋 Model Parameters & Diagnostics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{garch_result.get('aic', 0):.1f}")
                    st.metric("BIC", f"{garch_result.get('bic', 0):.1f}")
                with col2:
                    st.metric("Quality Score", f"{garch_result.get('model_quality_score', 0):.1f}/100")
                    st.metric("Forecast Score", f"{garch_result.get('forecast_accuracy_score', 0):.1f}/100")
                with col3:
                    st.metric("Converged", "✅" if garch_result.get("converged") else "❌")
                    st.metric("Stationary", "✅" if garch_result.get("stationary") else "❌")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ==================== COMPREHENSIVE VOLATILITY CHART ====================
                st.subheader("📈 Comprehensive Volatility Analysis Chart")
                
                # Get volatility components
                cond_vol = garch_result.get("cond_volatility")
                forecast_vol = garch_result.get("forecast_volatility")
                forecast_upper = garch_result.get("forecast_upper")
                forecast_lower = garch_result.get("forecast_lower")
                
                # Create comprehensive volatility chart
                fig = self.vol_viz.create_comprehensive_volatility_chart(
                    returns,
                    cond_vol,
                    forecast_vol,
                    forecast_upper,
                    forecast_lower,
                    title=f"{selected_asset} - Comprehensive Volatility Analysis",
                    show_20d_rv=config["chart_settings"]["show_20d_rv"],
                    show_60d_rv=config["chart_settings"]["show_60d_rv"],
                    show_garch=config["chart_settings"]["show_garch"],
                    show_confidence=config["chart_settings"]["show_confidence"]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ==================== VOLATILITY COMPARISON CHART ====================
                st.subheader("🔄 Volatility Comparison")
                
                # Calculate realized volatilities
                rv_20 = returns.rolling(20).std() * np.sqrt(252) * 100
                rv_60 = returns.rolling(60).std() * np.sqrt(252) * 100
                
                # Create comparison chart
                comp_fig = self.vol_viz.create_volatility_comparison_chart(
                    rv_20,
                    rv_60,
                    cond_vol,
                    title=f"{selected_asset} - Volatility Distribution Comparison"
                )
                
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # ==================== DETAILED DIAGNOSTICS ====================
                with st.expander("🔬 Detailed Model Diagnostics", expanded=False):
                    if "diagnostics" in garch_result:
                        diagnostics = garch_result["diagnostics"]
                        
                        # Ljung-Box tests
                        if "ljung_box_squared" in diagnostics:
                            st.write("**Ljung-Box Tests on Squared Residuals:**")
                            lb_data = []
                            for test in diagnostics["ljung_box_squared"]:
                                lb_data.append({
                                    "Lag": test["lag"],
                                    "Statistic": test["statistic"],
                                    "p-value": test["pvalue"],
                                    "Significant": test["pvalue"] < 0.05
                                })
                            st.dataframe(pd.DataFrame(lb_data))
                        
                        # ARCH-LM test
                        if "arch_lm" in diagnostics:
                            st.write("**ARCH-LM Test:**")
                            arch_data = diagnostics["arch_lm"]
                            st.write(f"LM Statistic: {arch_data.get('lm_statistic', 'N/A'):.4f}")
                            st.write(f"p-value: {arch_data.get('lm_pvalue', 'N/A'):.4f}")
                        
                        # Normality tests
                        if "normality_tests" in diagnostics:
                            st.write("**Normality Tests:**")
                            norm_data = diagnostics["normality_tests"]
                            if "jarque_bera" in norm_data:
                                st.write(f"Jarque-Bera: {norm_data['jarque_bera']['statistic']:.4f} "
                                        f"(p={norm_data['jarque_bera']['pvalue']:.4f})")
                
                # ==================== VOLATILITY DECOMPOSITION ====================
                with st.expander("📊 Volatility Decomposition", expanded=False):
                    if "volatility_decomposition" in garch_result:
                        vol_decomp = garch_result["volatility_decomposition"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Volatility Persistence", 
                                     f"{vol_decomp.get('volatility_persistence', 0):.3f}")
                            st.metric("Avg Conditional Vol", 
                                     f"{vol_decomp.get('avg_conditional_vol', 0):.2f}%")
                        with col2:
                            st.metric("Std Conditional Vol", 
                                     f"{vol_decomp.get('std_conditional_vol', 0):.2f}%")
                            st.metric("Volatility Clustering", 
                                     f"{vol_decomp.get('volatility_clustering', 0):.3f}")
                        
                        # Regime distribution
                        if "regime_distribution" in vol_decomp:
                            st.write("**Volatility Regime Distribution:**")
                            regime_df = pd.DataFrame.from_dict(
                                vol_decomp["regime_distribution"], 
                                orient="index", 
                                columns=["Count"]
                            )
                            st.dataframe(regime_df)
                
                # ==================== FORECAST DETAILS ====================
                with st.expander("🔮 Volatility Forecast Details", expanded=False):
                    if forecast_vol is not None:
                        st.write(f"**{config['forecast_horizon']}-Day Volatility Forecast:**")
                        
                        # Create forecast chart
                        if cond_vol is not None:
                            hist_vol = cond_vol * 100
                            forecast_fig = self.vol_viz.create_volatility_forecast_chart(
                                hist_vol,
                                forecast_vol * 100,
                                forecast_upper * 100 if forecast_upper is not None else forecast_vol * 100 * 1.96,
                                forecast_lower * 100 if forecast_lower is not None else forecast_vol * 100 / 1.96,
                                title=f"{selected_asset} - Volatility Forecast"
                            )
                            st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Forecast table
                        forecast_df = pd.DataFrame({
                            "Day": range(1, len(forecast_vol) + 1),
                            "Forecast Volatility (%)": forecast_vol * 100,
                            "Upper Bound (%)": forecast_upper * 100 if forecast_upper is not None else forecast_vol * 100 * 1.96,
                            "Lower Bound (%)": forecast_lower * 100 if forecast_lower is not None else forecast_vol * 100 / 1.96
                        })
                        st.dataframe(
                            forecast_df.style.format({
                                "Forecast Volatility (%)": "{:.2f}",
                                "Upper Bound (%)": "{:.2f}",
                                "Lower Bound (%)": "{:.2f}"
                            }),
                            use_container_width=True
                        )
        
        else:
            st.info("Run grid search to select a champion GARCH model first.")
    
    def display_volatility_analysis(self, config: Dict[str, Any]):
        """Display comprehensive volatility analysis"""
        st.header("📈 Comprehensive Volatility Analysis")
        
        if not st.session_state.volatility_metrics:
            st.warning("No volatility metrics available")
            return
        
        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset for Volatility Analysis",
            options=st.session_state.selected_assets,
            key="vol_asset_select"
        )
        
        if selected_asset not in st.session_state.volatility_metrics:
            st.warning("Selected asset not found")
            return
        
        metrics = st.session_state.volatility_metrics[selected_asset]
        
        # Display volatility metrics
        st.subheader("📊 Volatility Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("20D Realized Vol", f"{metrics.get('rv_20_mean', 0):.1f}%")
        with col2:
            st.metric("60D Realized Vol", f"{metrics.get('rv_60_mean', 0):.1f}%")
        with col3:
            st.metric("Vol Ratio (20/60)", f"{metrics.get('vol_ratio_20_60', 0):.2f}")
        with col4:
            st.metric("Vol Persistence", f"{metrics.get('volatility_persistence', 0):.3f}")
        
        # Volatility clusters
        if "volatility_clusters" in metrics:
            clusters = metrics["volatility_clusters"]
            st.write(f"**Volatility Clusters:** {clusters.get('num_clusters', 0)} clusters, "
                    f"Average duration: {clusters.get('avg_duration_days', 0):.1f} days")
        
        # Create volatility regime chart
        if selected_asset in st.session_state.asset_data:
            df = st.session_state.asset_data[selected_asset]
            rv_20 = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
            
            regime_fig = self.vol_viz.create_volatility_regime_chart(
                rv_20,
                metrics.get("volatility_regimes"),
                title=f"{selected_asset} - Volatility Regimes"
            )
            
            st.plotly_chart(regime_fig, use_container_width=True)
        
        # Extreme volatility analysis
        st.subheader("⚠️ Extreme Volatility Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Extreme Up Days", metrics.get("extreme_up_days", 0))
        with col2:
            st.metric("Extreme Down Days", metrics.get("extreme_down_days", 0))
        
        # Volatility skewness and kurtosis
        st.write(f"**Volatility Distribution:** Skewness: {metrics.get('volatility_skew', 0):.2f}, "
                f"Kurtosis: {metrics.get('volatility_kurtosis', 0):.2f}")
    
    def display_regime_analysis(self, config: Dict[str, Any]):
        """Display regime analysis"""
        st.header("🔄 Market Regime Detection")
        
        # Implementation would go here
        st.info("Regime analysis module under development")
    
    def display_reports(self, config: Dict[str, Any]):
        """Display reporting interface"""
        st.header("📋 Institutional Reports")
        
        # Implementation would go here
        st.info("Reporting module under development")

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
        dashboard = SuperiorInstitutionalDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
