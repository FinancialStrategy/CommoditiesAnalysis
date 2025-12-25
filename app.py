"""
🏛️ Institutional Commodities Analytics Platform v7.2 (Ultra)
Enhanced Scientific Analytics • Robust Correlations (incl. Ledoit–Wolf) • Professional Risk Metrics
Institutional-Grade Computational Finance Platform (Streamlit Single-File Edition)

Key Upgrades (v7.2)
- ✅ Correct correlation matrix + PSD-safe nearest-correlation fix (Higham-style)
- ✅ Optional Ledoit–Wolf shrinkage correlation (scikit-learn)
- ✅ New Institutional Signal tab:
      (EWMA 22D Vol) / (EWMA 33D Vol + EWMA 99D Vol)
      + Bollinger Bands + Green/Orange/Red risk bands
- ✅ Real benchmark-based Treynor + Information Ratio (no random benchmark)
- ✅ Hard crash fixes: `import scipy` + Higham DataFrame-safe implementation
- ✅ NEW (added without removing core platform features):
      • Interactive Tracking Error tab with green/orange/red band zones
      • Rolling Beta tab
      • Relative VaR / CVaR / ES vs benchmark chart with band zones
"""

# =============================================================================
# IMPORTS (DO NOT MOVE st.set_page_config BELOW IMPORTS THAT REQUIRE st)
# =============================================================================
import os
import math
import json
import time
import warnings
import traceback
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Scientific stack
import scipy  # ✅ REQUIRED because we reference scipy.__version__
from scipy import stats

# Optional visualization extras
try:
    import seaborn as sns  # not required, but kept for compatibility in user environments
except Exception:
    sns = None

# =============================================================================
# STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =============================================================================
st.set_page_config(
    page_title="Institutional Commodities Analytics Platform v7.2",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["NUMEXPR_MAX_THREADS"] = os.environ.get("NUMEXPR_MAX_THREADS", "8")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

# =============================================================================
# STYLE (INSTITUTIONAL UI)
# =============================================================================
def _inject_css() -> None:
    css = """
    <style>
    :root{
        --bg:#0b1220;
        --card:#111a2e;
        --card2:#0f172a;
        --stroke:rgba(255,255,255,0.08);
        --text:#e5e7eb;
        --muted:#9ca3af;
        --accent:#60a5fa;
        --green:#22c55e;
        --orange:#f59e0b;
        --red:#ef4444;
        --purple:#a78bfa;
        --cyan:#22d3ee;
    }
    .block-container{padding-top:1.2rem;}
    .institutional-hero{
        border:1px solid var(--stroke);
        background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(167,139,250,0.12));
        padding: 1.2rem 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .institutional-hero h1{
        margin:0;
        color: var(--text);
        font-size: 1.8rem;
        letter-spacing: 0.2px;
    }
    .institutional-hero p{
        margin:.3rem 0 0 0;
        color: var(--muted);
        font-size: .95rem;
    }
    .section-header{
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:1rem;
        border:1px solid var(--stroke);
        background: rgba(255,255,255,0.02);
        padding: .8rem 1rem;
        border-radius: 14px;
        margin: 0.3rem 0 0.9rem 0;
    }
    .section-header h2{
        margin:0;
        color: var(--text);
        font-size:1.2rem;
    }
    .section-actions{display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end;}
    .scientific-badge{
        display:inline-flex;
        align-items:center;
        padding:.25rem .55rem;
        border-radius:999px;
        border:1px solid var(--stroke);
        font-size:.78rem;
        color: var(--text);
        background: rgba(255,255,255,0.03);
        white-space:nowrap;
    }
    .scientific-badge.info{border-color:rgba(96,165,250,0.35); background:rgba(96,165,250,0.12);}
    .scientific-badge.low-risk{border-color:rgba(34,197,94,0.35); background:rgba(34,197,94,0.12);}
    .scientific-badge.medium-risk{border-color:rgba(245,158,11,0.35); background:rgba(245,158,11,0.12);}
    .scientific-badge.high-risk{border-color:rgba(239,68,68,0.35); background:rgba(239,68,68,0.12);}
    .institutional-card{
        border:1px solid var(--stroke);
        background: rgba(17,26,46,0.55);
        padding: 1rem;
        border-radius: 16px;
        margin-bottom: 0.8rem;
    }
    .metric-title{color:var(--muted); font-size:.85rem; margin-bottom:.2rem;}
    .metric-value{color:var(--text); font-size:1.6rem; font-weight:700;}
    .subtle{color:var(--muted);}
    hr{border-color: rgba(255,255,255,0.08)!important;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

_inject_css()

# =============================================================================
# DEPENDENCY MANAGER (OPTIONAL MODULES)
# =============================================================================
class ScientificDependencyManager:
    def __init__(self):
        self._cache: Dict[str, bool] = {}

    def is_available(self, name: str) -> bool:
        if name in self._cache:
            return self._cache[name]
        try:
            __import__(name)
            self._cache[name] = True
        except Exception:
            self._cache[name] = False
        return self._cache[name]

sci_dep_manager = ScientificDependencyManager()

# =============================================================================
# HELPERS
# =============================================================================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            if np.isnan(x):
                return default
            return float(x)
        val = float(x)
        if np.isnan(val):
            return default
        return val
    except Exception:
        return default

def _annualize_mean(daily_mean: float, trading_days: int = 252) -> float:
    return daily_mean * trading_days

def _annualize_vol(daily_std: float, trading_days: int = 252) -> float:
    return daily_std * math.sqrt(trading_days)

def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns is None or returns.dropna().empty:
        return 0.0
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())

def _ewma_vol_annualized(returns: pd.Series, span: int, trading_days: int = 252) -> pd.Series:
    r = returns.astype(float)
    ewm_var = (r ** 2).ewm(span=span, adjust=False, min_periods=max(10, int(span * 0.7))).mean()
    return np.sqrt(ewm_var) * np.sqrt(trading_days)

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ScientificAnalysisConfiguration:
    lookback_years: int = 5
    interval: str = "1d"
    annual_trading_days: int = 252
    risk_free_rate: float = 0.03  # annual

    correlation_method: str = field(default="pearson")
    ewma_lambda: float = 0.94
    ensure_psd_corr: bool = True

    var_confidence: float = 0.95
    var_horizon_days: int = 1
    use_student_t_parametric: bool = True

    rolling_beta_window: int = 63
    tracking_error_window: int = 63

    vol_ratio_green_max: float = 0.35
    vol_ratio_orange_max: float = 0.55

    te_green_max: float = 0.04
    te_orange_max: float = 0.08

    relvar_green_max: float = 1.0
    relvar_orange_max: float = 2.0

# =============================================================================
# DATA MANAGER
# =============================================================================
class ScientificDataManager:
    def __init__(self, cfg: ScientificAnalysisConfiguration):
        self.cfg = cfg

    @st.cache_data(show_spinner=False)
    def fetch_prices(_self, tickers: Tuple[str, ...], start: str, end: str, interval: str) -> pd.DataFrame:
        try:
            df = yf.download(
                list(tickers),
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if df is None or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" in df.columns.get_level_values(0):
                    close = df["Close"].copy()
                else:
                    close = df.xs("Close", axis=1, level=0, drop_level=False)
                    if isinstance(close, pd.DataFrame) and close.shape[1] > 0:
                        close.columns = close.columns.get_level_values(-1)
                close = close.sort_index()
                return close
            if "Close" in df.columns:
                out = df[["Close"]].copy()
                out.columns = [tickers[0]]
                return out
            return df
        except Exception:
            return pd.DataFrame()

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            return pd.DataFrame()
        rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        return rets

    def calculate_scientific_features(self, prices: pd.Series) -> pd.DataFrame:
        if prices is None or prices.dropna().empty:
            return pd.DataFrame()

        df = pd.DataFrame(index=prices.index)
        df["Price"] = prices.astype(float)
        df["Returns"] = df["Price"].pct_change()

        try:
            if df["Returns"].notna().sum() >= 120:
                r = df["Returns"].copy()

                df["EWMA_Vol_22"] = _ewma_vol_annualized(r, 22, self.cfg.annual_trading_days) * 100.0
                df["EWMA_Vol_33"] = _ewma_vol_annualized(r, 33, self.cfg.annual_trading_days) * 100.0
                df["EWMA_Vol_99"] = _ewma_vol_annualized(r, 99, self.cfg.annual_trading_days) * 100.0

                denom = (df["EWMA_Vol_33"] + df["EWMA_Vol_99"]) + 1e-12
                df["EWMA_Vol_Ratio_22_over_33_99"] = df["EWMA_Vol_22"] / denom

                ratio = df["EWMA_Vol_Ratio_22_over_33_99"]
                bb_n = 20
                if ratio.notna().sum() >= bb_n:
                    mid = ratio.rolling(bb_n, min_periods=int(bb_n * 0.8)).mean()
                    sd = ratio.rolling(bb_n, min_periods=int(bb_n * 0.8)).std()
                    df["EWMA_Ratio_BB_Mid"] = mid
                    df["EWMA_Ratio_BB_Upper"] = mid + 2.0 * sd
                    df["EWMA_Ratio_BB_Lower"] = mid - 2.0 * sd
        except Exception:
            pass

        return df

# =============================================================================
# CORRELATION ENGINE
# =============================================================================
class ScientificCorrelationEngine:
    def __init__(self, cfg: ScientificAnalysisConfiguration):
        self.cfg = cfg

    def _calculate_ewma_cov(self, data: pd.DataFrame, lam: float) -> np.ndarray:
        X = data.dropna().values
        if X.shape[0] < 5:
            return np.cov(data.dropna().values, rowvar=False)
        n = X.shape[0]
        w = np.array([(1 - lam) * (lam ** (n - 1 - i)) for i in range(n)], dtype=float)
        w = w / (w.sum() + 1e-12)
        mean = np.average(X, axis=0, weights=w)
        Xc = X - mean
        cov = (Xc.T * w) @ Xc
        return cov

    def _cov_to_corr(self, cov: np.ndarray, cols: List[str]) -> pd.DataFrame:
        d = np.sqrt(np.diag(cov))
        denom = np.outer(d, d) + 1e-12
        corr = cov / denom
        corr = np.clip(corr, -0.9999, 0.9999)
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=cols, columns=cols)

    def _calculate_ledoit_wolf_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        if not sci_dep_manager.is_available("sklearn"):
            st.warning("Ledoit-Wolf requires scikit-learn. Falling back to Pearson correlation.")
            return data.corr(method="pearson")
        try:
            from sklearn.covariance import LedoitWolf
            X = data.dropna().values
            if X.shape[0] < 30:
                return data.corr(method="pearson")
            lw = LedoitWolf().fit(X)
            cov = lw.covariance_
            return self._cov_to_corr(cov, list(data.columns))
        except Exception as e:
            st.warning(f"Ledoit-Wolf correlation failed: {e}. Falling back to Pearson.")
            return data.corr(method="pearson")

    def _calculate_correlation_method(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        m = (method or "pearson").lower().strip()
        if m in ("pearson", "spearman", "kendall"):
            return data.corr(method=m)
        if m == "ewma":
            cov = self._calculate_ewma_cov(data, self.cfg.ewma_lambda)
            return self._cov_to_corr(cov, list(data.columns))
        if m == "ledoit_wolf":
            return self._calculate_ledoit_wolf_correlation(data)
        return data.corr(method="pearson")

    def _higham_nearest_correlation(self, A: Union[np.ndarray, pd.DataFrame], max_iter: int = 100) -> pd.DataFrame:
        if isinstance(A, pd.DataFrame):
            idx = A.index
            cols = A.columns
            X = A.values.copy()
        else:
            X = np.array(A, dtype=float, copy=True)
            idx = list(range(X.shape[0]))
            cols = list(range(X.shape[1]))

        X = (X + X.T) / 2.0
        for _ in range(max_iter):
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            try:
                eigvals, eigvecs = np.linalg.eigh(Y)
                eigvals = np.maximum(eigvals, 0.0)
                X_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except Exception:
                break
            if np.linalg.norm(X_new - X, "fro") < 1e-10:
                X = X_new
                break
            X = X_new

        np.fill_diagonal(X, 1.0)
        X = np.clip(X, -0.9999, 0.9999)
        np.fill_diagonal(X, 1.0)
        return pd.DataFrame(X, index=idx, columns=cols)

    def ensure_psd(self, corr: pd.DataFrame) -> pd.DataFrame:
        if corr is None or corr.empty:
            return pd.DataFrame()
        corr = corr.copy()
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr.values, 1.0)
        try:
            eig = np.linalg.eigvalsh(corr.values)
            if np.min(eig) < -1e-10:
                corr = self._higham_nearest_correlation(corr)
        except Exception:
            corr = self._higham_nearest_correlation(corr)
        corr = corr.clip(-0.9999, 0.9999)
        np.fill_diagonal(corr.values, 1.0)
        return corr

    def compute_correlation(self, returns: pd.DataFrame, method: str) -> pd.DataFrame:
        if returns is None or returns.empty:
            return pd.DataFrame()
        data = returns.dropna(how="all").dropna(axis=1, how="all")
        if data.shape[1] < 2:
            return pd.DataFrame()
        corr = self._calculate_correlation_method(data, method)
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        if self.cfg.ensure_psd_corr:
            corr = self.ensure_psd(corr)
        return corr

# =============================================================================
# ANALYTICS ENGINE
# =============================================================================
class ScientificAnalyticsEngine:
    def __init__(self, cfg: ScientificAnalysisConfiguration):
        self.cfg = cfg

    def calculate_scientific_risk_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        if returns is None or returns.dropna().empty:
            return {}
        r = returns.dropna().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return {}

        ann_ret = _annualize_mean(r.mean(), self.cfg.annual_trading_days)
        ann_vol = _annualize_vol(r.std(ddof=1), self.cfg.annual_trading_days)
        sharpe = (ann_ret - self.cfg.risk_free_rate) / (ann_vol + 1e-12)

        downside = r[r < 0.0]
        downside_vol = _annualize_vol(downside.std(ddof=1), self.cfg.annual_trading_days) if len(downside) > 5 else 0.0
        sortino = (ann_ret - self.cfg.risk_free_rate) / (downside_vol + 1e-12) if downside_vol > 0 else 0.0

        mdd = _max_drawdown_from_returns(r)

        var_h, cvar_h = self._historical_var_cvar(r, self.cfg.var_confidence, self.cfg.var_horizon_days)
        var_p, cvar_p = self._parametric_var_cvar(r, self.cfg.var_confidence, self.cfg.var_horizon_days, use_t=self.cfg.use_student_t_parametric)

        beta, alpha = self._capm_beta_alpha(r, benchmark_returns)
        treynor = self._calculate_treynor_ratio(r, benchmark_returns)
        info_ratio = self._calculate_information_ratio(r, benchmark_returns)
        tracking_error = self._tracking_error(r, benchmark_returns)

        return {
            "Ann_Return": float(ann_ret),
            "Ann_Vol": float(ann_vol),
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "Max_Drawdown": float(mdd),
            "Hist_VaR": float(var_h),
            "Hist_CVaR_ES": float(cvar_h),
            "Param_VaR": float(var_p),
            "Param_CVaR": float(cvar_p),
            "Beta": float(beta),
            "Alpha": float(alpha),
            "Treynor_Ratio": float(treynor),
            "Information_Ratio": float(info_ratio),
            "Tracking_Error": float(tracking_error),
        }

    def _historical_var_cvar(self, r: pd.Series, confidence: float, horizon: int) -> Tuple[float, float]:
        rr = r.dropna()
        if rr.empty:
            return 0.0, 0.0
        scaled = rr * math.sqrt(max(1, int(horizon)))
        q = np.quantile(scaled, 1 - confidence)
        var = -float(q)
        tail = scaled[scaled <= q]
        cvar = -float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    def _parametric_var_cvar(self, r: pd.Series, confidence: float, horizon: int, use_t: bool = True) -> Tuple[float, float]:
        rr = r.dropna()
        if rr.empty:
            return 0.0, 0.0
        mu = rr.mean()
        sigma = rr.std(ddof=1)
        if sigma < 1e-12:
            return 0.0, 0.0
        h = math.sqrt(max(1, int(horizon)))
        mu_h = mu * max(1, int(horizon))
        sigma_h = sigma * h

        if use_t and len(rr) >= 80:
            try:
                df, loc, scale = stats.t.fit(rr.values)
                q = stats.t.ppf(1 - confidence, df, loc=loc, scale=scale)
                var = -(q * h)
                tail = rr[rr <= q]
                if len(tail) > 10:
                    cvar = -float(tail.mean()) * h
                else:
                    cvar = float(var)
                return float(var), float(cvar)
            except Exception:
                pass

        z = stats.norm.ppf(1 - confidence)
        var = -(mu_h + z * sigma_h)
        pdf = stats.norm.pdf(z)
        cvar = -(mu_h - sigma_h * pdf / (1 - confidence))
        return float(var), float(cvar)

    def _capm_beta_alpha(self, r: pd.Series, bench: Optional[pd.Series]) -> Tuple[float, float]:
        if bench is None or bench.dropna().empty:
            return 0.0, 0.0
        aligned = pd.DataFrame({"a": r, "m": bench}).dropna()
        if len(aligned) < 30:
            return 0.0, 0.0
        cov = aligned["a"].cov(aligned["m"])
        var_m = aligned["m"].var()
        beta = cov / var_m if var_m > 1e-12 else 0.0
        ann_a = _annualize_mean(aligned["a"].mean(), self.cfg.annual_trading_days)
        ann_m = _annualize_mean(aligned["m"].mean(), self.cfg.annual_trading_days)
        alpha = (ann_a - self.cfg.risk_free_rate) - beta * (ann_m - self.cfg.risk_free_rate)
        return float(beta), float(alpha)

    def _calculate_treynor_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        if benchmark_returns is None or benchmark_returns.dropna().empty:
            return 0.0
        aligned = pd.DataFrame({"a": returns, "m": benchmark_returns}).dropna()
        if len(aligned) < 30:
            return 0.0
        cov = aligned["a"].cov(aligned["m"])
        var_m = aligned["m"].var()
        beta = cov / var_m if var_m > 1e-12 else 0.0
        if abs(beta) < 1e-12:
            return 0.0
        ann_ret = _annualize_mean(aligned["a"].mean(), self.cfg.annual_trading_days)
        return float((ann_ret - self.cfg.risk_free_rate) / beta)

    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        if benchmark_returns is None or benchmark_returns.dropna().empty:
            return 0.0
        aligned = pd.DataFrame({"a": returns, "b": benchmark_returns}).dropna()
        if len(aligned) < 30:
            return 0.0
        active = aligned["a"] - aligned["b"]
        te = active.std(ddof=1) * math.sqrt(self.cfg.annual_trading_days)
        if te < 1e-12:
            return 0.0
        ann_active = _annualize_mean(active.mean(), self.cfg.annual_trading_days)
        return float(ann_active / te)

    def _tracking_error(self, returns: pd.Series, benchmark_returns: Optional[pd.Series]) -> float:
        if benchmark_returns is None or benchmark_returns.dropna().empty:
            return 0.0
        aligned = pd.DataFrame({"a": returns, "b": benchmark_returns}).dropna()
        if len(aligned) < 30:
            return 0.0
        active = aligned["a"] - aligned["b"]
        te = active.std(ddof=1) * math.sqrt(self.cfg.annual_trading_days)
        return float(te)

    def rolling_beta(self, returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
        df = pd.DataFrame({"a": returns, "m": benchmark_returns}).dropna()
        if df.shape[0] < window + 5:
            return pd.Series(index=returns.index, dtype=float)
        cov = df["a"].rolling(window).cov(df["m"])
        var = df["m"].rolling(window).var()
        beta = cov / (var + 1e-12)
        return beta.reindex(returns.index)

    def rolling_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
        df = pd.DataFrame({"a": returns, "b": benchmark_returns}).dropna()
        if df.shape[0] < window + 5:
            return pd.Series(index=returns.index, dtype=float)
        active = df["a"] - df["b"]
        te = active.rolling(window).std(ddof=1) * math.sqrt(self.cfg.annual_trading_days)
        return te.reindex(returns.index)

    def relative_var_cvar_es(self, returns: pd.Series, benchmark_returns: pd.Series, confidence: float, horizon: int) -> Dict[str, float]:
        df = pd.DataFrame({"a": returns, "b": benchmark_returns}).dropna()
        if df.shape[0] < 60:
            return {"Rel_Hist_VaR": 0.0, "Rel_Hist_CVaR_ES": 0.0, "Rel_Param_VaR": 0.0, "Rel_Param_CVaR": 0.0}
        active = df["a"] - df["b"]
        var_h, cvar_h = self._historical_var_cvar(active, confidence, horizon)
        var_p, cvar_p = self._parametric_var_cvar(active, confidence, horizon, use_t=self.cfg.use_student_t_parametric)
        scale = math.sqrt(self.cfg.annual_trading_days)
        return {
            "Rel_Hist_VaR": float(var_h * scale * 100.0),
            "Rel_Hist_CVaR_ES": float(cvar_h * scale * 100.0),
            "Rel_Param_VaR": float(var_p * scale * 100.0),
            "Rel_Param_CVaR": float(cvar_p * scale * 100.0),
        }

# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================
class ScientificVisualizationEngine:
    def __init__(self, cfg: ScientificAnalysisConfiguration):
        self.cfg = cfg

    def _create_empty_plot(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(height=420, template="plotly_white")
        return fig

    def create_correlation_heatmap(self, corr: pd.DataFrame, title: str) -> go.Figure:
        if corr is None or corr.empty:
            return self._create_empty_plot("No correlation data.")
        fig = px.imshow(
            corr,
            text_auto=False,
            aspect="auto",
            origin="lower",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        fig.update_layout(title=dict(text=title, x=0.5), height=720, template="plotly_white")
        return fig

    def create_volatility_ratio_signal_chart(self, features_df: pd.DataFrame, symbol: str, green_max: float, orange_max: float, title: str) -> go.Figure:
        if features_df is None or features_df.empty:
            return self._create_empty_plot("No features data.")
        if "EWMA_Vol_Ratio_22_over_33_99" not in features_df.columns:
            return self._create_empty_plot("EWMA ratio missing. Run analysis.")
        df = features_df.dropna(subset=["EWMA_Vol_Ratio_22_over_33_99"]).copy()
        if df.empty:
            return self._create_empty_plot("No valid EWMA ratio data.")
        ratio = df["EWMA_Vol_Ratio_22_over_33_99"]
        if "EWMA_Ratio_BB_Upper" not in df.columns or "EWMA_Ratio_BB_Lower" not in df.columns:
            bb_n = 20
            mid = ratio.rolling(bb_n, min_periods=int(bb_n * 0.8)).mean()
            sd = ratio.rolling(bb_n, min_periods=int(bb_n * 0.8)).std()
            df["EWMA_Ratio_BB_Mid"] = mid
            df["EWMA_Ratio_BB_Upper"] = mid + 2.0 * sd
            df["EWMA_Ratio_BB_Lower"] = mid - 2.0 * sd

        fig = go.Figure()
        ymax = max(1.5, float(ratio.max()) * 1.15) if ratio.notna().any() else 1.5

        fig.add_hrect(y0=0.0, y1=green_max, fillcolor="rgba(34,197,94,0.12)", line_width=0,
                      annotation_text="GREEN", annotation_position="top left")
        fig.add_hrect(y0=green_max, y1=orange_max, fillcolor="rgba(245,158,11,0.12)", line_width=0,
                      annotation_text="ORANGE", annotation_position="top left")
        fig.add_hrect(y0=orange_max, y1=ymax, fillcolor="rgba(239,68,68,0.12)", line_width=0,
                      annotation_text="RED", annotation_position="top left")

        fig.add_trace(go.Scatter(x=df.index, y=ratio, name="EWMA Vol Ratio (22 / (33+99))", mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["EWMA_Ratio_BB_Upper"], name="BB Upper", mode="lines", line=dict(width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=df["EWMA_Ratio_BB_Lower"], name="BB Lower", mode="lines", line=dict(width=1.5, dash="dash")))
        fig.add_hline(y=green_max, line_dash="dot", opacity=0.6, annotation_text=f"Green max = {green_max:.2f}")
        fig.add_hline(y=orange_max, line_dash="dot", opacity=0.6, annotation_text=f"Orange max = {orange_max:.2f}")

        last_x = df.index[-1]
        last_y = float(ratio.iloc[-1])
        fig.add_trace(go.Scatter(x=[last_x], y=[last_y], mode="markers", name="Latest", marker=dict(size=10)))

        fig.update_layout(
            title=dict(text=f"{title} — {symbol}", x=0.5),
            template="plotly_white",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
        )
        fig.update_yaxes(title_text="Ratio (unitless)", range=[0, ymax])
        fig.update_xaxes(title_text="Date")
        return fig

    def create_tracking_error_chart(self, te_series: pd.Series, symbol: str, green_max: float, orange_max: float, title: str = "Tracking Error (Annualized)") -> go.Figure:
        if te_series is None or te_series.dropna().empty:
            return self._create_empty_plot("No tracking error series available.")
        s = te_series.dropna().astype(float)
        ymax = max(float(s.max()) * 1.25, orange_max * 1.4, 0.10)

        fig = go.Figure()
        fig.add_hrect(y0=0.0, y1=green_max, fillcolor="rgba(34,197,94,0.12)", line_width=0,
                      annotation_text="GREEN", annotation_position="top left")
        fig.add_hrect(y0=green_max, y1=orange_max, fillcolor="rgba(245,158,11,0.12)", line_width=0,
                      annotation_text="ORANGE", annotation_position="top left")
        fig.add_hrect(y0=orange_max, y1=ymax, fillcolor="rgba(239,68,68,0.12)", line_width=0,
                      annotation_text="RED", annotation_position="top left")

        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Tracking Error", line=dict(width=2)))
        fig.add_hline(y=green_max, line_dash="dot", opacity=0.6, annotation_text=f"Green max = {green_max:.2%}")
        fig.add_hline(y=orange_max, line_dash="dot", opacity=0.6, annotation_text=f"Orange max = {orange_max:.2%}")
        fig.add_trace(go.Scatter(x=[s.index[-1]], y=[float(s.iloc[-1])], mode="markers", name="Latest", marker=dict(size=10)))

        fig.update_layout(
            title=dict(text=f"{title} — {symbol}", x=0.5),
            template="plotly_white",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
        )
        fig.update_yaxes(title_text="Tracking Error (decimal)", range=[0, ymax])
        fig.update_xaxes(title_text="Date")
        return fig

    def create_rolling_beta_chart(self, beta: pd.Series, symbol: str, title: str = "Rolling Beta") -> go.Figure:
        if beta is None or beta.dropna().empty:
            return self._create_empty_plot("No rolling beta available.")
        s = beta.dropna().astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Rolling Beta", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=[s.index[-1]], y=[float(s.iloc[-1])], mode="markers", name="Latest", marker=dict(size=10)))
        fig.add_hline(y=1.0, line_dash="dot", opacity=0.6, annotation_text="Beta = 1.0")
        fig.update_layout(
            title=dict(text=f"{title} — {symbol}", x=0.5),
            template="plotly_white",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
        )
        fig.update_yaxes(title_text="Beta")
        fig.update_xaxes(title_text="Date")
        return fig

    def create_relative_risk_chart(self, rel_risk_df: pd.DataFrame, symbol: str, green_max: float, orange_max: float, title: str = "Relative Risk vs Benchmark") -> go.Figure:
        if rel_risk_df is None or rel_risk_df.empty:
            return self._create_empty_plot("No relative risk history available.")
        df = rel_risk_df.dropna(how="all").copy()
        if df.empty:
            return self._create_empty_plot("No valid relative risk values.")
        series_candidates = [c for c in ["Rel_Hist_VaR", "Rel_Hist_CVaR_ES", "Rel_Param_VaR", "Rel_Param_CVaR"] if c in df.columns]
        if not series_candidates:
            return self._create_empty_plot("Relative risk columns missing.")
        primary = series_candidates[0]
        s = df[primary].dropna()
        ymax = max(float(s.max()) * 1.25, orange_max * 1.4, 3.0)

        fig = go.Figure()
        fig.add_hrect(y0=0.0, y1=green_max, fillcolor="rgba(34,197,94,0.12)", line_width=0,
                      annotation_text="GREEN", annotation_position="top left")
        fig.add_hrect(y0=green_max, y1=orange_max, fillcolor="rgba(245,158,11,0.12)", line_width=0,
                      annotation_text="ORANGE", annotation_position="top left")
        fig.add_hrect(y0=orange_max, y1=ymax, fillcolor="rgba(239,68,68,0.12)", line_width=0,
                      annotation_text="RED", annotation_position="top left")

        for c in series_candidates:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[c], mode="lines",
                name=c.replace("_", " "),
                line=dict(width=2 if c == primary else 1.5, dash="solid" if c == primary else "dash")
            ))
        fig.add_hline(y=green_max, line_dash="dot", opacity=0.6, annotation_text=f"Green max = {green_max:.2f}%")
        fig.add_hline(y=orange_max, line_dash="dot", opacity=0.6, annotation_text=f"Orange max = {orange_max:.2f}%")
        last = df[primary].dropna()
        if not last.empty:
            fig.add_trace(go.Scatter(x=[last.index[-1]], y=[float(last.iloc[-1])], mode="markers", name="Latest (primary)", marker=dict(size=10)))

        fig.update_layout(
            title=dict(text=f"{title} — {symbol}", x=0.5),
            template="plotly_white",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
        )
        fig.update_yaxes(title_text="Relative Risk (annualized %, proxy)", range=[0, ymax])
        fig.update_xaxes(title_text="Date")
        return fig

# =============================================================================
# MAIN PLATFORM
# =============================================================================
class ScientificCommoditiesPlatform:
    def __init__(self):
        self.cfg = ScientificAnalysisConfiguration()
        self.data_manager = ScientificDataManager(self.cfg)
        self.corr_engine = ScientificCorrelationEngine(self.cfg)
        self.analytics = ScientificAnalyticsEngine(self.cfg)
        self.viz = ScientificVisualizationEngine(self.cfg)

        if "selected_assets" not in st.session_state:
            st.session_state.selected_assets = ["GC=F", "SI=F", "CL=F", "HG=F"]
        if "selected_benchmarks" not in st.session_state:
            st.session_state.selected_benchmarks = ["^GSPC"]
        if "sc_results" not in st.session_state:
            st.session_state.sc_results = {}

        if "vol_ratio_thresholds" not in st.session_state:
            st.session_state.vol_ratio_thresholds = {"green_max": self.cfg.vol_ratio_green_max, "orange_max": self.cfg.vol_ratio_orange_max}
        if "te_thresholds" not in st.session_state:
            st.session_state.te_thresholds = {"green_max": self.cfg.te_green_max, "orange_max": self.cfg.te_orange_max}
        if "relrisk_thresholds" not in st.session_state:
            st.session_state.relrisk_thresholds = {"green_max": self.cfg.relvar_green_max, "orange_max": self.cfg.relvar_orange_max}

    def render_sidebar(self):
        st.sidebar.markdown("## ⚙️ Configuration")

        st.sidebar.markdown("### 📌 Asset Universe")
        default_assets = st.session_state.selected_assets
        assets = st.sidebar.multiselect(
            "Assets (tickers)",
            options=[
                "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F",
                "ZW=F", "ZC=F", "ZS=F", "KC=F", "CC=F",
                "^BCOM", "DX-Y.NYB"
            ],
            default=default_assets,
            help="Commodities futures (Yahoo tickers). Add/remove as needed.",
            key="assets_multiselect"
        )
        if len(assets) == 0:
            assets = default_assets
        st.session_state.selected_assets = assets

        st.sidebar.markdown("### 🧭 Benchmark")
        bench = st.sidebar.selectbox(
            "Benchmark (market proxy for Beta/Treynor/IR/TE/Relative Risk)",
            options=["^GSPC", "^NDX", "DXY", "XU100.IS", "^BSESN", "^N225"],
            index=0,
            key="benchmark_select"
        )
        st.session_state.selected_benchmarks = [bench]

        st.sidebar.markdown("### 🗓️ Time Range")
        lookback_years = st.sidebar.slider("Lookback Years", 1, 15, int(self.cfg.lookback_years), 1, key="lookback_years")
        self.cfg.lookback_years = lookback_years

        st.sidebar.markdown("### 🔗 Correlation Controls")
        corr_method = st.sidebar.selectbox(
            "Correlation Method",
            options=["pearson", "spearman", "kendall", "ewma", "ledoit_wolf"],
            index=0,
            help="ledoit_wolf requires scikit-learn. ewma uses decay lambda.",
            key="corr_method"
        )
        self.cfg.correlation_method = corr_method
        self.cfg.ensure_psd_corr = st.sidebar.checkbox("Force PSD correlation", value=True, key="psd_corr")
        if corr_method == "ewma":
            self.cfg.ewma_lambda = st.sidebar.slider("EWMA Lambda (decay)", 0.80, 0.99, float(self.cfg.ewma_lambda), 0.01, key="ewma_lambda")

        st.sidebar.markdown("### 📉 VaR / CVaR / ES")
        self.cfg.var_confidence = st.sidebar.slider("Confidence Level", 0.90, 0.99, float(self.cfg.var_confidence), 0.01, key="var_conf")
        self.cfg.var_horizon_days = st.sidebar.slider("Horizon (days)", 1, 20, int(self.cfg.var_horizon_days), 1, key="var_hor")
        self.cfg.use_student_t_parametric = st.sidebar.checkbox("Student-t Parametric VaR", value=True, key="use_t")

        st.sidebar.markdown("### 🧮 Rolling Windows")
        self.cfg.rolling_beta_window = st.sidebar.slider("Rolling Beta Window (days)", 20, 252, int(self.cfg.rolling_beta_window), 1, key="beta_win")
        self.cfg.tracking_error_window = st.sidebar.slider("Tracking Error Window (days)", 20, 252, int(self.cfg.tracking_error_window), 1, key="te_win")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🟢🟠🔴 Vol Ratio Risk Bands")
        gmax = st.sidebar.slider("Green max threshold (Ratio)", 0.10, 1.00, float(st.session_state.vol_ratio_thresholds.get("green_max", 0.35)), 0.01, key="vr_green")
        omax = st.sidebar.slider("Orange max threshold (Ratio)", min(1.50, gmax + 0.01), 1.50, max(float(st.session_state.vol_ratio_thresholds.get("orange_max", 0.55)), gmax + 0.01), 0.01, key="vr_orange")
        st.session_state.vol_ratio_thresholds = {"green_max": gmax, "orange_max": omax}

        st.sidebar.markdown("### 🟢🟠🔴 Tracking Error Bands (Annualized)")
        tg = st.sidebar.slider("Green max (TE)", 0.01, 0.20, float(st.session_state.te_thresholds.get("green_max", 0.04)), 0.005, key="te_green")
        to = st.sidebar.slider("Orange max (TE)", min(0.30, tg + 0.005), 0.30, max(float(st.session_state.te_thresholds.get("orange_max", 0.08)), tg + 0.005), 0.005, key="te_orange")
        st.session_state.te_thresholds = {"green_max": tg, "orange_max": to}

        st.sidebar.markdown("### 🟢🟠🔴 Relative Risk Bands (Annualized %)")
        rg = st.sidebar.slider("Green max (Relative risk %)", 0.25, 5.0, float(st.session_state.relrisk_thresholds.get("green_max", 1.0)), 0.05, key="rr_green")
        ro = st.sidebar.slider("Orange max (Relative risk %)", min(10.0, rg + 0.05), 10.0, max(float(st.session_state.relrisk_thresholds.get("orange_max", 2.0)), rg + 0.05), 0.05, key="rr_orange")
        st.session_state.relrisk_thresholds = {"green_max": rg, "orange_max": ro}

        st.sidebar.markdown("---")
        st.sidebar.markdown("### ▶️ Execute")
        run = st.sidebar.button("Run Scientific Analysis", key="run_analysis_btn")
        return run

    def run_scientific_analysis(self):
        assets = st.session_state.selected_assets
        bench = st.session_state.selected_benchmarks[0] if st.session_state.selected_benchmarks else "^GSPC"
        tickers = list(dict.fromkeys(list(assets) + [bench]))

        end = datetime.utcnow().date()
        start = end - timedelta(days=int(self.cfg.lookback_years * 365.25))

        prices = self.data_manager.fetch_prices(tuple(tickers), start=str(start), end=str(end), interval=self.cfg.interval)
        if prices is None or prices.empty:
            st.error("❌ No data downloaded. Please check tickers / internet / Yahoo availability.")
            return

        returns = self.data_manager.compute_returns(prices)

        bench_ret = None
        if bench in returns.columns:
            bench_ret = returns[bench].dropna()

        features: Dict[str, pd.DataFrame] = {}
        metrics: Dict[str, Dict[str, Any]] = {}
        rolling_beta: Dict[str, pd.Series] = {}
        rolling_te: Dict[str, pd.Series] = {}
        relrisk_hist: Dict[str, pd.DataFrame] = {}

        for a in assets:
            if a not in prices.columns:
                continue
            ser = prices[a].dropna()
            feat = self.data_manager.calculate_scientific_features(ser)
            features[a] = feat

            r = returns[a].dropna() if a in returns.columns else pd.Series(dtype=float)
            metrics[a] = self.analytics.calculate_scientific_risk_metrics(r, benchmark_returns=bench_ret)

            if bench_ret is not None and not bench_ret.empty and not r.empty:
                rolling_beta[a] = self.analytics.rolling_beta(r, bench_ret, window=self.cfg.rolling_beta_window)
                rolling_te[a] = self.analytics.rolling_tracking_error(r, bench_ret, window=self.cfg.tracking_error_window)

                win = max(120, int(self.cfg.tracking_error_window))
                df_ab = pd.DataFrame({"a": r, "b": bench_ret}).dropna()
                if df_ab.shape[0] >= win + 10:
                    rr_rows, idx = [], []
                    for i in range(win, df_ab.shape[0]):
                        sub = df_ab.iloc[i - win:i]
                        rr_rows.append(self.analytics.relative_var_cvar_es(sub["a"], sub["b"], self.cfg.var_confidence, self.cfg.var_horizon_days))
                        idx.append(sub.index[-1])
                    relrisk_hist[a] = pd.DataFrame(rr_rows, index=pd.Index(idx, name="Date"))
                else:
                    relrisk_hist[a] = pd.DataFrame()

        corr_in = returns[assets].dropna(how="all")
        corr = self.corr_engine.compute_correlation(corr_in, method=self.cfg.correlation_method)

        st.session_state.sc_results = {
            "prices": prices,
            "returns": returns,
            "benchmark": bench,
            "benchmark_returns": bench_ret,
            "features": features,
            "metrics": metrics,
            "corr": corr,
            "rolling_beta": rolling_beta,
            "rolling_te": rolling_te,
            "relrisk_hist": relrisk_hist,
            "config_snapshot": dict(self.cfg.__dict__),
            "timestamp": datetime.utcnow().isoformat()
        }

    def render(self):
        st.markdown(
            """
            <div class="institutional-hero">
              <h1>🏛️ Institutional Commodities Analytics Platform <span class="subtle">v7.2</span></h1>
              <p>Robust correlations • Institutional risk metrics • EWMA volatility risk signal • Tracking Error • Rolling Beta • Relative VaR/CVaR/ES</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        run_clicked = self.render_sidebar()
        if run_clicked:
            with st.spinner("Running scientific analysis..."):
                try:
                    self.run_scientific_analysis()
                    st.success("✅ Analysis complete.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.code(traceback.format_exc())

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "📈 Risk Analytics",
            "🧭 EWMA Vol Ratio Signal",
            "🔗 Correlation Analysis",
            "🎯 Tracking Error",
            "🧷 Rolling Beta",
            "⚖️ Relative VaR/CVaR/ES"
        ])

        with tab1:
            self.render_overview()
        with tab2:
            self.render_risk_analytics()
        with tab3:
            self.render_vol_ratio_signal()
        with tab4:
            self.render_correlation_analysis()
        with tab5:
            self.render_tracking_error()
        with tab6:
            self.render_rolling_beta()
        with tab7:
            self.render_relative_risk()

        st.markdown("---")
        self.render_data_validation()

    def render_overview(self):
        st.markdown(
            """
            <div class="section-header">
                <h2>📊 Overview</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">v7.2 Ultra</span>
                    <span class="scientific-badge">SciPy: {}</span>
                    <span class="scientific-badge">Plotly</span>
                </div>
            </div>
            """.format(scipy.__version__),
            unsafe_allow_html=True
        )

        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis from the sidebar to populate metrics, correlation, and signal tabs.")
            return

        prices: pd.DataFrame = res.get("prices", pd.DataFrame())
        returns: pd.DataFrame = res.get("returns", pd.DataFrame())
        bench = res.get("benchmark", "")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Assets Selected</div>
                <div class="metric-value">{len(st.session_state.selected_assets)}</div>
                <div class="subtle">{", ".join(st.session_state.selected_assets[:4])}{("..." if len(st.session_state.selected_assets)>4 else "")}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Benchmark</div>
                <div class="metric-value">{bench}</div>
                <div class="subtle">Used for Beta/Treynor/IR/TE/Relative Risk</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            nobs = int(prices.shape[0]) if prices is not None else 0
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Data Points</div>
                <div class="metric-value">{nobs}</div>
                <div class="subtle">{res.get("timestamp","")[:19]} UTC</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Correlation Method</div>
                <div class="metric-value">{self.cfg.correlation_method}</div>
                <div class="subtle">PSD enforced: {self.cfg.ensure_psd_corr}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Price Snapshot")
        if prices is not None and not prices.empty:
            st.line_chart(prices[st.session_state.selected_assets].dropna(how="all"))
        else:
            st.warning("No prices to display.")

        st.markdown("### Returns Snapshot (last 250)")
        if returns is not None and not returns.empty:
            st.dataframe(returns[st.session_state.selected_assets].tail(250), use_container_width=True)
        else:
            st.warning("No returns to display.")

    def render_risk_analytics(self):
        st.markdown(
            """
            <div class="section-header">
                <h2>📈 Risk Analytics</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">Sharpe • Sortino</span>
                    <span class="scientific-badge medium-risk">VaR/CVaR/ES</span>
                    <span class="scientific-badge">Treynor • IR (real benchmark)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return

        metrics: Dict[str, Dict[str, Any]] = res.get("metrics", {})
        if not metrics:
            st.warning("No metrics computed.")
            return

        df = pd.DataFrame(metrics).T
        for c in ["Ann_Return", "Ann_Vol", "Hist_VaR", "Hist_CVaR_ES", "Param_VaR", "Param_CVaR", "Tracking_Error"]:
            if c in df.columns:
                df[c] = df[c] * 100.0 if c in ["Ann_Return", "Ann_Vol"] else df[c] * 100.0

        st.dataframe(df.sort_index(), use_container_width=True)

        if "Sharpe" in df.columns:
            st.markdown("### Sharpe Comparison")
            fig = px.bar(df.reset_index().rename(columns={"index": "Asset"}), x="Asset", y="Sharpe")
            fig.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    def render_vol_ratio_signal(self):
        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return

        st.markdown(
            """
            <div class="section-header">
                <h2>🧭 EWMA Volatility Ratio Signal</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">(EWMA22)/(EWMA33+EWMA99)</span>
                    <span class="scientific-badge medium-risk">Bollinger Bands</span>
                    <span class="scientific-badge high-risk">Risk Zones</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        features: Dict[str, pd.DataFrame] = res.get("features", {})
        symbols = list(features.keys())
        if not symbols:
            st.warning("No features available.")
            return

        symbol = st.selectbox("Select Asset for Signal", options=symbols, index=0, key="vol_ratio_symbol")
        df = features.get(symbol, pd.DataFrame())
        if df is None or df.empty:
            st.warning("No data for selected asset.")
            return

        thr = st.session_state.get("vol_ratio_thresholds", {"green_max": 0.35, "orange_max": 0.55})
        green_max = float(thr.get("green_max", 0.35))
        orange_max = float(thr.get("orange_max", 0.55))

        if "EWMA_Vol_Ratio_22_over_33_99" in df.columns and df["EWMA_Vol_Ratio_22_over_33_99"].dropna().shape[0] > 0:
            last_val = float(df["EWMA_Vol_Ratio_22_over_33_99"].dropna().iloc[-1])
            if last_val <= green_max:
                zone, badge = "GREEN", "low-risk"
            elif last_val <= orange_max:
                zone, badge = "ORANGE", "medium-risk"
            else:
                zone, badge = "RED", "high-risk"

            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Latest Risk Signal</div>
                <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                    <div>
                        <div class="metric-value">{last_val:.3f}</div>
                        <div class="subtle">EWMA Vol Ratio (unitless)</div>
                    </div>
                    <div><span class="scientific-badge {badge}">{zone} ZONE</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        fig = self.viz.create_volatility_ratio_signal_chart(df, symbol, green_max, orange_max, "Institutional Risk Signal: EWMA Vol Ratio")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Signal Data (Last 80 rows)", expanded=False):
            cols = [c for c in ["EWMA_Vol_22", "EWMA_Vol_33", "EWMA_Vol_99",
                                "EWMA_Vol_Ratio_22_over_33_99",
                                "EWMA_Ratio_BB_Mid", "EWMA_Ratio_BB_Upper", "EWMA_Ratio_BB_Lower"] if c in df.columns]
            st.dataframe(df[cols].tail(80), use_container_width=True)

    def render_correlation_analysis(self):
        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return
        st.markdown(
            """
            <div class="section-header">
                <h2>🔗 Correlation Analysis</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">Correct alignment</span>
                    <span class="scientific-badge">PSD safe</span>
                    <span class="scientific-badge">Ledoit-Wolf optional</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        corr = res.get("corr", pd.DataFrame())
        if corr is None or corr.empty:
            st.warning("No correlation matrix computed.")
            return
        fig = self.viz.create_correlation_heatmap(corr, f"Asset Correlations — method: {self.cfg.correlation_method}")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📋 Correlation Table", expanded=False):
            st.dataframe(corr.round(4), use_container_width=True)

    def render_tracking_error(self):
        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return
        st.markdown(
            """
            <div class="section-header">
                <h2>🎯 Tracking Error</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">Active risk vs benchmark</span>
                    <span class="scientific-badge medium-risk">Green/Orange/Red Zones</span>
                    <span class="scientific-badge">Rolling window</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        te_map: Dict[str, pd.Series] = res.get("rolling_te", {})
        if not te_map:
            st.warning("Tracking error series not available (need benchmark + enough data).")
            return
        symbols = list(te_map.keys())
        symbol = st.selectbox("Select Asset", options=symbols, index=0, key="te_symbol")
        te_series = te_map.get(symbol, pd.Series(dtype=float))
        thr = st.session_state.get("te_thresholds", {"green_max": 0.04, "orange_max": 0.08})
        green_max, orange_max = float(thr.get("green_max", 0.04)), float(thr.get("orange_max", 0.08))

        if te_series is not None and te_series.dropna().shape[0] > 0:
            last = float(te_series.dropna().iloc[-1])
            if last <= green_max:
                zone, badge = "GREEN", "low-risk"
            elif last <= orange_max:
                zone, badge = "ORANGE", "medium-risk"
            else:
                zone, badge = "RED", "high-risk"
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Latest Tracking Error (annualized)</div>
                <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                    <div>
                        <div class="metric-value">{last:.2%}</div>
                        <div class="subtle">Rolling window: {self.cfg.tracking_error_window} days</div>
                    </div>
                    <div><span class="scientific-badge {badge}">{zone} ZONE</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        fig = self.viz.create_tracking_error_chart(te_series, symbol, green_max, orange_max, f"Tracking Error (Annualized) — Window {self.cfg.tracking_error_window}D")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📋 Tracking Error Data (Last 120 rows)", expanded=False):
            st.dataframe(te_series.dropna().to_frame("Tracking_Error").tail(120), use_container_width=True)

    def render_rolling_beta(self):
        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return
        st.markdown(
            """
            <div class="section-header">
                <h2>🧷 Rolling Beta</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">Rolling CAPM beta</span>
                    <span class="scientific-badge">Benchmark-linked</span>
                    <span class="scientific-badge">Window configurable</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        beta_map: Dict[str, pd.Series] = res.get("rolling_beta", {})
        if not beta_map:
            st.warning("Rolling beta not available (need benchmark + enough data).")
            return
        symbols = list(beta_map.keys())
        symbol = st.selectbox("Select Asset", options=symbols, index=0, key="beta_symbol")
        beta = beta_map.get(symbol, pd.Series(dtype=float))
        if beta is None or beta.dropna().empty:
            st.warning("No rolling beta data for this asset.")
            return
        last = float(beta.dropna().iloc[-1])
        st.markdown(f"""
        <div class="institutional-card">
            <div class="metric-title">Latest Rolling Beta</div>
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                <div>
                    <div class="metric-value">{last:.3f}</div>
                    <div class="subtle">Window: {self.cfg.rolling_beta_window} days</div>
                </div>
                <div><span class="scientific-badge info">Beta vs {res.get("benchmark","")}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = self.viz.create_rolling_beta_chart(beta, symbol, f"Rolling Beta — Window {self.cfg.rolling_beta_window}D")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📋 Beta Data (Last 120 rows)", expanded=False):
            st.dataframe(beta.dropna().to_frame("Rolling_Beta").tail(120), use_container_width=True)

    def render_relative_risk(self):
        res = st.session_state.get("sc_results", {})
        if not res:
            st.info("Run analysis first.")
            return
        st.markdown(
            """
            <div class="section-header">
                <h2>⚖️ Relative VaR / CVaR / ES vs Benchmark</h2>
                <div class="section-actions">
                    <span class="scientific-badge info">Active returns risk</span>
                    <span class="scientific-badge medium-risk">Band zones</span>
                    <span class="scientific-badge">Rolling history</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        rel_map: Dict[str, pd.DataFrame] = res.get("relrisk_hist", {})
        if not rel_map:
            st.warning("Relative risk history not available.")
            return
        symbols = [s for s, df in rel_map.items() if df is not None]
        if not symbols:
            st.warning("No relative risk frames computed.")
            return
        symbol = st.selectbox("Select Asset", options=symbols, index=0, key="rel_symbol")
        df = rel_map.get(symbol, pd.DataFrame())
        if df is None or df.empty:
            st.warning("Not enough data for rolling relative risk. Increase lookback or reduce window.")
            return
        thr = st.session_state.get("relrisk_thresholds", {"green_max": 1.0, "orange_max": 2.0})
        green_max, orange_max = float(thr.get("green_max", 1.0)), float(thr.get("orange_max", 2.0))

        primary = "Rel_Hist_VaR" if "Rel_Hist_VaR" in df.columns else df.columns[0]
        last = df[primary].dropna()
        if not last.empty:
            last_val = float(last.iloc[-1])
            if last_val <= green_max:
                zone, badge = "GREEN", "low-risk"
            elif last_val <= orange_max:
                zone, badge = "ORANGE", "medium-risk"
            else:
                zone, badge = "RED", "high-risk"
            st.markdown(f"""
            <div class="institutional-card">
                <div class="metric-title">Latest Relative Risk (primary: {primary})</div>
                <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                    <div>
                        <div class="metric-value">{last_val:.2f}%</div>
                        <div class="subtle">Annualized risk proxy from active returns</div>
                    </div>
                    <div><span class="scientific-badge {badge}">{zone} ZONE</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        fig = self.viz.create_relative_risk_chart(df, symbol, green_max, orange_max, f"Relative VaR/CVaR/ES vs {res.get('benchmark','')}")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📋 Relative Risk Data (Last 120 rows)", expanded=False):
            st.dataframe(df.tail(120), use_container_width=True)

    def render_data_validation(self):
        res = st.session_state.get("sc_results", {})
        st.markdown(
            """
            <div class="section-header">
                <h2>📋 Data & Validation</h2>
                <div class="section-actions">
                    <span class="scientific-badge">Quality checks</span>
                    <span class="scientific-badge">Overlap / NA</span>
                    <span class="scientific-badge">Diagnostics</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if not res:
            st.info("No results yet.")
            return

        prices: pd.DataFrame = res.get("prices", pd.DataFrame())
        returns: pd.DataFrame = res.get("returns", pd.DataFrame())
        assets = st.session_state.selected_assets
        bench = res.get("benchmark", "")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Missingness (Prices)")
            if prices is not None and not prices.empty:
                miss = (prices[assets + [bench]].isna().mean() * 100.0).sort_values(ascending=False)
                st.dataframe(miss.to_frame("Missing %").round(2), use_container_width=True)
            else:
                st.warning("No price data.")
        with c2:
            st.markdown("#### Missingness (Returns)")
            if returns is not None and not returns.empty:
                miss = (returns[assets + [bench]].isna().mean() * 100.0).sort_values(ascending=False)
                st.dataframe(miss.to_frame("Missing %").round(2), use_container_width=True)
            else:
                st.warning("No return data.")

        st.markdown("#### Notes")
        st.write(
            "- Correlations computed after aligning returns and dropping all-NA columns.
"
            "- PSD enforcement ensures correlation matrix is numerically valid for risk engines.
"
            "- Treynor / Information Ratio / Tracking Error / Rolling Beta are computed vs selected benchmark.
"
            "- Relative VaR/CVaR/ES uses active returns (asset - benchmark) and is shown as an annualized % proxy."
        )

# =============================================================================
# MAIN
# =============================================================================
def main():
    try:
        app = ScientificCommoditiesPlatform()
        app.render()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
