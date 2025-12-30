"""
🏛️ Institutional Commodities Analytics Platform v7.4 (Cloud-Safe, Institutional)
Integrated Portfolio Analytics • Correct Correlations • Robust VaR/CVaR/ES • GARCH • Optional Regimes
Single-file Streamlit application designed for Streamlit Cloud reliability.

Key fixes included:
- Correlation matrix: correct alignment + variance checks + optional Ledoit–Wolf shrinkage + PSD enforcement (Higham)
- Risk metrics: VaR/CVaR/ES fixed (left-tail), no NaNs from mis-specified quantiles
- Backward compatibility: stress_test(shock=...) accepted; garch_analysis(p=..., q=...) accepted; detect_regimes(n_states=...) accepted
- Reporting: Excel export with engine fallback (openpyxl → xlsxwriter → csv)
"""

from __future__ import annotations

import os
import gc
import math
import time
import json
import logging
import warnings
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("institutional_commodities")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

class DependencyManager:
    """Lazy optional imports with safe fallbacks."""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_available(self, name: str) -> bool:
        self._load(name)
        return bool(self._cache.get(name, {}).get("available", False))

    def get(self, name: str, key: str, default=None):
        self._load(name)
        return self._cache.get(name, {}).get(key, default)

    def _load(self, name: str):
        if name in self._cache:
            return
        rec: Dict[str, Any] = {"available": False, "error": None}
        try:
            if name == "sklearn":
                from sklearn.covariance import LedoitWolf
                rec.update({"available": True, "LedoitWolf": LedoitWolf})
            elif name == "arch":
                from arch import arch_model
                rec.update({"available": True, "arch_model": arch_model})
            elif name == "hmmlearn":
                # hmmlearn requires scikit-learn
                from hmmlearn.hmm import GaussianHMM
                from sklearn.preprocessing import StandardScaler
                rec.update({"available": True, "GaussianHMM": GaussianHMM, "StandardScaler": StandardScaler})
            elif name == "openpyxl":
                import openpyxl  # noqa: F401
                rec.update({"available": True})
            elif name == "xlsxwriter":
                import xlsxwriter  # noqa: F401
                rec.update({"available": True})
            elif name == "psutil":
                import psutil  # noqa: F401
                rec.update({"available": True})
            elif name == "statsmodels":
                import statsmodels.api as sm
                rec.update({"available": True, "sm": sm})
            else:
                rec.update({"available": False, "error": f"Unknown dependency key: {name}"})
        except Exception as e:
            rec.update({"available": False, "error": str(e)})
        self._cache[name] = rec


dep = DependencyManager()


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AnalysisConfiguration:
    assets: List[str] = field(default_factory=lambda: ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"])
    benchmark: str = "SPY"
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365*3))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    risk_free_ticker: str = "^IRX"  # 13-week T-bill proxy (yfinance)
    price_field: str = "Adj Close"
    min_obs: int = 150

    # Correlation settings
    corr_method: str = "pearson"      # pearson/spearman/kendall/ewma
    corr_shrinkage: str = "none"      # none/ledoitwolf
    corr_psd: bool = True
    ewma_lambda: float = 0.94

    # VaR settings
    var_confidence: float = 0.95
    var_method: str = "historical"    # historical/parametric/modified/student_t
    var_horizon: int = 1

    # Institutional signal settings
    ewma_fast: int = 22
    ewma_mid: int = 33
    ewma_slow: int = 99
    bb_window: int = 60
    bb_k: float = 2.0
    sig_green: float = 0.90
    sig_orange: float = 1.10

    # Beta / TE
    beta_window: int = 252
    te_window: int = 252

    # GARCH
    garch_p_range: Tuple[int, int] = (1, 2)
    garch_q_range: Tuple[int, int] = (1, 2)
    garch_forecast_horizon: int = 10


# =============================================================================
# UTILITIES
# =============================================================================

def _to_datetime(x) -> datetime:
    if isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return datetime.now()

def _pct(x: float) -> float:
    try:
        return float(x) * 100.0
    except Exception:
        return 0.0

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)

def _as_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError("Expected Series or single-column DataFrame")
    return x


# =============================================================================
# DATA MANAGER
# =============================================================================

class EnhancedDataManager:
    def __init__(self):
        self.asset_prices: pd.DataFrame = pd.DataFrame()
        self.benchmark_prices: pd.Series = pd.Series(dtype=float)
        self.returns: pd.DataFrame = pd.DataFrame()
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.risk_free: pd.Series = pd.Series(dtype=float)
        self.quality_report: pd.DataFrame = pd.DataFrame()

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=60*60)
    def _download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        # yfinance is fastest in one bulk call
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return df

    def load(self, cfg: AnalysisConfiguration) -> Dict[str, Any]:
        t0 = time.time()
        assets = [t.strip().upper() for t in cfg.assets if t and isinstance(t, str)]
        bench = cfg.benchmark.strip().upper() if cfg.benchmark else ""
        rf = cfg.risk_free_ticker.strip().upper() if cfg.risk_free_ticker else ""

        tickers = sorted(list(set(assets + ([bench] if bench else []) + ([rf] if rf else []))))
        start = _to_datetime(cfg.start_date).strftime("%Y-%m-%d")
        end = (_to_datetime(cfg.end_date) + timedelta(days=1)).strftime("%Y-%m-%d")

        if not tickers:
            return {"ok": False, "error": "No tickers selected."}

        raw = self._download(tickers=tickers, start=start, end=end)

        # Normalize raw format: when multiple tickers, yfinance uses column multiindex (field, ticker) or (ticker, field)
        prices = pd.DataFrame(index=raw.index)
        field = cfg.price_field

        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Try both orientations
                if field in raw.columns.get_level_values(0):
                    # columns: (field, ticker)
                    for t in tickers:
                        if (field, t) in raw.columns:
                            prices[t] = raw[(field, t)]
                else:
                    # columns: (ticker, field)
                    for t in tickers:
                        if (t, field) in raw.columns:
                            prices[t] = raw[(t, field)]
            else:
                # Single ticker, flat columns
                if field in raw.columns:
                    prices[tickers[0]] = raw[field]
        except Exception as e:
            return {"ok": False, "error": f"Failed parsing yfinance response: {e}"}

        prices = prices.sort_index().dropna(how="all")
        if prices.empty or len(prices) < 50:
            return {"ok": False, "error": "Insufficient price data downloaded."}

        # Split assets / benchmark / rf
        self.asset_prices = prices[[t for t in assets if t in prices.columns]].copy()
        self.benchmark_prices = prices[bench].copy() if bench in prices.columns else pd.Series(dtype=float)
        rf_prices = prices[rf].copy() if rf in prices.columns else pd.Series(dtype=float)

        # Compute returns (simple returns)
        self.returns = self.asset_prices.pct_change().replace([np.inf, -np.inf], np.nan)
        self.benchmark_returns = self.benchmark_prices.pct_change().replace([np.inf, -np.inf], np.nan) if not self.benchmark_prices.empty else pd.Series(dtype=float)

        # Risk-free: ^IRX is a yield (%). Convert to daily rate approx: rf_daily ≈ (yield/100)/252
        if not rf_prices.empty:
            rf_yield = rf_prices / 100.0
            self.risk_free = (rf_yield / 252.0).rename("rf_daily")
        else:
            self.risk_free = pd.Series(dtype=float)

        # Quality report
        self.quality_report = self._build_quality_report(cfg)

        load_time = time.time() - t0
        return {
            "ok": True,
            "load_time_s": load_time,
            "assets_loaded": list(self.asset_prices.columns),
            "n_obs": int(self.returns.shape[0]) if isinstance(self.returns, pd.DataFrame) else 0,
        }

    def _build_quality_report(self, cfg: AnalysisConfiguration) -> pd.DataFrame:
        rep = []
        df = self.returns.copy()
        if df.empty:
            return pd.DataFrame()

        for c in df.columns:
            s = df[c]
            rep.append({
                "ticker": c,
                "obs": int(s.notna().sum()),
                "missing_pct": float(100 * s.isna().mean()),
                "stdev": float(s.std(skipna=True)),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
                "last_date": str(s.dropna().index.max().date()) if s.notna().any() else None
            })

        rep_df = pd.DataFrame(rep).sort_values(["obs"], ascending=False)
        # Flags
        rep_df["flag_low_obs"] = rep_df["obs"] < int(cfg.min_obs)
        rep_df["flag_constant"] = rep_df["stdev"] < 1e-12
        return rep_df


# =============================================================================
# ANALYTICS
# =============================================================================

class InstitutionalAnalytics:
    def __init__(self, annual_trading_days: int = 252):
        self.annual_trading_days = annual_trading_days

    # ------------------------------
    # Correlation (Correct + PSD)
    # ------------------------------
    @staticmethod
    def _symmetrize(a: np.ndarray) -> np.ndarray:
        return (a + a.T) / 2.0

    @staticmethod
    def _project_psd(a: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        a = InstitutionalAnalytics._symmetrize(a)
        vals, vecs = np.linalg.eigh(a)
        vals = np.maximum(vals, epsilon)
        return InstitutionalAnalytics._symmetrize((vecs * vals) @ vecs.T)

    def _higham_nearest_correlation(self, corr: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
        """Higham (2002) nearest correlation matrix (Frobenius) via alternating projections."""
        y = corr.copy()
        np.fill_diagonal(y, 1.0)
        y = self._symmetrize(y)

        delta_s = np.zeros_like(y)
        base_norm = np.linalg.norm(y, ord="fro") + 1e-12

        for _ in range(max_iter):
            r = y - delta_s
            x = self._project_psd(r, epsilon=1e-10)
            delta_s = x - r
            y_new = x.copy()
            np.fill_diagonal(y_new, 1.0)
            y_new = self._symmetrize(y_new)

            rel_err = np.linalg.norm(y_new - y, ord="fro") / base_norm
            y = y_new
            if rel_err < tol:
                break

        # clamp numeric noise
        y = np.clip(y, -1.0, 1.0)
        np.fill_diagonal(y, 1.0)
        return self._symmetrize(y)

    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = "pearson",
        shrinkage: str = "none",
        ensure_psd: bool = True,
        ewma_lambda: float = 0.94,
        min_pair_obs: int = 60,
    ) -> pd.DataFrame:
        """
        Correct correlation computation:
        - Align by date and drop constant / all-NA series
        - Optional Ledoit–Wolf covariance shrinkage -> correlation
        - Optional PSD repair (Higham nearest correlation)
        """
        if returns_df is None or not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
            return pd.DataFrame()

        # Only numeric
        df = returns_df.copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

        # Drop columns with too few obs or near-constant variance
        keep = []
        for c in df.columns:
            s = df[c].dropna()
            if len(s) >= min_pair_obs and s.std() > 1e-12:
                keep.append(c)
        df = df[keep].copy()

        if df.shape[1] < 2:
            return pd.DataFrame()

        method = (method or "pearson").lower().strip()
        shrinkage = (shrinkage or "none").lower().strip()

        # Pairwise correlation (pandas handles pairwise deletion)
        if method in {"pearson", "spearman", "kendall"}:
            corr = df.corr(method=method, min_periods=min_pair_obs)
        elif method == "ewma":
            # EWMA covariance then convert to correlation
            # Compute demeaned returns
            x = df - df.mean()
            # EWMA weights (most recent highest): w_t = (1-lam)*lam^(T-1-t)
            lam = float(ewma_lambda)
            T = x.shape[0]
            w = np.array([(1 - lam) * (lam ** (T - 1 - i)) for i in range(T)], dtype=float)
            w = w / (w.sum() + 1e-12)
            X = x.values
            cov = (X.T * w) @ X  # weighted covariance, demeaned
            d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
            corr_np = cov / np.outer(d, d)
            corr_np = np.clip(corr_np, -1.0, 1.0)
            np.fill_diagonal(corr_np, 1.0)
            corr = pd.DataFrame(corr_np, index=df.columns, columns=df.columns)
        else:
            corr = df.corr(method="pearson", min_periods=min_pair_obs)

        # Optional Ledoit–Wolf shrinkage (cov shrinkage -> corr)
        if shrinkage == "ledoitwolf" and dep.is_available("sklearn"):
            try:
                LedoitWolf = dep.get("sklearn", "LedoitWolf")
                lw = LedoitWolf().fit(df.dropna().values)  # uses complete cases
                cov = lw.covariance_
                d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
                corr_np = cov / np.outer(d, d)
                corr_np = np.clip(corr_np, -1.0, 1.0)
                np.fill_diagonal(corr_np, 1.0)
                corr = pd.DataFrame(corr_np, index=df.columns, columns=df.columns)
            except Exception as e:
                logger.warning(f"LedoitWolf shrinkage failed (fallback to {method}): {e}")

        # Clean NaNs off-diagonal where insufficient overlap
        corr = corr.reindex(index=df.columns, columns=df.columns)
        corr = corr.astype(float)

        # ensure diag=1
        np.fill_diagonal(corr.values, 1.0)

        if ensure_psd:
            try:
                corr_np = corr.values
                corr_psd = self._higham_nearest_correlation(corr_np, max_iter=120, tol=1e-7)
                corr = pd.DataFrame(corr_psd, index=corr.index, columns=corr.columns)
            except Exception as e:
                logger.warning(f"PSD repair failed: {e}")

        return corr

    # ------------------------------
    # Performance metrics
    # ------------------------------
    def calculate_performance_metrics(self, returns: pd.Series, rf_daily: Optional[pd.Series] = None) -> Dict[str, Any]:
        s = _as_series(returns).dropna()
        if s.empty or len(s) < 20:
            return {}

        rf = rf_daily.dropna() if isinstance(rf_daily, pd.Series) else pd.Series(dtype=float)
        if not rf.empty:
            aligned = pd.concat([s, rf], axis=1, join="inner").dropna()
            if not aligned.empty:
                s = aligned.iloc[:, 0]
                rf = aligned.iloc[:, 1]
            else:
                rf = pd.Series(dtype=float)

        cum = (1 + s).cumprod()
        total_ret = cum.iloc[-1] - 1.0
        ann_ret = (1 + total_ret) ** (self.annual_trading_days / max(len(s), 1)) - 1.0
        ann_vol = s.std() * math.sqrt(self.annual_trading_days)

        # drawdown
        peak = cum.cummax()
        dd = (cum / peak - 1.0)
        max_dd = dd.min()

        # downside / sortino
        downside = s[s < 0].std() * math.sqrt(self.annual_trading_days)
        if downside <= 0:
            downside = np.nan

        if not rf.empty:
            rf_ann = rf.mean() * self.annual_trading_days
        else:
            rf_ann = 0.0

        sharpe = (ann_ret - rf_ann) / (ann_vol + 1e-12)
        sortino = (ann_ret - rf_ann) / (downside + 1e-12) if not np.isnan(downside) else np.nan

        return {
            "n_obs": int(len(s)),
            "total_return": float(total_ret),
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "max_drawdown": float(max_dd),
            "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        }

    # ------------------------------
    # VaR / CVaR / ES (Fixed)
    # ------------------------------
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
        horizon: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Returns VaR/CVaR as POSITIVE loss percentages:
          VaR = -Q_alpha(returns), alpha = 1 - confidence
          CVaR/ES = -E[returns | returns <= Q_alpha]
        Fixes common bug: using right-tail quantiles (ppf(confidence)) which can flip sign.
        """
        r = _as_series(returns).dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 100:
            logger.warning(f"Insufficient data for VaR: {len(r)} observations")
            return {}

        alpha = 1.0 - float(confidence_level)
        method = (method or "historical").lower().strip()

        # Horizon handling (approx). For more exact: use aggregated horizon returns.
        if int(horizon) > 1:
            r_scaled = r * math.sqrt(int(horizon))
        else:
            r_scaled = r

        try:
            if method == "historical":
                q = np.nanpercentile(r_scaled.values, alpha * 100.0)
            elif method == "parametric":
                mu = float(r_scaled.mean())
                sigma = float(r_scaled.std())
                z = float(stats.norm.ppf(alpha))
                q = mu + sigma * z
            elif method == "modified":
                mu = float(r_scaled.mean())
                sigma = float(r_scaled.std())
                skew = float(r_scaled.skew())
                kurt = float(r_scaled.kurtosis())
                z = float(stats.norm.ppf(alpha))
                z_cf = (z
                        + (z**2 - 1.0) * skew / 6.0
                        + (z**3 - 3.0*z) * kurt / 24.0
                        - (2.0*z**3 - 5.0*z) * (skew**2) / 36.0)
                q = mu + sigma * z_cf
            elif method == "student_t":
                from scipy.stats import t
                df, loc, scale = t.fit(r_scaled.values)
                q = float(t.ppf(alpha, df, loc=loc, scale=scale))
            else:
                q = np.nanpercentile(r_scaled.values, alpha * 100.0)

            # VaR/CVaR as positive losses
            var = -float(q)
            tail = r_scaled[r_scaled <= q]
            cvar = -float(tail.mean()) if len(tail) > 0 else var

            # Exceedances (actual returns below quantile)
            exceed = int((r_scaled <= q).sum())
            exceed_rate = exceed / max(len(r_scaled), 1)

            backtest = {}
            if len(r_scaled) > 250:
                expected = alpha * len(r_scaled)
                dev = abs(exceed - expected) / (expected + 1e-12)
                backtest = {
                    "expected_exceedances": float(expected),
                    "actual_exceedances": float(exceed),
                    "deviation_pct": float(dev * 100.0),
                    "pass": bool(dev < 0.2),
                }

            return {
                "var": float(var) * 100.0,
                "cvar": float(cvar) * 100.0,
                "quantile_return": float(q) * 100.0,
                "confidence_level": float(confidence_level),
                "alpha": float(alpha),
                "method": method,
                "horizon": int(horizon),
                "observations": int(len(r_scaled)),
                "exceedances": exceed,
                "exceedance_rate": float(exceed_rate) * 100.0,
                "backtest": backtest,
            }
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return {}

    # ------------------------------
    # Stress test (compat: shock=...)
    # ------------------------------
    def stress_test(
        self,
        returns: pd.Series,
        scenarios: Optional[List[float]] = None,
        include_historical: bool = True,
        historical_percentiles: Optional[List[float]] = None,
        shock: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Stress testing. Supports legacy call: stress_test(returns, shock=-0.05)."""
        r = _as_series(returns).dropna()
        if len(r) < 100:
            return {}

        if scenarios is None:
            scenarios = [-0.01, -0.02, -0.05, -0.10, -0.20]
        if shock is not None:
            # Accept single shock as override
            scenarios = [float(shock)]

        if historical_percentiles is None:
            historical_percentiles = [0.01, 0.05, 0.10]

        out: Dict[str, Any] = {}
        for sh in scenarios:
            shocked = r + float(sh)
            mets = self.calculate_performance_metrics(shocked)
            var_95 = self.calculate_var(shocked, confidence_level=0.95, method="historical")
            out[f"shock_{abs(sh)*100:.0f}%"] = {
                "shock_pct": float(sh) * 100.0,
                "mean_return_pct": float(shocked.mean() * 100.0),
                "ann_vol_pct": float(shocked.std() * math.sqrt(self.annual_trading_days) * 100.0),
                "max_drawdown_pct": float(mets.get("max_drawdown", np.nan) * 100.0) if mets else np.nan,
                "var_95_pct": float(var_95.get("var", np.nan)) if var_95 else np.nan,
                "cvar_95_pct": float(var_95.get("cvar", np.nan)) if var_95 else np.nan,
            }

        if include_historical:
            for p in historical_percentiles:
                q = float(np.nanpercentile(r.values, p * 100.0))
                shocked = r + q
                mets = self.calculate_performance_metrics(shocked)
                out[f"historical_{int(p*100)}pct"] = {
                    "shock_pct": float(q) * 100.0,
                    "percentile": float(p),
                    "max_drawdown_pct": float(mets.get("max_drawdown", np.nan) * 100.0) if mets else np.nan,
                }

        return out

    # ------------------------------
    # Tracking error & rolling beta
    # ------------------------------
    def tracking_error(self, returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
        r = _as_series(returns)
        b = _as_series(benchmark)
        df = pd.concat([r, b], axis=1, join="inner").dropna()
        if df.empty:
            return pd.Series(dtype=float)
        active = df.iloc[:, 0] - df.iloc[:, 1]
        te = active.rolling(window=window, min_periods=max(20, window//3)).std() * math.sqrt(self.annual_trading_days)
        te.name = "tracking_error"
        return te

    def rolling_beta(self, returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
        r = _as_series(returns)
        b = _as_series(benchmark)
        df = pd.concat([r, b], axis=1, join="inner").dropna()
        if df.empty:
            return pd.Series(dtype=float)
        rr = df.iloc[:, 0]
        bb = df.iloc[:, 1]
        cov = rr.rolling(window=window, min_periods=max(40, window//3)).cov(bb)
        var = bb.rolling(window=window, min_periods=max(40, window//3)).var()
        beta = cov / (var + 1e-12)
        beta.name = "rolling_beta"
        return beta

    # ------------------------------
    # Relative VaR (active returns)
    # ------------------------------
    def relative_var(self, returns: pd.Series, benchmark: pd.Series, confidence_level: float = 0.95, method: str = "historical", horizon: int = 1) -> Dict[str, Any]:
        r = _as_series(returns)
        b = _as_series(benchmark)
        df = pd.concat([r, b], axis=1, join="inner").dropna()
        if df.empty or len(df) < 100:
            return {}
        active = df.iloc[:, 0] - df.iloc[:, 1]
        res = self.calculate_var(active, confidence_level=confidence_level, method=method, horizon=horizon)
        res["series"] = "active_returns"
        return res

    # ------------------------------
    # EWMA vol ratio signal
    # ------------------------------
    def ewma_vol(self, returns: pd.Series, span: int) -> pd.Series:
        r = _as_series(returns).dropna()
        # EWMA volatility of returns (sqrt of EWMA variance)
        var = r.ewm(span=span, adjust=False, min_periods=max(20, span//3)).var(bias=False)
        vol = np.sqrt(var) * math.sqrt(self.annual_trading_days)
        vol.name = f"ewma_vol_{span}"
        return vol

    def ewma_vol_ratio_signal(self, returns: pd.Series, fast: int = 22, mid: int = 33, slow: int = 99) -> pd.Series:
        v_fast = self.ewma_vol(returns, fast)
        v_mid = self.ewma_vol(returns, mid)
        v_slow = self.ewma_vol(returns, slow)
        df = pd.concat([v_fast, v_mid, v_slow], axis=1, join="inner").dropna()
        if df.empty:
            return pd.Series(dtype=float)
        ratio = df.iloc[:, 0] / (df.iloc[:, 1] + df.iloc[:, 2] + 1e-12)
        ratio.name = "ewma_vol_ratio"
        return ratio

    # ------------------------------
    # GARCH (compat: p=..., q=...)
    # ------------------------------
    def garch_analysis(
        self,
        returns: pd.Series,
        p_range: Tuple[int, int] = (1, 2),
        q_range: Tuple[int, int] = (1, 2),
        distributions: Optional[List[str]] = None,
        include_forecast: bool = True,
        forecast_horizon: int = 10,
        p: Optional[int] = None,
        q: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Grid-search GARCH(p,q) across distributions. Accepts legacy p/q args."""
        if p is not None:
            p_range = (int(p), int(p))
        if q is not None:
            q_range = (int(q), int(q))

        if distributions is None:
            distributions = ["normal", "t", "skewt"]

        r = _as_series(returns).dropna()
        if len(r) < 300:
            return {"available": False, "message": "Insufficient data (need >= 300 obs)."}

        if not dep.is_available("arch"):
            return {"available": False, "message": "ARCH package not installed. Add 'arch' to requirements."}

        arch_model = dep.get("arch", "arch_model")

        r_scaled = (r * 100.0).astype(float)

        candidates: List[Dict[str, Any]] = []
        best = None

        for pp in range(p_range[0], p_range[1] + 1):
            for qq in range(q_range[0], q_range[1] + 1):
                for dist in distributions:
                    try:
                        model = arch_model(r_scaled, mean="Constant", vol="GARCH", p=pp, q=qq, dist=dist)
                        fit = model.fit(disp="off", show_warning=False, update_freq=0, options={"maxiter": 1000})
                        aic = float(fit.aic)
                        bic = float(fit.bic)
                        rec = {
                            "p": int(pp), "q": int(qq), "dist": str(dist),
                            "aic": aic, "bic": bic,
                            "params": {k: float(v) for k, v in fit.params.items()},
                            "converged": bool(getattr(fit, "convergence_flag", 0) == 0),
                        }
                        candidates.append(rec)
                        if best is None or aic < best["aic"]:
                            best = rec.copy()
                            best["_fit"] = fit
                    except Exception:
                        continue

        if not candidates or best is None:
            return {"available": False, "message": "No GARCH model could be fitted."}

        out: Dict[str, Any] = {
            "available": True,
            "best_model": {k: v for k, v in best.items() if k != "_fit"},
            "candidates": sorted(candidates, key=lambda x: x["aic"])[:10],
        }

        if include_forecast and best.get("_fit") is not None:
            try:
                fit = best["_fit"]
                fc = fit.forecast(horizon=int(forecast_horizon), reindex=False)
                # variance forecasts for each horizon step
                var_fc = fc.variance.values[-1, :]
                vol_fc = np.sqrt(np.maximum(var_fc, 1e-12)) / 100.0 * math.sqrt(self.annual_trading_days)
                out["forecast_vol_annualized"] = [float(x) for x in vol_fc]
            except Exception as e:
                out["forecast_error"] = str(e)

        return out

    # ------------------------------
    # Regime detection (compat: n_states=...)
    # ------------------------------
    def detect_regimes(
        self,
        returns: pd.Series,
        n_regimes: int = 2,
        n_states: Optional[int] = None,
        include_predictions: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Optional HMM regime detection; accepts legacy n_states arg."""
        if n_states is not None:
            n_regimes = int(n_states)

        r = _as_series(returns).dropna()
        if len(r) < 300:
            return {"available": False, "message": "Insufficient data (need >= 300 obs)."}

        if not dep.is_available("hmmlearn"):
            return {
                "available": False,
                "message": "HMM optional. Install hmmlearn + scikit-learn to enable regimes."
            }

        GaussianHMM = dep.get("hmmlearn", "GaussianHMM")
        StandardScaler = dep.get("hmmlearn", "StandardScaler")

        # Features: return, abs(return), rolling vol
        vol = r.rolling(20, min_periods=20).std()
        X = pd.concat([r, r.abs(), vol], axis=1).dropna()
        if X.empty or X.shape[0] < 200:
            return {"available": False, "message": "Not enough feature data for HMM."}

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        hmm = GaussianHMM(n_components=int(n_regimes), covariance_type="full", n_iter=300, random_state=42)
        hmm.fit(Xs)

        states = hmm.predict(Xs)
        probs = hmm.predict_proba(Xs)

        # Label regimes by mean return
        labels = {}
        for k in range(int(n_regimes)):
            mu = float(X.iloc[states == k, 0].mean()) if np.any(states == k) else 0.0
            labels[k] = {"mean_return": mu, "name": "Risk-On" if mu >= 0 else "Risk-Off"}

        res: Dict[str, Any] = {
            "available": True,
            "n_regimes": int(n_regimes),
            "states": states.tolist(),
            "probabilities": probs.tolist(),
            "index": [str(d.date()) for d in X.index],
            "labels": labels,
            "transition_matrix": hmm.transmat_.tolist(),
        }

        if include_predictions:
            last_state = int(states[-1])
            next_probs = hmm.transmat_[last_state].tolist()
            res["predicted_next_regime"] = {
                "last_state": last_state,
                "next_state_probabilities": next_probs,
                "most_likely_next": int(np.argmax(next_probs)),
            }

        return res


# =============================================================================
# VISUALS
# =============================================================================

class InstitutionalVisualizer:
    def __init__(self):
        self.template = "plotly_dark"

    def _empty(self, msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(height=420, template=self.template)
        return fig

    def correlation_heatmap(self, corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        if corr is None or corr.empty:
            return self._empty("No correlation data.")
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1, zmax=1, zmid=0,
            colorscale="RdBu",
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y} vs %{x}</b><br>ρ=%{z:.3f}<extra></extra>"
        ))
        fig.update_layout(
            template=self.template,
            height=max(520, 60 * len(corr.columns)),
            title=dict(text=title, x=0.5),
            xaxis_tickangle=45,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def ewma_signal_chart(
        self,
        ratio: pd.Series,
        bb_window: int = 60,
        bb_k: float = 2.0,
        green: float = 0.90,
        orange: float = 1.10,
        title: str = "Institutional Signal: EWMA Vol Ratio",
    ) -> go.Figure:
        if ratio is None or ratio.empty:
            return self._empty("No EWMA ratio data.")
        s = ratio.dropna()
        if len(s) < max(40, bb_window):
            return self._empty("Not enough data for Bollinger bands.")

        ma = s.rolling(bb_window, min_periods=bb_window//2).mean()
        sd = s.rolling(bb_window, min_periods=bb_window//2).std()
        upper = ma + bb_k * sd
        lower = ma - bb_k * sd

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="EWMA Ratio", mode="lines"))
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, name=f"BB Mean({bb_window})", mode="lines"))
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, name="BB Upper", mode="lines"))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, name="BB Lower", mode="lines"))

        # Risk bands
        fig.add_hrect(y0=0, y1=green, opacity=0.12, line_width=0, annotation_text="GREEN", annotation_position="top left")
        fig.add_hrect(y0=green, y1=orange, opacity=0.12, line_width=0, annotation_text="ORANGE", annotation_position="top left")
        fig.add_hrect(y0=orange, y1=max(float(s.max())*1.1, orange*1.1), opacity=0.12, line_width=0, annotation_text="RED", annotation_position="top left")

        fig.update_layout(
            template=self.template,
            height=520,
            title=dict(text=title, x=0.5),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_yaxes(title_text="Ratio (Fast / (Mid+Slow))")
        return fig

    def line_with_bands(self, series: pd.Series, title: str, y_title: str, bands: Optional[List[Tuple[float, float, str]]] = None) -> go.Figure:
        if series is None or series.empty:
            return self._empty("No data.")
        s = series.dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=y_title))
        if bands:
            for (y0, y1, label) in bands:
                fig.add_hrect(y0=y0, y1=y1, opacity=0.10, line_width=0, annotation_text=label, annotation_position="top left")
        fig.update_layout(template=self.template, height=520, title=dict(text=title, x=0.5), hovermode="x unified")
        fig.update_yaxes(title_text=y_title)
        return fig

    def bar_table(self, df: pd.DataFrame, title: str = "Table") -> go.Figure:
        if df is None or df.empty:
            return self._empty("No data.")
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), align="left"),
            cells=dict(values=[df[c].tolist() for c in df.columns], align="left")
        )])
        fig.update_layout(template=self.template, height=520, title=dict(text=title, x=0.5))
        return fig


# =============================================================================
# REPORTING / EXPORT
# =============================================================================

def export_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> Tuple[Optional[bytes], str]:
    """Return (bytes, mime). Falls back if openpyxl not available."""
    from io import BytesIO
    bio = BytesIO()

    # Prefer openpyxl, then xlsxwriter, else csv zip-like fallback
    engine = None
    if dep.is_available("openpyxl"):
        engine = "openpyxl"
    elif dep.is_available("xlsxwriter"):
        engine = "xlsxwriter"

    if engine is None:
        # fallback: single CSV (first sheet)
        if not sheets:
            return None, "text/plain"
        first_name = list(sheets.keys())[0]
        data = sheets[first_name]
        return data.to_csv(index=True).encode("utf-8"), "text/csv"

    with pd.ExcelWriter(bio, engine=engine) as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            safe = str(name)[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=safe, index=True)
    return bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# =============================================================================
# UI HELPERS
# =============================================================================

def inject_css():
    st.markdown(
        """
        <style>
        .kpi { padding: 14px 16px; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
        .kpi .label { font-size: 0.85rem; opacity: 0.78; margin-bottom: 6px; }
        .kpi .value { font-size: 1.55rem; font-weight: 800; line-height: 1.1; }
        .kpi .sub { font-size: 0.82rem; opacity: 0.68; margin-top: 8px; }
        .badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; border:1px solid rgba(255,255,255,0.12); margin-right:6px; }
        .muted { opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True
    )

def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

class InstitutionalCommoditiesDashboard:
    def __init__(self):
        self.data = EnhancedDataManager()
        self.analytics = InstitutionalAnalytics()
        self.viz = InstitutionalVisualizer()
        self.start_time = datetime.now()

        self._ensure_session_state()

    def _ensure_session_state(self):
        ss = st.session_state
        ss.setdefault("data_loaded", False)
        ss.setdefault("last_load_summary", {})
        ss.setdefault("cfg", AnalysisConfiguration())
        ss.setdefault("selected_asset", None)
        ss.setdefault("selected_benchmark", None)

    def sidebar(self) -> AnalysisConfiguration:
        cfg: AnalysisConfiguration = st.session_state.get("cfg", AnalysisConfiguration())

        st.sidebar.title("⚙️ Controls")

        # Assets (comma-separated)
        assets_str = st.sidebar.text_area(
            "Assets (comma-separated tickers)",
            value=",".join(cfg.assets),
            height=75,
            help="Examples: GC=F, SI=F, CL=F, NG=F, HG=F"
        )
        assets = [x.strip().upper() for x in assets_str.split(",") if x.strip()]

        benchmark = st.sidebar.text_input("Benchmark", value=cfg.benchmark)
        rf = st.sidebar.text_input("Risk-free proxy", value=cfg.risk_free_ticker)

        c1, c2 = st.sidebar.columns(2)
        with c1:
            start = st.date_input("Start", value=cfg.start_date.date())
        with c2:
            end = st.date_input("End", value=cfg.end_date.date())

        st.sidebar.markdown("---")
        st.sidebar.subheader("Correlation")
        corr_method = st.sidebar.selectbox("Method", ["pearson", "spearman", "kendall", "ewma"], index=["pearson","spearman","kendall","ewma"].index(cfg.corr_method))
        shrink = st.sidebar.selectbox("Shrinkage", ["none", "ledoitwolf"], index=["none","ledoitwolf"].index(cfg.corr_shrinkage))
        corr_psd = st.sidebar.checkbox("PSD enforce (Higham)", value=cfg.corr_psd)
        ewma_lam = st.sidebar.slider("EWMA λ", min_value=0.80, max_value=0.995, value=float(cfg.ewma_lambda), step=0.005)

        st.sidebar.markdown("---")
        st.sidebar.subheader("VaR / CVaR")
        var_conf = st.sidebar.slider("Confidence", 0.80, 0.99, float(cfg.var_confidence), step=0.01)
        var_method = st.sidebar.selectbox("Method", ["historical", "parametric", "modified", "student_t"], index=["historical","parametric","modified","student_t"].index(cfg.var_method))
        var_h = st.sidebar.slider("Horizon (days)", 1, 20, int(cfg.var_horizon), step=1)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Institutional Signal")
        sig_green = st.sidebar.slider("Green threshold", 0.40, 1.50, float(cfg.sig_green), step=0.05)
        sig_orange = st.sidebar.slider("Orange threshold", 0.50, 2.00, float(cfg.sig_orange), step=0.05)

        st.sidebar.markdown("---")
        load_clicked = st.sidebar.button("📥 Load Data", key="btn_load_data")
        if load_clicked:
            cfg = AnalysisConfiguration(
                assets=assets,
                benchmark=benchmark.strip().upper(),
                start_date=_to_datetime(start),
                end_date=_to_datetime(end),
                risk_free_ticker=rf.strip().upper(),
                corr_method=corr_method,
                corr_shrinkage=shrink,
                corr_psd=corr_psd,
                ewma_lambda=float(ewma_lam),
                var_confidence=float(var_conf),
                var_method=var_method,
                var_horizon=int(var_h),
                sig_green=float(sig_green),
                sig_orange=float(sig_orange),
            )
            st.session_state["cfg"] = cfg
            self._load(cfg)

        return st.session_state.get("cfg", cfg)

    def _load(self, cfg: AnalysisConfiguration):
        with st.spinner("Downloading and preparing data..."):
            summary = self.data.load(cfg)
        st.session_state["last_load_summary"] = summary
        st.session_state["data_loaded"] = bool(summary.get("ok", False))
        if summary.get("ok", False):
            st.success(f"✅ Loaded in {summary.get('load_time_s', 0):.2f}s • Assets: {len(summary.get('assets_loaded', []))} • Obs: {summary.get('n_obs', 0)}")
        else:
            st.error(f"❌ Load failed: {summary.get('error', 'Unknown error')}")

    def run(self):
        st.set_page_config(page_title="Institutional Commodities Analytics", page_icon="🏛️", layout="wide")
        inject_css()

        cfg = self.sidebar()

        st.title("🏛️ Institutional Commodities Analytics Platform v7.4")
        st.caption("Correct correlations • Robust VaR/CVaR/ES • GARCH • Optional regimes • Cloud-safe reporting")

        if not st.session_state.get("data_loaded", False):
            st.info("Use the sidebar to select tickers and click **Load Data**.")
            last = st.session_state.get("last_load_summary", {})
            if last and not last.get("ok", True):
                st.code(last, language="json")
            return

        returns_df = self.data.returns.copy()
        bench = self.data.benchmark_returns.copy()

        # Asset selection for single-series tabs
        asset_list = list(returns_df.columns) if isinstance(returns_df, pd.DataFrame) else []
        if asset_list:
            st.session_state["selected_asset"] = st.session_state.get("selected_asset") or asset_list[0]
        if not bench.empty:
            st.session_state["selected_benchmark"] = st.session_state.get("selected_benchmark") or cfg.benchmark

        tabs = st.tabs([
            "📊 Dashboard",
            "🧮 Risk Analytics",
            "🔗 Correlations",
            "🧭 Institutional Signal",
            "🎯 Tracking Error",
            "🧷 Rolling Beta",
            "⚖️ Relative VaR/CVaR/ES",
            "📈 Advanced (GARCH/HMM)",
            "📑 Reporting",
            "🧪 Data Quality"
        ])

        with tabs[0]:
            self.tab_dashboard(cfg, returns_df, bench)

        with tabs[1]:
            self.tab_risk(cfg, returns_df, bench)

        with tabs[2]:
            self.tab_correlations(cfg, returns_df)

        with tabs[3]:
            self.tab_signal(cfg, returns_df)

        with tabs[4]:
            self.tab_tracking_error(cfg, returns_df, bench)

        with tabs[5]:
            self.tab_rolling_beta(cfg, returns_df, bench)

        with tabs[6]:
            self.tab_relative_var(cfg, returns_df, bench)

        with tabs[7]:
            self.tab_advanced(cfg, returns_df)

        with tabs[8]:
            self.tab_reporting(cfg, returns_df, bench)

        with tabs[9]:
            self.tab_quality()

        # memory cleanup
        gc.collect()

    # ------------------------------
    # Tabs
    # ------------------------------
    def tab_dashboard(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Overview")

        # Portfolio: equal weight for overview
        port = returns_df.mean(axis=1, skipna=True)

        mets = self.analytics.calculate_performance_metrics(port, rf_daily=self.data.risk_free)
        vol_ratio = np.nan
        if mets:
            ann_vol = mets.get("annual_vol", np.nan)
            hist_vol = returns_df.stack().std() * math.sqrt(252) if not returns_df.empty else np.nan
            vol_ratio = ann_vol / (hist_vol + 1e-12) if np.isfinite(ann_vol) and np.isfinite(hist_vol) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Portfolio Total Return", f"{_pct(mets.get('total_return', np.nan)):.2f}%", "Equal-weight portfolio")
        with c2:
            kpi("Annual Return", f"{_pct(mets.get('annual_return', np.nan)):.2f}%", f"Obs: {mets.get('n_obs', 0)}")
        with c3:
            kpi("Annual Volatility", f"{_pct(mets.get('annual_vol', np.nan)):.2f}%", f"Vol ratio: {vol_ratio:.2f}x" if np.isfinite(vol_ratio) else "")
        with c4:
            kpi("Max Drawdown", f"{_pct(mets.get('max_drawdown', np.nan)):.2f}%", f"Sharpe: {mets.get('sharpe', np.nan):.2f}" if mets else "")

        # Simple cumulative chart
        st.markdown("#### Cumulative Performance")
        cum = (1 + returns_df.fillna(0)).cumprod()
        fig = go.Figure()
        for c in returns_df.columns:
            if c in cum.columns:
                fig.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines", name=c))
        fig.update_layout(template=self.viz.template, height=520, hovermode="x unified", title=dict(text="Assets Cumulative Index (Base=1)", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        if not bench.empty:
            st.markdown("#### Portfolio vs Benchmark (Equal Weight)")
            aligned = pd.concat([port.rename("Portfolio"), bench.rename("Benchmark")], axis=1, join="inner").dropna()
            if not aligned.empty:
                c = (1 + aligned).cumprod()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=c.index, y=c["Portfolio"], mode="lines", name="Portfolio"))
                fig2.add_trace(go.Scatter(x=c.index, y=c["Benchmark"], mode="lines", name=cfg.benchmark))
                fig2.update_layout(template=self.viz.template, height=520, hovermode="x unified", title=dict(text="Portfolio vs Benchmark (Base=1)", x=0.5))
                st.plotly_chart(fig2, use_container_width=True)

    def tab_risk(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Risk Analytics")

        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_risk")
        r = returns_df[asset].dropna()

        mets = self.analytics.calculate_performance_metrics(r, rf_daily=self.data.risk_free)
        var = self.analytics.calculate_var(r, confidence_level=cfg.var_confidence, method=cfg.var_method, horizon=cfg.var_horizon)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Total Return", f"{_pct(mets.get('total_return', np.nan)):.2f}%", "Asset")
        with c2:
            kpi("Annual Vol", f"{_pct(mets.get('annual_vol', np.nan)):.2f}%", f"Sharpe: {mets.get('sharpe', np.nan):.2f}")
        with c3:
            kpi(f"VaR ({int(cfg.var_confidence*100)}%)", f"{var.get('var', np.nan):.2f}%", f"Method: {cfg.var_method}")
        with c4:
            kpi(f"CVaR/ES ({int(cfg.var_confidence*100)}%)", f"{var.get('cvar', np.nan):.2f}%", f"Exceed: {var.get('exceedance_rate', np.nan):.2f}%")

        st.markdown("#### Stress Testing")
        shock = st.slider("Single shock scenario (daily return add-on)", -0.25, 0.10, -0.05, step=0.01)
        st_res = self.analytics.stress_test(r, shock=shock)
        if st_res:
            st.dataframe(pd.DataFrame(st_res).T, use_container_width=True)
        else:
            st.info("Not enough data for stress test.")

    def tab_correlations(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame):
        st.subheader("Correlation Analysis (Correct & Institutional)")
        st.caption("Correlation is computed after alignment and dropping constant / low-observation series. Optional shrinkage + PSD enforcement available.")

        corr = self.analytics.compute_correlation_matrix(
            returns_df=returns_df,
            method=cfg.corr_method,
            shrinkage=cfg.corr_shrinkage,
            ensure_psd=cfg.corr_psd,
            ewma_lambda=cfg.ewma_lambda,
            min_pair_obs=max(60, cfg.min_obs // 3),
        )
        st.plotly_chart(self.viz.correlation_heatmap(corr, title=f"Correlation • method={cfg.corr_method} • shrink={cfg.corr_shrinkage} • psd={cfg.corr_psd}"), use_container_width=True)

        if not corr.empty:
            tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            tri = tri[np.isfinite(tri)]
            if len(tri) > 0:
                st.markdown("#### Summary")
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi("Avg Correlation", f"{np.mean(tri):.2f}", "Lower = better diversification")
                with c2:
                    kpi("Median Correlation", f"{np.median(tri):.2f}", "")
                with c3:
                    kpi("Max Pair Correlation", f"{np.max(tri):.2f}", "")

    def tab_signal(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame):
        st.subheader("Institutional Signal: EWMA Vol Ratio + Bollinger + Bands")
        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_signal")
        r = returns_df[asset].dropna()

        ratio = self.analytics.ewma_vol_ratio_signal(r, fast=cfg.ewma_fast, mid=cfg.ewma_mid, slow=cfg.ewma_slow)
        fig = self.viz.ewma_signal_chart(
            ratio=ratio,
            bb_window=cfg.bb_window,
            bb_k=cfg.bb_k,
            green=cfg.sig_green,
            orange=cfg.sig_orange,
            title=f"{asset} • EWMA({cfg.ewma_fast}) / (EWMA({cfg.ewma_mid}) + EWMA({cfg.ewma_slow}))"
        )
        st.plotly_chart(fig, use_container_width=True)

        if not ratio.empty:
            last = float(ratio.dropna().iloc[-1])
            if last < cfg.sig_green:
                st.success(f"GREEN zone: ratio={last:.3f} (< {cfg.sig_green})")
            elif last < cfg.sig_orange:
                st.warning(f"ORANGE zone: ratio={last:.3f} (between {cfg.sig_green} and {cfg.sig_orange})")
            else:
                st.error(f"RED zone: ratio={last:.3f} (≥ {cfg.sig_orange})")

    def tab_tracking_error(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Tracking Error (Rolling)")

        if bench is None or bench.empty:
            st.info("Benchmark returns not available. Load a benchmark ticker.")
            return

        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_te")
        r = returns_df[asset].dropna()

        te = self.analytics.tracking_error(r, bench, window=cfg.te_window)
        bands = [(0.0, 0.10, "GREEN"), (0.10, 0.20, "ORANGE"), (0.20, max(float(te.max())*1.1 if not te.empty else 0.3, 0.3), "RED")]
        fig = self.viz.line_with_bands(te, title=f"{asset} vs {cfg.benchmark} • Tracking Error", y_title="Tracking Error (ann.)", bands=bands)
        st.plotly_chart(fig, use_container_width=True)

    def tab_rolling_beta(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Rolling Beta (Benchmark-based)")

        if bench is None or bench.empty:
            st.info("Benchmark returns not available. Load a benchmark ticker.")
            return

        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_beta")
        r = returns_df[asset].dropna()

        beta = self.analytics.rolling_beta(r, bench, window=cfg.beta_window)
        bands = [(-5.0, 0.5, "Low/Defensive"), (0.5, 1.5, "Market-like"), (1.5, 5.0, "High/Beta")]
        fig = self.viz.line_with_bands(beta, title=f"{asset} vs {cfg.benchmark} • Rolling Beta", y_title="Beta", bands=bands)
        st.plotly_chart(fig, use_container_width=True)

    def tab_relative_var(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Relative VaR/CVaR/ES (Active Returns vs Benchmark)")

        if bench is None or bench.empty:
            st.info("Benchmark returns not available. Load a benchmark ticker.")
            return

        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_relvar")
        r = returns_df[asset].dropna()

        rel = self.analytics.relative_var(r, bench, confidence_level=cfg.var_confidence, method=cfg.var_method, horizon=cfg.var_horizon)
        if not rel:
            st.info("Not enough data for relative VaR.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi(f"Active VaR ({int(cfg.var_confidence*100)}%)", f"{rel.get('var', np.nan):.2f}%", "Positive loss")
        with c2:
            kpi(f"Active CVaR/ES ({int(cfg.var_confidence*100)}%)", f"{rel.get('cvar', np.nan):.2f}%", "")
        with c3:
            kpi("Active Quantile Return", f"{rel.get('quantile_return', np.nan):.2f}%", "Left-tail return")

        # Show active return series and bands
        aligned = pd.concat([r.rename("asset"), bench.rename("bench")], axis=1, join="inner").dropna()
        active = (aligned["asset"] - aligned["bench"]).rename("active_returns")
        bands = [(-5.0, -0.5, "Left-tail risk"), (-0.5, 0.5, "Neutral"), (0.5, 5.0, "Positive active")]
        fig = self.viz.line_with_bands(active.cumsum(), title=f"{asset} - {cfg.benchmark} • Active PnL (cum)", y_title="Cumulative Active Return", bands=None)
        st.plotly_chart(fig, use_container_width=True)

    def tab_advanced(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame):
        st.subheader("Advanced Analytics")
        asset = st.selectbox("Select asset", options=list(returns_df.columns), key="sel_asset_adv")
        r = returns_df[asset].dropna()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### GARCH")
            p1, p2 = cfg.garch_p_range
            q1, q2 = cfg.garch_q_range
            p_range = st.slider("p range", 1, 5, (p1, p2), key="garch_p_range")
            q_range = st.slider("q range", 1, 5, (q1, q2), key="garch_q_range")
            fh = st.slider("forecast horizon", 1, 30, int(cfg.garch_forecast_horizon), key="garch_fh")

            if st.button("Run GARCH", key="btn_run_garch"):
                with st.spinner("Fitting GARCH models..."):
                    res = self.analytics.garch_analysis(
                        r,
                        p_range=tuple(map(int, p_range)),
                        q_range=tuple(map(int, q_range)),
                        forecast_horizon=int(fh),
                        include_forecast=True
                    )
                st.session_state["garch_res"] = res

            res = st.session_state.get("garch_res", {})
            if res:
                st.json({k: v for k, v in res.items() if k != "candidates"})
                if res.get("available") and "candidates" in res:
                    st.dataframe(pd.DataFrame(res["candidates"]), use_container_width=True)

        with col2:
            st.markdown("#### Regime Detection (HMM optional)")
            n = st.slider("Number of regimes", 2, 5, 2, key="hmm_n")
            if st.button("Run HMM Regimes", key="btn_run_hmm"):
                with st.spinner("Running regime detection..."):
                    res = self.analytics.detect_regimes(r, n_regimes=int(n), include_predictions=True)
                st.session_state["hmm_res"] = res

            res = st.session_state.get("hmm_res", {})
            if res:
                if not res.get("available", False):
                    st.warning(res.get("message", "HMM not available"))
                else:
                    st.success("Regimes detected")
                    labels = res.get("labels", {})
                    st.json(labels)
                    # Plot states
                    idx = pd.to_datetime(res.get("index", []))
                    states = pd.Series(res.get("states", []), index=idx, name="state")
                    fig = self.viz.line_with_bands(states, title="Regime State over Time", y_title="State", bands=None)
                    st.plotly_chart(fig, use_container_width=True)

    def tab_reporting(self, cfg: AnalysisConfiguration, returns_df: pd.DataFrame, bench: pd.Series):
        st.subheader("Reporting / Export")

        # Summary tables
        summary_rows = []
        for c in returns_df.columns:
            mets = self.analytics.calculate_performance_metrics(returns_df[c], rf_daily=self.data.risk_free)
            if mets:
                summary_rows.append({
                    "ticker": c,
                    "total_return_%": _pct(mets["total_return"]),
                    "annual_return_%": _pct(mets["annual_return"]),
                    "annual_vol_%": _pct(mets["annual_vol"]),
                    "max_dd_%": _pct(mets["max_drawdown"]),
                    "sharpe": mets["sharpe"],
                    "sortino": mets["sortino"],
                })
        summary = pd.DataFrame(summary_rows).set_index("ticker") if summary_rows else pd.DataFrame()

        corr = self.analytics.compute_correlation_matrix(
            returns_df, method=cfg.corr_method, shrinkage=cfg.corr_shrinkage, ensure_psd=cfg.corr_psd, ewma_lambda=cfg.ewma_lambda
        )

        sheets = {
            "summary": summary,
            "correlation": corr,
            "returns": returns_df,
        }

        excel_bytes, mime = export_excel_bytes(sheets)
        if excel_bytes is None:
            st.error("No data to export.")
            return

        filename = f"institutional_commodities_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if mime == "text/csv":
            st.download_button("⬇️ Download CSV (fallback)", data=excel_bytes, file_name=filename + ".csv", mime=mime, key="dl_csv")
        else:
            st.download_button("⬇️ Download Excel", data=excel_bytes, file_name=filename + ".xlsx", mime=mime, key="dl_xlsx")

        st.markdown("#### Preview: Summary")
        if not summary.empty:
            st.dataframe(summary.round(4), use_container_width=True)
        else:
            st.info("No summary data to show.")

    def tab_quality(self):
        st.subheader("Data Quality Report")
        qr = self.data.quality_report
        if qr is None or qr.empty:
            st.info("No quality report available.")
            return
        st.dataframe(qr, use_container_width=True)
        if qr["flag_low_obs"].any():
            st.warning("Some assets have low observations. Consider extending date range or removing those tickers.")
        if qr["flag_constant"].any():
            st.warning("Some assets appear constant (zero variance). Correlations for these are meaningless.")


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    try:
        app = InstitutionalCommoditiesDashboard()
        app.run()
    except Exception as e:
        st.error("## 🚨 Application Error\nAn unexpected error occurred.")
        st.code(traceback.format_exc())
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
