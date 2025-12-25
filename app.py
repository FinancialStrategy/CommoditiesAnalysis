# ==============================================================================
# 🏛️ Institutional Commodities Analytics Platform v7.1 (Cloud-Safe, Long Complete)
# Enhanced Scientific Analytics • Advanced Correlation Methods • Professional Risk Metrics
# ------------------------------------------------------------------------------
# Key upgrades (v7.1)
# - FIXED correlation reporting: strict overlap alignment + robust NaN handling
# - Added Ledoit-Wolf shrinkage correlation (with safe fallback if sklearn missing)
# - Added SMART SIGNAL TAB:
#   Ratio = (EWMA 22D Vol) / (EWMA 33D Vol + EWMA 99D Vol)
#   + Bollinger Bands (upper/lower lines) + Alarm Zones (green/orange/red bands)
# - Added fully-implemented Stress Testing tab to prevent missing-method errors
# - Cloud-safe: no hard dependency on psutil / hmmlearn / arch / quantstats (optional)
# - Stronger data-quality gating + reproducibility config
#
# NOTE:
# - For Streamlit Cloud: add sklearn to requirements.txt to enable shrinkage corr.
# - If you want HMM regimes: add hmmlearn to requirements.txt.
# ==============================================================================

import os
import math
import json
import warnings
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy import stats

warnings.filterwarnings("ignore")

# ==============================================================================
# STREAMLIT PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==============================================================================
st.set_page_config(
    page_title="Institutional Commodities Analytics Platform v7.1",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# OPTIONAL DEPENDENCIES (CLOUD SAFE)
# ==============================================================================
HAS_SKLEARN = False
HAS_ARCH = False
HAS_QS = False
HAS_HMMLEARN = False

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

try:
    import quantstats as qs
    HAS_QS = True
except Exception:
    HAS_QS = False

try:
    import hmmlearn  # noqa: F401
    HAS_HMMLEARN = True
except Exception:
    HAS_HMMLEARN = False

# ==============================================================================
# UTILITIES
# ==============================================================================
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"

def safe_float(x: float, fmt: str = "{:.4f}") -> str:
    if x is None:
        return "—"
    try:
        if np.isnan(x) or np.isinf(x):
            return "—"
        return fmt.format(float(x))
    except Exception:
        return "—"

def clamp_int(x: int, lo: int, hi: int) -> int:
    try:
        xi = int(x)
    except Exception:
        xi = lo
    return max(lo, min(hi, xi))

def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a

def df_nonempty(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and (df.shape[0] > 0) and (df.shape[1] > 0)

def series_nonempty(s: Optional[pd.Series]) -> bool:
    return isinstance(s, pd.Series) and (s.dropna().shape[0] > 0)

# ==============================================================================
# INSTITUTIONAL STYLE (LIGHT UI; you can flip to dark if you want)
# ==============================================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
    .kpi-card {
        border-radius: 14px;
        padding: 14px 14px 12px 14px;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(255,255,255,0.75);
        box-shadow: 0 2px 14px rgba(0,0,0,0.05);
    }
    .kpi-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
    .kpi-value { font-size: 1.35rem; font-weight: 700; margin: 0; }
    .kpi-sub   { font-size: 0.78rem; opacity: 0.70; margin-top: 4px; }
    .small-note { font-size: 0.85rem; opacity: 0.75; }
    .section-title { font-size: 1.25rem; font-weight: 800; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class AppConfig:
    title: str = "Institutional Commodities Analytics Platform v7.1"
    # Data
    default_years_back: int = 5
    min_overlap_days: int = 90
    max_missing_frac: float = 0.15
    max_gap_days: int = 10
    # Returns
    returns_mode: str = "log"  # "simple" or "log"
    # Risk
    var_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    annualization: int = 252
    # Ratio Signal defaults
    ewma_short: int = 22
    ewma_mid: int = 33
    ewma_long: int = 99
    boll_window: int = 20
    boll_k: float = 2.0
    # Alarm thresholds (ratio)
    ratio_green_max: float = 0.80
    ratio_orange_max: float = 1.00
    # Portfolio
    default_weight_mode: str = "equal"

CFG = AppConfig()

# ==============================================================================
# UNIVERSE (Classified)
# ==============================================================================
UNIVERSE: Dict[str, Dict[str, str]] = {
    "Precious Metals": {
        "Gold (GC=F)": "GC=F",
        "Silver (SI=F)": "SI=F",
        "Platinum (PL=F)": "PL=F",
        "Palladium (PA=F)": "PA=F",
    },
    "Energy": {
        "Crude Oil WTI (CL=F)": "CL=F",
        "Brent (BZ=F)": "BZ=F",
        "Natural Gas (NG=F)": "NG=F",
        "Heating Oil (HO=F)": "HO=F",
        "Gasoline RBOB (RB=F)": "RB=F",
    },
    "Industrial Metals": {
        "Copper (HG=F)": "HG=F",
        "Aluminum (ALI=F)*": "ALI=F",
        "Nickel (NICKEL=F)*": "NICKEL=F",
    },
    "Agriculture": {
        "Corn (ZC=F)": "ZC=F",
        "Wheat (ZW=F)": "ZW=F",
        "Soybeans (ZS=F)": "ZS=F",
        "Soybean Oil (ZL=F)": "ZL=F",
        "Soybean Meal (ZM=F)": "ZM=F",
        "Rice (ZR=F)": "ZR=F",
    },
    "Softs": {
        "Coffee (KC=F)": "KC=F",
        "Sugar (SB=F)": "SB=F",
        "Cotton (CT=F)": "CT=F",
        "Cocoa (CC=F)": "CC=F",
        "Orange Juice (OJ=F)": "OJ=F",
    },
    "Rates / FX Proxies": {
        "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
        "EURUSD (EURUSD=X)": "EURUSD=X",
        "USDJPY (JPY=X)": "JPY=X",
    },
}

def flatten_universe(universe: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    out = {}
    for cat, mp in universe.items():
        for label, ticker in mp.items():
            out[f"{cat} — {label}"] = ticker
    return out

UNIVERSE_FLAT = flatten_universe(UNIVERSE)

# ==============================================================================
# DATA ENGINE
# ==============================================================================
class DataEngine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    @staticmethod
    def _yf_download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            threads=True,
            progress=False,
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            level1 = df.columns.get_level_values(0)
            fields = list(dict.fromkeys(level1))
            prefer = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else fields[0])
            out = df[prefer].copy()
        else:
            if "Adj Close" in df.columns:
                out = df[["Adj Close"]].copy()
                out.columns = [tickers[0]]
            elif "Close" in df.columns:
                out = df[["Close"]].copy()
                out.columns = [tickers[0]]
            else:
                out = df.copy()
        if out.columns.dtype != object:
            out.columns = out.columns.astype(str)
        return out

    @staticmethod
    def _max_consecutive_nan_run(s: pd.Series) -> int:
        is_na = s.isna().astype(int).values
        if is_na.size == 0:
            return 0
        max_run = 0
        run = 0
        for v in is_na:
            if v == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return int(max_run)

    def data_quality_gate(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"dropped": {}, "kept": []}
        if not df_nonempty(prices):
            return prices, info

        out = prices.copy().sort_index()
        out = out.replace([np.inf, -np.inf], np.nan)

        for c in list(out.columns):
            s = out[c]
            missing_frac = float(s.isna().mean())
            max_gap = self._max_consecutive_nan_run(s)
            if missing_frac > self.cfg.max_missing_frac or max_gap > self.cfg.max_gap_days:
                info["dropped"][c] = {"missing_frac": missing_frac, "max_gap": max_gap}
                out = out.drop(columns=[c])
            else:
                info["kept"].append(c)

        out = out.ffill(limit=self.cfg.max_gap_days)
        return out, info

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if not df_nonempty(prices):
            return pd.DataFrame()
        px_ = prices.copy()
        if self.cfg.returns_mode == "log":
            rets = np.log(px_ / px_.shift(1))
        else:
            rets = px_.pct_change()
        rets = rets.replace([np.inf, -np.inf], np.nan)
        return rets.dropna(how="all")

    def fetch(self, tickers: List[str], start: str, end: str) -> Dict[str, Any]:
        raw = self._yf_download(tickers, start, end)
        cleaned, dq = self.data_quality_gate(raw)
        rets = self.compute_returns(cleaned)
        return {"raw_prices": raw, "prices": cleaned, "returns": rets, "dq": dq}

# ==============================================================================
# ANALYTICS
# ==============================================================================
class Analytics:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def ewma_vol(self, returns: pd.Series, span: int) -> pd.Series:
        if not series_nonempty(returns):
            return pd.Series(dtype=float)
        return returns.ewm(span=int(span), adjust=False).std(bias=False)

    def ratio_signal(self, returns: pd.Series) -> pd.DataFrame:
        if not series_nonempty(returns):
            return pd.DataFrame()

        vol_s = self.ewma_vol(returns, self.cfg.ewma_short)
        vol_m = self.ewma_vol(returns, self.cfg.ewma_mid)
        vol_l = self.ewma_vol(returns, self.cfg.ewma_long)

        denom = (vol_m + vol_l).replace(0, np.nan)
        ratio = (vol_s / denom).replace([np.inf, -np.inf], np.nan)

        df = pd.DataFrame({
            "ret": returns,
            f"ewma_vol_{self.cfg.ewma_short}": vol_s,
            f"ewma_vol_{self.cfg.ewma_mid}": vol_m,
            f"ewma_vol_{self.cfg.ewma_long}": vol_l,
            "ratio": ratio,
        }).dropna(how="all")

        w = int(self.cfg.boll_window)
        k = float(self.cfg.boll_k)
        ma = df["ratio"].rolling(w).mean()
        sd = df["ratio"].rolling(w).std(ddof=0)
        df["bb_mid"] = ma
        df["bb_upper"] = ma + k * sd
        df["bb_lower"] = ma - k * sd
        return df

    def _overlap_align_returns(self, rets: pd.DataFrame, min_overlap_days: int) -> pd.DataFrame:
        if not df_nonempty(rets):
            return pd.DataFrame()
        df = rets.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        keep_cols = [c for c in df.columns if df[c].dropna().shape[0] >= min_overlap_days]
        df = df[keep_cols]
        if df.shape[1] < 2:
            return pd.DataFrame()
        df = df.dropna(axis=0, how="any")
        if df.shape[0] < min_overlap_days:
            return pd.DataFrame()
        return df

    def correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: str = "pearson",
        min_overlap_days: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        meta = {
            "method": method,
            "min_overlap_days": min_overlap_days if min_overlap_days is not None else self.cfg.min_overlap_days,
            "n_obs": 0,
            "assets": 0,
            "notes": [],
        }

        if not df_nonempty(returns):
            meta["notes"].append("Empty returns.")
            return pd.DataFrame(), meta

        min_ov = int(min_overlap_days if min_overlap_days is not None else self.cfg.min_overlap_days)
        aligned = self._overlap_align_returns(returns, min_ov)
        if not df_nonempty(aligned) or aligned.shape[1] < 2:
            meta["notes"].append("Insufficient overlap after alignment.")
            return pd.DataFrame(), meta

        meta["n_obs"] = int(aligned.shape[0])
        meta["assets"] = int(aligned.shape[1])

        m = method.lower()
        if m in ["pearson", "spearman", "kendall"]:
            corr = aligned.corr(method=m)
            return corr, meta

        if m in ["ledoitwolf", "lw", "shrinkage"]:
            if not HAS_SKLEARN:
                meta["notes"].append("sklearn not available; fallback to Pearson.")
                corr = aligned.corr(method="pearson")
                return corr, meta

            lw = LedoitWolf().fit(aligned.values)
            cov = lw.covariance_
            d = np.sqrt(np.diag(cov))
            denom = np.outer(d, d)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = cov / denom
            corr = np.clip(corr, -1.0, 1.0)
            corr_df = pd.DataFrame(corr, index=aligned.columns, columns=aligned.columns)
            meta["notes"].append("Ledoit-Wolf shrinkage correlation computed from shrinkage covariance.")
            return corr_df, meta

        meta["notes"].append(f"Unknown method '{method}', fallback Pearson.")
        corr = aligned.corr(method="pearson")
        return corr, meta

    def var_historical(self, r: pd.Series, alpha: float) -> float:
        if not series_nonempty(r):
            return np.nan
        q = np.nanquantile(r.dropna().values, 1.0 - alpha)
        return float(-q)

    def cvar_historical(self, r: pd.Series, alpha: float) -> float:
        if not series_nonempty(r):
            return np.nan
        arr = r.dropna().values
        q = np.nanquantile(arr, 1.0 - alpha)
        tail = arr[arr <= q]
        if tail.size == 0:
            return np.nan
        return float(-np.nanmean(tail))

    def var_parametric_normal(self, r: pd.Series, alpha: float) -> float:
        if not series_nonempty(r):
            return np.nan
        mu = float(r.mean())
        sig = float(r.std(ddof=1))
        z = stats.norm.ppf(1.0 - alpha)
        return float(-(mu + z * sig))

    def cvar_parametric_normal(self, r: pd.Series, alpha: float) -> float:
        if not series_nonempty(r):
            return np.nan
        mu = float(r.mean())
        sig = float(r.std(ddof=1))
        z = stats.norm.ppf(1.0 - alpha)
        pdf = stats.norm.pdf(z)
        cvar = -(mu - sig * pdf / (1.0 - alpha))
        return float(cvar)

    def var_cornish_fisher(self, r: pd.Series, alpha: float) -> float:
        if not series_nonempty(r):
            return np.nan
        x = r.dropna().values
        mu = np.mean(x)
        sig = np.std(x, ddof=1)
        if sig <= 0:
            return np.nan
        s = stats.skew(x, bias=False)
        k = stats.kurtosis(x, fisher=True, bias=False)  # excess
        z = stats.norm.ppf(1.0 - alpha)
        z_cf = (
            z
            + (1/6) * (z**2 - 1) * s
            + (1/24) * (z**3 - 3*z) * k
            - (1/36) * (2*z**3 - 5*z) * (s**2)
        )
        return float(-(mu + z_cf * sig))

    def equal_weights(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([])
        return np.ones(n) / n

# ==============================================================================
# VISUALIZATION
# ==============================================================================
class Viz:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def kpi_row(self, kpis: List[Tuple[str, str, str]]):
        cols = st.columns(len(kpis))
        for i, (t, v, s) in enumerate(kpis):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-title">{t}</div>
                      <p class="kpi-value">{v}</p>
                      <div class="kpi-sub">{s}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def line_chart(self, df: pd.DataFrame, title: str, y_cols: List[str], height: int = 520) -> go.Figure:
        fig = go.Figure()
        for c in y_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig

    def price_chart(self, prices: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure()
        for c in prices.columns:
            fig.add_trace(go.Scatter(x=prices.index, y=prices[c], mode="lines", name=c))
        fig.update_layout(
            title=title,
            height=520,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig

    def correlation_heatmap(self, corr: pd.DataFrame, title: str) -> go.Figure:
        fig = px.imshow(
            corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            aspect="auto",
            text_auto=".2f",
        )
        fig.update_layout(
            title=title,
            height=650,
            template="plotly_white",
            margin=dict(l=30, r=30, t=70, b=30),
        )
        return fig

    def ratio_signal_chart(
        self,
        df_ratio: pd.DataFrame,
        title: str,
        green_max: float,
        orange_max: float,
        show_bbands: bool = True,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_ratio.index, y=df_ratio["ratio"],
            mode="lines", name="Ratio (EWMA22 / (EWMA33+EWMA99))"
        ))

        if show_bbands and ("bb_upper" in df_ratio.columns):
            fig.add_trace(go.Scatter(
                x=df_ratio.index, y=df_ratio["bb_upper"], mode="lines", name="BB Upper", line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_ratio.index, y=df_ratio["bb_mid"], mode="lines", name="BB Mid", line=dict(width=1, dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df_ratio.index, y=df_ratio["bb_lower"], mode="lines", name="BB Lower", line=dict(width=2)
            ))

        y_min = float(np.nanmin(df_ratio["ratio"].values)) if df_ratio["ratio"].notna().any() else 0.0
        y_max = float(np.nanmax(df_ratio["ratio"].values)) if df_ratio["ratio"].notna().any() else 1.0
        pad = 0.08 * (y_max - y_min) if (y_max > y_min) else 0.2
        y_min2 = y_min - pad
        y_max2 = y_max + pad

        fig.add_hrect(
            y0=y_min2, y1=green_max,
            fillcolor="rgba(0, 200, 0, 0.08)", line_width=0,
            annotation_text="GREEN (OK)", annotation_position="top left"
        )
        fig.add_hrect(
            y0=green_max, y1=orange_max,
            fillcolor="rgba(255, 165, 0, 0.10)", line_width=0,
            annotation_text="ORANGE (Watch)", annotation_position="top left"
        )
        fig.add_hrect(
            y0=orange_max, y1=y_max2,
            fillcolor="rgba(255, 0, 0, 0.08)", line_width=0,
            annotation_text="RED (Risk)", annotation_position="top left"
        )

        fig.update_layout(
            title=title,
            height=650,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=30, r=30, t=70, b=30),
        )
        fig.update_yaxes(title="Ratio", showgrid=True)
        fig.update_xaxes(showgrid=True)
        return fig

# ==============================================================================
# DASHBOARD APP
# ==============================================================================
class InstitutionalCommoditiesDashboard:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.data = DataEngine(cfg)
        self.an = Analytics(cfg)
        self.viz = Viz(cfg)

    def _sidebar(self) -> Dict[str, Any]:
        st.sidebar.markdown(f"### 🏛️ {self.cfg.title}")
        st.sidebar.caption(f"Run time: **{_now_str()}**")

        end = datetime.today().date()
        start_default = (datetime.today() - timedelta(days=365 * self.cfg.default_years_back)).date()

        st.sidebar.markdown("#### Data Window")
        start_date = st.sidebar.date_input("Start date", value=start_default, key="sb_start")
        end_date = st.sidebar.date_input("End date", value=end, key="sb_end")
        if start_date >= end_date:
            st.sidebar.warning("Start date must be earlier than end date. Resetting to defaults.")
            start_date = start_default
            end_date = end

        st.sidebar.markdown("#### Universe Selection")
        categories = list(UNIVERSE.keys())
        cat = st.sidebar.selectbox("Category", categories, index=0, key="sb_cat")

        labels = list(UNIVERSE[cat].keys())
        default_sel = labels[: min(4, len(labels))]
        selected_labels = st.sidebar.multiselect(
            "Select Instruments",
            options=labels,
            default=default_sel,
            key="sb_assets",
        )
        tickers = [UNIVERSE[cat][lbl] for lbl in selected_labels] if selected_labels else []

        st.sidebar.markdown("#### Data Quality Gate")
        min_overlap = st.sidebar.slider(
            "Min overlap days (correlation)",
            min_value=60,
            max_value=520,
            value=self.cfg.min_overlap_days,
            step=10,
            key="sb_min_overlap",
        )
        max_missing = st.sidebar.slider(
            "Max missing fraction per series",
            min_value=0.05,
            max_value=0.50,
            value=self.cfg.max_missing_frac,
            step=0.01,
            key="sb_max_missing",
        )
        max_gap = st.sidebar.slider(
            "Max consecutive missing days",
            min_value=2,
            max_value=30,
            value=self.cfg.max_gap_days,
            step=1,
            key="sb_max_gap",
        )

        st.sidebar.markdown("#### Signal Ratio Thresholds")
        green_max = st.sidebar.number_input(
            "Green max (<=)",
            min_value=0.05,
            max_value=5.00,
            value=float(self.cfg.ratio_green_max),
            step=0.05,
            key="sb_green_max",
        )
        orange_max = st.sidebar.number_input(
            "Orange max (<=)",
            min_value=green_max + 0.01,
            max_value=7.00,
            value=float(self.cfg.ratio_orange_max),
            step=0.05,
            key="sb_orange_max",
        )

        st.sidebar.markdown("#### Correlation Method")
        corr_method = st.sidebar.selectbox(
            "Method",
            options=["pearson", "spearman", "kendall", "ledoitwolf"],
            index=3 if HAS_SKLEARN else 0,
            key="sb_corr_method",
        )
        if corr_method == "ledoitwolf" and not HAS_SKLEARN:
            st.sidebar.warning("sklearn not installed. Ledoit-Wolf will fallback to Pearson.")

        st.sidebar.markdown("#### Portfolio Weights")
        weight_mode = st.sidebar.selectbox(
            "Weight mode",
            options=["equal", "custom"],
            index=0,
            key="sb_wmode",
        )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "tickers": tickers,
            "labels": selected_labels,
            "category": cat,
            "min_overlap": int(min_overlap),
            "max_missing": float(max_missing),
            "max_gap": int(max_gap),
            "green_max": float(green_max),
            "orange_max": float(orange_max),
            "corr_method": corr_method,
            "weight_mode": weight_mode,
        }

    @st.cache_data(show_spinner=False)
    def _load_data_cached(self, tickers: Tuple[str, ...], start: str, end: str, max_missing: float, max_gap: int) -> Dict[str, Any]:
        cfg2 = AppConfig(**{**self.cfg.__dict__})
        cfg2.max_missing_frac = float(max_missing)
        cfg2.max_gap_days = int(max_gap)
        de = DataEngine(cfg2)
        return de.fetch(list(tickers), start=start, end=end)

    def _display_overview(self, prices: pd.DataFrame, returns: pd.DataFrame, dq: Dict[str, Any], key_ns: str = "ov"):
        st.markdown("## Overview")
        if not df_nonempty(prices):
            st.warning("No data loaded. Please select instruments in the sidebar.")
            return

        last_date = prices.index.max()
        n_assets = prices.shape[1]
        n_obs = prices.shape[0]
        daily_rets = returns.dropna(how="all")
        k1 = ("Assets", f"{n_assets}", f"Obs: {n_obs}")
        k2 = ("Last Date", f"{last_date.date()}", "Data end")
        if n_assets >= 1:
            w = np.ones(n_assets) / n_assets
            p_rets = pd.Series(daily_rets.values @ w, index=daily_rets.index) if df_nonempty(daily_rets) else pd.Series(dtype=float)
            ann_ret = float(p_rets.mean() * self.cfg.annualization) if series_nonempty(p_rets) else np.nan
            ann_vol = float(p_rets.std(ddof=1) * math.sqrt(self.cfg.annualization)) if series_nonempty(p_rets) else np.nan
            sharpe = ann_ret / ann_vol if (ann_vol and ann_vol > 0 and not np.isnan(ann_ret)) else np.nan
            k3 = ("EqW Ann Return", safe_pct(ann_ret), "Return (annualized)")
            k4 = ("EqW Ann Vol", safe_pct(ann_vol), "Volatility (annualized)")
            k5 = ("EqW Sharpe", safe_float(sharpe, "{:.2f}"), "No RF adjustment")
            self.viz.kpi_row([k1, k2, k3, k4, k5])
        else:
            self.viz.kpi_row([k1, k2])

        st.markdown("### Data Quality Gate Report")
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("**Kept series**")
            st.write(dq.get("kept", []))
        with colB:
            st.markdown("**Dropped series**")
            st.write(dq.get("dropped", {}))

        st.markdown("### Prices (normalized)")
        norm = prices / prices.iloc[0]
        st.plotly_chart(self.viz.price_chart(norm, "Normalized Prices (Base=1.0)"), use_container_width=True)

        st.markdown("### Latest Returns Snapshot")
        st.dataframe(returns.tail(10), use_container_width=True)

    def _display_prices(self, prices: pd.DataFrame, returns: pd.DataFrame, labels: List[str], key_ns: str = "px"):
        st.markdown("## Prices & Returns")
        if not df_nonempty(prices):
            st.warning("No data loaded.")
            return

        st.plotly_chart(self.viz.price_chart(prices, "Prices (Auto-Adjusted)"), use_container_width=True)

        st.markdown("### Return Distribution (Selected)")
        if df_nonempty(returns):
            cols = st.columns(2)
            with cols[0]:
                pick = st.selectbox("Asset", list(prices.columns), index=0, key=f"{key_ns}_asset_sel")
            with cols[1]:
                bins = st.slider("Histogram bins", 10, 200, 60, 5, key=f"{key_ns}_bins")

            r = returns[pick].dropna()
            fig = px.histogram(r, nbins=bins, title=f"Return Histogram: {pick}")
            fig.update_layout(template="plotly_white", height=450, margin=dict(l=30, r=30, t=60, b=30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Summary Stats")
            s = pd.Series({
                "mean": r.mean(),
                "std": r.std(ddof=1),
                "skew": stats.skew(r.values, bias=False) if r.shape[0] > 10 else np.nan,
                "kurt_excess": stats.kurtosis(r.values, fisher=True, bias=False) if r.shape[0] > 10 else np.nan,
            })
            st.dataframe(s.to_frame("value"), use_container_width=True)

    def _display_ratio_signal(self, returns: pd.DataFrame, green_max: float, orange_max: float, key_ns: str = "ratio"):
        st.markdown("## Smart Signal: EWMA Volatility Ratio")
        st.markdown(
            f"""
            **Definition**  
            Ratio = **EWMA({self.cfg.ewma_short}D Vol)** / ( **EWMA({self.cfg.ewma_mid}D Vol)** + **EWMA({self.cfg.ewma_long}D Vol)** )

            **Interpretation (institutional risk signal)**  
            - Lower ratio → short-term volatility is modest relative to mid+long regime  
            - Higher ratio → short-term volatility is dominating → potential stress / acceleration
            """.strip()
        )

        if not df_nonempty(returns):
            st.warning("No returns available.")
            return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            asset = st.selectbox("Asset", list(returns.columns), index=0, key=f"{key_ns}_asset")
        with col2:
            show_bbands = st.checkbox("Show Bollinger Bands", value=True, key=f"{key_ns}_bb")
        with col3:
            bb_k = st.number_input("BB k", min_value=0.5, max_value=4.0, value=float(self.cfg.boll_k), step=0.1, key=f"{key_ns}_bbk")

        self.cfg.boll_k = float(bb_k)

        df_ratio = self.an.ratio_signal(returns[asset].dropna())
        if not df_nonempty(df_ratio):
            st.warning("Ratio series could not be computed (insufficient data).")
            return

        fig = self.viz.ratio_signal_chart(
            df_ratio=df_ratio,
            title=f"EWMA Ratio Signal — {asset}",
            green_max=green_max,
            orange_max=orange_max,
            show_bbands=show_bbands,
        )
        st.plotly_chart(fig, use_container_width=True)

        last_ratio = float(df_ratio["ratio"].dropna().iloc[-1]) if df_ratio["ratio"].dropna().shape[0] else np.nan
        if np.isnan(last_ratio):
            zone = "—"
        elif last_ratio <= green_max:
            zone = "GREEN"
        elif last_ratio <= orange_max:
            zone = "ORANGE"
        else:
            zone = "RED"

        self.viz.kpi_row([
            ("Last Ratio", safe_float(last_ratio, "{:.4f}"), "Most recent"),
            ("Zone", zone, f"Thresholds: green ≤ {green_max:.2f}, orange ≤ {orange_max:.2f}"),
            ("Obs", f"{df_ratio.shape[0]}", "Ratio history length"),
        ])

        st.markdown("### Under the hood (latest rows)")
        st.dataframe(df_ratio.tail(15), use_container_width=True)

    def _display_correlations(self, returns: pd.DataFrame, method: str, min_overlap: int, key_ns: str = "corr"):
        st.markdown("## Correlations (Robust Reporting)")
        if not df_nonempty(returns) or returns.shape[1] < 2:
            st.warning("Need at least 2 instruments with valid returns.")
            return

        st.caption("This correlation uses **strict overlap alignment** (rows where any series is missing are removed).")

        corr, meta = self.an.correlation_matrix(
            returns=returns,
            method=method,
            min_overlap_days=int(min_overlap),
        )

        if not df_nonempty(corr):
            st.error("Correlation could not be computed with the current overlap constraints.")
            st.write(meta)
            return

        st.plotly_chart(self.viz.correlation_heatmap(corr, f"Asset Correlations ({meta['method']})"), use_container_width=True)
        st.markdown("### Correlation Table")
        st.dataframe(corr.round(4), use_container_width=True)
        st.markdown("### Metadata / Diagnostics")
        st.json(meta)

    def _display_risk_analytics(self, returns: pd.DataFrame, weight_mode: str, key_ns: str = "risk"):
        st.markdown("## Risk Analytics (VaR / CVaR / ES)")
        if not df_nonempty(returns):
            st.warning("No returns available.")
            return

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            target = st.selectbox("Target", options=["Single Asset", "Equal-Weight Portfolio"], index=1, key=f"{key_ns}_target")
        with c2:
            alpha = st.selectbox("Confidence", options=list(self.cfg.var_levels), index=1, key=f"{key_ns}_alpha")
        with c3:
            horizon = st.slider("Horizon (days)", 1, 20, 1, 1, key=f"{key_ns}_h")

        if target == "Single Asset":
            asset = st.selectbox("Asset", list(returns.columns), index=0, key=f"{key_ns}_asset")
            r = returns[asset].dropna()
            label = asset
        else:
            cols = list(returns.columns)
            df = returns[cols].dropna(how="any")
            if not df_nonempty(df):
                st.warning("Insufficient strict-overlap returns for portfolio.")
                return
            w = self.an.equal_weights(df.shape[1])
            r = pd.Series(df.values @ w, index=df.index)
            label = "Equal-Weight Portfolio"

        if r.shape[0] < 60:
            st.warning("Short history may lead to unstable VaR estimates.")

        h = int(horizon)
        scale = math.sqrt(h)

        var_hist = self.an.var_historical(r, alpha) * scale
        cvar_hist = self.an.cvar_historical(r, alpha) * scale
        var_norm = self.an.var_parametric_normal(r, alpha) * scale
        cvar_norm = self.an.cvar_parametric_normal(r, alpha) * scale
        var_cf = self.an.var_cornish_fisher(r, alpha) * scale

        out = pd.DataFrame({
            "Metric": ["VaR (Hist)", "CVaR/ES (Hist)", "VaR (Normal)", "CVaR/ES (Normal)", "VaR (Cornish-Fisher)"],
            "Value": [var_hist, cvar_hist, var_norm, cvar_norm, var_cf],
        })
        out["Value %"] = out["Value"].apply(lambda x: safe_pct(x))
        st.dataframe(out, use_container_width=True)

        st.markdown("### Return Series & Tail Threshold")
        rr = r.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rr.index, y=rr.values, mode="lines", name="Returns"))
        q = np.nanquantile(rr.dropna().values, 1.0 - float(alpha))
        fig.add_hline(y=q, line_width=2, line_dash="dash", annotation_text=f"q(1-α)={q:.4f}")
        fig.update_layout(template="plotly_white", height=480, title=f"Returns — {label}", margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Multi-day VaR scaling uses √h approximation (institutional shortcut).")

    def _display_stress_testing(self, prices: pd.DataFrame, returns: pd.DataFrame, key_ns: str = "stress"):
        st.markdown("## Stress Testing & Shock Simulator")
        if not df_nonempty(prices) or not df_nonempty(returns):
            st.warning("Need prices and returns to run stress tests.")
            return

        st.markdown("### A) Parametric Shock Scenarios (instant shock)")
        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            shock_type = st.selectbox("Shock type", ["Return Shock", "Price Shock"], index=0, key=f"{key_ns}_stype")
        with cols[1]:
            shock = st.number_input("Shock magnitude (e.g., -0.05 = -5%)", value=-0.05, step=0.01, format="%.4f", key=f"{key_ns}_shock")
        with cols[2]:
            apply_to = st.selectbox("Apply to", ["All", "Single Asset"], index=0, key=f"{key_ns}_applyto")
        with cols[3]:
            asset = st.selectbox("Asset", list(prices.columns), index=0, key=f"{key_ns}_asset")

        last_prices = prices.iloc[-1].copy()

        if apply_to == "All":
            stressed = last_prices * (1.0 + shock)
        else:
            stressed = last_prices.copy()
            stressed.loc[asset] = last_prices.loc[asset] * (1.0 + shock)

        res = pd.DataFrame({
            "Last": last_prices,
            "Stressed": stressed,
            "Δ": stressed - last_prices,
            "Δ%": (stressed / last_prices - 1.0),
        })
        res["Δ%"] = res["Δ%"].apply(lambda x: safe_pct(float(x)))
        st.dataframe(res, use_container_width=True)

        st.markdown("### B) Historical Stress (worst days)")
        df = returns.dropna(how="any")
        if df_nonempty(df) and df.shape[1] >= 2:
            w = np.ones(df.shape[1]) / df.shape[1]
            p = pd.Series(df.values @ w, index=df.index)
            k = st.slider("Show worst N days", 5, 50, 10, 1, key=f"{key_ns}_worstn")
            worst = p.sort_values().head(k)
            st.write("Worst portfolio days (equal-weight, strict overlap):")
            st.dataframe(worst.to_frame("portfolio_return"), use_container_width=True)
        else:
            st.info("Historical portfolio stress requires strict-overlap returns with ≥2 assets.")

        st.markdown("### C) Drawdown Stress")
        norm = prices / prices.iloc[0]
        dd = norm / norm.cummax() - 1.0
        st.plotly_chart(self.viz.line_chart(dd, "Drawdown (per asset)", list(dd.columns), height=520), use_container_width=True)

    def _display_export(self, prices: pd.DataFrame, returns: pd.DataFrame, key_ns: str = "export"):
        st.markdown("## Export")
        if df_nonempty(prices):
            st.download_button(
                "Download Prices (CSV)",
                data=prices.to_csv().encode("utf-8"),
                file_name="prices.csv",
                mime="text/csv",
                key=f"{key_ns}_px",
            )
        if df_nonempty(returns):
            st.download_button(
                "Download Returns (CSV)",
                data=returns.to_csv().encode("utf-8"),
                file_name="returns.csv",
                mime="text/csv",
                key=f"{key_ns}_ret",
            )

    def run(self):
        st.markdown(f"# 🏛️ {self.cfg.title}")
        st.caption("Institutional-grade commodity analytics with robust correlation reporting and smart risk signals.")

        params = self._sidebar()
        self.cfg.max_missing_frac = float(params["max_missing"])
        self.cfg.max_gap_days = int(params["max_gap"])

        if len(params["tickers"]) == 0:
            st.info("Select instruments from the sidebar to start.")
            return

        start = str(params["start_date"])
        end = str(params["end_date"] + timedelta(days=1))

        with st.spinner("Downloading market data..."):
            pack = self._load_data_cached(tuple(params["tickers"]), start, end, params["max_missing"], params["max_gap"])

        prices = pack.get("prices", pd.DataFrame())
        returns = pack.get("returns", pd.DataFrame())
        dq = pack.get("dq", {})

        if not HAS_SKLEARN:
            st.info("ℹ️ For Ledoit-Wolf shrinkage correlation, add **scikit-learn** to requirements.txt.")
        if not HAS_HMMLEARN:
            st.caption("HMM regimes optional. Install **hmmlearn** to enable regimes (not required for v7.1 core).")
        if not HAS_ARCH:
            st.caption("GARCH optional. Install **arch** to enable GARCH models (not required for v7.1 core).")

        tab_over, tab_px, tab_ratio, tab_corr, tab_risk, tab_stress, tab_export = st.tabs([
            "Overview",
            "Prices & Returns",
            "Smart Signal (EWMA Ratio)",
            "Correlations",
            "Risk (VaR/CVaR/ES)",
            "Stress Testing",
            "Export",
        ])

        with tab_over:
            self._display_overview(prices, returns, dq, key_ns="ov")
        with tab_px:
            self._display_prices(prices, returns, params["labels"], key_ns="px")
        with tab_ratio:
            self._display_ratio_signal(returns, params["green_max"], params["orange_max"], key_ns="ratio")
        with tab_corr:
            self._display_correlations(returns, params["corr_method"], params["min_overlap"], key_ns="corr")
        with tab_risk:
            self._display_risk_analytics(returns, params["weight_mode"], key_ns="risk")
        with tab_stress:
            self._display_stress_testing(prices, returns, key_ns="stress")
        with tab_export:
            self._display_export(prices, returns, key_ns="export")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    try:
        app = InstitutionalCommoditiesDashboard(CFG)
        app.run()
    except Exception as e:
        st.error("🚨 Application Error")
        st.code(str(e))
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
