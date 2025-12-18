"""
Institutional Commodities Analytics Platform (Streamlit Cloud-ready) — v3
--------------------------------------------------------------------------------
Implements:
- Portfolio tab (weights, exposures vs SPX/DXY, ES risk contributions)
- Champion GARCH selection (grid → champion → refit vol + forecast)
- Regime-conditioned risk (HMM regimes → VaR/ES by regime)
- Scenario library (systematic shocks via betas + vol shocks + multi-day profiles)
- VaR backtests (Kupiec POF + Christoffersen independence for Historical/Normal VaR)

Data: Yahoo Finance via yfinance. Expect occasional gaps/delays.
"""

import os, math, warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Diagnostics / backtests
try:
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# HMM regimes
try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# QuantStats optional
try:
    import quantstats as qs
    QS_AVAILABLE = True
except Exception:
    QS_AVAILABLE = False

st.set_page_config(page_title="Institutional Commodities Platform v3", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
  .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 1.7rem;border-radius: 15px;color: white;margin-bottom: 1rem;box-shadow: 0 10px 30px rgba(0,0,0,0.10);}
  .metric-card {background: white;padding: 1.15rem;border-radius: 12px;box-shadow: 0 4px 15px rgba(0,0,0,0.05);border-left: 4px solid #667eea;}
  .metric-value {font-size: 1.85rem;font-weight: 750;color: #243447;margin: 0.35rem 0;}
  .metric-label {font-size: 0.82rem;color: #7f8c8d;text-transform: uppercase;letter-spacing: 0.9px;}
  .positive {color: #27ae60;font-weight: 700;}
  .negative {color: #e74c3c;font-weight: 700;}
  .neutral  {color: #f39c12;font-weight: 700;}
  .sidebar-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 0.9rem;border-radius: 10px;color: white;margin-bottom: 1rem;text-align: center;}
  .status-badge {display:inline-block;padding:0.2rem 0.7rem;border-radius:18px;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.4px;}
  .status-success {background:#d4edda;color:#155724;}
  .status-warning {background:#fff3cd;color:#856404;}
</style>
""",
    unsafe_allow_html=True,
)

COMMODITIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Precious Metals": {"GC=F": {"name": "Gold Futures"}, "SI=F": {"name": "Silver Futures"}, "PL=F": {"name": "Platinum Futures"}, "PA=F": {"name": "Palladium Futures"}},
    "Industrial Metals": {"HG=F": {"name": "Copper Futures"}, "ALI=F": {"name": "Aluminum Futures"}},
    "Energy": {"CL=F": {"name": "Crude Oil WTI"}, "BZ=F": {"name": "Brent Crude"}, "NG=F": {"name": "Natural Gas"}},
    "Agriculture": {"ZC=F": {"name": "Corn Futures"}, "ZW=F": {"name": "Wheat Futures"}, "ZS=F": {"name": "Soybean Futures"}},
}
BENCHMARKS = {"^GSPC": {"name": "S&P 500 (SPX proxy)"}, "DX-Y.NYB": {"name": "US Dollar Index (DXY)"}}

def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def format_number(x: Any, d: int = 3) -> str:
    v = safe_float(x, np.nan)
    return "—" if not np.isfinite(v) else f"{v:.{d}f}"

def format_percentage(x: Any, d: int = 2) -> str:
    v = safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{d}f}%"

def badge(text: str, success: bool) -> str:
    cls = "status-success" if success else "status-warning"
    return f'<span class="status-badge {cls}">{text}</span>'

def annualize_vol(s: pd.Series, ann: int = 252) -> pd.Series:
    return s * math.sqrt(ann)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yf(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=["Adj Close"])
    df.index = pd.to_datetime(df.index)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Returns"] = df["Adj Close"].pct_change()
    df["SMA20"] = df["Adj Close"].rolling(20).mean()
    df["SMA50"] = df["Adj Close"].rolling(50).mean()
    bb_std = df["Adj Close"].rolling(20).std()
    df["BB_U"] = df["SMA20"] + 2 * bb_std
    df["BB_L"] = df["SMA20"] - 2 * bb_std
    df["RV20"] = df["Returns"].rolling(20).std()
    df["RV60"] = df["Returns"].rolling(60).std()
    delta = df["Adj Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    return df

def returns_series(df: pd.DataFrame) -> pd.Series:
    return df["Adj Close"].pct_change().dropna()

class Analytics:
    @staticmethod
    def perf_metrics(r: pd.Series) -> Dict[str, Any]:
        r = r.dropna()
        if len(r) < 60:
            return {}
        if QS_AVAILABLE:
            qs.extend_pandas()
            try:
                return {"total_ret": float(qs.stats.comp(r)), "vol": float(qs.stats.volatility(r)), "sharpe": float(qs.stats.sharpe(r)), "max_dd": float(qs.stats.max_drawdown(r))}
            except Exception:
                pass
        ann = 252.0
        total = float((1 + r).prod() - 1)
        vol = float(r.std(ddof=1) * math.sqrt(ann))
        mu = float(r.mean() * ann)
        sharpe = float(mu / vol) if vol > 1e-12 else np.nan
        eq = (1 + r).cumprod()
        max_dd = float((eq / eq.cummax() - 1).min())
        return {"total_ret": total, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}

    @staticmethod
    def arch_lm(r: pd.Series, lag: int = 5) -> float:
        if not STATSMODELS_AVAILABLE:
            return np.nan
        x = r.dropna().values
        if len(x) < lag + 40:
            return np.nan
        try:
            _, pval, _, _ = het_arch(x - np.mean(x), maxlag=lag)
            return float(pval)
        except Exception:
            return np.nan

    @staticmethod
    def ljung_box(r: pd.Series, lag: int = 10) -> float:
        if not STATSMODELS_AVAILABLE:
            return np.nan
        x = r.dropna().values
        if len(x) < lag + 40:
            return np.nan
        try:
            df = acorr_ljungbox(x, lags=[lag], return_df=True)
            return float(df["lb_pvalue"].iloc[0])
        except Exception:
            return np.nan

    @staticmethod
    def rolling_beta(asset: pd.Series, bench: pd.Series, window: int = 60) -> pd.DataFrame:
        df = pd.concat([asset.rename("a"), bench.rename("b")], axis=1, join="inner").dropna()
        if len(df) < window + 5:
            return pd.DataFrame()
        cov = df["a"].rolling(window).cov(df["b"])
        var = df["b"].rolling(window).var().replace(0.0, np.nan)
        beta = cov / var
        corr = df["a"].rolling(window).corr(df["b"])
        return pd.DataFrame({"beta": beta, "r2": corr**2}).dropna()

    @staticmethod
    def static_beta(asset: pd.Series, bench: Optional[pd.Series], window: int = 252) -> float:
        if bench is None:
            return 0.0
        df = pd.concat([asset.rename("a"), bench.rename("b")], axis=1, join="inner").dropna().tail(window)
        if len(df) < 60:
            return np.nan
        v = df["b"].var()
        return float(df["a"].cov(df["b"]) / v) if v and v > 1e-18 else np.nan

    @staticmethod
    def hist_var_es(r: pd.Series, conf: float) -> Tuple[float, float]:
        x = r.dropna().values
        if len(x) < 60:
            return (np.nan, np.nan)
        var = float(np.quantile(x, 1 - conf))
        es = float(np.mean(x[x <= var])) if (x <= var).any() else np.nan
        return var, es

    @staticmethod
    def normal_var_es(r: pd.Series, conf: float) -> Tuple[float, float]:
        x = r.dropna().values
        if len(x) < 60:
            return (np.nan, np.nan)
        mu = float(np.mean(x))
        sig = float(np.std(x, ddof=1))
        z = stats.norm.ppf(1 - conf)
        var = float(mu + sig * z)
        es = float(mu - sig * stats.norm.pdf(z) / (1 - conf))
        return var, es

    @staticmethod
    def kupiec_pof(hits: np.ndarray, p: float) -> Tuple[float, float]:
        hits = np.asarray(hits, dtype=bool)
        n = hits.size
        x = hits.sum()
        if n <= 0:
            return (np.nan, np.nan)
        phat = min(max(x / n, 1e-12), 1 - 1e-12)
        p = min(max(p, 1e-12), 1 - 1e-12)
        L0 = (1 - p) ** (n - x) * (p ** x)
        L1 = (1 - phat) ** (n - x) * (phat ** x)
        LR = float(-2 * np.log(L0 / L1))
        return LR, float(1 - stats.chi2.cdf(LR, df=1))

    @staticmethod
    def christoffersen_independence(hits: np.ndarray) -> Tuple[float, float]:
        h = np.asarray(hits, dtype=int)
        if h.size < 2:
            return (np.nan, np.nan)
        h0, h1 = h[:-1], h[1:]
        n00 = int(((h0 == 0) & (h1 == 0)).sum())
        n01 = int(((h0 == 0) & (h1 == 1)).sum())
        n10 = int(((h0 == 1) & (h1 == 0)).sum())
        n11 = int(((h0 == 1) & (h1 == 1)).sum())

        def rate(num, den):
            r = (num / den) if den else 0.0
            return min(max(r, 1e-12), 1 - 1e-12)

        p01 = rate(n01, n00 + n01)
        p11 = rate(n11, n10 + n11)
        p1 = rate(n01 + n11, n00 + n01 + n10 + n11)

        L0 = ((1 - p1) ** (n00 + n10)) * (p1 ** (n01 + n11))
        L1 = ((1 - p01) ** n00) * (p01 ** n01) * ((1 - p11) ** n10) * (p11 ** n11)
        LR = float(-2 * np.log(L0 / L1))
        return LR, float(1 - stats.chi2.cdf(LR, df=1))

    @staticmethod
    def rolling_var_backtest(r: pd.Series, method: str, window: int, conf: float) -> pd.DataFrame:
        r = r.dropna()
        if len(r) < window + 50:
            return pd.DataFrame()
        alpha = 1 - conf
        rets = r.values
        idx = r.index
        var = np.full_like(rets, np.nan, dtype=float)
        if method == "historical":
            for t in range(window, len(rets)):
                var[t] = np.quantile(rets[t-window:t], alpha)
        else:
            z = stats.norm.ppf(alpha)
            for t in range(window, len(rets)):
                w = rets[t-window:t]
                var[t] = np.mean(w) + np.std(w, ddof=1) * z
        df = pd.DataFrame({"ret": rets, "var": var}, index=idx).dropna()
        df["hit"] = df["ret"] < df["var"]
        return df

    @staticmethod
    def garch_grid(r: pd.Series, vol_models: Tuple[str, ...], p_max: int, q_max: int, dists: Tuple[str, ...], cap: int) -> pd.DataFrame:
        if not ARCH_AVAILABLE:
            return pd.DataFrame()
        r = r.dropna()
        if len(r) < 450:
            return pd.DataFrame()
        combos = []
        for vol in vol_models:
            for p in range(1, p_max + 1):
                for q in range(1, q_max + 1):
                    for dist in dists:
                        combos.append((vol, p, q, 0, dist))
                        if vol == "GARCH":
                            combos.append((vol, p, q, 1, dist))
        combos = combos[:cap]
        scaled = r.values * 100.0
        rows = []
        for vol, p, q, o, dist in combos:
            try:
                m = arch_model(scaled, mean="Constant", vol=vol, p=p, o=o, q=q, dist=dist)
                fit = m.fit(disp="off", show_warning=False)
                rows.append({"vol": vol, "p": p, "q": q, "o": o, "dist": dist, "aic": float(fit.aic), "bic": float(fit.bic), "converged": bool(getattr(fit, "convergence_flag", 0) == 0)})
            except Exception:
                rows.append({"vol": vol, "p": p, "q": q, "o": o, "dist": dist, "aic": np.nan, "bic": np.nan, "converged": False})
        return pd.DataFrame(rows).sort_values(["converged", "aic", "bic"], ascending=[False, True, True], na_position="last")

    @staticmethod
    def pick_champion(grid: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if grid.empty:
            return None
        g = grid[np.isfinite(grid["aic"])]
        if g.empty:
            return None
        conv = g[g["converged"] == True]
        row = conv.iloc[0] if not conv.empty else g.iloc[0]
        return {"vol": row["vol"], "p": int(row["p"]), "q": int(row["q"]), "o": int(row["o"]), "dist": row["dist"]}

    @staticmethod
    def fit_garch_once(r: pd.Series, spec: Dict[str, Any], horizon: int) -> Optional[Dict[str, Any]]:
        if not ARCH_AVAILABLE or spec is None:
            return None
        r = r.dropna()
        if len(r) < 300:
            return None
        scaled = r.values * 100.0
        try:
            m = arch_model(scaled, mean="Constant", vol=spec["vol"], p=spec["p"], o=spec.get("o", 0), q=spec["q"], dist=spec["dist"])
            fit = m.fit(disp="off", show_warning=False)
            cond_vol = pd.Series(fit.conditional_volatility, index=r.index) / 100.0
            fc = fit.forecast(horizon=horizon, reindex=False)
            var_fc = np.array(fc.variance.values[-1, :], dtype=float)
            vol_fc = np.sqrt(np.maximum(var_fc, 0.0)) / 100.0
            fc_idx = pd.date_range(start=r.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
            return {"desc": f"{spec['vol']}({spec['p']},{spec.get('o',0)},{spec['q']}) dist={spec['dist']}", "aic": float(fit.aic), "bic": float(fit.bic), "converged": bool(getattr(fit, "convergence_flag", 0) == 0), "cond_vol": cond_vol, "vol_forecast": pd.Series(vol_fc, index=fc_idx)}
        except Exception:
            return None

    @staticmethod
    def hmm_regimes(r: pd.Series, rv: Optional[pd.Series], n_states: int, use_rv: bool, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not HMM_AVAILABLE:
            return pd.DataFrame(), pd.DataFrame()
        r = r.dropna()
        if len(r) < 260:
            return pd.DataFrame(), pd.DataFrame()
        feat = pd.DataFrame({"ret": r})
        feat["rv"] = (rv.reindex(r.index) if (use_rv and rv is not None) else r.rolling(20).std())
        feat = feat.dropna()
        if len(feat) < 260:
            return pd.DataFrame(), pd.DataFrame()
        X = StandardScaler().fit_transform(feat[["ret", "rv"]].values)
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, random_state=seed)
        model.fit(X)
        feat["state"] = model.predict(X)

        stats_rows = []
        for s in sorted(feat["state"].unique()):
            sub = feat[feat["state"] == s]
            stats_rows.append({"state": int(s), "freq": float(len(sub)/len(feat)), "mean_daily": float(sub["ret"].mean()), "ann_vol": float(sub["ret"].std(ddof=1)*math.sqrt(252))})
        df_stats = pd.DataFrame(stats_rows).sort_values("mean_daily")
        ordered = list(df_stats["state"].values)
        labels = {ordered[0]: "Risk-Off", ordered[-1]: "Risk-On"}
        for mid in ordered[1:-1]:
            labels[mid] = "Neutral"
        feat["regime"] = feat["state"].map(labels)
        df_stats["regime"] = df_stats["state"].map(labels)
        return feat, df_stats

    @staticmethod
    def regime_conditioned_risk(df_states: pd.DataFrame, confs: Tuple[float, ...]) -> pd.DataFrame:
        if df_states.empty:
            return pd.DataFrame()
        rows = []
        for reg in sorted(df_states["regime"].dropna().unique()):
            sub = df_states[df_states["regime"] == reg]["ret"]
            if len(sub) < 60:
                continue
            for conf in confs:
                hvar, hes = Analytics.hist_var_es(sub, conf)
                nvar, nes = Analytics.normal_var_es(sub, conf)
                rows.append({"regime": reg, "conf": conf, "n_obs": int(len(sub)), "Hist_VaR": hvar, "Hist_ES": hes, "Norm_VaR": nvar, "Norm_ES": nes})
        return pd.DataFrame(rows)

    @staticmethod
    def es_contributions(ret_df: pd.DataFrame, w: np.ndarray, conf: float) -> pd.DataFrame:
        df = ret_df.dropna()
        if df.empty or len(df) < 200:
            return pd.DataFrame()
        w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)
        port = df.values @ w
        var = np.quantile(port, 1-conf)
        tail = port <= var
        if tail.sum() < 10:
            return pd.DataFrame()
        contrib = df.values * w
        es_c = contrib[tail].mean(axis=0)
        share = np.abs(es_c) / (np.sum(np.abs(es_c)) + 1e-12)
        return pd.DataFrame({"ES_contrib": es_c, "AbsShare": share}, index=df.columns).sort_values("AbsShare", ascending=False)

SCENARIOS = {
    "Baseline (no shock)": {"spx": 0.0, "dxy": 0.0, "vol": 1.0, "profile": "flat"},
    "Equity risk-off (SPX -5%)": {"spx": -0.05, "dxy": 0.0, "vol": 1.2, "profile": "flat"},
    "USD spike (DXY +2%)": {"spx": 0.0, "dxy": 0.02, "vol": 1.15, "profile": "flat"},
    "Vol shock (×1.5 vol)": {"spx": 0.0, "dxy": 0.0, "vol": 1.5, "profile": "flat"},
    "Gap shock (day1 -8%, days2-5 -2%)": {"spx": 0.0, "dxy": 0.0, "vol": 1.25, "profile": "gap_down"},
}

def profile_array(profile: str, horizon: int, day1: float) -> np.ndarray:
    a = np.zeros(horizon, dtype=float)
    if horizon <= 0:
        return a
    if profile == "gap_down":
        a[0] = day1
        for t in range(1, min(horizon, 5)):
            a[t] = day1 * 0.25
    else:
        a[0] = day1
    return a

def simulate_paths(hist_ret: np.ndarray, w: np.ndarray, betas_spx: np.ndarray, betas_dxy: np.ndarray,
                   horizon: int, sims: int, method: str, spx_shock: float, dxy_shock: float, vol_scale: float, prof: str, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, k = hist_ret.shape
    if T < 220:
        return np.zeros((0, 0))
    if method == "normal":
        mu = hist_ret.mean(axis=0)
        cov = np.cov(hist_ret.T)
        try:
            samp = rng.multivariate_normal(mu, cov, size=(sims, horizon))
        except Exception:
            sig = hist_ret.std(axis=0, ddof=1)
            samp = rng.normal(mu, sig, size=(sims, horizon, k))
    else:
        idx = rng.integers(0, T, size=(sims, horizon))
        samp = hist_ret[idx]
    samp = samp * float(vol_scale)
    spx_prof = profile_array(prof, horizon, spx_shock)
    dxy_prof = profile_array(prof, horizon, dxy_shock)
    for t in range(horizon):
        if spx_prof[t] != 0:
            samp[:, t, :] += spx_prof[t] * betas_spx.reshape(1, -1)
        if dxy_prof[t] != 0:
            samp[:, t, :] += dxy_prof[t] * betas_dxy.reshape(1, -1)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w)/len(w)
    paths = np.ones((sims, horizon + 1), dtype=float)
    for t in range(horizon):
        pr = samp[:, t, :] @ w
        paths[:, t+1] = paths[:, t] * (1 + pr)
    return paths

class Viz:
    @staticmethod
    def price_tech(df: pd.DataFrame, title: str) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], subplot_titles=("Price", "RSI(14)"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Price", line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], name="BB Lower", line=dict(dash="dot"), fill="tonexty", opacity=0.2), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", opacity=0.35, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", opacity=0.35, row=2, col=1)
        fig.update_layout(height=720, template="plotly_white", hovermode="x unified", title=title)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        return fig

    @staticmethod
    def cum_returns(ret_df: pd.DataFrame, title: str) -> go.Figure:
        cum = (1 + ret_df).cumprod()
        fig = go.Figure()
        for c in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c))
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title)
        return fig

    @staticmethod
    def vol_plot(r: pd.Series, fit: Dict[str, Any], title: str) -> go.Figure:
        rv20 = r.rolling(20).std()
        rv60 = r.rolling(60).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv20.index, y=annualize_vol(rv20), name="Realized (20d)"))
        fig.add_trace(go.Scatter(x=rv60.index, y=annualize_vol(rv60), name="Realized (60d)", line=dict(dash="dash")))
        if fit:
            fig.add_trace(go.Scatter(x=fit["cond_vol"].index, y=annualize_vol(fit["cond_vol"]), name="GARCH Conditional", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=fit["vol_forecast"].index, y=annualize_vol(fit["vol_forecast"]), name="Forecast", line=dict(dash="dot")))
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title, yaxis_title="Annualized vol")
        return fig

    @staticmethod
    def rolling_beta_plot(beta_df: pd.DataFrame, title: str) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], subplot_titles=("Beta", "R²"))
        fig.add_trace(go.Scatter(x=beta_df.index, y=beta_df["beta"], name="Beta", line=dict(width=2)), row=1, col=1)
        fig.add_hline(y=0, opacity=0.35, row=1, col=1)
        fig.add_trace(go.Scatter(x=beta_df.index, y=beta_df["r2"], name="R²", line=dict(dash="dash")), row=2, col=1)
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title)
        return fig

    @staticmethod
    def backtest_plot(df_bt: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt["ret"], name="Return"))
        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt["var"], name="VaR", line=dict(dash="dash")))
        hits = df_bt[df_bt["hit"]]
        fig.add_trace(go.Scatter(x=hits.index, y=hits["ret"], name="Breach", mode="markers", marker=dict(size=7)))
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title, yaxis_title="Return")
        return fig

    @staticmethod
    def regime_price_plot(price: pd.Series, regime: pd.Series, title: str) -> go.Figure:
        df = pd.DataFrame({"price": price, "regime": regime}).dropna()
        codes = df["regime"].astype("category").cat.codes
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", opacity=0.55))
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="markers", marker=dict(size=5, color=codes), name="Regime"))
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title, showlegend=False)
        return fig

    @staticmethod
    def scenario_band(paths: np.ndarray, title: str) -> go.Figure:
        p05 = np.quantile(paths, 0.05, axis=0)
        p50 = np.quantile(paths, 0.50, axis=0)
        p95 = np.quantile(paths, 0.95, axis=0)
        t = np.arange(paths.shape[1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=p50, name="Median", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=t, y=p95, name="P95", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=t, y=p05, name="P05", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=list(t)+list(t[::-1]), y=list(p95)+list(p05[::-1]), fill="toself", opacity=0.15, line=dict(width=0), name="90% band"))
        fig.update_layout(height=520, template="plotly_white", hovermode="x unified", title=title, xaxis_title="Days ahead", yaxis_title="Portfolio value")
        return fig

class Dashboard:
    def __init__(self):
        self.an = Analytics()
        self.viz = Viz()

    def metric_card(self, label: str, value: str, good: Optional[bool] = None):
        cls = "neutral" if good is None else ("positive" if good else "negative")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {cls}">{value}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    def header(self):
        st.markdown(
            """
<div class="main-header">
  <h1 style="margin:0;font-size:2.5rem;">🏦 Institutional Commodities Analytics — v3</h1>
  <p style="margin:8px 0 0 0;opacity:0.95;">
    Portfolio • Champion GARCH • Regime Risk • Scenario Library • VaR Backtests
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
        if not ARCH_AVAILABLE:
            st.warning("`arch` not available → GARCH features disabled.")
        if not HMM_AVAILABLE:
            st.info("HMM regimes optional. Install `hmmlearn` + `scikit-learn` to enable regimes.")

    def sidebar(self) -> Dict[str, Any]:
        with st.sidebar:
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Controls")
            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                start = st.date_input("Start", datetime.now().date() - timedelta(days=1095))
            with c2:
                end = st.date_input("End", datetime.now().date())
            if start >= end:
                st.error("End must be > start")
                st.stop()

            cat = st.selectbox("Category", list(COMMODITIES.keys()))
            syms = st.multiselect(
                "Assets",
                list(COMMODITIES[cat].keys()),
                default=list(COMMODITIES[cat].keys())[:3],
                format_func=lambda s: f"{COMMODITIES[cat][s]['name']} ({s})",
            )

            st.markdown("---")
            bench = st.multiselect("Benchmarks", list(BENCHMARKS.keys()), default=["^GSPC", "DX-Y.NYB"], format_func=lambda b: BENCHMARKS[b]["name"])

            st.markdown("---")
            arch_lags = st.slider("ARCH LM lags", 1, 20, 5, 1)
            garch_h = st.slider("GARCH forecast horizon", 5, 60, 20, 5)
            grid_cap = st.slider("Champion grid cap (fits)", 6, 36, 18, 2)
            p_max = st.slider("p max", 1, 4, 2, 1)
            q_max = st.slider("q max", 1, 4, 2, 1)
            vol_models = st.multiselect("Vol models", ["GARCH", "EGARCH"], default=["GARCH", "EGARCH"])
            dists = st.multiselect("Dists", ["normal", "t", "skewt"], default=["t", "normal"])

            st.markdown("---")
            confs = st.multiselect("Confidence", [0.90, 0.95, 0.99], default=[0.95, 0.99])
            bt_window = st.slider("VaR backtest window", 100, 750, 250, 25)
            beta_window = st.slider("Rolling beta window", 20, 252, 60, 5)

            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        return {
            "start": start, "end": end, "syms": syms,
            "bench": bench, "arch_lags": arch_lags, "garch_h": garch_h,
            "grid_cap": grid_cap, "p_max": p_max, "q_max": q_max,
            "vol_models": tuple(vol_models) if vol_models else ("GARCH",),
            "dists": tuple(dists) if dists else ("t",),
            "confs": tuple(sorted(confs)) if confs else (0.95,),
            "bt_window": bt_window,
            "beta_window": beta_window,
        }

    def run(self):
        self.header()
        p = self.sidebar()
        if not p["syms"]:
            st.warning("Select at least one asset.")
            return

        data, rets, metrics, diag, bench_rets = {}, {}, {}, {}, {}
        with st.spinner("Downloading & preparing data..."):
            for sym in p["syms"]:
                df = fetch_yf(sym, p["start"], p["end"])
                if df.empty or len(df) < 180:
                    st.warning(f"Insufficient data for {sym}.")
                    continue
                df = add_features(df)
                data[sym] = df
                r = returns_series(df)
                rets[sym] = r
                metrics[sym] = self.an.perf_metrics(r)
                diag[sym] = {"arch_p": self.an.arch_lm(r, p["arch_lags"]), "lb_p": self.an.ljung_box(r, 10)}

            for b in p["bench"]:
                bdf = fetch_yf(b, p["start"], p["end"])
                if not bdf.empty and len(bdf) >= 180:
                    bench_rets[b] = bdf["Adj Close"].pct_change().dropna()

        if not rets:
            st.error("No data loaded.")
            return

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Assets", "🧺 Portfolio", "🧠 Volatility", "🧩 Regimes", "⚠️ Scenarios", "✅ VaR Backtests"])
        with tab1:
            self.tab_assets(data, rets, metrics)
        with tab2:
            self.tab_portfolio(rets, bench_rets, p)
        with tab3:
            self.tab_volatility(rets, diag, p)
        with tab4:
            self.tab_regimes(data, rets, p)
        with tab5:
            self.tab_scenarios(rets, bench_rets, p)
        with tab6:
            self.tab_backtests(rets, p)

    def tab_assets(self, data, rets, metrics):
        st.markdown("### 📈 Asset Overview")
        for sym, df in data.items():
            st.plotly_chart(self.viz.price_tech(df, f"{sym} — Price & Technicals"), use_container_width=True)
            m = metrics.get(sym, {})
            c1, c2, c3, c4 = st.columns(4)
            with c1: self.metric_card("Total Return", format_percentage(safe_float(m.get("total_ret"))*100, 2), good=safe_float(m.get("total_ret")) > 0)
            with c2: self.metric_card("Sharpe", format_number(m.get("sharpe"), 3), good=(safe_float(m.get("sharpe"), np.nan) > 1))
            with c3: self.metric_card("Max DD", format_percentage(safe_float(m.get("max_dd"))*100, 2))
            with c4: self.metric_card("Vol", format_percentage(safe_float(m.get("vol"))*100, 2))
            st.markdown("---")
        aligned = pd.concat(rets.values(), axis=1, join="inner")
        aligned.columns = list(rets.keys())
        st.plotly_chart(self.viz.cum_returns(aligned, "Cumulative Returns (aligned)"), use_container_width=True)

    def tab_portfolio(self, rets, bench_rets, p):
        st.markdown("### 🧺 Portfolio Builder")
        assets = list(rets.keys())
        ret_df = pd.concat([rets[a].rename(a) for a in assets], axis=1, join="inner").dropna()
        if ret_df.empty or len(ret_df) < 180:
            st.warning("Not enough overlap for portfolio.")
            return
        mode = st.radio("Weights", ["Equal", "Custom"], horizontal=True, key="pf_weights_mode")
        if mode == "Equal":
            w = np.ones(len(assets)) / len(assets)
        else:
            cols = st.columns(min(4, len(assets)))
            w_list = []
            for i, a in enumerate(assets):
                with cols[i % len(cols)]:
                    w_list.append(st.slider(a, 0.0, 1.0, 1.0/len(assets), 0.01))
            w = np.array(w_list, dtype=float)
            w = (w / w.sum()) if w.sum() > 1e-12 else np.ones(len(assets)) / len(assets)

        port = pd.Series(ret_df.values @ w, index=ret_df.index, name="PORTFOLIO")
        m = self.an.perf_metrics(port)
        c1, c2, c3, c4 = st.columns(4)
        with c1: self.metric_card("Total Return", format_percentage(safe_float(m.get("total_ret"))*100, 2), good=safe_float(m.get("total_ret")) > 0)
        with c2: self.metric_card("Sharpe", format_number(m.get("sharpe"), 3), good=(safe_float(m.get("sharpe"), np.nan) > 1))
        with c3: self.metric_card("Max DD", format_percentage(safe_float(m.get("max_dd"))*100, 2))
        with c4: self.metric_card("Vol", format_percentage(safe_float(m.get("vol"))*100, 2))
        st.plotly_chart(self.viz.cum_returns(pd.DataFrame({"PORTFOLIO": port}), "Portfolio Cumulative Return"), use_container_width=True)

        conf = float(st.selectbox("ES confidence", list(p["confs"]), index=min(1, len(p["confs"])-1)))
        contrib = self.an.es_contributions(ret_df, w, conf)
        if contrib.empty:
            st.info("ES contributions unavailable (not enough tail).")
        else:
            st.dataframe(contrib.style.format({"ES_contrib": "{:.4%}", "AbsShare": "{:.1%}"}), use_container_width=True)

        if bench_rets:
            for b, br in bench_rets.items():
                beta_df = self.an.rolling_beta(port, br, window=p["beta_window"])
                if not beta_df.empty:
                    st.plotly_chart(self.viz.rolling_beta_plot(beta_df, f"PORTFOLIO vs {b}"), use_container_width=True)

    def tab_volatility(self, rets, diag, p):
        st.markdown("### 🧠 Volatility (Champion GARCH)")
        target = st.selectbox("Target", list(rets.keys()), index=0, key="vol_target")
        r = rets[target]
        c1, c2, c3 = st.columns(3)
        with c1: self.metric_card("ARCH LM p", format_number(diag[target]["arch_p"], 4), good=(safe_float(diag[target]["arch_p"], 1) < 0.05))
        with c2: self.metric_card("LB p (lag10)", format_number(diag[target]["lb_p"], 4), good=(safe_float(diag[target]["lb_p"], 0) > 0.05))
        with c3: st.markdown(f"**Obs:** {len(r)}")

        if not ARCH_AVAILABLE:
            st.info("Install `arch` for champion selection.")
            return

        champ_key, grid_key = f"champ::{target}", f"grid::{target}"
        if st.button("🏁 Run champion search", use_container_width=True):
            grid = self.an.garch_grid(r, p["vol_models"], p["p_max"], p["q_max"], p["dists"], cap=p["grid_cap"])
            st.session_state[grid_key] = grid
            st.session_state[champ_key] = self.an.pick_champion(grid)

        grid = st.session_state.get(grid_key, pd.DataFrame())
        champ = st.session_state.get(champ_key, None)

        if not grid.empty:
            st.dataframe(grid.style.format({"aic": "{:.2f}", "bic": "{:.2f}"}), use_container_width=True)

        if champ:
            st.success(f"Champion: {champ['vol']}({champ['p']},{champ['o']},{champ['q']}) dist={champ['dist']}")
            fit = self.an.fit_garch_once(r, champ, horizon=p["garch_h"])
            if fit:
                st.markdown(f"**Model:** {fit['desc']} • AIC {fit['aic']:.2f} • BIC {fit['bic']:.2f} • {badge('Converged' if fit['converged'] else 'Unstable', fit['converged'])}", unsafe_allow_html=True)
                st.plotly_chart(self.viz.vol_plot(r, {"cond_vol": fit["cond_vol"], "vol_forecast": fit["vol_forecast"]}, f"{target} — Volatility"), use_container_width=True)
            else:
                st.warning("Fit failed.")
        else:
            st.info("Run champion search to select a model.")

    def tab_regimes(self, data, rets, p):
        st.markdown("### 🧩 Regime-conditioned Risk (HMM)")
        if not HMM_AVAILABLE:
            st.info("Install `hmmlearn` + `scikit-learn` to enable regimes.")
            return
        target = st.selectbox("Target", list(rets.keys()), index=0, key="reg_target")
        n_states = st.slider("States", 2, 5, 3, 1)
        use_rv = st.checkbox("Use RV20 feature", True)
        seed = st.number_input("Seed", 0, 10000, 7, 1)
        df_states, df_stats = self.an.hmm_regimes(rets[target], data[target].get("RV20") if target in data else None, n_states=n_states, use_rv=use_rv, seed=int(seed))
        if df_states.empty:
            st.warning("Not enough data for HMM.")
            return
        price = data[target]["Adj Close"].reindex(df_states.index)
        st.plotly_chart(self.viz.regime_price_plot(price, df_states["regime"], f"{target} — Regimes"), use_container_width=True)
        st.dataframe(df_stats.style.format({"freq": "{:.2%}", "mean_daily": "{:.4%}", "ann_vol": "{:.2%}"}), use_container_width=True)
        rc = self.an.regime_conditioned_risk(df_states, p["confs"])
        if not rc.empty:
            st.dataframe(rc.style.format({"Hist_VaR": "{:.4%}", "Hist_ES": "{:.4%}", "Norm_VaR": "{:.4%}", "Norm_ES": "{:.4%}"}), use_container_width=True)

    def tab_scenarios(self, rets, bench_rets, p):
        st.markdown("### ⚠️ Scenario Library (Portfolio-ready)")
        assets = list(rets.keys())
        ret_df = pd.concat([rets[a].rename(a) for a in assets], axis=1, join="inner").dropna()
        if ret_df.empty or len(ret_df) < 220:
            st.warning("Not enough overlap for scenario simulation.")
            return
        mode = st.radio("Weights", ["Equal", "Custom"], horizontal=True, key="sc_mode")
        if mode == "Equal":
            w = np.ones(len(assets)) / len(assets)
        else:
            cols = st.columns(min(4, len(assets)))
            w_list = []
            for i, a in enumerate(assets):
                with cols[i % len(cols)]:
                    w_list.append(st.slider(f"{a}", 0.0, 1.0, 1.0/len(assets), 0.01, key=f"scw_{a}"))
            w = np.array(w_list, dtype=float)
            w = (w / w.sum()) if w.sum() > 1e-12 else np.ones(len(assets)) / len(assets)

        scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), index=1, key="sc_scenario")
        base = SCENARIOS[scenario]
        c1, c2, c3 = st.columns(3)
        with c1: horizon = st.slider("Horizon", 5, 90, 20, 5)
        with c2: sims = st.slider("Simulations", 500, 12000, 3000, 500)
        with c3: method = st.selectbox("Sampling", ["bootstrap", "normal"], index=0)

        spx = bench_rets.get("^GSPC")
        dxy = bench_rets.get("DX-Y.NYB")
        betas_spx = np.array([self.an.static_beta(rets[a], spx) for a in assets], dtype=float)
        betas_dxy = np.array([self.an.static_beta(rets[a], dxy) for a in assets], dtype=float)

        k1, k2, k3 = st.columns(3)
        with k1: spx_shock = st.slider("SPX shock day-1", -0.20, 0.20, float(base["spx"]), 0.01)
        with k2: dxy_shock = st.slider("DXY shock day-1", -0.10, 0.10, float(base["dxy"]), 0.005)
        with k3: vol_scale = st.slider("Vol scale (×)", 0.5, 2.5, float(base["vol"]), 0.05)

        if st.button("Run scenario", use_container_width=True):
            paths = simulate_paths(ret_df.values, w, betas_spx, betas_dxy, horizon, sims, method, spx_shock, dxy_shock, vol_scale, base["profile"])
            st.session_state["paths"] = paths

        paths = st.session_state.get("paths", None)
        if paths is None or not isinstance(paths, np.ndarray) or paths.size == 0:
            st.info("Run a scenario to see results.")
            return
        st.plotly_chart(self.viz.scenario_band(paths, f"Portfolio Scenario — {scenario}"), use_container_width=True)

        conf = float(st.selectbox("Tail confidence", list(p["confs"]), index=min(1, len(p["confs"])-1)))
        hret = paths[:, -1] - 1.0
        var = np.quantile(hret, 1-conf)
        es = hret[hret <= var].mean() if (hret <= var).any() else np.nan
        a, b, c = st.columns(3)
        with a: st.metric("Horizon VaR", format_percentage(var*100, 2))
        with b: st.metric("Horizon ES", format_percentage(es*100, 2))
        with c: st.metric("Median", format_percentage(np.median(hret)*100, 2))

    def tab_backtests(self, rets, p):
        st.markdown("### ✅ VaR Backtests (Kupiec + Christoffersen)")
        target = st.selectbox("Target", list(rets.keys()), index=0, key="bt_target")
        method = st.radio("VaR method", ["historical", "normal"], horizontal=True)
        conf = float(st.selectbox("Confidence", list(p["confs"]), index=min(1, len(p["confs"])-1), key="bt_conf"))
        df_bt = self.an.rolling_var_backtest(rets[target], method, int(p["bt_window"]), conf)
        if df_bt.empty:
            st.warning("Not enough data for backtest.")
            return
        hits = df_bt["hit"].values
        alpha = 1 - conf
        lr_pof, p_pof = self.an.kupiec_pof(hits, alpha)
        lr_ind, p_ind = self.an.christoffersen_independence(hits)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Obs", f"{len(df_bt):,}")
        with c2: st.metric("Breaches", f"{hits.sum():,}")
        with c3: st.metric("Breach rate", format_percentage(hits.mean()*100, 2))
        with c4: st.metric("Expected", format_percentage(alpha*100, 2))

        st.dataframe(pd.DataFrame([{"Test": "Kupiec POF", "LR": lr_pof, "p": p_pof}, {"Test": "Christoffersen IND", "LR": lr_ind, "p": p_ind}]).style.format({"LR": "{:.3f}", "p": "{:.4f}"}), use_container_width=True)
        st.plotly_chart(self.viz.backtest_plot(df_bt, f"{target} — {method.title()} VaR ({int(conf*100)}%)"), use_container_width=True)

def main():
    Dashboard().run()

if __name__ == "__main__":
    main()
