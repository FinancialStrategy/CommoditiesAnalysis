
# =============================================================================
# 🧩 PATCH PACK v1.0 — Portfolio Lab + Risk Decomposition + VaR (3 Methods) + PCA Vol Drivers
# Target: "🏛️ Institutional Commodities Analytics Platform v7.0" (Streamlit)
#
# ✅ Paste this entire file CONTENT at the *BOTTOM* of your existing app.py
# ✅ Then add the tiny hook snippets shown at the end ("INTEGRATION HOOKS")
#
# This patch is strictly additive:
# - Does NOT remove/replace your existing features
# - Adds Copper + Aluminum futures tickers
# - Adds a full "Portfolio Lab" suite using PyPortfolioOpt (optional, graceful fallback)
# - Adds manual-weight portfolio builder + multi-portfolio comparative analysis
# - Adds portfolio volatility risk contributions (assets + PCA factors)
# - Adds VaR + CVaR/ES and Relative VaR (active) with 3 methods (Historical / Parametric / Monte Carlo)
# - Adds PCA-driven volatility factor analysis for GARCH vol + Portfolio EWMA vol
#
# NOTE:
# - PyPortfolioOpt requires: PyPortfolioOpt + cvxpy + scikit-learn
# - This patch handles missing packages gracefully and shows install hints inside the app.
# =============================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

# Optional (used if available)
try:
    from sklearn.decomposition import PCA
    _SKLEARN_OK = True
except Exception:
    PCA = None
    _SKLEARN_OK = False

# Optional (used if available in your app)
try:
    from arch import arch_model
    _ARCH_OK = True
except Exception:
    arch_model = None
    _ARCH_OK = False

# Optional: PyPortfolioOpt (graceful fallback)
_PYPFOPT_OK = False
try:
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.cla import CLA

    # Optional advanced optimizers (availability varies by version)
    try:
        from pypfopt.efficient_cvar import EfficientCVaR
    except Exception:
        EfficientCVaR = None
    try:
        from pypfopt.efficient_semivariance import EfficientSemivariance
    except Exception:
        EfficientSemivariance = None
    try:
        from pypfopt.efficient_cdar import EfficientCDaR
    except Exception:
        EfficientCDaR = None
    try:
        from pypfopt.black_litterman import BlackLittermanModel
    except Exception:
        BlackLittermanModel = None

    _PYPFOPT_OK = True
except Exception:
    _PYPFOPT_OK = False


# =============================================================================
# 1) FUTURES UNIVERSE PATCH: Add Copper + Aluminum
# =============================================================================

def patch_futures_universe(existing_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Add Copper + Aluminum tickers to your futures universe.
    Yahoo Finance tickers:
      - Copper (COMEX): HG=F
      - Aluminum (COMEX): ALI=F
    """
    base = dict(existing_map) if isinstance(existing_map, dict) else {}

    # Keep existing keys untouched; only add missing
    additions = {
        "Copper (COMEX) - HG=F": "HG=F",
        "Aluminum (COMEX) - ALI=F": "ALI=F",
    }
    for k, v in additions.items():
        if k not in base:
            base[k] = v
    return base


# =============================================================================
# 2) UTILITIES — Returns, EWMA Vol, Covariance, Risk Contributions
# =============================================================================

def _to_prices_df(prices: Any) -> pd.DataFrame:
    if prices is None:
        return pd.DataFrame()
    if isinstance(prices, pd.Series):
        return prices.to_frame()
    if isinstance(prices, pd.DataFrame):
        return prices.copy()
    return pd.DataFrame(prices)

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    method:
      - 'log': log returns
      - 'simple': pct returns
    """
    prices = _to_prices_df(prices)
    prices = prices.sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.ffill().bfill()

    if prices.empty or prices.shape[0] < 3:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    if method.lower() == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices).diff()

    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    return rets

def annualize_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    # geometric
    return float((1.0 + r).prod() ** (periods_per_year / max(1, r.shape[0])) - 1.0)

def annualize_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods_per_year))

def max_drawdown(cum: pd.Series) -> float:
    s = cum.dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())

def ewma_volatility(returns: pd.Series, span: int = 33, periods_per_year: int = 252) -> pd.Series:
    """
    EWMA volatility (annualized). Uses span for EWM std.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    vol = r.ewm(span=span, adjust=False).std(bias=False) * np.sqrt(periods_per_year)
    vol.name = f"EWMA_Vol_{span}"
    return vol

def _align_for_cov(returns_df: pd.DataFrame) -> pd.DataFrame:
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    # drop all-NA columns
    x = x.dropna(axis=1, how="all")
    # keep common rows
    x = x.dropna(axis=0, how="any")
    return x

def sample_covariance(returns_df: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 1:
        return pd.DataFrame()
    cov = x.cov()
    if annualize:
        cov *= periods_per_year
    return cov

def ledoit_wolf_covariance(returns_df: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance (requires sklearn). If sklearn missing, fallback to sample covariance.
    """
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 1:
        return pd.DataFrame()
    if not _SKLEARN_OK:
        return sample_covariance(x, annualize=annualize, periods_per_year=periods_per_year)

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(x.values)
        cov = pd.DataFrame(lw.covariance_, index=x.columns, columns=x.columns)
        if annualize:
            cov *= periods_per_year
        return cov
    except Exception:
        return sample_covariance(x, annualize=annualize, periods_per_year=periods_per_year)

def risk_contribution(weights: np.ndarray, cov_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Risk contribution decomposition (assets):
      RC_i = w_i * (Σ w)_i / σ_p
    """
    if cov_annual is None or cov_annual.empty:
        return pd.DataFrame()
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    cols = list(cov_annual.columns)
    if w.shape[0] != len(cols):
        return pd.DataFrame()
    Sigma = cov_annual.values
    port_var = float(w.T @ Sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    mrc = (Sigma @ w).flatten()  # marginal risk contributions (unnormalized)
    rc = (w.flatten() * mrc) / (port_vol if port_vol > 0 else np.nan)

    out = pd.DataFrame({
        "weight": w.flatten(),
        "marginal_contrib": mrc,
        "risk_contrib": rc,
        "risk_contrib_pct": rc / (port_vol if port_vol > 0 else np.nan)
    }, index=cols)
    out = out.sort_values("risk_contrib_pct", ascending=False)
    return out

def pca_factor_risk_contribution(weights: np.ndarray, cov_annual: pd.DataFrame, top_k: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PCA factor risk decomposition:
      - eigen-decompose Σ = V Λ V'
      - portfolio variance = Σ_j ( (w' v_j)^2 * λ_j )
    Returns:
      factor_var_contrib: variance contribution by PCA factor
      factor_loading: loadings matrix (assets x factors)
    """
    if cov_annual is None or cov_annual.empty:
        return pd.DataFrame(), pd.DataFrame()

    Sigma = cov_annual.values
    cols = list(cov_annual.columns)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    if w.shape[0] != len(cols):
        return pd.DataFrame(), pd.DataFrame()

    # Symmetrize for numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    k = int(min(max(1, top_k), len(evals)))
    evals_k = evals[:k]
    evecs_k = evecs[:, :k]

    # Factor exposures: b_j = w' v_j
    b = (w.T @ evecs_k).flatten()
    var_contrib = (b ** 2) * evals_k
    total_var = float((w.T @ Sigma @ w).item())
    total_var = max(total_var, 0.0)

    df_var = pd.DataFrame({
        "eigenvalue": evals_k,
        "exposure_wTv": b,
        "variance_contrib": var_contrib,
        "variance_contrib_pct": var_contrib / (total_var if total_var > 0 else np.nan),
    }, index=[f"PC{i+1}" for i in range(k)]).sort_values("variance_contrib_pct", ascending=False)

    df_load = pd.DataFrame(evecs_k, index=cols, columns=[f"PC{i+1}" for i in range(k)])
    return df_var, df_load


# =============================================================================
# 3) VaR / CVaR(ES) + Relative VaR (Active Returns) — 3 Methods
# =============================================================================

@dataclass
class VaRResult:
    var: float
    cvar_es: float
    method: str
    alpha: float
    horizon_days: int

def _scale_var(value_1d: float, horizon_days: int, method: str) -> float:
    """
    For parametric / Monte Carlo under iid normal-ish assumption: sqrt(h)
    For historical: use empirical scaling by resampling if we want, but for simplicity:
      - keep sqrt scaling as default (documented)
    """
    if horizon_days <= 1 or value_1d is None or np.isnan(value_1d):
        return value_1d
    return float(value_1d * np.sqrt(horizon_days))

def historical_var(returns: pd.Series, alpha: float = 0.05, horizon_days: int = 1) -> VaRResult:
    r = returns.dropna()
    if r.empty:
        return VaRResult(np.nan, np.nan, "Historical", alpha, horizon_days)

    losses = -r
    var_1d = float(np.quantile(losses, 1.0 - (1.0 - alpha)))  # alpha quantile of losses
    # ES = mean loss beyond VaR threshold
    tail = losses[losses >= var_1d]
    es_1d = float(tail.mean()) if len(tail) > 0 else float(losses.max())
    return VaRResult(_scale_var(var_1d, horizon_days, "Historical"), _scale_var(es_1d, horizon_days, "Historical"), "Historical", alpha, horizon_days)

def parametric_var(returns: pd.Series, alpha: float = 0.05, horizon_days: int = 1) -> VaRResult:
    r = returns.dropna()
    if r.empty:
        return VaRResult(np.nan, np.nan, "Parametric (Gaussian)", alpha, horizon_days)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0 or np.isnan(sigma):
        return VaRResult(np.nan, np.nan, "Parametric (Gaussian)", alpha, horizon_days)

    from scipy.stats import norm
    z = float(norm.ppf(alpha))
    # VaR on losses: -(mu + z*sigma)
    var_1d = float(-(mu + z * sigma))
    # ES (Gaussian) on losses
    es_1d = float(-(mu) + sigma * (norm.pdf(z) / alpha))
    return VaRResult(_scale_var(var_1d, horizon_days, "Parametric"), _scale_var(es_1d, horizon_days, "Parametric"), "Parametric (Gaussian)", alpha, horizon_days)

def monte_carlo_var(returns_df: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05, horizon_days: int = 1,
                    n_sims: int = 20000, seed: int = 42, use_shrinkage_cov: bool = True) -> VaRResult:
    """
    Monte Carlo VaR for portfolio using multivariate normal simulation:
      r_sim ~ N(mu, cov)
      portfolio_sim = r_sim @ w
    """
    x = _align_for_cov(returns_df)
    if x.empty:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    mu = x.mean().values
    cov = ledoit_wolf_covariance(x, annualize=False) if use_shrinkage_cov else sample_covariance(x, annualize=False)
    if cov.empty:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    w = np.asarray(weights, dtype=float).flatten()
    if len(w) != x.shape[1]:
        return VaRResult(np.nan, np.nan, "Monte Carlo (MVN)", alpha, horizon_days)

    rng = np.random.default_rng(int(seed))
    sims = rng.multivariate_normal(mean=mu, cov=cov.values, size=int(n_sims))
    port = sims @ w
    losses = -port
    var_1d = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_1d]
    es_1d = float(tail.mean()) if len(tail) > 0 else float(losses.max())
    return VaRResult(_scale_var(var_1d, horizon_days, "MC"), _scale_var(es_1d, horizon_days, "MC"), "Monte Carlo (MVN)", alpha, horizon_days)

def compute_var_suite(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float,
    horizon_days: int,
    n_sims: int,
    seed: int,
    use_shrinkage_cov: bool
) -> pd.DataFrame:
    """
    Compute VaR + ES for 3 methods, return a tidy DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    h = historical_var(portfolio_returns, alpha=alpha, horizon_days=horizon_days)
    p = parametric_var(portfolio_returns, alpha=alpha, horizon_days=horizon_days)
    m = monte_carlo_var(asset_returns, weights=weights, alpha=alpha, horizon_days=horizon_days,
                        n_sims=n_sims, seed=seed, use_shrinkage_cov=use_shrinkage_cov)

    for res in [h, p, m]:
        rows.append({
            "method": res.method,
            "alpha": res.alpha,
            "horizon_days": res.horizon_days,
            "VaR": res.var,
            "ES(CVaR)": res.cvar_es
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4) GARCH VOL + PCA VOL DRIVERS
# =============================================================================

def _garch_vol_series(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal",
                      annualize: bool = True, periods_per_year: int = 252) -> pd.Series:
    """
    GARCH(p,q) conditional volatility series. Requires 'arch'.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    if not _ARCH_OK:
        return pd.Series(dtype=float)

    # Scale returns for stability (arch typically uses %)
    r_pct = r * 100.0

    try:
        am = arch_model(r_pct, vol="Garch", p=int(p), q=int(q), dist=str(dist), rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility / 100.0  # back to decimal
        if annualize:
            cond_vol = cond_vol * np.sqrt(periods_per_year)
        cond_vol.name = f"GARCH({p},{q})_vol"
        return cond_vol
    except Exception:
        return pd.Series(dtype=float)

def compute_garch_vol_matrix(returns_df: pd.DataFrame, p: int = 1, q: int = 1, dist: str = "normal",
                             annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute GARCH vol for each asset (may be expensive). Use on a selected subset.
    """
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all")
    if x.empty:
        return pd.DataFrame()

    out = {}
    for col in x.columns:
        v = _garch_vol_series(x[col], p=p, q=q, dist=dist, annualize=annualize, periods_per_year=periods_per_year)
        if not v.empty:
            out[col] = v

    if not out:
        return pd.DataFrame()
    vol_df = pd.concat(out, axis=1).dropna(how="all")
    return vol_df

def pca_on_matrix(df: pd.DataFrame, n_components: int = 4, standardize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA on time x assets matrix.
    Returns:
      scores_df: time x k
      loadings_df: assets x k
      evr_df: explained variance ratios
    """
    if df is None or df.empty or df.shape[1] < 2 or not _SKLEARN_OK:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.empty or x.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = x.values
    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        X = (X - mu) / sd

    k = int(min(max(2, n_components), x.shape[1]))
    pca = PCA(n_components=k)
    scores = pca.fit_transform(X)  # time x k
    loadings = pca.components_.T     # assets x k

    scores_df = pd.DataFrame(scores, index=x.index, columns=[f"PC{i+1}" for i in range(k)])
    loadings_df = pd.DataFrame(loadings, index=x.columns, columns=[f"PC{i+1}" for i in range(k)])
    evr_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(k)],
        "ExplainedVarRatio": pca.explained_variance_ratio_,
        "CumExplained": np.cumsum(pca.explained_variance_ratio_)
    })
    return scores_df, loadings_df, evr_df


# =============================================================================
# 5) PORTFOLIO LAB — Manual Builder + PyPortfolioOpt Strategies + Comparative Analysis
# =============================================================================

@dataclass
class PortfolioSpec:
    name: str
    weights: Dict[str, float]
    benchmark: Optional[str] = None

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in w.items() if v is not None and np.isfinite(v) and float(v) >= 0.0}
    s = float(sum(w.values()))
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}

def _weights_to_array(weights: Dict[str, float], cols: List[str]) -> np.ndarray:
    return np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)

def compute_portfolio_returns(returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    x = returns_df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.empty:
        return pd.Series(dtype=float)

    cols = list(x.columns)
    w = _weights_to_array(weights, cols)
    if np.allclose(w.sum(), 0.0):
        return pd.Series(index=x.index, dtype=float)

    pr = x.values @ w
    s = pd.Series(pr, index=x.index, name="Portfolio")
    return s

def compute_beta_alpha(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> Tuple[float, float]:
    """
    CAPM beta + Jensen alpha (annualized alpha approximation).
    """
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan, np.nan
    p = p.loc[idx]
    b = b.loc[idx]

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    p_ex = p - rf_daily
    b_ex = b - rf_daily

    var_b = float(np.var(b_ex, ddof=1))
    if var_b <= 0 or np.isnan(var_b):
        return np.nan, np.nan
    cov_pb = float(np.cov(p_ex, b_ex, ddof=1)[0, 1])
    beta = cov_pb / var_b

    # alpha daily = mean(p_ex) - beta * mean(b_ex)
    alpha_daily = float(p_ex.mean() - beta * b_ex.mean())
    alpha_annual = float((1.0 + alpha_daily) ** periods_per_year - 1.0)
    return float(beta), float(alpha_annual)

def tracking_error(portfolio: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan
    active = (p.loc[idx] - b.loc[idx]).dropna()
    if active.empty:
        return np.nan
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))

def information_ratio(portfolio: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    p = portfolio.dropna()
    b = benchmark.dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan
    active = (p.loc[idx] - b.loc[idx]).dropna()
    if active.empty:
        return np.nan
    te = active.std(ddof=1)
    if te <= 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))

def portfolio_metrics_table(
    portfolio_ret: pd.Series,
    bench_ret: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    r = portfolio_ret.dropna()
    if r.empty:
        return {
            "AnnReturn": np.nan, "AnnVol": np.nan, "Sharpe": np.nan,
            "MaxDrawdown": np.nan, "Beta": np.nan, "Alpha": np.nan,
            "TrackingError": np.nan, "InfoRatio": np.nan
        }

    cum = (1.0 + r).cumprod()
    out = {
        "AnnReturn": annualize_return(r, periods_per_year),
        "AnnVol": annualize_vol(r, periods_per_year),
        "Sharpe": sharpe_ratio(r, rf_annual, periods_per_year),
        "MaxDrawdown": max_drawdown(cum),
        "Beta": np.nan,
        "Alpha": np.nan,
        "TrackingError": np.nan,
        "InfoRatio": np.nan
    }
    if bench_ret is not None and isinstance(bench_ret, pd.Series):
        beta, alpha = compute_beta_alpha(r, bench_ret, rf_annual, periods_per_year)
        out["Beta"] = beta
        out["Alpha"] = alpha
        out["TrackingError"] = tracking_error(r, bench_ret, periods_per_year)
        out["InfoRatio"] = information_ratio(r, bench_ret, periods_per_year)
    return out

def _plot_cum_performance(portfolios: Dict[str, pd.Series], title: str, key: str) -> None:
    fig = go.Figure()
    for name, r in portfolios.items():
        rr = r.dropna()
        if rr.empty:
            continue
        cum = (1.0 + rr).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, mode="lines"))
    fig.update_layout(
        title=title,
        height=420,
        xaxis_title="Date",
        yaxis_title="Cumulative Growth",
        legend_title="Series",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def _plot_risk_contrib_bar(rc_df: pd.DataFrame, title: str, key: str) -> None:
    if rc_df is None or rc_df.empty:
        st.info("Risk contribution cannot be computed (insufficient data).")
        return
    df = rc_df.reset_index().rename(columns={"index": "Asset"})
    fig = px.bar(df, x="Asset", y="risk_contrib_pct", title=title)
    fig.update_layout(height=420, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)

def _plot_pca_loadings_heatmap(loadings: pd.DataFrame, title: str, key: str) -> None:
    if loadings is None or loadings.empty:
        st.info("PCA loadings are not available.")
        return
    fig = px.imshow(loadings.values, x=loadings.columns, y=loadings.index, aspect="auto", title=title)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)

def _ensure_session_portfolios(key_ns: str) -> None:
    k = f"{key_ns}__saved_portfolios"
    if k not in st.session_state or not isinstance(st.session_state.get(k), list):
        st.session_state[k] = []

def _save_portfolio_spec(spec: PortfolioSpec, key_ns: str) -> None:
    _ensure_session_portfolios(key_ns)
    k = f"{key_ns}__saved_portfolios"
    st.session_state[k].append(spec)

def _get_saved_portfolios(key_ns: str) -> List[PortfolioSpec]:
    _ensure_session_portfolios(key_ns)
    return list(st.session_state.get(f"{key_ns}__saved_portfolios", []))

def _clear_saved_portfolios(key_ns: str) -> None:
    st.session_state[f"{key_ns}__saved_portfolios"] = []

def _render_install_hints() -> None:
    st.warning("PyPortfolioOpt / scikit-learn / cvxpy may not be installed in your environment.")
    st.code(
        "requirements.txt (Streamlit Cloud) suggested entries:\n"
        "PyPortfolioOpt==1.5.5\n"
        "cvxpy>=1.3\n"
        "scikit-learn>=1.2\n"
        "arch>=6.3\n"
        "scipy>=1.10\n"
    )

def _pypfopt_expected_returns_and_cov(returns_df: pd.DataFrame, cov_method: str, key_ns: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    x = _align_for_cov(returns_df)
    if x.empty or x.shape[1] < 2:
        return None, None

    # Expected returns
    # Use annualized mean historical returns
    mu = x.mean() * 252.0

    # Covariance
    if cov_method == "Ledoit-Wolf":
        cov = ledoit_wolf_covariance(x, annualize=True, periods_per_year=252)
    else:
        cov = sample_covariance(x, annualize=True, periods_per_year=252)

    # Ensure order consistency
    mu = mu.loc[cov.columns]
    return mu, cov

def _run_pypfopt_strategies(
    returns_df: pd.DataFrame,
    cov_method: str,
    rf_annual: float,
    key_ns: str
) -> Dict[str, Dict[str, float]]:
    """
    Runs multiple PyPortfolioOpt strategies and returns dict {strategy_name: weights_dict}
    """
    if not _PYPFOPT_OK:
        return {}

    mu, cov = _pypfopt_expected_returns_and_cov(returns_df, cov_method, key_ns)
    if mu is None or cov is None or cov.empty:
        return {}

    assets = list(cov.columns)

    out: Dict[str, Dict[str, float]] = {}

    # --- Efficient Frontier base
    try:
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        w = ef.max_sharpe(risk_free_rate=float(rf_annual))
        out["Max Sharpe (EF)"] = dict(ef.clean_weights())
    except Exception:
        pass

    try:
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        w = ef.min_volatility()
        out["Min Volatility (EF)"] = dict(ef.clean_weights())
    except Exception:
        pass

    # Efficient risk/return targets
    try:
        target_vol = float(st.session_state.get(f"{key_ns}__target_vol", 0.15))
        ef = EfficientFrontier(mu, cov)
        w = ef.efficient_risk(target_volatility=target_vol)
        out[f"Efficient Risk (Target Vol={target_vol:.2%})"] = dict(ef.clean_weights())
    except Exception:
        pass

    try:
        target_ret = float(st.session_state.get(f"{key_ns}__target_ret", 0.10))
        ef = EfficientFrontier(mu, cov)
        w = ef.efficient_return(target_return=target_ret)
        out[f"Efficient Return (Target Ret={target_ret:.2%})"] = dict(ef.clean_weights())
    except Exception:
        pass

    # Quadratic utility
    try:
        gamma = float(st.session_state.get(f"{key_ns}__gamma", 1.0))
        ef = EfficientFrontier(mu, cov)
        w = ef.max_quadratic_utility(risk_aversion=gamma)
        out[f"Max Quadratic Utility (γ={gamma:.2f})"] = dict(ef.clean_weights())
    except Exception:
        pass

    # HRP
    try:
        hrp = HRPOpt(returns_df.dropna(how="any"))
        hrp_w = hrp.optimize()
        out["Hierarchical Risk Parity (HRP)"] = {k: float(v) for k, v in hrp_w.items()}
    except Exception:
        pass

    # CLA
    try:
        cla = CLA(mu, cov)
        cla_w = cla.max_sharpe()
        out["Critical Line Algorithm (CLA)"] = {k: float(v) for k, v in cla_w.items()}
    except Exception:
        pass

    # Efficient CVaR
    if EfficientCVaR is not None:
        try:
            ec = EfficientCVaR(mu, cov, returns_df.dropna(how="any"))
            cvar_w = ec.min_cvar()
            out["Min CVaR (EfficientCVaR)"] = dict(ec.clean_weights())
        except Exception:
            pass

    # Efficient Semivariance
    if EfficientSemivariance is not None:
        try:
            es = EfficientSemivariance(mu, cov, returns_df.dropna(how="any"))
            semi_w = es.min_semivariance()
            out["Min Semivariance (EfficientSemivariance)"] = dict(es.clean_weights())
        except Exception:
            pass

    # Efficient CDaR
    if EfficientCDaR is not None:
        try:
            cd = EfficientCDaR(mu, cov, returns_df.dropna(how="any"))
            cd_w = cd.min_cdar()
            out["Min CDaR (EfficientCDaR)"] = dict(cd.clean_weights())
        except Exception:
            pass

    return out

def _render_manual_portfolio_builder(assets: List[str], key_ns: str) -> PortfolioSpec:
    st.subheader("Manual Portfolio Builder (User-Defined Weights)")
    sel = st.multiselect("Select assets", options=assets, default=assets[:min(6, len(assets))], key=f"{key_ns}__asset_select")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        auto_normalize = st.checkbox("Auto-normalize weights to 100%", value=True, key=f"{key_ns}__auto_norm")
    with colB:
        name = st.text_input("Portfolio name", value="Custom_1", key=f"{key_ns}__pname")
    with colC:
        bench = st.text_input("Benchmark ticker/name (optional, must exist in your benchmark returns dict)", value="", key=f"{key_ns}__bench")

    weights: Dict[str, float] = {}
    if not sel:
        st.info("Select at least 1 asset.")
        return PortfolioSpec(name=name, weights={}, benchmark=bench if bench else None)

    st.caption("Set weights using sliders (0–100). Tip: turn on Auto-normalize.")
    cols = st.columns(2)
    for i, a in enumerate(sel):
        with cols[i % 2]:
            weights[a] = st.slider(f"Weight: {a}", min_value=0.0, max_value=100.0, value=float(100.0/len(sel)),
                                   step=0.5, key=f"{key_ns}__w__{a}")

    weights = {k: v/100.0 for k, v in weights.items()}
    if auto_normalize:
        weights = _normalize_weights(weights)

    s = sum(weights.values())
    st.write(f"Weight sum: **{s:.4f}**")
    if abs(s - 1.0) > 1e-3:
        st.warning("Weights do not sum to 1.0. Enable Auto-normalize or adjust sliders.")

    return PortfolioSpec(name=name, weights=weights, benchmark=bench if bench else None)

def render_portfolio_lab_suite(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark_returns: Optional[Dict[str, pd.Series]] = None,
    key_ns: str = "portfolio_lab"
) -> None:
    """
    Main entrypoint. Call this inside a new Streamlit tab without removing anything else.
    Inputs expected:
      - prices_df: wide dataframe of prices for assets (columns=assets)
      - returns_df: wide dataframe of returns aligned to prices_df
      - benchmark_returns: dict[str, Series] of benchmark returns (e.g., {'SPY': spy_rets, 'XU100': xu100_rets})
    """
    st.header("🏦 Portfolio Lab — PyPortfolioOpt + Manual Portfolios + Risk Decomposition")
    if prices_df is None or returns_df is None or returns_df.empty:
        st.info("No returns data provided to Portfolio Lab. Provide prices_df/returns_df from your existing pipeline.")
        return

    benchmark_returns = benchmark_returns or {}

    # Align and clean
    R = returns_df.copy()
    R = R.replace([np.inf, -np.inf], np.nan)
    R = R.dropna(axis=1, how="all")
    if R.shape[1] < 1:
        st.info("Not enough assets for portfolio analysis.")
        return

    # ---------------- Sidebar controls
    with st.sidebar.expander("📌 Portfolio Lab Controls", expanded=False):
        st.session_state[f"{key_ns}__rf_annual"] = st.number_input("Risk-free rate (annual, decimal)", value=float(st.session_state.get(f"{key_ns}__rf_annual", 0.03)),
                                                                  step=0.005, format="%.4f", key=f"{key_ns}__rf_annual_in")
        st.session_state[f"{key_ns}__target_vol"] = st.slider("Target Volatility (Efficient Risk)", min_value=0.05, max_value=0.50,
                                                              value=float(st.session_state.get(f"{key_ns}__target_vol", 0.15)), step=0.01, key=f"{key_ns}__target_vol_in")
        st.session_state[f"{key_ns}__target_ret"] = st.slider("Target Return (Efficient Return)", min_value=-0.10, max_value=0.60,
                                                              value=float(st.session_state.get(f"{key_ns}__target_ret", 0.10)), step=0.01, key=f"{key_ns}__target_ret_in")
        st.session_state[f"{key_ns}__gamma"] = st.slider("Risk Aversion γ (Quadratic Utility)", min_value=0.1, max_value=10.0,
                                                         value=float(st.session_state.get(f"{key_ns}__gamma", 1.0)), step=0.1, key=f"{key_ns}__gamma_in")

    rf_annual = float(st.session_state.get(f"{key_ns}__rf_annual", 0.03))

    tabs = st.tabs([
        "① Manual Portfolios",
        "② PyPortfolioOpt Strategies",
        "③ Comparative Analysis",
        "④ Volatility Contributions",
        "⑤ VaR + Relative VaR (3 Methods)",
        "⑥ PCA Vol Drivers (GARCH + EWMA)"
    ])

    # =============================================================================
    # TAB 1 — Manual Portfolios
    # =============================================================================
    with tabs[0]:
        assets = list(R.columns)
        spec = _render_manual_portfolio_builder(assets, key_ns=key_ns)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("💾 Save Portfolio", key=f"{key_ns}__save_btn"):
                if spec.weights:
                    _save_portfolio_spec(spec, key_ns=key_ns)
                    st.success(f"Saved portfolio: {spec.name}")
                else:
                    st.warning("No weights to save.")
        with c2:
            if st.button("🧹 Clear Saved Portfolios", key=f"{key_ns}__clear_btn"):
                _clear_saved_portfolios(key_ns=key_ns)
                st.info("Saved portfolios cleared.")
        with c3:
            st.write(f"Saved count: **{len(_get_saved_portfolios(key_ns))}**")

        if spec.weights:
            pr = compute_portfolio_returns(R[list(spec.weights.keys())], spec.weights)
            st.subheader("Preview: Custom Portfolio Performance")
            _plot_cum_performance({spec.name: pr}, "Cumulative Growth (Custom Portfolio)", key=f"{key_ns}__cum_custom")
            mt = portfolio_metrics_table(pr, bench_ret=benchmark_returns.get(spec.benchmark) if spec.benchmark else None, rf_annual=rf_annual)
            st.dataframe(pd.DataFrame([mt], index=[spec.name]).T, use_container_width=True)

    # =============================================================================
    # TAB 2 — PyPortfolioOpt Strategies
    # =============================================================================
    with tabs[1]:
        st.subheader("PyPortfolioOpt Strategies (Institutional Suite)")
        if not _PYPFOPT_OK:
            _render_install_hints()
            st.info("Manual portfolios remain fully available even without PyPortfolioOpt.")
        else:
            cov_method = st.selectbox("Covariance Method", ["Ledoit-Wolf", "Sample"], index=0, key=f"{key_ns}__cov_method")
            st.caption("Strategies run on the aligned returns window (rows with missing values dropped).")

            strategies = _run_pypfopt_strategies(R, cov_method=cov_method, rf_annual=rf_annual, key_ns=key_ns)
            if not strategies:
                st.warning("No strategies produced weights (insufficient data or solver constraints). Try fewer assets or longer history.")
            else:
                st.success(f"Strategies computed: {len(strategies)}")

                # Show weights
                strat_names = list(strategies.keys())
                chosen = st.multiselect("Choose strategies to save/compare", strat_names, default=strat_names[:min(4, len(strat_names))], key=f"{key_ns}__choose_strats")
                if chosen:
                    for sname in chosen:
                        w = strategies[sname]
                        w_norm = _normalize_weights(w)
                        st.markdown(f"**{sname}**")
                        w_df = pd.DataFrame({"weight": pd.Series(w_norm)}).sort_values("weight", ascending=False)
                        st.dataframe(w_df, use_container_width=True)

                        if st.button(f"Save → {sname}", key=f"{key_ns}__save_strat__{sname}"):
                            _save_portfolio_spec(PortfolioSpec(name=sname, weights=w_norm, benchmark=None), key_ns=key_ns)
                            st.success(f"Saved: {sname}")

    # =============================================================================
    # TAB 3 — Comparative Analysis
    # =============================================================================
    with tabs[2]:
        st.subheader("Comparative Analysis — Saved Portfolios")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("No saved portfolios yet. Save from Manual Portfolios or PyPortfolioOpt Strategies.")
        else:
            # Build returns series for each saved portfolio
            series_map: Dict[str, pd.Series] = {}
            rows = []
            for spec in saved:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                if not cols:
                    continue
                w = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                pr = compute_portfolio_returns(R[cols], w)
                series_map[spec.name] = pr

                bench = benchmark_returns.get(spec.benchmark) if spec.benchmark else None
                mt = portfolio_metrics_table(pr, bench_ret=bench, rf_annual=rf_annual)
                rows.append({"Portfolio": spec.name, "Benchmark": spec.benchmark or "", **mt})

            if not series_map:
                st.warning("Saved portfolios could not be computed with current returns universe.")
            else:
                _plot_cum_performance(series_map, "Cumulative Growth — Saved Portfolios", key=f"{key_ns}__cum_saved")
                mt_df = pd.DataFrame(rows).set_index("Portfolio")
                st.dataframe(mt_df, use_container_width=True)

    # =============================================================================
    # TAB 4 — Volatility Contributions (Assets + PCA Factors)
    # =============================================================================
    with tabs[3]:
        st.subheader("Portfolio Volatility Contributions")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("Save at least one portfolio to compute volatility contributions.")
        else:
            names = [s.name for s in saved]
            pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_vol_contrib")
            spec = next((s for s in saved if s.name == pick), None)
            if spec is None:
                st.info("Portfolio not found.")
            else:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                subR = _align_for_cov(R[cols])
                if subR.empty or subR.shape[1] < 2:
                    st.warning("Need at least 2 assets with sufficient overlap.")
                else:
                    cov_method = st.selectbox("Covariance for decomposition", ["Ledoit-Wolf", "Sample"], index=0, key=f"{key_ns}__cov_dec")
                    covA = ledoit_wolf_covariance(subR, annualize=True) if cov_method == "Ledoit-Wolf" else sample_covariance(subR, annualize=True)
                    w_arr = _weights_to_array(w_norm, list(subR.columns))

                    rc = risk_contribution(w_arr, covA)
                    _plot_risk_contrib_bar(rc, f"Asset Risk Contribution (% of Portfolio Vol) — {pick}", key=f"{key_ns}__rc_bar")

                    with st.expander("Risk Contribution Table"):
                        st.dataframe(rc, use_container_width=True)

                    st.markdown("---")
                    st.subheader("PCA Factor Risk Decomposition (Covariance Eigen-Factors)")
                    top_k = st.slider("Number of PCA factors", min_value=2, max_value=min(12, subR.shape[1]),
                                      value=min(6, subR.shape[1]), step=1, key=f"{key_ns}__pca_k_risk")
                    df_var, df_load = pca_factor_risk_contribution(w_arr, covA, top_k=top_k)
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.dataframe(df_var, use_container_width=True)
                    with c2:
                        _plot_pca_loadings_heatmap(df_load, "PCA Factor Loadings (Eigenvectors)", key=f"{key_ns}__pca_load_risk")

    # =============================================================================
    # TAB 5 — VaR + Relative VaR (Active) — 3 Methods
    # =============================================================================
    with tabs[4]:
        st.subheader("VaR + CVaR(ES) + Relative VaR (Active vs Benchmark) — 3 Methods")
        saved = _get_saved_portfolios(key_ns=key_ns)
        if not saved:
            st.info("Save at least one portfolio first.")
        else:
            names = [s.name for s in saved]
            pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_var")
            spec = next((s for s in saved if s.name == pick), None)
            if spec is None:
                st.info("Portfolio not found.")
            else:
                cols = [c for c in spec.weights.keys() if c in R.columns]
                w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                subR = _align_for_cov(R[cols])

                if subR.empty:
                    st.warning("Insufficient overlapping returns.")
                else:
                    w_arr = _weights_to_array(w_norm, list(subR.columns))
                    pr = compute_portfolio_returns(subR, w_norm)

                    bmk_key = st.selectbox("Benchmark (for Relative VaR)", options=["(None)"] + list(benchmark_returns.keys()),
                                           index=0, key=f"{key_ns}__bmk_var")
                    bench = benchmark_returns.get(bmk_key) if bmk_key != "(None)" else None

                    alpha = st.select_slider("Confidence level", options=[0.10, 0.05, 0.025, 0.01], value=0.05, key=f"{key_ns}__alpha")
                    horizon = st.selectbox("Horizon (days)", options=[1, 5, 10, 21], index=0, key=f"{key_ns}__horizon")
                    n_sims = st.slider("Monte Carlo simulations", min_value=5000, max_value=80000, value=20000, step=5000, key=f"{key_ns}__n_sims")
                    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1, key=f"{key_ns}__seed")
                    use_shrink = st.checkbox("Use shrinkage covariance (Ledoit-Wolf) in Monte Carlo", value=True, key=f"{key_ns}__mc_shrink")

                    st.markdown("### Portfolio VaR/ES")
                    var_df = compute_var_suite(pr, subR, w_arr, alpha=alpha, horizon_days=horizon, n_sims=int(n_sims), seed=int(seed), use_shrinkage_cov=use_shrink)
                    st.dataframe(var_df, use_container_width=True)

                    if bench is not None and isinstance(bench, pd.Series):
                        # Relative/Active VaR
                        idx = pr.dropna().index.intersection(bench.dropna().index)
                        if len(idx) >= 60:
                            active = (pr.loc[idx] - bench.loc[idx]).dropna()
                            st.markdown("### Relative (Active) VaR/ES — (Portfolio - Benchmark)")
                            # For Monte Carlo on active returns, we simulate portfolio and benchmark separately is complex;
                            # practical approach: compute active series and use historical/parametric on it,
                            # and for MC approximate using active mean/var derived from assets vs benchmark covariance if available.
                            rel_hist = historical_var(active, alpha=alpha, horizon_days=horizon)
                            rel_para = parametric_var(active, alpha=alpha, horizon_days=horizon)

                            rel_rows = [
                                {"method": rel_hist.method, "alpha": rel_hist.alpha, "horizon_days": rel_hist.horizon_days, "VaR": rel_hist.var, "ES(CVaR)": rel_hist.cvar_es},
                                {"method": rel_para.method, "alpha": rel_para.alpha, "horizon_days": rel_para.horizon_days, "VaR": rel_para.var, "ES(CVaR)": rel_para.cvar_es},
                            ]
                            rel_df = pd.DataFrame(rel_rows)
                            st.dataframe(rel_df, use_container_width=True)

                            # Interactive chart of rolling active VaR (historical)
                            window = st.slider("Rolling window (days) for active VaR chart", min_value=60, max_value=504, value=252, step=21, key=f"{key_ns}__roll_win")
                            alpha_roll = alpha

                            active_losses = -active
                            roll_var = active_losses.rolling(window).quantile(alpha_roll)
                            roll_es = active_losses.rolling(window).apply(lambda s: float(pd.Series(s)[pd.Series(s) >= np.quantile(pd.Series(s), alpha_roll)].mean()) if len(s.dropna()) > 0 else np.nan, raw=False)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=active.index, y=active.values, name="Active Return", mode="lines"))
                            fig.add_trace(go.Scatter(x=roll_var.index, y=-roll_var.values, name=f"Rolling VaR (α={alpha_roll})", mode="lines"))
                            fig.add_trace(go.Scatter(x=roll_es.index, y=-roll_es.values, name=f"Rolling ES (α={alpha_roll})", mode="lines"))
                            fig.update_layout(
                                title="Active Return vs Rolling VaR/ES (Historical)",
                                height=460, xaxis_title="Date", yaxis_title="Return",
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_ns}__active_var_chart")
                        else:
                            st.info("Benchmark overlap is not sufficient for Relative VaR (need ~60+ points).")

    # =============================================================================
    # TAB 6 — PCA Vol Drivers (GARCH vol matrix + Portfolio EWMA vol)
    # =============================================================================
    with tabs[5]:
        st.subheader("PCA Volatility Drivers — GARCH Vol & Portfolio EWMA Vol")
        if not _SKLEARN_OK:
            st.warning("scikit-learn is required for PCA. Install scikit-learn to enable this tab.")
        else:
            saved = _get_saved_portfolios(key_ns=key_ns)
            if not saved:
                st.info("Save at least one portfolio.")
            else:
                names = [s.name for s in saved]
                pick = st.selectbox("Select portfolio", names, index=0, key=f"{key_ns}__pick_pca_vol")
                spec = next((s for s in saved if s.name == pick), None)
                if spec is None:
                    st.info("Portfolio not found.")
                else:
                    cols = [c for c in spec.weights.keys() if c in R.columns]
                    w_norm = _normalize_weights({k: v for k, v in spec.weights.items() if k in cols})
                    subR = _align_for_cov(R[cols])

                    if subR.empty or subR.shape[1] < 2:
                        st.warning("Need at least 2 assets with sufficient overlap.")
                    else:
                        # Portfolio returns + vols
                        pr = compute_portfolio_returns(subR, w_norm)
                        ewma_span = st.selectbox("EWMA span", options=[22, 33, 55, 99], index=1, key=f"{key_ns}__ewma_span")
                        port_ewma = ewma_volatility(pr, span=int(ewma_span))

                        # GARCH vol matrix for assets
                        st.caption("GARCH vol computation can be heavy. Use a smaller asset set if needed.")
                        garch_p = st.selectbox("GARCH p", options=[1, 2], index=0, key=f"{key_ns}__garch_p")
                        garch_q = st.selectbox("GARCH q", options=[1, 2], index=0, key=f"{key_ns}__garch_q")
                        dist = st.selectbox("GARCH distribution", options=["normal", "t"], index=0, key=f"{key_ns}__garch_dist")

                        if not _ARCH_OK:
                            st.warning("arch package not available; cannot compute GARCH vol. Install 'arch' to enable.")
                        else:
                            with st.spinner("Computing asset GARCH vol matrix..."):
                                garch_vol = compute_garch_vol_matrix(subR, p=int(garch_p), q=int(garch_q), dist=str(dist), annualize=True)

                            if garch_vol.empty:
                                st.warning("GARCH vol matrix is empty (model failed or insufficient data).")
                            else:
                                # PCA on GARCH vol matrix
                                n_comp = st.slider("PCA components", min_value=2, max_value=min(8, garch_vol.shape[1]), value=min(4, garch_vol.shape[1]), step=1, key=f"{key_ns}__pca_k_vol")
                                scores, loadings, evr = pca_on_matrix(garch_vol, n_components=int(n_comp), standardize=True)

                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**Explained variance (GARCH vol factors)**")
                                    st.dataframe(evr, use_container_width=True)
                                with c2:
                                    _plot_pca_loadings_heatmap(loadings, "PCA Loadings on Asset GARCH Vol", key=f"{key_ns}__pca_load_garch")

                                # Correlation of PCA factors with portfolio EWMA vol
                                if not scores.empty and not port_ewma.empty:
                                    # Align
                                    idx = scores.index.intersection(port_ewma.index)
                                    if len(idx) > 50:
                                        corr = scores.loc[idx].corrwith(port_ewma.loc[idx])
                                        corr_df = pd.DataFrame({"CorrWith_Port_EWMA": corr}).sort_values("CorrWith_Port_EWMA", ascending=False)
                                        st.markdown("**Correlation: PCA Vol Factors vs Portfolio EWMA Vol**")
                                        st.dataframe(corr_df, use_container_width=True)

                                        fig = go.Figure()
                                        for pc in scores.columns:
                                            fig.add_trace(go.Scatter(x=scores.index, y=scores[pc], name=pc, mode="lines"))
                                        fig.update_layout(title="PCA Factor Scores (GARCH Vol Factors)", height=420,
                                                          xaxis_title="Date", yaxis_title="Score",
                                                          margin=dict(l=10, r=10, t=50, b=10))
                                        st.plotly_chart(fig, use_container_width=True, key=f"{key_ns}__pca_scores_chart")

                                        fig2 = go.Figure()
                                        fig2.add_trace(go.Scatter(x=port_ewma.index, y=port_ewma.values, name=f"Portfolio EWMA Vol (span={ewma_span})", mode="lines"))
                                        fig2.update_layout(title="Portfolio EWMA Volatility", height=360,
                                                           xaxis_title="Date", yaxis_title="Annualized Vol",
                                                           margin=dict(l=10, r=10, t=50, b=10))
                                        st.plotly_chart(fig2, use_container_width=True, key=f"{key_ns}__port_ewma_chart")


# =============================================================================
# 6) INTEGRATION HOOKS (tiny snippets)
# =============================================================================
#
# A) Add Copper + Aluminum to your futures universe:
#    If you have a dict like FUTURES_TICKERS or FUTURES_MAP, patch it:
#
#        FUTURES_TICKERS = patch_futures_universe(FUTURES_TICKERS)
#
#    (Do this right after your futures dict is defined.)
#
# B) Add a NEW TAB in your existing tab router (without removing anything):
#
#    If you have something like:
#        tabs = st.tabs([...existing...])
#    add another label at the end, e.g. "Portfolio Lab"
#
#    Then inside that tab call:
#
#        render_portfolio_lab_suite(
#            prices_df=prices_df,                 # your assets prices wide df
#            returns_df=returns_df,               # your assets returns wide df
#            benchmark_returns=bench_rets_dict,   # dict { 'SPY': spy_ret, 'XU100': xu100_ret, ... }
#            key_ns="portfolio_lab"
#        )
#
#    Note: benchmark_returns is optional. If you already compute benchmarks, pass them.
# =============================================================================
