"""
🏛️ Institutional Commodities Analytics Platform v3.0
Advanced GARCH, Regime Detection, Portfolio Analytics & Stress Testing
Streamlit Cloud Optimized
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
from arch import arch_model
import warnings
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Institutional configuration management"""
    # Data settings
    MIN_DATA_DAYS = 50
    MAX_GAP_DAYS = 5
    OUTLIER_THRESHOLD = 5.0  # Standard deviations
    
    # Model settings
    ROLLING_WINDOW = 252  # 1 year
    HMM_N_STATES = 3
    CONFIDENCE_LEVEL = 0.95
    
    # Risk settings
    VAR_LEVELS = [0.95, 0.99]
    STRESS_SCENARIOS = {
        "USD_Spike": {"DXY": 0.05, "vol_scale": 1.5},
        "Risk_Off": {"SPY": -0.08, "vol_scale": 2.0},
        "Commodity_Crash": {"CL=F": -0.15, "GC=F": -0.08},
        "Vol_Shock": {"vol_scale": 3.0},
        "Liquidity_Crisis": {"vol_scale": 2.5, "correlation_scale": 1.3}
    }
    
    # Performance
    CACHE_TTL = 3600  # 1 hour
    MAX_ASSETS = 20

# ============================================================================
# Advanced Data Manager
# ============================================================================

class InstitutionalDataManager:
    """Robust data management with quality checks"""
    
    def __init__(self):
        self.config = Config()
        self.align_index = None
    
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def bulk_fetch_with_quality_check(_self, symbols: List[str], 
                                      start_date: datetime, 
                                      end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Bulk fetch with comprehensive quality checks"""
        data_dict = {}
        
        with st.spinner(f"Fetching {len(symbols)} assets..."):
            for symbol in symbols:
                try:
                    df = _self._fetch_single_asset(symbol, start_date, end_date)
                    if df is not None and _self._passes_quality_check(df, symbol):
                        data_dict[symbol] = df
                except Exception as e:
                    st.warning(f"Failed {symbol}: {str(e)[:80]}")
        
        # Align indices
        if data_dict:
            data_dict = _self._align_data_indices(data_dict)
        
        return data_dict
    
    def _fetch_single_asset(self, symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch single asset with extended features"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
            
            if df.empty or len(df) < self.config.MIN_DATA_DAYS:
                return None
            
            # Basic calculations
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Extended features
            df = self._add_technical_features(df)
            df = self._add_volatility_features(df)
            df = self._add_momentum_features(df)
            
            # Quality flags
            df['Data_Quality'] = 1.0
            df.loc[df['Volume'] == 0, 'Data_Quality'] = 0.5
            
            return df
            
        except Exception as e:
            return None
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical features"""
        # Moving averages
        periods = [5, 20, 50, 200]
        for p in periods:
            df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()
            df[f'EMA_{p}'] = df['Close'].ewm(span=p).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        returns = df['Returns'].dropna()
        
        # Historical volatility
        windows = [5, 20, 60]
        for w in windows:
            df[f'Vol_{w}D'] = returns.rolling(w).std() * np.sqrt(252)
        
        # Parkinson volatility (high-low based)
        df['HL_Ratio'] = np.log(df['High'] / df['Low'])
        df['Parkinson_Vol'] = df['HL_Ratio'].rolling(20).std() * np.sqrt(252 / (4 * np.log(2)))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        # Price momentum
        for p in [5, 20, 60]:
            df[f'Momentum_{p}D'] = df['Close'].pct_change(p)
        
        # Rate of Change
        df['ROC_10D'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        return df
    
    def _passes_quality_check(self, df: pd.DataFrame, symbol: str) -> bool:
        """Comprehensive data quality check"""
        checks = []
        
        # Length check
        checks.append(len(df) >= self.config.MIN_DATA_DAYS)
        
        # Missing data check
        missing_pct = df['Close'].isnull().sum() / len(df)
        checks.append(missing_pct < 0.05)
        
        # Gap check
        date_diff = df.index.to_series().diff().dt.days
        max_gap = date_diff.max()
        checks.append(max_gap <= self.config.MAX_GAP_DAYS)
        
        # Outlier check
        returns = df['Returns'].dropna()
        z_scores = np.abs(stats.zscore(returns.fillna(0)))
        outlier_pct = (z_scores > self.config.OUTLIER_THRESHOLD).mean()
        checks.append(outlier_pct < 0.01)
        
        # Volume check
        zero_volume_pct = (df['Volume'] == 0).mean()
        checks.append(zero_volume_pct < 0.1)
        
        return all(checks)
    
    def _align_data_indices(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all data to common index"""
        if not data_dict:
            return data_dict
        
        # Get common index (inner join)
        common_idx = None
        for df in data_dict.values():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        
        # Filter to common dates
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned_dict[symbol] = df.loc[common_idx]
        
        return aligned_dict
    
    def calculate_returns_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create aligned returns matrix"""
        returns_dict = {}
        for symbol, df in data_dict.items():
            if 'Returns' in df.columns:
                returns_dict[symbol] = df['Returns']
        
        if not returns_dict:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df

# ============================================================================
# Advanced GARCH Modeling Engine
# ============================================================================

class InstitutionalGARCHModeler:
    """Advanced GARCH modeling with diagnostics"""
    
    def __init__(self):
        self.config = Config()
    
    def perform_comprehensive_garch_analysis(self, returns: pd.Series, 
                                           garch_type: str = 'GARCH',
                                           p: int = 1, q: int = 1,
                                           distribution: str = 't') -> Dict[str, Any]:
        """Complete GARCH analysis with diagnostics"""
        results = {}
        
        # 1. Data preparation
        returns_clean = returns.dropna()
        if len(returns_clean) < 200:
            return {"error": "Insufficient data for GARCH"}
        
        # Scale returns
        returns_scaled = returns_clean * 100
        
        # 2. ARCH effects test
        arch_test = self._test_arch_effects(returns_scaled)
        results['arch_test'] = arch_test
        
        if not arch_test['arch_present']:
            results['warning'] = "No significant ARCH effects detected"
        
        # 3. Fit GARCH model
        try:
            model = arch_model(
                returns_scaled,
                mean='Constant',
                vol=garch_type,
                p=p,
                q=q,
                dist=distribution
            )
            
            garch_fit = model.fit(disp='off', show_warning=False, 
                                 options={'maxiter': 1000, 'ftol': 1e-10})
            
            # 4. Model results
            results['model'] = garch_fit
            results['params'] = dict(garch_fit.params)
            results['converged'] = garch_fit.convergence_flag == 0
            
            # 5. Conditional volatility
            results['cond_volatility'] = garch_fit.conditional_volatility / 100
            
            # 6. Standardized residuals
            std_resid = garch_fit.resid / garch_fit.conditional_volatility
            results['std_residuals'] = std_resid
            
            # 7. Comprehensive diagnostics
            results['diagnostics'] = self._calculate_garch_diagnostics(garch_fit, std_resid)
            
            # 8. Stationarity check
            alpha = results['params'].get('alpha[1]', 0)
            beta = results['params'].get('beta[1]', 0)
            results['stationary'] = alpha + beta < 1
            
            # 9. Information criteria
            results['aic'] = garch_fit.aic
            results['bic'] = garch_fit.bic
            results['log_likelihood'] = garch_fit.loglikelihood
            
            # 10. Forecast
            forecast, lower, upper = self._forecast_volatility(garch_fit, steps=30)
            results['forecast'] = forecast
            results['forecast_lower'] = lower
            results['forecast_upper'] = upper
            
            # 11. Model quality score
            results['model_score'] = self._calculate_model_score(results)
            
        except Exception as e:
            results['error'] = f"GARCH fitting failed: {str(e)}"
        
        return results
    
    def _test_arch_effects(self, returns: pd.Series, lags: int = 10) -> Dict:
        """Test for ARCH effects"""
        try:
            LM, LM_p, F, F_p = het_arch(returns, maxlag=lags)
            
            return {
                "arch_present": LM_p < 0.05,
                "lm_statistic": LM,
                "lm_p_value": LM_p,
                "f_statistic": F,
                "f_p_value": F_p
            }
        except:
            return {"arch_present": False, "lm_p_value": 1.0}
    
    def _calculate_garch_diagnostics(self, model, std_resid: pd.Series) -> Dict:
        """Calculate comprehensive model diagnostics"""
        diagnostics = {}
        
        # Ljung-Box tests
        try:
            lb_resid = acorr_ljungbox(std_resid, lags=[5, 10, 20], return_df=True)
            diagnostics['lb_resid_pvals'] = lb_resid['lb_pvalue'].tolist()
        except:
            diagnostics['lb_resid_pvals'] = [1.0, 1.0, 1.0]
        
        try:
            lb_squared = acorr_ljungbox(std_resid**2, lags=[5, 10, 20], return_df=True)
            diagnostics['lb_squared_pvals'] = lb_squared['lb_pvalue'].tolist()
        except:
            diagnostics['lb_squared_pvals'] = [1.0, 1.0, 1.0]
        
        # Normality tests
        try:
            jb_stat, jb_pval = stats.jarque_bera(std_resid.dropna())
            diagnostics['jarque_bera'] = {"stat": jb_stat, "pval": jb_pval}
        except:
            diagnostics['jarque_bera'] = {"stat": 0, "pval": 1.0}
        
        # ARCH-LM test on residuals
        try:
            LM, LM_p, F, F_p = het_arch(std_resid, maxlag=10)
            diagnostics['arch_lm_pval'] = LM_p
        except:
            diagnostics['arch_lm_pval'] = 1.0
        
        # Parameter significance
        try:
            t_stats = model.tvalues
            p_values = model.pvalues
            diagnostics['param_significance'] = {
                param: {"t": t_stats[param], "p": p_values[param]}
                for param in model.params.index
            }
        except:
            diagnostics['param_significance'] = {}
        
        return diagnostics
    
    def _forecast_volatility(self, model, steps: int = 30):
        """Forecast volatility with confidence intervals"""
        try:
            # Point forecast
            forecast = model.forecast(horizon=steps)
            point_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100
            
            # Calculate confidence intervals using asymptotic distribution
            alpha = model.params.get('alpha[1]', 0.1)
            beta = model.params.get('beta[1]', 0.8)
            
            # Simplified confidence intervals
            if steps == 1:
                se = point_forecast * 0.1
            else:
                # For multi-step, variance increases
                se = point_forecast * 0.15 * np.sqrt(np.arange(1, steps + 1))
            
            lower = point_forecast - 1.96 * se
            upper = point_forecast + 1.96 * se
            
            return point_forecast, lower, upper
            
        except:
            return np.array([]), np.array([]), np.array([])
    
    def _calculate_model_score(self, results: Dict) -> float:
        """Calculate overall model quality score"""
        score = 0.0
        weights = {
            'converged': 0.2,
            'stationary': 0.2,
            'lb_squared': 0.2,
            'aic': 0.2,
            'param_signif': 0.2
        }
        
        # Convergence
        if results.get('converged', False):
            score += weights['converged']
        
        # Stationarity
        if results.get('stationary', False):
            score += weights['stationary']
        
        # Ljung-Box squared residuals (want high p-value)
        diag = results.get('diagnostics', {})
        lb_squared_pvals = diag.get('lb_squared_pvals', [1.0, 1.0, 1.0])
        avg_lb_pval = np.mean(lb_squared_pvals)
        score += weights['lb_squared'] * avg_lb_pval
        
        # AIC (normalized)
        aic = results.get('aic', 0)
        if aic != 0:
            # Lower AIC is better
            score += weights['aic'] * max(0, 1 - abs(aic) / 10000)
        
        # Parameter significance
        param_sig = diag.get('param_significance', {})
        if param_sig:
            sig_params = sum(1 for p in param_sig.values() if p.get('p', 1) < 0.05)
            score += weights['param_signif'] * (sig_params / len(param_sig))
        
        return min(score * 100, 100)
    
    def grid_search_garch(self, returns: pd.Series, 
                         p_range: List[int] = [1, 2], 
                         q_range: List[int] = [1, 2],
                         garch_types: List[str] = ['GARCH', 'EGARCH', 'GJR'],
                         distributions: List[str] = ['normal', 't']) -> pd.DataFrame:
        """Grid search for optimal GARCH specification"""
        results = []
        
        with st.spinner("Running GARCH grid search..."):
            for garch_type in garch_types:
                for p in p_range:
                    for q in q_range:
                        for dist in distributions:
                            try:
                                model_result = self.perform_comprehensive_garch_analysis(
                                    returns, garch_type, p, q, dist
                                )
                                
                                if model_result.get('converged', False):
                                    results.append({
                                        'garch_type': garch_type,
                                        'p': p,
                                        'q': q,
                                        'distribution': dist,
                                        'aic': model_result.get('aic', 0),
                                        'bic': model_result.get('bic', 0),
                                        'log_likelihood': model_result.get('log_likelihood', 0),
                                        'model_score': model_result.get('model_score', 0),
                                        'stationary': model_result.get('stationary', False),
                                        'alpha': model_result.get('params', {}).get('alpha[1]', 0),
                                        'beta': model_result.get('params', {}).get('beta[1]', 0),
                                        'omega': model_result.get('params', {}).get('omega', 0),
                                        'converged': True
                                    })
                            except:
                                continue
        
        if results:
            df_results = pd.DataFrame(results)
            # Rank by model score (higher is better)
            df_results = df_results.sort_values('model_score', ascending=False)
            return df_results
        
        return pd.DataFrame()

# ============================================================================
# Regime Detection Engine
# ============================================================================

class RegimeDetector:
    """Advanced regime detection with HMM"""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
    
    def detect_regimes(self, returns: pd.Series, volatility: pd.Series) -> Dict:
        """Detect market regimes using HMM"""
        try:
            # Prepare features
            features = pd.DataFrame({
                'returns': returns.values,
                'volatility': volatility.values,
                'abs_returns': np.abs(returns.values)
            }).dropna()
            
            if len(features) < 100:
                return {"error": "Insufficient data for regime detection"}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(features)
            
            # Fit HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            self.model.fit(X_scaled)
            
            # Predict states
            states = self.model.predict(X_scaled)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(returns, volatility, states)
            
            # Transition matrix
            transmat = self.model.transmat_
            
            # Regime persistence
            persistence = self._calculate_regime_persistence(states)
            
            # Regime labels
            regime_labels = self._label_regimes(regime_stats)
            
            return {
                "states": states,
                "regime_stats": regime_stats,
                "regime_labels": regime_labels,
                "transition_matrix": transmat,
                "persistence": persistence,
                "model": self.model,
                "features": features
            }
            
        except Exception as e:
            return {"error": f"Regime detection failed: {str(e)}"}
    
    def _calculate_regime_statistics(self, returns: pd.Series, 
                                   volatility: pd.Series, 
                                   states: np.ndarray) -> pd.DataFrame:
        """Calculate statistics for each regime"""
        stats_list = []
        
        for state in range(self.n_states):
            mask = states == state
            regime_returns = returns.iloc[mask]
            regime_vol = volatility.iloc[mask]
            
            if len(regime_returns) > 10:
                stats_list.append({
                    'regime': state,
                    'count': len(regime_returns),
                    'pct_total': len(regime_returns) / len(states) * 100,
                    'mean_return': regime_returns.mean() * 100,
                    'std_return': regime_returns.std() * 100,
                    'mean_vol': regime_vol.mean() * 100,
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'var_95': np.percentile(regime_returns, 5) * 100,
                    'cvar_95': regime_returns[regime_returns <= np.percentile(regime_returns, 5)].mean() * 100
                })
        
        return pd.DataFrame(stats_list)
    
    def _calculate_regime_persistence(self, states: np.ndarray) -> Dict:
        """Calculate regime persistence metrics"""
        persistence = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            state_changes = np.diff(state_mask.astype(int))
            
            # Count state entries
            state_entries = np.sum(state_changes == 1) + (state_mask[0] == True)
            
            if state_entries > 0:
                avg_duration = len(states[state_mask]) / state_entries
            else:
                avg_duration = 0
            
            persistence[state] = {
                'avg_duration': avg_duration,
                'occurrences': state_entries,
                'total_days': len(states[state_mask])
            }
        
        return persistence
    
    def _label_regimes(self, regime_stats: pd.DataFrame) -> Dict[int, str]:
        """Label regimes based on characteristics"""
        if regime_stats.empty:
            return {}
        
        labels = {}
        
        for _, row in regime_stats.iterrows():
            regime = row['regime']
            
            if row['mean_return'] > 0.05 and row['mean_vol'] < 15:
                labels[regime] = "Bull Low Vol"
            elif row['mean_return'] > 0.05 and row['mean_vol'] >= 15:
                labels[regime] = "Bull High Vol"
            elif row['mean_return'] < -0.05 and row['mean_vol'] > 20:
                labels[regime] = "Bear Crisis"
            elif row['mean_return'] < 0 and row['mean_vol'] > 15:
                labels[regime] = "Bear High Vol"
            elif abs(row['mean_return']) < 0.03 and row['mean_vol'] < 10:
                labels[regime] = "Sideways Low Vol"
            else:
                labels[regime] = "Transition"
        
        return labels

# ============================================================================
# Portfolio Analytics Engine
# ============================================================================

class PortfolioAnalytics:
    """Comprehensive portfolio analytics"""
    
    def __init__(self):
        self.config = Config()
    
    def analyze_portfolio(self, returns_df: pd.DataFrame, 
                         weights: Dict[str, float] = None) -> Dict:
        """Analyze portfolio with given weights"""
        if returns_df.empty or len(returns_df.columns) < 2:
            return {}
        
        # Default to equal weights if not provided
        if weights is None:
            n_assets = len(returns_df.columns)
            weights = {asset: 1/n_assets for asset in returns_df.columns}
        
        # Ensure all weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Portfolio returns
        weighted_returns = pd.DataFrame()
        for asset, weight in weights.items():
            if asset in returns_df.columns:
                weighted_returns[asset] = returns_df[asset] * weight
        
        portfolio_returns = weighted_returns.sum(axis=1)
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(portfolio_returns)
        metrics['weights'] = weights
        
        # Risk contributions
        risk_contrib = self._calculate_risk_contributions(returns_df, weights)
        metrics['risk_contributions'] = risk_contrib
        
        # Drawdown analysis
        drawdown_stats = self._analyze_drawdowns(portfolio_returns)
        metrics['drawdown_analysis'] = drawdown_stats
        
        # Correlation analysis
        corr_matrix = returns_df.corr()
        metrics['correlation_matrix'] = corr_matrix
        
        # Beta to benchmarks (if available)
        beta_analysis = self._calculate_betas(returns_df, portfolio_returns)
        metrics['beta_analysis'] = beta_analysis
        
        return metrics
    
    def _calculate_portfolio_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = ((1 + portfolio_returns).prod() - 1) * 100
        metrics['annual_return'] = ((1 + portfolio_returns.mean()) ** 252 - 1) * 100
        metrics['annual_volatility'] = portfolio_returns.std() * np.sqrt(252) * 100
        metrics['sharpe_ratio'] = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        metrics['sortino_ratio'] = (portfolio_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # VaR and CVaR
        for level in self.config.VAR_LEVELS:
            var_level = int(level * 100)
            var = np.percentile(portfolio_returns, 100 - var_level) * 100
            cvar = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 100 - var_level)].mean() * 100
            
            metrics[f'var_{var_level}'] = var
            metrics[f'cvar_{var_level}'] = cvar
        
        # Higher moments
        metrics['skewness'] = portfolio_returns.skew()
        metrics['kurtosis'] = portfolio_returns.kurtosis()
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean() * 100
        metrics['win_rate'] = win_rate
        
        # Gain/Loss ratio
        gain = portfolio_returns[portfolio_returns > 0].mean() * 100
        loss = portfolio_returns[portfolio_returns < 0].mean() * 100
        metrics['gain_loss_ratio'] = abs(gain / loss) if loss != 0 else float('inf')
        
        return metrics
    
    def _calculate_risk_contributions(self, returns_df: pd.DataFrame, 
                                     weights: Dict[str, float]) -> Dict:
        """Calculate risk contributions using marginal VaR"""
        if returns_df.empty:
            return {}
        
        # Convert weights to array
        weight_vec = np.array([weights.get(asset, 0) for asset in returns_df.columns])
        
        # Covariance matrix
        cov_matrix = returns_df.cov().values * 252
        
        # Portfolio variance
        portfolio_variance = weight_vec.T @ cov_matrix @ weight_vec
        
        if portfolio_variance <= 0:
            return {}
        
        # Marginal contributions to risk
        marginal_risk = (cov_matrix @ weight_vec) / np.sqrt(portfolio_variance)
        
        # Risk contributions
        risk_contrib = {}
        for i, asset in enumerate(returns_df.columns):
            rc = weight_vec[i] * marginal_risk[i]
            risk_contrib[asset] = {
                'weight': weight_vec[i] * 100,
                'marginal_risk': marginal_risk[i] * 100,
                'risk_contribution': rc * 100,
                'risk_percent': (rc / np.sqrt(portfolio_variance)) * 100 if np.sqrt(portfolio_variance) > 0 else 0
            }
        
        return risk_contrib
    
    def _analyze_drawdowns(self, returns: pd.Series) -> Dict:
        """Analyze portfolio drawdowns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min() * 100
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None
        
        # Drawdown statistics
        dd_periods = drawdown[drawdown < 0]
        avg_dd = dd_periods.mean() * 100 if len(dd_periods) > 0 else 0
        std_dd = dd_periods.std() * 100 if len(dd_periods) > 1 else 0
        
        # Recovery analysis
        recovery_stats = self._analyze_recoveries(drawdown)
        
        return {
            'max_drawdown': max_dd,
            'max_dd_date': max_dd_date,
            'avg_drawdown': avg_dd,
            'std_drawdown': std_dd,
            'drawdown_count': len(dd_periods),
            'recovery_stats': recovery_stats,
            'ulcer_index': np.sqrt((dd_periods ** 2).mean()) * 100 if len(dd_periods) > 0 else 0
        }
    
    def _analyze_recoveries(self, drawdown: pd.Series) -> Dict:
        """Analyze drawdown recoveries"""
        # Find drawdown periods
        in_drawdown = False
        drawdown_start = None
        recoveries = []
        
        for date, dd in drawdown.items():
            if dd < -0.05 and not in_drawdown:  # Start of drawdown (>5%)
                in_drawdown = True
                drawdown_start = date
                drawdown_depth = dd
            elif dd >= -0.01 and in_drawdown:  # Recovery (<1% from peak)
                in_drawdown = False
                if drawdown_start:
                    duration = (date - drawdown_start).days
                    recoveries.append({
                        'start': drawdown_start,
                        'end': date,
                        'duration_days': duration,
                        'depth': drawdown_depth * 100
                    })
        
        if recoveries:
            avg_recovery = np.mean([r['duration_days'] for r in recoveries])
            max_recovery = max([r['duration_days'] for r in recoveries])
        else:
            avg_recovery = 0
            max_recovery = 0
        
        return {
            'avg_recovery_days': avg_recovery,
            'max_recovery_days': max_recovery,
            'recovery_count': len(recoveries)
        }
    
    def _calculate_betas(self, asset_returns: pd.DataFrame, 
                        portfolio_returns: pd.Series) -> Dict:
        """Calculate betas to common benchmarks"""
        betas = {}
        
        # Common benchmark tickers
        benchmarks = ['SPY', 'DXY', 'TLT', 'GLD']
        
        for benchmark in benchmarks:
            if benchmark in asset_returns.columns:
                # Use existing data
                benchmark_returns = asset_returns[benchmark]
            else:
                # Could fetch if needed, but for now skip
                continue
            
            # Calculate rolling beta (60-day window)
            rolling_beta = []
            for i in range(60, len(portfolio_returns)):
                window_port = portfolio_returns.iloc[i-60:i]
                window_bench = benchmark_returns.iloc[i-60:i]
                
                if len(window_port.dropna()) > 30 and len(window_bench.dropna()) > 30:
                    cov = np.cov(window_port.dropna(), window_bench.dropna())[0, 1]
                    var = np.var(window_bench.dropna())
                    beta = cov / var if var > 0 else 0
                    rolling_beta.append(beta)
                else:
                    rolling_beta.append(np.nan)
            
            if rolling_beta:
                betas[benchmark] = {
                    'current_beta': rolling_beta[-1] if not np.isnan(rolling_beta[-1]) else 0,
                    'avg_beta': np.nanmean(rolling_beta),
                    'beta_std': np.nanstd(rolling_beta),
                    'rolling_beta': rolling_beta
                }
        
        return betas

# ============================================================================
# Stress Testing Engine
# ============================================================================

class StressTester:
    """Institutional stress testing engine"""
    
    def __init__(self):
        self.config = Config()
    
    def run_stress_tests(self, returns_df: pd.DataFrame, 
                        weights: Dict[str, float],
                        scenario: str = "USD_Spike") -> Dict:
        """Run stress test scenario"""
        if returns_df.empty:
            return {}
        
        # Get scenario parameters
        scenario_params = self.config.STRESS_SCENARIOS.get(scenario, {})
        
        # Apply stress scenario
        stressed_returns = self._apply_stress_scenario(returns_df, scenario_params)
        
        # Calculate portfolio impact
        if stressed_returns is not None:
            portfolio_analytics = PortfolioAnalytics()
            stressed_metrics = portfolio_analytics.analyze_portfolio(stressed_returns, weights)
            
            # Calculate stress impact
            baseline_metrics = portfolio_analytics.analyze_portfolio(returns_df, weights)
            
            impact = {}
            for key in ['var_95', 'cvar_95', 'max_drawdown', 'annual_volatility']:
                if key in stressed_metrics and key in baseline_metrics:
                    impact[f'{key}_change'] = stressed_metrics[key] - baseline_metrics[key]
            
            return {
                'scenario': scenario,
                'parameters': scenario_params,
                'stressed_metrics': stressed_metrics,
                'baseline_metrics': baseline_metrics,
                'impact': impact,
                'stressed_returns': stressed_returns
            }
        
        return {}
    
    def _apply_stress_scenario(self, returns_df: pd.DataFrame, 
                              params: Dict) -> Optional[pd.DataFrame]:
        """Apply stress scenario to returns"""
        try:
            stressed_returns = returns_df.copy()
            
            # Apply volatility shock
            if 'vol_scale' in params:
                vol_scale = params['vol_scale']
                stressed_returns = stressed_returns * vol_scale
            
            # Apply specific asset shocks
            for asset, shock in params.items():
                if asset in stressed_returns.columns:
                    if isinstance(shock, (int, float)):
                        # Add shock to returns
                        stressed_returns[asset] = stressed_returns[asset] + shock
            
            # Apply correlation increase
            if 'correlation_scale' in params:
                corr_scale = params['correlation_scale']
                # Simple implementation: increase off-diagonal correlations
                corr_matrix = stressed_returns.corr()
                n_assets = len(corr_matrix)
                
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        corr_matrix.iloc[i, j] = min(0.9, corr_matrix.iloc[i, j] * corr_scale)
                        corr_matrix.iloc[j, i] = corr_matrix.iloc[i, j]
                
                # Cholesky decomposition to generate correlated returns
                # (simplified for demonstration)
                pass
            
            return stressed_returns
            
        except Exception as e:
            st.warning(f"Stress test failed: {str(e)}")
            return None
    
    def calculate_reverse_stress(self, returns_df: pd.DataFrame,
                                weights: Dict[str, float],
                                target_loss: float = -10.0) -> Dict:
        """Calculate shock needed to achieve target loss"""
        if returns_df.empty:
            return {}
        
        # Simple implementation: find volatility shock needed
        portfolio_analytics = PortfolioAnalytics()
        baseline_metrics = portfolio_analytics.analyze_portfolio(returns_df, weights)
        
        current_var = baseline_metrics.get('var_95', 0)
        target_var = target_loss  # Negative value
        
        if current_var >= target_var:
            # Already at or below target
            return {"shock_required": 0, "status": "Already vulnerable"}
        
        # Calculate required volatility scaling
        # Simplified: var scales roughly with volatility
        vol_required = (target_var / current_var) if current_var != 0 else 2.0
        
        return {
            "target_loss_pct": target_loss,
            "current_var_95": current_var,
            "volatility_shock_required": max(vol_required, 1.0),
            "interpretation": f"Volatility needs to increase {vol_required:.1f}x to achieve {target_loss}% VaR"
        }

# ============================================================================
# Main Application
# ============================================================================

class InstitutionalCommoditiesPlatform:
    """Main institutional platform"""
    
    def __init__(self):
        self.data_manager = InstitutionalDataManager()
        self.garch_modeler = InstitutionalGARCHModeler()
        self.regime_detector = RegimeDetector()
        self.portfolio_analytics = PortfolioAnalytics()
        self.stress_tester = StressTester()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_loaded': False,
            'asset_data': {},
            'returns_matrix': pd.DataFrame(),
            'selected_assets': [],
            'portfolio_weights': {},
            'garch_results': {},
            'regime_results': {},
            'portfolio_results': {},
            'stress_test_results': {},
            'champion_garch': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application"""
        # Display header
        self._display_header()
        
        # Setup sidebar
        sidebar_config = self._setup_sidebar()
        
        if sidebar_config:
            start_date, end_date, selected_assets = sidebar_config
            
            # Main content area
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dashboard", 
                "⚡ GARCH Analysis", 
                "🔄 Regime Detection",
                "📈 Portfolio Analytics",
                "💥 Stress Testing",
                "📋 Reports"
            ])
            
            with tab1:
                self._display_dashboard(selected_assets, start_date, end_date)
            
            with tab2:
                self._display_garch_analysis(selected_assets)
            
            with tab3:
                self._display_regime_detection(selected_assets)
            
            with tab4:
                self._display_portfolio_analytics()
            
            with tab5:
                self._display_stress_testing()
            
            with tab6:
                self._display_reports()
    
    def _display_header(self):
        """Display professional header"""
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        '>
            <h1 style='margin:0; font-size:2.5rem;'>Institutional Commodities Analytics v3.0</h1>
            <p style='margin:0; padding-top:0.5rem; font-size:1.2rem; opacity:0.9;'>
                Advanced GARCH • Portfolio Analytics • Stress Testing • Regime Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _setup_sidebar(self):
        """Setup sidebar with configuration options"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start Date", start_date, key="config_start")
            with col2:
                end = st.date_input("End Date", end_date, key="config_end")
            
            if start >= end:
                st.error("Start date must be before end date")
                return None
            
            st.markdown("---")
            st.markdown("### 📊 Asset Selection")
            
            # Asset selection
            selected_assets = []
            for category, assets in COMMODITIES.items():
                with st.expander(f"{category}", expanded=True):
                    for symbol, name in assets.items():
                        if st.checkbox(f"{name}", key=f"asset_{symbol}"):
                            selected_assets.append(symbol)
            
            # Add benchmark selection
            st.markdown("### 📈 Benchmarks")
            benchmarks = list(BENCHMARKS.keys())
            for benchmark in benchmarks:
                if st.checkbox(f"{BENCHMARKS[benchmark]}", key=f"bench_{benchmark}"):
                    selected_assets.append(benchmark)
            
            st.markdown("---")
            
            # Load data button
            if st.button("📥 Load Market Data", type="primary", use_container_width=True):
                with st.spinner("Loading data..."):
                    self._load_data(selected_assets, start, end)
            
            # Clear cache button
            if st.session_state.data_loaded:
                if st.button("🔄 Clear Cache", type="secondary", use_container_width=True):
                    st.cache_data.clear()
                    for key in ['data_loaded', 'asset_data', 'returns_matrix']:
                        if key in st.session_state:
                            st.session_state[key] = None if key == 'returns_matrix' else False
                    st.rerun()
            
            return start, end, selected_assets
    
    def _load_data(self, symbols: List[str], start_date, end_date):
        """Load data for selected symbols"""
        if not symbols:
            st.warning("Please select at least one asset")
            return
        
        # Fetch data
        asset_data = self.data_manager.bulk_fetch_with_quality_check(
            symbols, start_date, end_date
        )
        
        if not asset_data:
            st.error("Failed to load data. Please try again.")
            return
        
        # Store in session state
        st.session_state.asset_data = asset_data
        st.session_state.selected_assets = list(asset_data.keys())
        
        # Create returns matrix
        returns_matrix = self.data_manager.calculate_returns_matrix(asset_data)
        st.session_state.returns_matrix = returns_matrix
        
        # Initialize equal weights
        if returns_matrix is not None and not returns_matrix.empty:
            n_assets = len(returns_matrix.columns)
            equal_weight = 1.0 / n_assets
            st.session_state.portfolio_weights = {
                asset: equal_weight for asset in returns_matrix.columns
            }
        
        st.session_state.data_loaded = True
        st.success(f"✓ Loaded {len(asset_data)} assets")
    
    def _display_dashboard(self, selected_assets, start_date, end_date):
        """Display main dashboard"""
        st.header("📊 Market Dashboard")
        
        if not st.session_state.data_loaded:
            st.info("Please load data from the sidebar to begin analysis")
            return
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets Loaded", len(st.session_state.asset_data))
        with col2:
            st.metric("Data Points", len(st.session_state.returns_matrix))
        with col3:
            avg_corr = st.session_state.returns_matrix.corr().mean().mean()
            st.metric("Avg Correlation", f"{avg_corr:.2%}")
        with col4:
            total_days = (end_date - start_date).days
            st.metric("Time Period", f"{total_days} days")
        
        # Asset performance table
        st.subheader("📈 Asset Performance Summary")
        
        performance_data = []
        for symbol, df in st.session_state.asset_data.items():
            if len(df) > 0:
                returns = df['Returns'].dropna()
                if len(returns) > 0:
                    perf = {
                        'Asset': symbol,
                        'Current Price': df['Close'].iloc[-1],
                        '1D Return': df['Returns'].iloc[-1] * 100,
                        'Annual Return': ((1 + returns.mean()) ** 252 - 1) * 100,
                        'Annual Vol': returns.std() * np.sqrt(252) * 100,
                        'Sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                        'Max DD': self._calculate_max_dd(returns) * 100
                    }
                    performance_data.append(perf)
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df.style.format({
                'Current Price': '{:.2f}',
                '1D Return': '{:.2f}%',
                'Annual Return': '{:.2f}%',
                'Annual Vol': '{:.2f}%',
                'Sharpe': '{:.2f}',
                'Max DD': '{:.2f}%'
            }), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("📊 Correlation Matrix")
        if not st.session_state.returns_matrix.empty:
            corr_matrix = st.session_state.returns_matrix.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}'
            ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_garch_analysis(self, selected_assets):
        """Display GARCH analysis interface"""
        st.header("⚡ Advanced GARCH Analysis")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first")
            return
        
        # Asset selection for GARCH
        selected_asset = st.selectbox(
            "Select Asset for GARCH Analysis",
            options=st.session_state.selected_assets,
            key="garch_asset_select"
        )
        
        if not selected_asset or selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        returns = df['Returns'].dropna()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            garch_type = st.selectbox("GARCH Type", ["GARCH", "EGARCH", "GJR"], key="garch_type")
        with col2:
            p_value = st.slider("ARCH Order (p)", 1, 3, 1, key="garch_p")
        with col3:
            q_value = st.slider("GARCH Order (q)", 1, 3, 1, key="garch_q")
        
        distribution = st.selectbox("Distribution", ["normal", "t", "skewt"], key="garch_dist")
        
        # Analysis buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Run GARCH Analysis", type="primary", use_container_width=True):
                with st.spinner("Running GARCH analysis..."):
                    garch_results = self.garch_modeler.perform_comprehensive_garch_analysis(
                        returns, garch_type, p_value, q_value, distribution
                    )
                    st.session_state.garch_results[selected_asset] = garch_results
        
        with col2:
            if st.button("🔍 Grid Search", type="secondary", use_container_width=True):
                with st.spinner("Running grid search..."):
                    grid_results = self.garch_modeler.grid_search_garch(
                        returns, 
                        p_range=[1, 2], 
                        q_range=[1, 2],
                        garch_types=['GARCH', 'EGARCH', 'GJR'],
                        distributions=['normal', 't']
                    )
                    
                    if not grid_results.empty:
                        st.subheader("Grid Search Results")
                        st.dataframe(grid_results.style.format({
                            'aic': '{:.1f}',
                            'bic': '{:.1f}',
                            'model_score': '{:.1f}',
                            'alpha': '{:.4f}',
                            'beta': '{:.4f}'
                        }), use_container_width=True)
                        
                        # Select champion model
                        if not grid_results.empty:
                            champion = grid_results.iloc[0]
                            st.session_state.champion_garch = {
                                'asset': selected_asset,
                                'type': champion['garch_type'],
                                'p': champion['p'],
                                'q': champion['q'],
                                'distribution': champion['distribution'],
                                'score': champion['model_score']
                            }
                            st.success(f"✓ Champion model selected: {champion['garch_type']}({champion['p']},{champion['q']})")
        
        # Display GARCH results if available
        if selected_asset in st.session_state.garch_results:
            garch_results = st.session_state.garch_results[selected_asset]
            
            if 'error' in garch_results:
                st.error(garch_results['error'])
            else:
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Score", f"{garch_results.get('model_score', 0):.1f}/100")
                with col2:
                    st.metric("Converged", "✓" if garch_results.get('converged') else "✗")
                with col3:
                    st.metric("Stationary", "✓" if garch_results.get('stationary') else "✗")
                
                # Parameters
                st.subheader("Model Parameters")
                params = garch_results.get('params', {})
                params_df = pd.DataFrame.from_dict(params, orient='index', columns=['Value'])
                st.dataframe(params_df.style.format({'Value': '{:.6f}'}))
                
                # Diagnostics
                st.subheader("Model Diagnostics")
                diag = garch_results.get('diagnostics', {})
                
                diag_cols = st.columns(2)
                with diag_cols[0]:
                    st.write("**Ljung-Box Tests (p-values)**")
                    if 'lb_squared_pvals' in diag:
                        for i, pval in enumerate(diag['lb_squared_pvals']):
                            st.write(f"Lag {[5,10,20][i]}: {pval:.4f}")
                
                with diag_cols[1]:
                    st.write("**Normality Tests**")
                    if 'jarque_bera' in diag:
                        jb = diag['jarque_bera']
                        st.write(f"JB Stat: {jb['stat']:.2f}")
                        st.write(f"p-value: {jb['pval']:.4f}")
                
                # Volatility forecast
                if 'forecast' in garch_results:
                    st.subheader("Volatility Forecast")
                    forecast = garch_results['forecast']
                    
                    if len(forecast) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=forecast * 100,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='#1a2980', width=2)
                        ))
                        
                        if 'forecast_lower' in garch_results:
                            fig.add_trace(go.Scatter(
                                y=garch_results['forecast_lower'] * 100,
                                mode='lines',
                                name='Lower 95%',
                                line=dict(color='gray', dash='dash')
                            ))
                        
                        if 'forecast_upper' in garch_results:
                            fig.add_trace(go.Scatter(
                                y=garch_results['forecast_upper'] * 100,
                                mode='lines',
                                name='Upper 95%',
                                line=dict(color='gray', dash='dash')
                            ))
                        
                        fig.update_layout(
                            title="30-Day Volatility Forecast",
                            yaxis_title="Volatility (%)",
                            xaxis_title="Days Ahead",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _display_regime_detection(self, selected_assets):
        """Display regime detection interface"""
        st.header("🔄 Market Regime Detection")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first")
            return
        
        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset for Regime Analysis",
            options=st.session_state.selected_assets,
            key="regime_asset_select"
        )
        
        if not selected_asset or selected_asset not in st.session_state.asset_data:
            return
        
        df = st.session_state.asset_data[selected_asset]
        returns = df['Returns'].dropna()
        
        # Calculate volatility (simplified)
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        if st.button("🔍 Detect Regimes", type="primary"):
            with st.spinner("Detecting market regimes..."):
                regime_results = self.regime_detector.detect_regimes(returns, volatility)
                st.session_state.regime_results[selected_asset] = regime_results
        
        # Display results if available
        if selected_asset in st.session_state.regime_results:
            regime_results = st.session_state.regime_results[selected_asset]
            
            if 'error' in regime_results:
                st.error(regime_results['error'])
            else:
                # Regime statistics
                st.subheader("Regime Statistics")
                regime_stats = regime_results.get('regime_stats', pd.DataFrame())
                
                if not regime_stats.empty:
                    st.dataframe(regime_stats.style.format({
                        'mean_return': '{:.2f}%',
                        'std_return': '{:.2f}%',
                        'mean_vol': '{:.2f}%',
                        'sharpe': '{:.2f}',
                        'var_95': '{:.2f}%',
                        'cvar_95': '{:.2f}%'
                    }), use_container_width=True)
                
                # Regime labels
                st.subheader("Regime Labels")
                regime_labels = regime_results.get('regime_labels', {})
                for regime, label in regime_labels.items():
                    st.write(f"**Regime {regime}**: {label}")
                
                # Regime plot
                st.subheader("Regime Visualization")
                states = regime_results.get('states', [])
                
                if len(states) > 0:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=("Price with Regimes", "Regime State"))
                    
                    # Price
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[-len(states):],
                            y=df['Close'].iloc[-len(states):],
                            name='Price',
                            line=dict(color='gray', width=1)
                        ),
                        row=1, col=1
                    )
                    
                    # Color by regime
                    for regime in np.unique(states):
                        mask = states == regime
                        regime_dates = df.index[-len(states):][mask]
                        regime_prices = df['Close'].iloc[-len(states):][mask]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=regime_dates,
                                y=regime_prices,
                                mode='markers',
                                name=f'Regime {regime}',
                                marker=dict(size=4)
                            ),
                            row=1, col=1
                        )
                    
                    # Regime states
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[-len(states):],
                            y=states,
                            name='Regime',
                            line=dict(color='#1a2980', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_portfolio_analytics(self):
        """Display portfolio analytics interface"""
        st.header("📈 Portfolio Analytics")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first")
            return
        
        # Portfolio weights configuration
        st.subheader("Portfolio Configuration")
        
        returns_df = st.session_state.returns_matrix
        if returns_df.empty:
            st.warning("No returns data available")
            return
        
        assets = returns_df.columns.tolist()
        
        # Weight input
        st.write("**Adjust Portfolio Weights**")
        weight_cols = st.columns(min(4, len(assets)))
        
        updated_weights = {}
        for i, asset in enumerate(assets):
            with weight_cols[i % 4]:
                current_weight = st.session_state.portfolio_weights.get(asset, 0)
                new_weight = st.number_input(
                    f"{asset}",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_weight,
                    step=0.01,
                    key=f"weight_{asset}"
                )
                updated_weights[asset] = new_weight
        
        # Normalize weights
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in updated_weights.items()}
            st.session_state.portfolio_weights = normalized_weights
        
        st.write(f"**Total Weight**: {sum(updated_weights.values()):.1%}")
        
        if st.button("📊 Analyze Portfolio", type="primary"):
            with st.spinner("Analyzing portfolio..."):
                portfolio_results = self.portfolio_analytics.analyze_portfolio(
                    returns_df, 
                    st.session_state.portfolio_weights
                )
                st.session_state.portfolio_results = portfolio_results
        
        # Display portfolio results
        if st.session_state.portfolio_results:
            results = st.session_state.portfolio_results
            
            # Key metrics
            st.subheader("Portfolio Performance")
            
            metric_cols = st.columns(4)
            metrics_to_show = [
                ('Annual Return', 'annual_return', '{:.2f}%'),
                ('Annual Vol', 'annual_volatility', '{:.2f}%'),
                ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
                ('Max Drawdown', 'drawdown_analysis.max_drawdown', '{:.2f}%')
            ]
            
            for col, (label, key, fmt) in zip(metric_cols, metrics_to_show):
                with col:
                    # Handle nested keys
                    if '.' in key:
                        parts = key.split('.')
                        value = results.get(parts[0], {}).get(parts[1], 0)
                    else:
                        value = results.get(key, 0)
                    
                    st.metric(label, fmt.format(value))
            
            # Risk metrics
            st.subheader("Risk Metrics")
            risk_cols = st.columns(4)
            with risk_cols[0]:
                st.metric("VaR 95%", f"{results.get('var_95', 0):.2f}%")
            with risk_cols[1]:
                st.metric("CVaR 95%", f"{results.get('cvar_95', 0):.2f}%")
            with risk_cols[2]:
                st.metric("Sortino", f"{results.get('sortino_ratio', 0):.2f}")
            with risk_cols[3]:
                st.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%")
            
            # Risk contributions
            st.subheader("Risk Contributions")
            risk_contrib = results.get('risk_contributions', {})
            
            if risk_contrib:
                contrib_data = []
                for asset, contrib in risk_contrib.items():
                    contrib_data.append({
                        'Asset': asset,
                        'Weight': f"{contrib['weight']:.1f}%",
                        'Risk Contribution': f"{contrib['risk_contribution']:.2f}%",
                        '% of Total Risk': f"{contrib['risk_percent']:.1f}%"
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                st.dataframe(contrib_df, use_container_width=True)
                
                # Visualize risk contributions
                fig = go.Figure(data=[go.Pie(
                    labels=list(risk_contrib.keys()),
                    values=[c['risk_contribution'] for c in risk_contrib.values()],
                    hole=0.4
                )])
                fig.update_layout(title="Risk Contribution Breakdown")
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_stress_testing(self):
        """Display stress testing interface"""
        st.header("💥 Stress Testing")
        
        if not st.session_state.data_loaded:
            st.info("Please load data first")
            return
        
        if not st.session_state.portfolio_results:
            st.warning("Please run portfolio analysis first")
            return
        
        returns_df = st.session_state.returns_matrix
        weights = st.session_state.portfolio_weights
        
        # Stress test scenarios
        st.subheader("Scenario Selection")
        
        scenario = st.selectbox(
            "Select Stress Scenario",
            options=list(Config.STRESS_SCENARIOS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
            key="stress_scenario"
        )
        
        if st.button("🚨 Run Stress Test", type="primary"):
            with st.spinner(f"Running {scenario} scenario..."):
                stress_results = self.stress_tester.run_stress_tests(
                    returns_df, weights, scenario
                )
                st.session_state.stress_test_results = stress_results
        
        # Display stress test results
        if st.session_state.stress_test_results:
            results = st.session_state.stress_test_results
            
            st.subheader(f"Stress Test Results: {results['scenario']}")
            
            # Impact metrics
            st.write("**Scenario Impact**")
            impact = results.get('impact', {})
            
            impact_cols = st.columns(len(impact))
            for col, (metric, change) in zip(impact_cols, impact.items()):
                with col:
                    metric_name = metric.replace('_change', '').replace('_', ' ').title()
                    st.metric(metric_name, f"{change:.2f}%")
            
            # Comparison table
            st.subheader("Baseline vs Stressed")
            
            comparison_data = []
            baseline = results.get('baseline_metrics', {})
            stressed = results.get('stressed_metrics', {})
            
            metrics_to_compare = ['annual_return', 'annual_volatility', 
                                'var_95', 'cvar_95', 'max_drawdown']
            
            for metric in metrics_to_compare:
                if metric in baseline and metric in stressed:
                    comparison_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Baseline': baseline[metric],
                        'Stressed': stressed[metric],
                        'Change': stressed[metric] - baseline[metric]
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df.style.format({
                    'Baseline': '{:.2f}%',
                    'Stressed': '{:.2f}%',
                    'Change': '{:.2f}%'
                }), use_container_width=True)
            
            # Reverse stress test
            st.subheader("Reverse Stress Test")
            
            target_loss = st.number_input(
                "Target Portfolio Loss (%)",
                min_value=-50.0,
                max_value=0.0,
                value=-10.0,
                step=1.0
            )
            
            if st.button("🔁 Calculate Required Shock"):
                reverse_stress = self.stress_tester.calculate_reverse_stress(
                    returns_df, weights, target_loss
                )
                
                if reverse_stress:
                    st.write(f"**Target Loss**: {target_loss}%")
                    st.write(f"**Current VaR 95%**: {reverse_stress['current_var_95']:.2f}%")
                    st.write(f"**Required Volatility Shock**: {reverse_stress['volatility_shock_required']:.1f}x")
                    st.info(reverse_stress['interpretation'])
    
    def _display_reports(self):
        """Display reporting interface"""
        st.header("📋 Institutional Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Snapshot Report")
            if st.button("📄 Generate Snapshot", use_container_width=True):
                self._generate_snapshot_report()
        
        with col2:
            st.subheader("Portfolio Tear Sheet")
            if st.button("📊 Generate Tear Sheet", use_container_width=True):
                self._generate_tear_sheet()
    
    def _generate_snapshot_report(self):
        """Generate snapshot report"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'assets_loaded': list(st.session_state.asset_data.keys()),
            'data_points': len(st.session_state.returns_matrix),
            'portfolio_weights': st.session_state.portfolio_weights,
            'champion_garch': st.session_state.get('champion_garch'),
            'portfolio_metrics': st.session_state.get('portfolio_results', {}),
            'stress_test_results': st.session_state.get('stress_test_results', {})
        }
        
        # Convert to JSON for display
        snapshot_json = json.dumps(snapshot, indent=2, default=str)
        
        st.subheader("Run Snapshot")
        st.code(snapshot_json, language='json')
        
        # Download button
        st.download_button(
            label="📥 Download Snapshot",
            data=snapshot_json,
            file_name=f"commodities_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _generate_tear_sheet(self):
        """Generate portfolio tear sheet"""
        if not st.session_state.portfolio_results:
            st.warning("No portfolio results available")
            return
        
        results = st.session_state.portfolio_results
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Portfolio Tear Sheet</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #1a2980; color: white; padding: 20px; border-radius: 10px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #26d0ce; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Portfolio Tear Sheet</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Overview</h2>
                <div class="metric-card">
                    <p><strong>Assets:</strong> {len(st.session_state.portfolio_weights)}</p>
                    <p><strong>Total Return:</strong> {results.get('total_return', 0):.2f}%</p>
                    <p><strong>Annual Volatility:</strong> {results.get('annual_volatility', 0):.2f}%</p>
                    <p><strong>Sharpe Ratio:</strong> {results.get('sharpe_ratio', 0):.2f}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <div class="metric-card">
                    <p><strong>VaR 95%:</strong> {results.get('var_95', 0):.2f}%</p>
                    <p><strong>CVaR 95%:</strong> {results.get('cvar_95', 0):.2f}%</p>
                    <p><strong>Max Drawdown:</strong> {results.get('drawdown_analysis', {}).get('max_drawdown', 0):.2f}%</p>
                    <p><strong>Win Rate:</strong> {results.get('win_rate', 0):.1f}%</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p><em>This report is generated for informational purposes only. 
                Past performance is not indicative of future results. 
                Consult with a financial advisor before making investment decisions.</em></p>
            </div>
        </body>
        </html>
        """
        
        st.subheader("Portfolio Tear Sheet Preview")
        st.components.v1.html(html_content, height=600)
        
        # Download button
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=f"portfolio_tear_sheet_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )
    
    def _calculate_max_dd(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() if not drawdown.empty else 0

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Institutional Commodities Analytics v3.0",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide Streamlit branding
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create and run app
    app = InstitutionalCommoditiesPlatform()
    app.run()
