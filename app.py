"""
🏛️ Institutional Commodities Analytics Platform v7.0 (Quantum Enhanced)
Integrated Portfolio Analytics • Advanced GARCH & Regime Detection • 
Hybrid AI Forecasting (LSTM, RNN, XGBoost) • Signal Intelligence • 
Black-Scholes Options • Macro Sensitivity • Professional Reporting
Streamlit Cloud Optimized with Superior Architecture & Performance
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, signal
from scipy.stats import norm

# Optional AI/ML dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Optional dependency (used only for some diagnostic plots)
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None
from io import BytesIO, StringIO
import base64

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Environment optimization
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(
    page_title="Institutional Commodities Platform v7.0 (Quantum)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/institutional-commodities',
        'Report a bug': "https://github.com/institutional-commodities/issues",
        'About': """🏛️ Institutional Commodities Analytics v7.0 Quantum Enhanced
                    Advanced analytics platform for institutional commodity trading
                    © 2024 Institutional Trading Analytics"""
    }
)

# =============================================================================
# ENHANCED DATA STRUCTURES & CONFIGURATION
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
    ai_forecast_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisConfiguration:
    """Comprehensive analysis configuration with AI enhancements"""
    start_date: datetime = field(default_factory=lambda: (datetime.now() - timedelta(days=1095)))
    end_date: datetime = field(default_factory=lambda: datetime.now())
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
    ai_lookback_days: int = 60
    ai_forecast_days: int = 15
    ai_ensemble_weight_lstm: float = 0.5
    ai_ensemble_weight_xgb: float = 0.5
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.start_date >= self.end_date:
            return False
        if not (0 <= self.risk_free_rate <= 1):
            return False
        if not all(0.5 <= cl <= 0.999 for cl in self.confidence_levels):
            return False
        return True

# Enhanced commodities universe with AI capabilities
COMMODITIES_UNIVERSE = {
    AssetCategory.PRECIOUS_METALS.value: {
        "GC=F": AssetMetadata(
            symbol="GC=F",
            name="Gold Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#FFD700",
            description="COMEX Gold Futures (100 troy ounces)",
            exchange="COMEX",
            contract_size="100 troy oz",
            margin_requirement=0.045,
            tick_size=0.10,
            risk_level="Low",
            ai_forecast_enabled=True
        ),
        "SI=F": AssetMetadata(
            symbol="SI=F",
            name="Silver Futures",
            category=AssetCategory.PRECIOUS_METALS,
            color="#C0C0C0",
            description="COMEX Silver Futures (5,000 troy ounces)",
            exchange="COMEX",
            contract_size="5,000 troy oz",
            margin_requirement=0.065,
            tick_size=0.005,
            risk_level="Medium",
            ai_forecast_enabled=True
        ),
    },
    AssetCategory.ENERGY.value: {
        "CL=F": AssetMetadata(
            symbol="CL=F",
            name="Crude Oil WTI",
            category=AssetCategory.ENERGY,
            color="#000000",
            description="NYMEX Light Sweet Crude Oil (1,000 barrels)",
            exchange="NYMEX",
            contract_size="1,000 barrels",
            margin_requirement=0.085,
            tick_size=0.01,
            risk_level="High",
            ai_forecast_enabled=True
        ),
        "NG=F": AssetMetadata(
            symbol="NG=F",
            name="Natural Gas",
            category=AssetCategory.ENERGY,
            color="#4169E1",
            description="NYMEX Natural Gas (10,000 MMBtu)",
            exchange="NYMEX",
            contract_size="10,000 MMBtu",
            margin_requirement=0.095,
            tick_size=0.001,
            risk_level="High",
            ai_forecast_enabled=True
        ),
    }
}

BENCHMARKS = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "type": "equity",
        "color": "#1E90FF",
        "description": "S&P 500 Equity Index"
    },
    "DX-Y.NYB": {
        "name": "US Dollar Index",
        "type": "currency",
        "color": "#32CD32",
        "description": "US Dollar Currency Index"
    },
    "TLT": {
        "name": "20+ Year Treasury ETF",
        "type": "fixed_income",
        "color": "#8A2BE2",
        "description": "Long-term US Treasury Bonds"
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "type": "commodity",
        "color": "#FFD700",
        "description": "Gold-backed ETF"
    },
    "DBC": {
        "name": "Invesco DB Commodity Index",
        "type": "commodity",
        "color": "#FF6347",
        "description": "Broad Commodities ETF"
    }
}

# =============================================================================
# QUANTUM AI ENGINE (NEW)
# =============================================================================

class QuantumAIEngine:
    """Hybrid AI forecasting engine with LSTM, RNN, and XGBoost ensemble"""
    
    def __init__(self, lookback_days: int = 60, forecast_days: int = 15):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if SKLEARN_AVAILABLE else None
        
    def prepare_sequences(self, data: pd.Series):
        """Prepare sequences for AI models"""
        if not SKLEARN_AVAILABLE or self.scaler is None:
            return None, None, None
            
        scaled = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X, y = [], []
        
        for i in range(self.lookback_days, len(scaled)):
            X.append(scaled[i-self.lookback_days:i, 0])
            y.append(scaled[i, 0])
            
        return np.array(X), np.array(y), scaled
    
    def train_lstm(self, X, y):
        """Train LSTM neural network"""
        if not TENSORFLOW_AVAILABLE or X is None or y is None:
            return None
            
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.lookback_days, 1)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return model
    
    def train_xgboost(self, X, y):
        """Train XGBoost model"""
        if not XGB_AVAILABLE or X is None or y is None:
            return None
            
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X, y)
        return model
    
    def run_ensemble_forecast(self, data: pd.Series, 
                            lstm_weight: float = 0.5,
                            xgb_weight: float = 0.5) -> Dict[str, Any]:
        """Run ensemble forecasting with multiple AI models"""
        
        if not data.empty and len(data) > self.lookback_days * 2:
            try:
                X, y, scaled_full = self.prepare_sequences(data)
                
                if X is None or y is None:
                    return {"available": False, "message": "AI dependencies not available"}
                
                # Train models
                lstm_model = self.train_lstm(X, y)
                xgb_model = self.train_xgboost(X, y)
                
                if lstm_model is None and xgb_model is None:
                    return {"available": False, "message": "No AI models trained successfully"}
                
                # Generate forecasts
                forecasts = {}
                
                # LSTM forecast
                if lstm_model is not None:
                    current_window_lstm = scaled_full[-self.lookback_days:].reshape((1, self.lookback_days, 1))
                    lstm_preds = []
                    
                    for _ in range(self.forecast_days):
                        pred = lstm_model.predict(current_window_lstm, verbose=0)[0, 0]
                        lstm_preds.append(pred)
                        current_window_lstm = np.append(
                            current_window_lstm[:, 1:, :], 
                            [[[pred]]], 
                            axis=1
                        )
                    
                    forecasts['lstm'] = self.scaler.inverse_transform(
                        np.array(lstm_preds).reshape(-1, 1)
                    ).flatten()
                
                # XGBoost forecast
                if xgb_model is not None:
                    current_window_xgb = scaled_full[-self.lookback_days:].reshape((1, -1))
                    xgb_preds = []
                    
                    for _ in range(self.forecast_days):
                        pred = xgb_model.predict(current_window_xgb)[0]
                        xgb_preds.append(pred)
                        current_window_xgb = np.append(
                            current_window_xgb[:, 1:], 
                            [[pred]], 
                            axis=1
                        )
                    
                    forecasts['xgb'] = self.scaler.inverse_transform(
                        np.array(xgb_preds).reshape(-1, 1)
                    ).flatten()
                
                # Ensemble forecast
                if 'lstm' in forecasts and 'xgb' in forecasts:
                    ensemble_forecast = (
                        forecasts['lstm'] * lstm_weight + 
                        forecasts['xgb'] * xgb_weight
                    )
                    forecasts['ensemble'] = ensemble_forecast
                
                # Calculate confidence intervals
                confidence_intervals = {}
                if 'ensemble' in forecasts:
                    # Simple confidence interval based on historical error
                    historical_errors = []
                    for i in range(len(X)):
                        if lstm_model and xgb_model:
                            lstm_pred = lstm_model.predict(X[i:i+1].reshape(1, self.lookback_days, 1), verbose=0)[0, 0]
                            xgb_pred = xgb_model.predict(X[i:i+1])[0]
                            pred = lstm_pred * lstm_weight + xgb_pred * xgb_weight
                            actual = y[i]
                            historical_errors.append(abs(pred - actual))
                    
                    if historical_errors:
                        error_std = np.std(historical_errors)
                        scaled_error_std = error_std * (data.max() - data.min())
                        
                        confidence_intervals = {
                            'upper': forecasts['ensemble'] + 1.96 * scaled_error_std,
                            'lower': forecasts['ensemble'] - 1.96 * scaled_error_std,
                            'confidence_level': 0.95
                        }
                
                return {
                    "available": True,
                    "forecasts": forecasts,
                    "confidence_intervals": confidence_intervals,
                    "lookback_days": self.lookback_days,
                    "forecast_days": self.forecast_days,
                    "current_price": data.iloc[-1],
                    "models_used": list(forecasts.keys())
                }
                
            except Exception as e:
                return {"available": False, "message": f"AI forecasting error: {str(e)}"}
        
        return {"available": False, "message": "Insufficient data for AI forecasting"}

# =============================================================================
# SIGNAL INTELLIGENCE ENGINE (NEW)
# =============================================================================

class SignalIntelligenceEngine:
    """Convert AI forecasts into actionable trading signals with risk management"""
    
    @staticmethod
    def generate_signal(
        current_price: float,
        ai_forecast: Dict[str, Any],
        annual_volatility: float,
        confidence_threshold: float = 0.03
    ) -> Dict[str, Any]:
        """Generate trading signal from AI forecast"""
        
        if not ai_forecast.get("available", False):
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "color": "#888888",
                "stop_loss": None,
                "take_profit": None,
                "expected_return": 0.0,
                "risk_reward_ratio": 0.0
            }
        
        ensemble_forecast = ai_forecast.get("forecasts", {}).get("ensemble")
        if ensemble_forecast is None or len(ensemble_forecast) == 0:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "color": "#888888",
                "stop_loss": None,
                "take_profit": None,
                "expected_return": 0.0,
                "risk_reward_ratio": 0.0
            }
        
        # Calculate expected return
        forecast_price = ensemble_forecast[-1]
        expected_return = (forecast_price - current_price) / current_price
        
        # Calculate daily volatility
        daily_vol = annual_volatility / math.sqrt(252)
        
        # Signal logic
        if expected_return > confidence_threshold:
            # Bullish signal
            stop_loss = current_price * (1 - daily_vol * 2.0)  # 2 sigma stop
            take_profit = current_price * (1 + daily_vol * 3.0)  # 3 sigma target
            
            return {
                "action": "BUY",
                "confidence": min(abs(expected_return) * 10, 1.0),
                "color": "#00FF88",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "expected_return": expected_return * 100,
                "risk_reward_ratio": abs((take_profit - current_price) / (current_price - stop_loss)) 
                if current_price > stop_loss else 0.0,
                "signal_strength": "STRONG" if expected_return > 0.05 else "MODERATE"
            }
            
        elif expected_return < -confidence_threshold:
            # Bearish signal
            stop_loss = current_price * (1 + daily_vol * 2.0)
            take_profit = current_price * (1 - daily_vol * 3.0)
            
            return {
                "action": "SELL",
                "confidence": min(abs(expected_return) * 10, 1.0),
                "color": "#FF3B3B",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "expected_return": expected_return * 100,
                "risk_reward_ratio": abs((current_price - take_profit) / (stop_loss - current_price)) 
                if stop_loss > current_price else 0.0,
                "signal_strength": "STRONG" if expected_return < -0.05 else "MODERATE"
            }
        
        else:
            # Neutral signal
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "color": "#888888",
                "stop_loss": None,
                "take_profit": None,
                "expected_return": expected_return * 100,
                "risk_reward_ratio": 0.0,
                "signal_strength": "NEUTRAL"
            }

# =============================================================================
# OPTIONS ANALYTICS ENGINE (NEW)
# =============================================================================

class OptionsAnalyticsEngine:
    """Black-Scholes options pricing and analytics"""
    
    @staticmethod
    def black_scholes(
        S: float,           # Current price
        K: float,           # Strike price
        T: float,           # Time to expiration (years)
        r: float,           # Risk-free rate
        sigma: float,       # Volatility
        option_type: str = "call"  # "call" or "put"
    ) -> Dict[str, float]:
        """Calculate Black-Scholes option price and Greeks"""
        
        if T <= 0:
            return {
                "price": 0.0,
                "delta": 0.0 if option_type == "call" else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = (K * T * np.exp(-r * T) * 
               (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))) / 100
        
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }
    
    @staticmethod
    def calculate_implied_volatility(
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str = "call",
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        
        def black_scholes_price(sigma):
            result = OptionsAnalyticsEngine.black_scholes(S, K, T, r, sigma, option_type)
            return result["price"]
        
        def black_scholes_vega(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T)
        
        # Initial guesses
        sigma = 0.3
        for _ in range(max_iterations):
            price = black_scholes_price(sigma)
            vega = black_scholes_vega(sigma)
            
            if vega == 0:
                break
                
            diff = market_price - price
            if abs(diff) < tolerance:
                break
                
            sigma = sigma + diff / vega
            sigma = max(0.01, min(2.0, sigma))  # Bound between 1% and 200%
        
        return sigma

# =============================================================================
# MACRO SENSITIVITY ANALYZER (NEW)
# =============================================================================

class MacroSensitivityAnalyzer:
    """Analyze commodity sensitivity to macroeconomic factors"""
    
    @staticmethod
    def fetch_macro_data() -> pd.DataFrame:
        """Fetch macroeconomic indicators"""
        try:
            macro_tickers = ["DX-Y.NYB", "^TNX", "^VIX", "DGS10"]
            macro_data = yf.download(macro_tickers, period="5y", progress=False)['Adj Close']
            macro_data.columns = ["DXY", "SP500", "VIX", "US10Y"]
            
            # Calculate returns
            macro_returns = macro_data.pct_change().dropna()
            
            return macro_returns
            
        except Exception as e:
            st.warning(f"Macro data fetch failed: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def analyze_sensitivity(
        commodity_returns: pd.Series,
        macro_returns: pd.DataFrame,
        window: int = 252
    ) -> Dict[str, Any]:
        """Analyze rolling sensitivity to macro factors"""
        
        if macro_returns.empty or commodity_returns.empty:
            return {"available": False}
        
        # Align data
        aligned_data = pd.concat([commodity_returns, macro_returns], axis=1, join='inner').dropna()
        
        if len(aligned_data) < window:
            return {"available": False}
        
        results = {
            "available": True,
            "correlation_matrix": aligned_data.corr(),
            "rolling_betas": {},
            "sensitivity_scores": {}
        }
        
        # Calculate rolling betas
        for macro_col in macro_returns.columns:
            rolling_beta = []
            dates = []
            
            for i in range(window, len(aligned_data)):
                window_data = aligned_data.iloc[i-window:i]
                
                if len(window_data) >= window // 2:  # Minimum samples
                    X = window_data[macro_col].values.reshape(-1, 1)
                    y = window_data.iloc[:, 0].values
                    
                    # Simple linear regression for beta
                    if len(np.unique(X)) > 1 and len(np.unique(y)) > 1:
                        beta = np.cov(y, X.flatten())[0, 1] / np.var(X)
                        rolling_beta.append(beta)
                        dates.append(aligned_data.index[i])
            
            if rolling_beta:
                results["rolling_betas"][macro_col] = pd.Series(rolling_beta, index=dates)
        
        # Calculate sensitivity scores
        corr_matrix = aligned_data.corr()
        if not commodity_returns.name:
            commodity_name = "Commodity"
        else:
            commodity_name = commodity_returns.name
            
        for macro_col in macro_returns.columns:
            if commodity_name in corr_matrix.index and macro_col in corr_matrix.columns:
                correlation = corr_matrix.loc[commodity_name, macro_col]
                sensitivity = abs(correlation)
                
                results["sensitivity_scores"][macro_col] = {
                    "correlation": correlation,
                    "sensitivity": sensitivity,
                    "interpretation": "High" if sensitivity > 0.5 else 
                                    "Medium" if sensitivity > 0.3 else "Low"
                }
        
        return results

# =============================================================================
# ENHANCED PORTFOLIO ANALYTICS
# =============================================================================

class EnhancedPortfolioAnalytics:
    """Enhanced portfolio analytics with risk parity and AI optimization"""
    
    @staticmethod
    def risk_parity_allocation(returns: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity allocation weights"""
        
        if returns.empty or len(returns.columns) < 2:
            n = len(returns.columns) if not returns.empty else 1
            return np.ones(n) / n
        
        # Calculate covariance matrix
        cov_matrix = returns.cov().values * 252
        
        n = len(returns.columns)
        
        def objective(weights):
            """Objective function to minimize risk concentration"""
            weights = weights.reshape(-1, 1)
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            if portfolio_variance <= 0:
                return 1e6
            
            # Calculate risk contributions
            marginal_contributions = (cov_matrix @ weights) / portfolio_variance
            risk_contributions = marginal_contributions * weights
            
            # Target equal risk contribution
            target_contribution = portfolio_variance / n
            
            # Calculate concentration
            concentration = np.sum((risk_contributions - target_contribution) ** 2)
            
            return concentration
        
        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        bounds = [(0.01, 0.5) for _ in range(n)]  # Limit concentration
        
        # Initial equal weights
        init_weights = np.ones(n) / n
        
        # Optimize
        try:
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimized_weights = result.x
                optimized_weights = optimized_weights / np.sum(optimized_weights)  # Normalize
                return optimized_weights
            else:
                return init_weights
                
        except Exception:
            return init_weights
    
    @staticmethod
    def calculate_risk_parity_metrics(
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate risk parity portfolio metrics"""
        
        if returns.empty or len(weights) == 0:
            return {}
        
        portfolio_returns = returns @ weights
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252 * 100
        annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        
        # Calculate risk contributions
        cov_matrix = returns.cov().values * 252
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        if portfolio_variance > 0:
            marginal_contributions = (cov_matrix @ weights) / portfolio_variance
            risk_contributions = marginal_contributions * weights
            risk_contributions_pct = risk_contributions * 100
            
            # Calculate diversification ratio
            weighted_vol = np.sum(weights * (returns.std() * np.sqrt(252)))
            diversification_ratio = weighted_vol / annual_volatility if annual_volatility > 0 else 1.0
        else:
            risk_contributions_pct = np.zeros_like(weights)
            diversification_ratio = 1.0
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": (annual_return / 100 - 0.02) / (annual_volatility / 100) 
            if annual_volatility > 0 else 0,
            "risk_contributions": dict(zip(returns.columns, risk_contributions_pct)),
            "diversification_ratio": diversification_ratio,
            "weights": dict(zip(returns.columns, weights * 100))
        }

# =============================================================================
# ENHANCED DASHBOARD WITH QUANTUM FEATURES
# =============================================================================

class QuantumEnhancedDashboard:
    """Enhanced dashboard with AI and quantum features"""
    
    def __init__(self):
        # Initialize engines
        self.data_manager = EnhancedDataManager()
        self.analytics = InstitutionalAnalytics()
        self.visualizer = InstitutionalVisualizer()
        self.ai_engine = QuantumAIEngine()
        self.signal_engine = SignalIntelligenceEngine()
        self.options_engine = OptionsAnalyticsEngine()
        self.macro_analyzer = MacroSensitivityAnalyzer()
        self.portfolio_analytics = EnhancedPortfolioAnalytics()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state with quantum features"""
        defaults = {
            'data_loaded': False,
            'selected_assets': [],
            'selected_benchmarks': [],
            'asset_data': {},
            'benchmark_data': {},
            'returns_data': {},
            'ai_forecasts': {},
            'trading_signals': {},
            'macro_sensitivity': {},
            'options_analytics': {},
            'risk_parity_weights': {},
            'analysis_config': AnalysisConfiguration()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def display_quantum_header(self):
        """Display quantum-enhanced header"""
        
        st.components.v1.html(f"""
        <div style="
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            padding: 1.8rem 2rem;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                font-size: 2.5rem; 
                font-weight: 850; 
                line-height: 1.1;
                margin-bottom: 0.5rem;
            ">
                🏛️ Institutional Commodities Analytics v7.0
            </div>
            <div style="
                font-size: 1.2rem;
                opacity: 0.9;
                background: rgba(255,255,255,0.1);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                display: inline-block;
                margin-top: 0.5rem;
            ">
                Quantum Enhanced • AI Forecasting • Signal Intelligence
            </div>
        </div>
        """, height=130)
    
    def display_ai_signals_tab(self, config: AnalysisConfiguration):
        """Display AI trading signals tab"""
        
        st.markdown('<div class="section-header"><h2>🤖 AI Trading Signals</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            st.info("Load data first to generate AI signals")
            return
        
        asset_data = st.session_state.get('asset_data', {})
        returns_data = st.session_state.get('returns_data', {})
        
        if not asset_data or not returns_data:
            return
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            ai_lookback = st.slider("AI Lookback Days", 30, 180, config.ai_lookback_days, key="ai_lookback")
        with col2:
            ai_forecast_days = st.slider("Forecast Horizon", 5, 30, config.ai_forecast_days, key="ai_forecast_days")
        with col3:
            confidence_threshold = st.slider("Signal Threshold (%)", 1.0, 10.0, 3.0, key="confidence_threshold")
        
        if st.button("🔮 Generate AI Signals", use_container_width=True):
            with st.spinner("Running AI forecasts..."):
                self.ai_engine.lookback_days = ai_lookback
                self.ai_engine.forecast_days = ai_forecast_days
                
                ai_forecasts = {}
                trading_signals = {}
                
                for symbol, df in asset_data.items():
                    if 'Close' in df.columns and not df.empty:
                        price_series = df['Close']
                        
                        # Run AI forecast
                        forecast_result = self.ai_engine.run_ensemble_forecast(price_series)
                        ai_forecasts[symbol] = forecast_result
                        
                        # Generate trading signal
                        if forecast_result.get("available", False):
                            current_price = price_series.iloc[-1]
                            annual_vol = returns_data[symbol].std() * np.sqrt(252) if symbol in returns_data.columns else 0.2
                            
                            signal = self.signal_engine.generate_signal(
                                current_price,
                                forecast_result,
                                annual_vol,
                                confidence_threshold / 100
                            )
                            trading_signals[symbol] = signal
                
                st.session_state.ai_forecasts = ai_forecasts
                st.session_state.trading_signals = trading_signals
        
        # Display signals
        trading_signals = st.session_state.get('trading_signals', {})
        if trading_signals:
            st.markdown("### 📊 Live Trading Signals")
            
            # Create signal cards
            cols = st.columns(min(4, len(trading_signals)))
            for idx, (symbol, signal) in enumerate(trading_signals.items()):
                col_idx = idx % len(cols)
                with cols[col_idx]:
                    action = signal.get("action", "HOLD")
                    color = signal.get("color", "#888888")
                    confidence = signal.get("confidence", 0.0)
                    
                    st.markdown(f"""
                    <div style="
                        background: {color}20;
                        border-left: 4px solid {color};
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-weight: bold; font-size: 1.2rem; color: {color}">
                            {symbol} - {action}
                        </div>
                        <div style="color: #666; font-size: 0.9rem;">
                            Confidence: {confidence:.1%}
                        </div>
                        {f'''
                        <div style="margin-top: 8px;">
                            <div style="font-size: 0.85rem;">TP: ${signal.get('take_profit', 0):.2f}</div>
                            <div style="font-size: 0.85rem;">SL: ${signal.get('stop_loss', 0):.2f}</div>
                        </div>
                        ''' if signal.get('stop_loss') else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display detailed forecasts
            st.markdown("### 📈 AI Forecast Details")
            selected_symbol = st.selectbox("Select Asset for Detailed Forecast", list(trading_signals.keys()))
            
            if selected_symbol in st.session_state.ai_forecasts:
                forecast_result = st.session_state.ai_forecasts[selected_symbol]
                
                if forecast_result.get("available", False):
                    # Create forecast chart
                    fig = go.Figure()
                    
                    # Historical price
                    hist_prices = asset_data[selected_symbol]['Close'].iloc[-60:]
                    fig.add_trace(go.Scatter(
                        x=hist_prices.index,
                        y=hist_prices.values,
                        name="Historical Price",
                        line=dict(color="#1E90FF", width=2)
                    ))
                    
                    # Forecast
                    forecast_dates = pd.date_range(
                        start=hist_prices.index[-1] + pd.Timedelta(days=1),
                        periods=config.ai_forecast_days,
                        freq='D'
                    )
                    
                    ensemble_forecast = forecast_result.get("forecasts", {}).get("ensemble")
                    if ensemble_forecast is not None:
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=ensemble_forecast,
                            name="AI Forecast",
                            line=dict(color="#FF6347", width=3, dash='dash')
                        ))
                        
                        # Confidence interval
                        ci = forecast_result.get("confidence_intervals", {})
                        if 'upper' in ci and 'lower' in ci:
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=ci['upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=ci['lower'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(255, 99, 71, 0.2)',
                                fill='tonexty',
                                name='95% Confidence'
                            ))
                    
                    fig.update_layout(
                        title=f"AI Forecast for {selected_symbol}",
                        height=400,
                        template=self.visualizer.template,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display model performance
                    st.markdown("#### 🤖 Model Performance")
                    models_used = forecast_result.get("models_used", [])
                    
                    if models_used:
                        cols = st.columns(len(models_used))
                        for idx, model in enumerate(models_used):
                            with cols[idx]:
                                st.metric(
                                    label=f"{model.upper()} Model",
                                    value="Active",
                                    delta="In Ensemble"
                                )
    
    def display_macro_sensitivity_tab(self, config: AnalysisConfiguration):
        """Display macro sensitivity analysis"""
        
        st.markdown('<div class="section-header"><h2>🌍 Macro Sensitivity Analysis</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            st.info("Load data first to analyze macro sensitivity")
            return
        
        returns_data = st.session_state.get('returns_data', {})
        
        if st.button("📊 Analyze Macro Sensitivity", use_container_width=True):
            with st.spinner("Fetching macro data and analyzing sensitivity..."):
                # Fetch macro data
                macro_returns = self.macro_analyzer.fetch_macro_data()
                
                if macro_returns.empty:
                    st.warning("Could not fetch macro data. Please try again.")
                    return
                
                sensitivity_results = {}
                
                # Analyze each asset
                for symbol in returns_data.columns:
                    if symbol in returns_data:
                        results = self.macro_analyzer.analyze_sensitivity(
                            returns_data[symbol],
                            macro_returns,
                            window=126
                        )
                        
                        if results.get("available", False):
                            sensitivity_results[symbol] = results
                
                st.session_state.macro_sensitivity = sensitivity_results
        
        sensitivity_results = st.session_state.get('macro_sensitivity', {})
        if sensitivity_results:
            # Display correlation heatmap
            st.markdown("### 📈 Macro Correlation Matrix")
            
            selected_symbol = st.selectbox("Select Asset", list(sensitivity_results.keys()))
            
            if selected_symbol in sensitivity_results:
                results = sensitivity_results[selected_symbol]
                
                # Create correlation heatmap
                corr_matrix = results.get("correlation_matrix", pd.DataFrame())
                
                if not corr_matrix.empty:
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title=f"Correlation Matrix - {selected_symbol}"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display sensitivity scores
                st.markdown("### 🎯 Sensitivity Scores")
                
                sensitivity_scores = results.get("sensitivity_scores", {})
                
                if sensitivity_scores:
                    scores_df = pd.DataFrame(sensitivity_scores).T
                    
                    # Create gauge charts for each macro factor
                    cols = st.columns(len(sensitivity_scores))
                    
                    for idx, (factor, score_data) in enumerate(sensitivity_scores.items()):
                        with cols[idx % len(cols)]:
                            correlation = score_data.get("correlation", 0)
                            sensitivity = score_data.get("sensitivity", 0)
                            interpretation = score_data.get("interpretation", "Low")
                            
                            # Create gauge indicator
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=sensitivity * 100,
                                title={'text': f"{factor}"},
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "#1E90FF"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "#00FF88"},
                                        {'range': [30, 70], 'color': "#FFA500"},
                                        {'range': [70, 100], 'color': "#FF4500"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            
                            fig.update_layout(height=200, margin=dict(t=50, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.caption(f"Correlation: {correlation:.3f} • {interpretation}")
                
                # Display rolling betas
                st.markdown("### 📊 Rolling Beta Analysis")
                
                rolling_betas = results.get("rolling_betas", {})
                
                if rolling_betas:
                    fig = go.Figure()
                    
                    for factor, beta_series in rolling_betas.items():
                        fig.add_trace(go.Scatter(
                            x=beta_series.index,
                            y=beta_series.values,
                            name=factor,
                            mode='lines'
                        ))
                    
                    fig.update_layout(
                        title=f"Rolling Beta - {selected_symbol}",
                        height=400,
                        yaxis_title="Beta",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_options_analytics_tab(self, config: AnalysisConfiguration):
        """Display options analytics tab"""
        
        st.markdown('<div class="section-header"><h2>📊 Options Analytics (Black-Scholes)</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            st.info("Load data first to calculate options")
            return
        
        asset_data = st.session_state.get('asset_data', {})
        
        if not asset_data:
            return
        
        # Options calculator
        st.markdown("### 🧮 Options Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_symbol = st.selectbox("Select Asset", list(asset_data.keys()))
            
            if selected_symbol in asset_data:
                current_price = asset_data[selected_symbol]['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
                
                # Get volatility
                returns_data = st.session_state.get('returns_data', {})
                if selected_symbol in returns_data.columns:
                    annual_vol = returns_data[selected_symbol].std() * np.sqrt(252)
                    st.metric("Annual Volatility", f"{annual_vol:.1%}")
                else:
                    annual_vol = 0.3  # Default
        
        with col2:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            strike_price = st.number_input(
                "Strike Price",
                min_value=0.01,
                value=round(current_price * 1.05, 2) if 'current_price' in locals() else 100.0,
                step=0.01
            )
            time_to_expiry = st.slider("Days to Expiry", 1, 365, 30)
            risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, config.risk_free_rate * 100) / 100
        
        if st.button("Calculate Option Price", use_container_width=True):
            # Convert days to years
            T = time_to_expiry / 365.0
            
            # Calculate option price
            option_result = self.options_engine.black_scholes(
                S=current_price,
                K=strike_price,
                T=T,
                r=risk_free_rate,
                sigma=annual_vol,
                option_type=option_type.lower()
            )
            
            st.session_state.options_analytics = {
                "current_price": current_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "option_type": option_type,
                "results": option_result
            }
        
        # Display results
        options_results = st.session_state.get('options_analytics', {})
        if options_results and "results" in options_results:
            results = options_results["results"]
            
            st.markdown("### 📈 Option Greeks")
            
            # Display Greeks in metrics
            cols = st.columns(5)
            greeks = [
                ("Price", f"${results['price']:.2f}", None),
                ("Delta", f"{results['delta']:.3f}", None),
                ("Gamma", f"{results['gamma']:.4f}", None),
                ("Theta", f"{results['theta']:.4f}", "per day"),
                ("Vega", f"{results['vega']:.4f}", "per 1% vol")
            ]
            
            for idx, (name, value, suffix) in enumerate(greeks):
                with cols[idx]:
                    st.metric(name, value, delta=suffix if suffix else "")
            
            # Create payoff diagram
            st.markdown("### 📊 Payoff Diagram")
            
            price_range = np.linspace(
                current_price * 0.7,
                current_price * 1.3,
                100
            )
            
            payoffs = []
            for S_T in price_range:
                if option_type.lower() == "call":
                    payoff = max(S_T - strike_price, 0) - results['price']
                else:
                    payoff = max(strike_price - S_T, 0) - results['price']
                payoffs.append(payoff)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_range,
                y=payoffs,
                mode='lines',
                name='Option Payoff',
                line=dict(color='#FF6347', width=3)
            ))
            
            # Add breakeven line
            if option_type.lower() == "call":
                breakeven = strike_price + results['price']
            else:
                breakeven = strike_price - results['price']
            
            fig.add_vline(
                x=breakeven,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Breakeven: ${breakeven:.2f}"
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                title=f"{option_type} Option Payoff Diagram",
                height=400,
                xaxis_title="Underlying Price at Expiry",
                yaxis_title="Profit/Loss",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_parity_tab(self, config: AnalysisConfiguration):
        """Display risk parity portfolio optimization"""
        
        st.markdown('<div class="section-header"><h2>⚖️ Risk Parity Portfolio</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            st.info("Load data first to calculate risk parity")
            return
        
        returns_data = st.session_state.get('returns_data', {})
        
        if returns_data.empty or len(returns_data.columns) < 2:
            st.warning("Need at least 2 assets for portfolio optimization")
            return
        
        if st.button("⚖️ Calculate Risk Parity", use_container_width=True):
            with st.spinner("Calculating risk parity allocation..."):
                # Calculate risk parity weights
                weights = self.portfolio_analytics.risk_parity_allocation(returns_data)
                
                # Calculate metrics
                metrics = self.portfolio_analytics.calculate_risk_parity_metrics(
                    returns_data,
                    weights
                )
                
                st.session_state.risk_parity_weights = metrics
        
        risk_parity_results = st.session_state.get('risk_parity_weights', {})
        if risk_parity_results:
            # Display portfolio metrics
            st.markdown("### 📊 Portfolio Metrics")
            
            cols = st.columns(4)
            cols[0].metric("Annual Return", f"{risk_parity_results.get('annual_return', 0):.2f}%")
            cols[1].metric("Annual Volatility", f"{risk_parity_results.get('annual_volatility', 0):.2f}%")
            cols[2].metric("Sharpe Ratio", f"{risk_parity_results.get('sharpe_ratio', 0):.2f}")
            cols[3].metric("Diversification Ratio", f"{risk_parity_results.get('diversification_ratio', 0):.2f}")
            
            # Display weights
            st.markdown("### 📈 Asset Allocation")
            
            weights_dict = risk_parity_results.get('weights', {})
            if weights_dict:
                weights_df = pd.DataFrame(
                    weights_dict.items(),
                    columns=['Asset', 'Weight (%)']
                ).sort_values('Weight (%)', ascending=False)
                
                # Create pie chart
                fig = px.pie(
                    weights_df,
                    values='Weight (%)',
                    names='Asset',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig.update_layout(
                    title="Risk Parity Allocation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display weights table
                st.dataframe(
                    weights_df.style.format({'Weight (%)': '{:.2f}%'}).bar(
                        subset=['Weight (%)'],
                        color='#1E90FF'
                    ),
                    use_container_width=True
                )
            
            # Display risk contributions
            st.markdown("### ⚠️ Risk Contributions")
            
            risk_contributions = risk_parity_results.get('risk_contributions', {})
            if risk_contributions:
                risk_df = pd.DataFrame(
                    risk_contributions.items(),
                    columns=['Asset', 'Risk Contribution (%)']
                )
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=risk_df['Asset'],
                        y=risk_df['Risk Contribution (%)'],
                        marker_color='#FF6347'
                    )
                ])
                
                fig.update_layout(
                    title="Risk Contribution by Asset",
                    height=400,
                    yaxis_title="Risk Contribution (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main app runner with quantum features"""
        try:
            self.display_quantum_header()
            
            # Sidebar
            sidebar_state = self._render_sidebar_controls()
            
            # Auto reload on changes
            if sidebar_state.get("auto_reload", False):
                self._load_sidebar_selection(sidebar_state)
            
            # Explicit load button
            if sidebar_state.get("load_clicked", False):
                self._load_sidebar_selection(sidebar_state)
            
            # Get config
            cfg = st.session_state.get("analysis_config", AnalysisConfiguration())
            
            if not st.session_state.get("data_loaded", False):
                self._display_welcome(cfg)
                return
            
            # Enhanced tabs with quantum features
            tab_labels = [
                "📊 Dashboard",
                "🤖 AI Signals",
                "🧮 Portfolio",
                "🌍 Macro Sensitivity",
                "📊 Options Analytics",
                "⚖️ Risk Parity",
                "📈 EWMA Ratio",
                "🎯 Tracking Error",
                "β Rolling Beta",
                "📉 Risk Analytics",
                "🧠 Advanced Analytics",
                "🧪 Stress Testing",
                "📑 Reporting"
            ]
            
            tabs = st.tabs(tab_labels)
            
            # Map tabs to functions
            with tabs[0]:
                self._display_dashboard(cfg)
            with tabs[1]:
                self.display_ai_signals_tab(cfg)
            with tabs[2]:
                self._display_portfolio(cfg)
            with tabs[3]:
                self.display_macro_sensitivity_tab(cfg)
            with tabs[4]:
                self.display_options_analytics_tab(cfg)
            with tabs[5]:
                self.display_risk_parity_tab(cfg)
            with tabs[6]:
                self._display_ewma_ratio_signal(cfg)
            with tabs[7]:
                self._display_tracking_error(cfg)
            with tabs[8]:
                self._display_rolling_beta(cfg)
            with tabs[9]:
                self._display_risk_analytics(cfg)
            with tabs[10]:
                self._display_advanced_analytics(cfg)
            with tabs[11]:
                self._display_stress_testing(cfg)
            with tabs[12]:
                self._display_reporting(cfg)
                
        except Exception as e:
            st.error(f"🚨 Application Error: {e}")
            st.code(traceback.format_exc())

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check for AI dependencies
    if not TENSORFLOW_AVAILABLE:
        st.sidebar.warning("TensorFlow not available. AI features limited.")
    if not XGB_AVAILABLE:
        st.sidebar.warning("XGBoost not available. Ensemble forecasting limited.")
    if not SKLEARN_AVAILABLE:
        st.sidebar.warning("Scikit-learn not available. Some features limited.")
    
    # Initialize and run dashboard
    dashboard = QuantumEnhancedDashboard()
    dashboard.run()
