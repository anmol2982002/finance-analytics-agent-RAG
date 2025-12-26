"""
Price Predictor with Uncertainty Quantification
Honest predictions with confidence intervals - never false precision.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import warnings
from src.utils import get_logger

logger = get_logger(__name__)

# Suppress warnings from statsmodels
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class PredictionBand:
    """Confidence band for predictions."""
    confidence_level: int  # 50, 80, 95
    upper: List[float]
    lower: List[float]


@dataclass
class RegimeForecast:
    """Market regime probabilities."""
    bull_probability: float
    bear_probability: float
    sideways_probability: float
    high_volatility_probability: float
    current_regime: str


@dataclass
class PredictionResult:
    """Complete prediction with uncertainty quantification."""
    ticker: str
    timestamp: str
    horizon_days: int
    
    # Point predictions (but emphasize bands!)
    median_prediction: List[float]
    
    # Confidence bands - the main output
    band_50: PredictionBand  # 50% CI
    band_80: PredictionBand  # 80% CI
    band_95: PredictionBand  # 95% CI
    
    # Regime analysis
    regime_forecast: RegimeForecast
    
    # Volatility forecast
    volatility_forecast: List[float]  # Expected volatility per day
    
    # Model diagnostics
    model_confidence: float  # How confident is the model itself
    prediction_quality: str  # 'HIGH', 'MEDIUM', 'LOW'
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "horizon_days": self.horizon_days,
            "median_prediction": self.median_prediction,
            "confidence_bands": {
                "50": {"upper": self.band_50.upper, "lower": self.band_50.lower},
                "80": {"upper": self.band_80.upper, "lower": self.band_80.lower},
                "95": {"upper": self.band_95.upper, "lower": self.band_95.lower}
            },
            "regime_forecast": {
                "bull_probability": self.regime_forecast.bull_probability,
                "bear_probability": self.regime_forecast.bear_probability,
                "sideways_probability": self.regime_forecast.sideways_probability,
                "high_volatility_probability": self.regime_forecast.high_volatility_probability,
                "current_regime": self.regime_forecast.current_regime
            },
            "volatility_forecast": self.volatility_forecast,
            "model_confidence": self.model_confidence,
            "prediction_quality": self.prediction_quality,
            "warnings": self.warnings
        }


class PricePredictor:
    """
    Honest price predictions with uncertainty quantification.
    
    Never gives single point predictions without uncertainty bounds.
    Uses ensemble of methods:
    - ARIMA for trend
    - GARCH for volatility
    - Monte Carlo for simulation
    - Regime detection for context
    """
    
    def __init__(
        self,
        n_simulations: int = 5000,
        use_garch: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            n_simulations: Number of Monte Carlo paths
            use_garch: Whether to use GARCH for volatility (requires arch)
        """
        self.n_simulations = n_simulations
        self.use_garch = use_garch
        
        # Check for optional dependencies
        self._has_arch = False
        self._has_statsmodels = False
        
        try:
            import arch
            self._has_arch = True
        except ImportError:
            logger.warning("arch package not found - using simpler volatility model")
        
        try:
            import statsmodels.api as sm
            self._has_statsmodels = True
        except ImportError:
            logger.warning("statsmodels not found - using simpler trend model")
        
        logger.info(f"Initialized PricePredictor with {n_simulations} simulations")
    
    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 7,
        ticker: str = "UNKNOWN"
    ) -> PredictionResult:
        """
        Generate price predictions with uncertainty quantification.
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Number of days to predict
            ticker: Ticker symbol for labeling
        
        Returns:
            PredictionResult with predictions and confidence bands
        """
        if df is None or len(df) < 30:
            return self._insufficient_data_result(ticker, horizon)
        
        logger.info(f"Generating {horizon}-day prediction for {ticker}")
        
        # Get current state
        current_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        
        # 1. Detect current regime
        regime = self._detect_regime(df)
        
        # 2. Estimate volatility (GARCH or simple)
        if self.use_garch and self._has_arch:
            volatility_forecast = self._garch_volatility(returns, horizon)
        else:
            volatility_forecast = self._simple_volatility(returns, horizon)
        
        # 3. Estimate drift (mean return)
        drift = self._estimate_drift(returns, regime)
        
        # 4. Run Monte Carlo simulation
        simulated_paths = self._monte_carlo_simulation(
            current_price=current_price,
            drift=drift,
            volatility=volatility_forecast,
            horizon=horizon
        )
        
        # 5. Calculate prediction bands
        median_prediction = []
        band_50_upper, band_50_lower = [], []
        band_80_upper, band_80_lower = [], []
        band_95_upper, band_95_lower = [], []
        
        for day in range(horizon):
            day_prices = simulated_paths[:, day]
            
            median_prediction.append(round(np.median(day_prices), 4))
            
            band_50_lower.append(round(np.percentile(day_prices, 25), 4))
            band_50_upper.append(round(np.percentile(day_prices, 75), 4))
            
            band_80_lower.append(round(np.percentile(day_prices, 10), 4))
            band_80_upper.append(round(np.percentile(day_prices, 90), 4))
            
            band_95_lower.append(round(np.percentile(day_prices, 2.5), 4))
            band_95_upper.append(round(np.percentile(day_prices, 97.5), 4))
        
        # 6. Assess prediction quality
        model_confidence, quality, warnings = self._assess_quality(
            df=df,
            volatility=volatility_forecast,
            regime=regime
        )
        
        return PredictionResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            horizon_days=horizon,
            median_prediction=median_prediction,
            band_50=PredictionBand(50, band_50_upper, band_50_lower),
            band_80=PredictionBand(80, band_80_upper, band_80_lower),
            band_95=PredictionBand(95, band_95_upper, band_95_lower),
            regime_forecast=regime,
            volatility_forecast=[round(v * 100, 2) for v in volatility_forecast],
            model_confidence=model_confidence,
            prediction_quality=quality,
            warnings=warnings
        )
    
    def _detect_regime(self, df: pd.DataFrame) -> RegimeForecast:
        """
        Detect current market regime using returns characteristics.
        """
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < 60:
            return RegimeForecast(
                bull_probability=33.3,
                bear_probability=33.3,
                sideways_probability=33.4,
                high_volatility_probability=50,
                current_regime="UNKNOWN"
            )
        
        # Recent trend (20-day)
        recent_return = returns.tail(20).sum()
        
        # Volatility regime
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        # Mean reversion indicator
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        distance_from_ma = (current_price - sma_50) / sma_50
        
        # Calculate regime probabilities
        if recent_return > 0.05:  # >5% gain in 20 days
            bull_prob = min(80, 50 + recent_return * 200)
            bear_prob = max(5, 25 - recent_return * 100)
            sideways_prob = 100 - bull_prob - bear_prob
            current_regime = "BULL_TREND"
        elif recent_return < -0.05:  # >5% loss
            bear_prob = min(80, 50 + abs(recent_return) * 200)
            bull_prob = max(5, 25 - abs(recent_return) * 100)
            sideways_prob = 100 - bull_prob - bear_prob
            current_regime = "BEAR_TREND"
        else:
            sideways_prob = 60
            bull_prob = 20
            bear_prob = 20
            current_regime = "SIDEWAYS"
        
        # High volatility flag
        high_vol_prob = min(90, vol_ratio * 50) if vol_ratio > 1 else max(10, vol_ratio * 50)
        
        if vol_ratio > 1.5:
            current_regime = "HIGH_VOLATILITY"
        
        return RegimeForecast(
            bull_probability=round(bull_prob, 1),
            bear_probability=round(bear_prob, 1),
            sideways_probability=round(sideways_prob, 1),
            high_volatility_probability=round(high_vol_prob, 1),
            current_regime=current_regime
        )
    
    def _garch_volatility(
        self,
        returns: pd.Series,
        horizon: int
    ) -> List[float]:
        """
        Forecast volatility using GARCH(1,1).
        """
        try:
            from arch import arch_model
            
            # Scale returns for numerical stability
            scaled_returns = returns * 100
            
            model = arch_model(
                scaled_returns.dropna(),
                vol='Garch',
                p=1, q=1,
                mean='Constant',
                rescale=False
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast variance
            forecast = result.forecast(horizon=horizon)
            variance_forecast = forecast.variance.values[-1, :]
            
            # Convert back to daily volatility
            volatility = np.sqrt(variance_forecast) / 100
            
            return list(volatility)
            
        except Exception as e:
            logger.warning(f"GARCH failed: {e}, falling back to simple volatility")
            return self._simple_volatility(returns, horizon)
    
    def _simple_volatility(
        self,
        returns: pd.Series,
        horizon: int
    ) -> List[float]:
        """
        Simple volatility forecast using exponential weighted average.
        """
        # EWMA volatility with decay
        ewma_vol = returns.ewm(span=20).std().iloc[-1]
        historical_vol = returns.std()
        
        # Blend current and historical
        volatility = []
        for day in range(horizon):
            # Mean revert towards historical vol
            weight = 0.9 ** day
            day_vol = ewma_vol * weight + historical_vol * (1 - weight)
            volatility.append(day_vol)
        
        return volatility
    
    def _estimate_drift(
        self,
        returns: pd.Series,
        regime: RegimeForecast
    ) -> float:
        """
        Estimate drift (expected return) based on regime.
        """
        historical_mean = returns.mean()
        recent_mean = returns.tail(20).mean()
        
        # Regime adjustment
        if regime.current_regime == "BULL_TREND":
            drift = historical_mean * 0.3 + recent_mean * 0.7
        elif regime.current_regime == "BEAR_TREND":
            drift = historical_mean * 0.3 + recent_mean * 0.7
        elif regime.current_regime == "HIGH_VOLATILITY":
            drift = historical_mean * 0.5  # Reduce confidence in trend
        else:
            drift = historical_mean * 0.7 + recent_mean * 0.3
        
        return drift
    
    def _monte_carlo_simulation(
        self,
        current_price: float,
        drift: float,
        volatility: List[float],
        horizon: int
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for price paths.
        """
        np.random.seed(None)  # Use random seed for each call
        
        # Generate random returns
        simulated_paths = np.zeros((self.n_simulations, horizon))
        
        for day in range(horizon):
            daily_vol = volatility[day] if day < len(volatility) else volatility[-1]
            
            # Geometric Brownian Motion with regime-adjusted drift
            random_returns = np.random.normal(drift, daily_vol, self.n_simulations)
            
            if day == 0:
                simulated_paths[:, day] = current_price * (1 + random_returns)
            else:
                simulated_paths[:, day] = simulated_paths[:, day-1] * (1 + random_returns)
        
        return simulated_paths
    
    def _assess_quality(
        self,
        df: pd.DataFrame,
        volatility: List[float],
        regime: RegimeForecast
    ) -> Tuple[float, str, List[str]]:
        """
        Assess the quality and reliability of predictions.
        """
        warnings = []
        confidence = 70  # Base confidence
        
        # Data quality check
        data_length = len(df)
        if data_length < 60:
            warnings.append("⚠️ Limited historical data - predictions less reliable")
            confidence -= 20
        elif data_length < 120:
            warnings.append("⚠️ Moderate historical data - consider longer history")
            confidence -= 10
        
        # Volatility check
        avg_vol = np.mean(volatility)
        if avg_vol > 0.03:  # >3% daily vol
            warnings.append("⚠️ High volatility environment - wider prediction bands")
            confidence -= 15
        elif avg_vol > 0.02:
            warnings.append("⚠️ Elevated volatility - predictions uncertain")
            confidence -= 5
        
        # Regime check
        if regime.current_regime == "HIGH_VOLATILITY":
            warnings.append("⚠️ High volatility regime - expect large price swings")
            confidence -= 10
        
        # Check for recent gaps or anomalies
        returns = df['Close'].pct_change().dropna()
        recent_max = returns.tail(10).abs().max()
        if recent_max > 0.05:  # >5% single day move recently
            warnings.append("⚠️ Recent large price moves - elevated uncertainty")
            confidence -= 10
        
        # Determine quality label
        confidence = max(20, min(95, confidence))
        
        if confidence >= 70:
            quality = "HIGH"
        elif confidence >= 50:
            quality = "MEDIUM"
        else:
            quality = "LOW"
        
        return round(confidence, 1), quality, warnings
    
    def _insufficient_data_result(
        self,
        ticker: str,
        horizon: int
    ) -> PredictionResult:
        """Return result when insufficient data."""
        return PredictionResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            horizon_days=horizon,
            median_prediction=[],
            band_50=PredictionBand(50, [], []),
            band_80=PredictionBand(80, [], []),
            band_95=PredictionBand(95, [], []),
            regime_forecast=RegimeForecast(33.3, 33.3, 33.4, 50, "UNKNOWN"),
            volatility_forecast=[],
            model_confidence=0,
            prediction_quality="LOW",
            warnings=["⚠️ Insufficient data for prediction - need at least 30 days of history"]
        )
    
    def get_price_targets(
        self,
        prediction: PredictionResult,
        risk_reward_ratio: float = 2.0
    ) -> Dict[str, Any]:
        """
        Calculate price targets based on prediction bands.
        
        Args:
            prediction: PredictionResult from predict()
            risk_reward_ratio: Desired risk/reward ratio
        
        Returns:
            Dict with entry, stop-loss, and target levels
        """
        if not prediction.median_prediction:
            return {"error": "No predictions available"}
        
        current = prediction.median_prediction[0] if prediction.median_prediction else 0
        final_median = prediction.median_prediction[-1]
        
        # Use 80% band for risk assessment
        final_lower_80 = prediction.band_80.lower[-1] if prediction.band_80.lower else current
        final_upper_80 = prediction.band_80.upper[-1] if prediction.band_80.upper else current
        
        # Calculate targets for bullish and bearish scenarios
        if final_median > current:
            # Bullish setup
            entry = current
            stop_loss = final_lower_80
            risk = entry - stop_loss
            target = entry + (risk * risk_reward_ratio)
            direction = "LONG"
        else:
            # Bearish setup
            entry = current
            stop_loss = final_upper_80
            risk = stop_loss - entry
            target = entry - (risk * risk_reward_ratio)
            direction = "SHORT"
        
        return {
            "direction": direction,
            "entry": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "target": round(target, 4),
            "risk_percent": round(abs(risk / entry) * 100, 2),
            "reward_percent": round(abs(target - entry) / entry * 100, 2),
            "risk_reward_ratio": risk_reward_ratio,
            "probability_based_on": "80% confidence band"
        }
