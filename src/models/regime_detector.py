"""
Market Regime Detector
Classifies market conditions using Hidden Markov Models and statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from sklearn.mixture import GaussianMixture
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeState:
    """Current market regime state."""
    regime: str  # BULL_TREND, BEAR_TREND, HIGH_VOLATILITY, LOW_VOLATILITY, CRASH
    confidence: float  # 0-100
    duration_days: int  # How long in this regime
    
    # Probabilities for each regime
    bull_probability: float
    bear_probability: float
    high_vol_probability: float
    low_vol_probability: float
    crash_probability: float
    
    # Transition probabilities
    probability_of_change: float
    most_likely_next_regime: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "duration_days": self.duration_days,
            "probabilities": {
                "bull": self.bull_probability,
                "bear": self.bear_probability,
                "high_volatility": self.high_vol_probability,
                "low_volatility": self.low_vol_probability,
                "crash": self.crash_probability
            },
            "probability_of_change": self.probability_of_change,
            "most_likely_next_regime": self.most_likely_next_regime
        }


@dataclass
class RegimeHistory:
    """Historical regime analysis."""
    regimes: List[Dict[str, Any]]  # List of regime periods
    avg_bull_duration: float
    avg_bear_duration: float
    current_cycle_position: str  # EARLY, MID, LATE
    regime_change_frequency: float  # Changes per year


class RegimeDetector:
    """
    Market regime classification using statistical methods.
    
    Regimes:
    - BULL_TREND: Strong upward momentum, low volatility
    - BEAR_TREND: Strong downward momentum
    - HIGH_VOLATILITY: Choppy markets, large swings
    - LOW_VOLATILITY: Calm, range-bound markets
    - CRASH: Extreme downside (tail risk event)
    
    Suggestions adapt to current regime!
    """
    
    # Regime definitions
    REGIMES = {
        'BULL_TREND': 0,
        'BEAR_TREND': 1,
        'HIGH_VOLATILITY': 2,
        'LOW_VOLATILITY': 3,
        'CRASH': 4
    }
    
    def __init__(
        self,
        lookback_days: int = 252,  # 1 year
        volatility_window: int = 20,
        trend_window: int = 50,
        crash_threshold: float = -0.15  # 15% decline
    ):
        """
        Initialize the regime detector.
        
        Args:
            lookback_days: Historical period for analysis
            volatility_window: Window for volatility calculation
            trend_window: Window for trend detection
            crash_threshold: Return threshold for crash detection
        """
        self.lookback_days = lookback_days
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.crash_threshold = crash_threshold
        
        logger.info("Initialized RegimeDetector")
    
    def detect_current_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect the current market regime.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            RegimeState with current regime and probabilities
        """
        if df is None or len(df) < self.trend_window:
            return self._unknown_regime()
        
        # Calculate features
        features = self._calculate_features(df)
        
        # Detect regime using rule-based approach
        regime, confidence, probabilities = self._classify_regime(features)
        
        # Calculate regime duration
        duration = self._calculate_regime_duration(df, regime)
        
        # Estimate transition probability
        prob_change, next_regime = self._estimate_transition(features, regime, duration)
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            duration_days=duration,
            bull_probability=probabilities['bull'],
            bear_probability=probabilities['bear'],
            high_vol_probability=probabilities['high_vol'],
            low_vol_probability=probabilities['low_vol'],
            crash_probability=probabilities['crash'],
            probability_of_change=prob_change,
            most_likely_next_regime=next_regime
        )
    
    def detect_with_gmm(self, df: pd.DataFrame, n_regimes: int = 4) -> RegimeState:
        """
        Detect regime using Gaussian Mixture Model (data-driven).
        
        More adaptive but less interpretable than rule-based approach.
        """
        if len(df) < 100:
            return self._unknown_regime()
        
        # Prepare features
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(self.volatility_window).std()
        
        # Create feature matrix
        X = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'returns_ma': returns.rolling(20).mean(),
            'vol_change': volatility.pct_change()
        }).dropna()
        
        if len(X) < 50:
            return self._unknown_regime()
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=3)
        X_scaled = (X - X.mean()) / X.std()
        gmm.fit(X_scaled)
        
        # Get current regime
        current_features = X_scaled.iloc[-1:].values
        probabilities = gmm.predict_proba(current_features)[0]
        current_regime_idx = np.argmax(probabilities)
        
        # Map to regime names based on cluster characteristics
        cluster_means = pd.DataFrame(
            gmm.means_,
            columns=['returns', 'volatility', 'returns_ma', 'vol_change']
        )
        
        # Assign names based on characteristics
        regime_names = self._assign_regime_names(cluster_means)
        current_regime_name = regime_names[current_regime_idx]
        
        # Build probabilities dict
        prob_dict = {
            'bull': 0, 'bear': 0, 'high_vol': 0, 'low_vol': 0, 'crash': 0
        }
        for i, name in enumerate(regime_names):
            key = name.lower().replace('_trend', '').replace('_volatility', '_vol')
            if key in prob_dict:
                prob_dict[key] = probabilities[i] * 100
        
        return RegimeState(
            regime=current_regime_name,
            confidence=round(np.max(probabilities) * 100, 1),
            duration_days=self._calculate_regime_duration(df, current_regime_name),
            bull_probability=prob_dict['bull'],
            bear_probability=prob_dict['bear'],
            high_vol_probability=prob_dict['high_vol'],
            low_vol_probability=prob_dict['low_vol'],
            crash_probability=prob_dict.get('crash', 0),
            probability_of_change=round((1 - np.max(probabilities)) * 100, 1),
            most_likely_next_regime=regime_names[np.argsort(probabilities)[-2]]
        )
    
    def analyze_regime_history(self, df: pd.DataFrame) -> RegimeHistory:
        """
        Analyze historical regime changes and patterns.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            RegimeHistory with regime periods and statistics
        """
        if len(df) < 100:
            return RegimeHistory(
                regimes=[],
                avg_bull_duration=0,
                avg_bear_duration=0,
                current_cycle_position="UNKNOWN",
                regime_change_frequency=0
            )
        
        regimes = []
        current_regime = None
        regime_start = 0
        
        # Detect regime at each point (simplified - rolling)
        for i in range(self.trend_window, len(df), 5):  # Check every 5 days
            temp_df = df.iloc[:i+1]
            features = self._calculate_features(temp_df)
            regime, _, _ = self._classify_regime(features)
            
            if regime != current_regime:
                if current_regime is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start_idx': regime_start,
                        'end_idx': i,
                        'duration': i - regime_start,
                        'start_date': df.index[regime_start].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(regime_start),
                        'end_date': df.index[i].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(i)
                    })
                current_regime = regime
                regime_start = i
        
        # Add current regime
        if current_regime:
            regimes.append({
                'regime': current_regime,
                'start_idx': regime_start,
                'end_idx': len(df) - 1,
                'duration': len(df) - 1 - regime_start,
                'start_date': df.index[regime_start].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(regime_start),
                'end_date': 'Present'
            })
        
        # Calculate statistics
        bull_durations = [r['duration'] for r in regimes if r['regime'] == 'BULL_TREND']
        bear_durations = [r['duration'] for r in regimes if r['regime'] == 'BEAR_TREND']
        
        avg_bull = np.mean(bull_durations) if bull_durations else 0
        avg_bear = np.mean(bear_durations) if bear_durations else 0
        
        # Regime change frequency
        num_changes = len(regimes) - 1
        total_days = len(df)
        changes_per_year = (num_changes / total_days) * 252 if total_days > 0 else 0
        
        # Current cycle position
        if regimes:
            current_duration = regimes[-1]['duration']
            avg_duration = (avg_bull + avg_bear) / 2 if (avg_bull + avg_bear) > 0 else current_duration
            
            if current_duration < avg_duration * 0.3:
                cycle_position = "EARLY"
            elif current_duration < avg_duration * 0.7:
                cycle_position = "MID"
            else:
                cycle_position = "LATE"
        else:
            cycle_position = "UNKNOWN"
        
        return RegimeHistory(
            regimes=regimes,
            avg_bull_duration=round(avg_bull, 1),
            avg_bear_duration=round(avg_bear, 1),
            current_cycle_position=cycle_position,
            regime_change_frequency=round(changes_per_year, 2)
        )
    
    def _calculate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for regime detection."""
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < self.volatility_window:
            return {}
        
        # Trend features
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50
        current_price = df['Close'].iloc[-1]
        
        # Returns
        return_20d = (current_price / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0
        return_50d = (current_price / df['Close'].iloc[-50] - 1) if len(df) >= 50 else return_20d
        
        # Volatility
        volatility_20d = returns.tail(20).std() * np.sqrt(252)
        volatility_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else volatility_20d
        
        # Drawdown
        peak = df['Close'].cummax()
        drawdown = (df['Close'] - peak) / peak
        current_drawdown = drawdown.iloc[-1]
        
        return {
            'price_vs_sma20': (current_price / sma_20 - 1) if sma_20 > 0 else 0,
            'price_vs_sma50': (current_price / sma_50 - 1) if sma_50 > 0 else 0,
            'price_vs_sma200': (current_price / sma_200 - 1) if pd.notna(sma_200) and sma_200 > 0 else 0,
            'sma20_vs_sma50': (sma_20 / sma_50 - 1) if sma_50 > 0 else 0,
            'return_20d': return_20d,
            'return_50d': return_50d,
            'volatility_20d': volatility_20d,
            'volatility_60d': volatility_60d,
            'volatility_ratio': volatility_20d / volatility_60d if volatility_60d > 0 else 1,
            'current_drawdown': current_drawdown,
            'rsi': self._calculate_rsi(df['Close'], 14)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _classify_regime(
        self,
        features: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify current regime based on features.
        """
        if not features:
            return "UNKNOWN", 0, {'bull': 25, 'bear': 25, 'high_vol': 25, 'low_vol': 25, 'crash': 0}
        
        # Extract features
        return_20d = features.get('return_20d', 0)
        return_50d = features.get('return_50d', 0)
        volatility = features.get('volatility_20d', 0.2)
        vol_ratio = features.get('volatility_ratio', 1)
        drawdown = features.get('current_drawdown', 0)
        price_vs_sma200 = features.get('price_vs_sma200', 0)
        rsi = features.get('rsi', 50)
        
        # Initialize scores
        scores = {
            'bull': 0,
            'bear': 0,
            'high_vol': 0,
            'low_vol': 0,
            'crash': 0
        }
        
        # Crash detection (highest priority)
        if drawdown < self.crash_threshold or (return_20d < -0.10 and volatility > 0.40):
            scores['crash'] = 80
            scores['bear'] = 15
            scores['high_vol'] = 5
        else:
            # Trend detection
            if return_50d > 0.10 and price_vs_sma200 > 0.05:
                scores['bull'] += 40
            elif return_50d > 0.05:
                scores['bull'] += 25
            elif return_50d < -0.10:
                scores['bear'] += 40
            elif return_50d < -0.05:
                scores['bear'] += 25
            else:
                scores['low_vol'] += 20
            
            # Short-term momentum
            if return_20d > 0.05:
                scores['bull'] += 20
            elif return_20d < -0.05:
                scores['bear'] += 20
            
            # Volatility regime
            if volatility > 0.35:
                scores['high_vol'] += 35
            elif volatility > 0.25:
                scores['high_vol'] += 20
            elif volatility < 0.15:
                scores['low_vol'] += 30
            elif volatility < 0.20:
                scores['low_vol'] += 15
            
            # Volatility expansion (recent increase)
            if vol_ratio > 1.5:
                scores['high_vol'] += 20
            elif vol_ratio < 0.7:
                scores['low_vol'] += 15
            
            # RSI extremes
            if rsi > 70:
                scores['bull'] += 10
            elif rsi < 30:
                scores['bear'] += 10
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            probs = {k: (v / total) * 100 for k, v in scores.items()}
        else:
            probs = {'bull': 25, 'bear': 25, 'high_vol': 25, 'low_vol': 25, 'crash': 0}
        
        # Determine regime
        max_regime = max(scores, key=scores.get)
        regime_map = {
            'bull': 'BULL_TREND',
            'bear': 'BEAR_TREND',
            'high_vol': 'HIGH_VOLATILITY',
            'low_vol': 'LOW_VOLATILITY',
            'crash': 'CRASH'
        }
        
        regime = regime_map[max_regime]
        confidence = probs[max_regime]
        
        return regime, round(confidence, 1), probs
    
    def _calculate_regime_duration(self, df: pd.DataFrame, current_regime: str) -> int:
        """Estimate how long we've been in the current regime."""
        if len(df) < self.trend_window + 20:
            return 0
        
        # Look back to find when regime started
        for i in range(len(df) - self.trend_window - 1, max(0, len(df) - 252), -5):
            temp_df = df.iloc[:i+1]
            if len(temp_df) < self.trend_window:
                break
            
            features = self._calculate_features(temp_df)
            regime, _, _ = self._classify_regime(features)
            
            if regime != current_regime:
                return len(df) - i - 1
        
        return min(len(df), 252)  # Default to max lookback
    
    def _estimate_transition(
        self,
        features: Dict[str, float],
        current_regime: str,
        duration: int
    ) -> Tuple[float, str]:
        """Estimate probability of regime change and likely next regime."""
        
        # Base transition probability increases with duration
        base_prob = min(50, duration / 10)  # Increases over time
        
        # Adjust based on regime
        if current_regime == 'CRASH':
            # Crashes typically short-lived
            prob_change = min(80, base_prob + 30)
            next_regime = 'HIGH_VOLATILITY'
        elif current_regime == 'HIGH_VOLATILITY':
            prob_change = min(60, base_prob + 15)
            # Check trend direction
            if features.get('return_20d', 0) > 0:
                next_regime = 'BULL_TREND'
            else:
                next_regime = 'BEAR_TREND'
        elif current_regime == 'BULL_TREND':
            # Bull markets can last long
            prob_change = max(10, base_prob - 10)
            if features.get('volatility_ratio', 1) > 1.3:
                next_regime = 'HIGH_VOLATILITY'
            else:
                next_regime = 'BEAR_TREND'
        elif current_regime == 'BEAR_TREND':
            prob_change = min(60, base_prob + 10)
            if features.get('volatility_ratio', 1) > 1.3:
                next_regime = 'HIGH_VOLATILITY'
            else:
                next_regime = 'BULL_TREND'
        else:  # LOW_VOLATILITY
            prob_change = base_prob
            if features.get('volatility_ratio', 1) > 1.2:
                next_regime = 'HIGH_VOLATILITY'
            elif features.get('return_20d', 0) > 0.03:
                next_regime = 'BULL_TREND'
            else:
                next_regime = 'BEAR_TREND'
        
        return round(prob_change, 1), next_regime
    
    def _assign_regime_names(self, cluster_means: pd.DataFrame) -> List[str]:
        """Assign interpretable names to GMM clusters."""
        names = []
        
        for i in range(len(cluster_means)):
            mean_return = cluster_means.loc[i, 'returns']
            mean_vol = cluster_means.loc[i, 'volatility']
            
            if mean_return > 0.5:  # Standardized
                if mean_vol > 0.5:
                    names.append('HIGH_VOLATILITY')
                else:
                    names.append('BULL_TREND')
            elif mean_return < -0.5:
                if mean_vol > 1.0:
                    names.append('CRASH')
                else:
                    names.append('BEAR_TREND')
            else:
                if mean_vol < -0.5:
                    names.append('LOW_VOLATILITY')
                else:
                    names.append('HIGH_VOLATILITY')
        
        # Ensure unique names
        seen = {}
        for i, name in enumerate(names):
            if name in seen:
                names[i] = f"{name}_{seen[name]}"
                seen[name] += 1
            else:
                seen[name] = 1
        
        return names
    
    def _unknown_regime(self) -> RegimeState:
        """Return unknown regime state."""
        return RegimeState(
            regime="UNKNOWN",
            confidence=0,
            duration_days=0,
            bull_probability=25,
            bear_probability=25,
            high_vol_probability=25,
            low_vol_probability=25,
            crash_probability=0,
            probability_of_change=50,
            most_likely_next_regime="UNKNOWN"
        )
    
    def get_regime_implications(self, regime: RegimeState) -> Dict[str, Any]:
        """
        Get trading implications for the current regime.
        
        Args:
            regime: Current RegimeState
        
        Returns:
            Dict with trading recommendations
        """
        implications = {
            'BULL_TREND': {
                'position_bias': 'LONG',
                'position_size': 'NORMAL to LARGE',
                'stop_loss_distance': 'WIDER',
                'take_profit': 'TRAIL or SCALE OUT',
                'strategies': ['Trend following', 'Breakout buying', 'Buy the dip'],
                'avoid': ['Shorting', 'Mean reversion']
            },
            'BEAR_TREND': {
                'position_bias': 'REDUCE EXPOSURE or SHORT',
                'position_size': 'SMALLER',
                'stop_loss_distance': 'TIGHTER',
                'take_profit': 'QUICK PROFITS',
                'strategies': ['Short selling', 'Put options', 'Cash'],
                'avoid': ['Buying dips too early', 'Large long positions']
            },
            'HIGH_VOLATILITY': {
                'position_bias': 'NEUTRAL or REDUCED',
                'position_size': 'SMALLER (half normal)',
                'stop_loss_distance': 'WIDER (2x ATR)',
                'take_profit': 'QUICK EXITS',
                'strategies': ['Options straddles', 'Scalping', 'Mean reversion'],
                'avoid': ['Large positions', 'Tight stops']
            },
            'LOW_VOLATILITY': {
                'position_bias': 'NEUTRAL',
                'position_size': 'NORMAL',
                'stop_loss_distance': 'NORMAL',
                'take_profit': 'RANGE TARGETS',
                'strategies': ['Range trading', 'Options selling', 'Covered calls'],
                'avoid': ['Breakout trades', 'Large directional bets']
            },
            'CRASH': {
                'position_bias': 'DEFENSIVE',
                'position_size': 'MINIMAL',
                'stop_loss_distance': 'VERY WIDE or NO STOPS',
                'take_profit': 'TAKE PROFITS EARLY',
                'strategies': ['Cash', 'Put options', 'VIX calls'],
                'avoid': ['Buying', 'Averaging down', 'Leverage']
            }
        }
        
        default = {
            'position_bias': 'NEUTRAL',
            'position_size': 'NORMAL',
            'stop_loss_distance': 'NORMAL',
            'take_profit': 'NORMAL',
            'strategies': ['Wait for clarity'],
            'avoid': ['Large positions']
        }
        
        return implications.get(regime.regime, default)
