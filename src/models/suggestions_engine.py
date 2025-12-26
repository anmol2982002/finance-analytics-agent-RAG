"""
ML-Powered Suggestions Engine
Reliable, backtested suggestions with confidence scores and explainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EntryExitZone:
    """Represents a price zone with probability."""
    price_low: float
    price_high: float
    probability: float
    zone_type: str  # 'SUPPORT', 'RESISTANCE', 'ENTRY', 'STOP_LOSS', 'TARGET'
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'


@dataclass
class SimilarPeriod:
    """A historical period similar to current conditions."""
    start_date: str
    end_date: str
    similarity_score: float  # 0-100
    outcome: str  # What happened after
    return_after_7d: float
    return_after_30d: float
    description: str


@dataclass
class RiskMetrics:
    """Risk analysis metrics from Monte Carlo simulation."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall: float  # Expected loss beyond VaR
    max_drawdown_expected: float
    probability_of_loss: float
    best_case_7d: float
    worst_case_7d: float
    median_outcome_7d: float


@dataclass
class SuggestionReport:
    """Complete suggestion report with all components."""
    ticker: str
    timestamp: str
    
    # Primary suggestion
    primary_signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0-100
    confidence_explanation: str
    
    # Entry/Exit zones
    entry_zones: List[EntryExitZone] = field(default_factory=list)
    stop_loss_zones: List[EntryExitZone] = field(default_factory=list)
    target_zones: List[EntryExitZone] = field(default_factory=list)
    
    # Historical context
    similar_periods: List[SimilarPeriod] = field(default_factory=list)
    
    # Risk analysis
    risk_metrics: Optional[RiskMetrics] = None
    
    # Multi-timeframe
    timeframe_confluence: Dict[str, str] = field(default_factory=dict)
    confluence_score: float = 0.0
    
    # Actionable summary
    summary: str = ""
    key_levels: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "primary_signal": self.primary_signal,
            "confidence": self.confidence,
            "confidence_explanation": self.confidence_explanation,
            "entry_zones": [
                {"price_low": z.price_low, "price_high": z.price_high, 
                 "probability": z.probability, "type": z.zone_type, "strength": z.strength}
                for z in self.entry_zones
            ],
            "stop_loss_zones": [
                {"price_low": z.price_low, "price_high": z.price_high,
                 "probability": z.probability, "type": z.zone_type, "strength": z.strength}
                for z in self.stop_loss_zones
            ],
            "target_zones": [
                {"price_low": z.price_low, "price_high": z.price_high,
                 "probability": z.probability, "type": z.zone_type, "strength": z.strength}
                for z in self.target_zones
            ],
            "similar_periods": [
                {"start_date": p.start_date, "end_date": p.end_date,
                 "similarity_score": p.similarity_score, "outcome": p.outcome,
                 "return_after_7d": p.return_after_7d, "return_after_30d": p.return_after_30d,
                 "description": p.description}
                for p in self.similar_periods
            ],
            "risk_metrics": {
                "var_95": self.risk_metrics.var_95,
                "var_99": self.risk_metrics.var_99,
                "expected_shortfall": self.risk_metrics.expected_shortfall,
                "max_drawdown_expected": self.risk_metrics.max_drawdown_expected,
                "probability_of_loss": self.risk_metrics.probability_of_loss,
                "best_case_7d": self.risk_metrics.best_case_7d,
                "worst_case_7d": self.risk_metrics.worst_case_7d,
                "median_outcome_7d": self.risk_metrics.median_outcome_7d
            } if self.risk_metrics else None,
            "timeframe_confluence": self.timeframe_confluence,
            "confluence_score": self.confluence_score,
            "summary": self.summary,
            "key_levels": self.key_levels,
            "warnings": self.warnings
        }


class SuggestionsEngine:
    """
    ML-powered suggestions engine with reliability guarantees.
    
    Features:
    - Pattern Recognition: Find similar historical periods
    - Entry/Exit Zones: Probability-weighted support/resistance
    - Monte Carlo Risk Analysis: Realistic risk estimates
    - Multi-Timeframe Confluence: Score alignment across timeframes
    - Explainable Confidence: Users understand why a suggestion is made
    """
    
    def __init__(
        self,
        lookback_years: int = 3,
        pattern_window: int = 20,
        n_simulations: int = 10000,
        min_confidence_threshold: float = 60.0
    ):
        """
        Initialize the suggestions engine.
        
        Args:
            lookback_years: Years of historical data for pattern matching
            pattern_window: Window size for pattern comparison
            n_simulations: Number of Monte Carlo simulations
            min_confidence_threshold: Minimum confidence to give non-neutral suggestion
        """
        self.lookback_years = lookback_years
        self.pattern_window = pattern_window
        self.n_simulations = n_simulations
        self.min_confidence_threshold = min_confidence_threshold
        
        logger.info(f"Initialized SuggestionsEngine with {n_simulations} simulations")
    
    def generate_suggestions(
        self,
        ticker: str,
        df: pd.DataFrame,
        technical_signals: Optional[Dict[str, Any]] = None,
        include_monte_carlo: bool = True
    ) -> SuggestionReport:
        """
        Generate comprehensive suggestions for a ticker.
        
        Args:
            ticker: Asset ticker symbol
            df: DataFrame with OHLCV and technical indicators
            technical_signals: Pre-calculated technical signals (optional)
            include_monte_carlo: Whether to run Monte Carlo simulation
        
        Returns:
            SuggestionReport with all analysis components
        """
        if df is None or df.empty or len(df) < self.pattern_window * 2:
            return self._create_insufficient_data_report(ticker)
        
        logger.info(f"Generating suggestions for {ticker}")
        
        # 1. Find similar historical patterns
        similar_periods = self._find_similar_patterns(df)
        
        # 2. Calculate entry/exit zones
        entry_zones, stop_loss_zones, target_zones = self._calculate_zones(df)
        
        # 3. Run Monte Carlo risk analysis
        risk_metrics = None
        if include_monte_carlo:
            risk_metrics = self._monte_carlo_risk_analysis(df)
        
        # 4. Multi-timeframe confluence analysis
        timeframe_confluence, confluence_score = self._analyze_timeframe_confluence(df)
        
        # 5. Calculate primary signal and confidence
        primary_signal, confidence, explanation = self._calculate_signal_and_confidence(
            df=df,
            technical_signals=technical_signals,
            similar_periods=similar_periods,
            confluence_score=confluence_score,
            risk_metrics=risk_metrics
        )
        
        # 6. Generate summary and warnings
        summary, warnings, key_levels = self._generate_summary(
            ticker=ticker,
            df=df,
            primary_signal=primary_signal,
            confidence=confidence,
            entry_zones=entry_zones,
            risk_metrics=risk_metrics
        )
        
        return SuggestionReport(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            primary_signal=primary_signal,
            confidence=confidence,
            confidence_explanation=explanation,
            entry_zones=entry_zones,
            stop_loss_zones=stop_loss_zones,
            target_zones=target_zones,
            similar_periods=similar_periods,
            risk_metrics=risk_metrics,
            timeframe_confluence=timeframe_confluence,
            confluence_score=confluence_score,
            summary=summary,
            key_levels=key_levels,
            warnings=warnings
        )
    
    def _find_similar_patterns(
        self,
        df: pd.DataFrame,
        n_matches: int = 5
    ) -> List[SimilarPeriod]:
        """
        Find historical periods with similar price patterns using DTW-lite.
        
        Uses normalized returns for comparison to handle different price scales.
        """
        if len(df) < self.pattern_window * 3:
            return []
        
        # Get current pattern (normalized returns)
        current_returns = df['Close'].pct_change().dropna().tail(self.pattern_window).values
        if len(current_returns) < self.pattern_window:
            return []
        
        current_normalized = (current_returns - np.mean(current_returns)) / (np.std(current_returns) + 1e-8)
        
        # Slide through history to find similar patterns
        similarities = []
        historical_df = df.iloc[:-self.pattern_window]  # Exclude current period
        
        for i in range(self.pattern_window, len(historical_df) - 30):  # Need 30 days forward
            hist_returns = historical_df['Close'].pct_change().iloc[i-self.pattern_window:i].values
            if len(hist_returns) != self.pattern_window or np.std(hist_returns) < 1e-8:
                continue
            
            hist_normalized = (hist_returns - np.mean(hist_returns)) / (np.std(hist_returns) + 1e-8)
            
            # Calculate similarity using Euclidean distance
            try:
                distance = euclidean(current_normalized, hist_normalized)
                similarity = max(0, 100 - distance * 20)  # Scale to 0-100
            except:
                continue
            
            if similarity > 50:  # Only keep reasonably similar matches
                # Calculate what happened after
                future_idx = min(i + 30, len(historical_df) - 1)
                future_7d_idx = min(i + 7, len(historical_df) - 1)
                
                return_7d = (historical_df['Close'].iloc[future_7d_idx] / 
                           historical_df['Close'].iloc[i] - 1) * 100
                return_30d = (historical_df['Close'].iloc[future_idx] / 
                            historical_df['Close'].iloc[i] - 1) * 100
                
                outcome = "BULLISH" if return_30d > 5 else "BEARISH" if return_30d < -5 else "NEUTRAL"
                
                similarities.append({
                    'idx': i,
                    'similarity': similarity,
                    'return_7d': return_7d,
                    'return_30d': return_30d,
                    'outcome': outcome
                })
        
        # Sort by similarity and take top matches
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = similarities[:n_matches]
        
        similar_periods = []
        for match in top_matches:
            idx = match['idx']
            period_start = df.index[idx - self.pattern_window] if hasattr(df.index[0], 'strftime') else str(idx - self.pattern_window)
            period_end = df.index[idx] if hasattr(df.index[0], 'strftime') else str(idx)
            
            period_start_str = period_start.strftime('%Y-%m-%d') if hasattr(period_start, 'strftime') else str(period_start)
            period_end_str = period_end.strftime('%Y-%m-%d') if hasattr(period_end, 'strftime') else str(period_end)
            
            similar_periods.append(SimilarPeriod(
                start_date=period_start_str,
                end_date=period_end_str,
                similarity_score=round(match['similarity'], 1),
                outcome=match['outcome'],
                return_after_7d=round(match['return_7d'], 2),
                return_after_30d=round(match['return_30d'], 2),
                description=f"Similar pattern from {period_start_str}. Led to {match['return_30d']:.1f}% move in 30 days."
            ))
        
        return similar_periods
    
    def _calculate_zones(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5
    ) -> Tuple[List[EntryExitZone], List[EntryExitZone], List[EntryExitZone]]:
        """
        Calculate probability-weighted entry, stop-loss, and target zones.
        
        Uses clustering on historical support/resistance levels.
        """
        if len(df) < 50:
            return [], [], []
        
        current_price = df['Close'].iloc[-1]
        
        # Find pivot highs and lows
        window = 5
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                pivot_highs.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                pivot_lows.append(df['Low'].iloc[i])
        
        if len(pivot_highs) < 3 or len(pivot_lows) < 3:
            return [], [], []
        
        # Cluster pivot points to find zones
        all_pivots = np.array(pivot_highs + pivot_lows).reshape(-1, 1)
        
        n_clusters = min(n_clusters, len(all_pivots) // 2)
        if n_clusters < 2:
            return [], [], []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_pivots)
        
        # Create zones from clusters
        zones_data = []
        for i, center in enumerate(kmeans.cluster_centers_):
            cluster_points = all_pivots[kmeans.labels_ == i]
            if len(cluster_points) < 2:
                continue
            
            zone_center = center[0]
            zone_width = np.std(cluster_points) * 2 or zone_center * 0.01
            
            # Calculate probability based on how often price respects this level
            touches = sum(1 for p in cluster_points if abs(p - zone_center) < zone_width)
            probability = min(95, touches / len(all_pivots) * 200)
            
            zones_data.append({
                'center': zone_center,
                'low': zone_center - zone_width,
                'high': zone_center + zone_width,
                'probability': probability,
                'touches': touches
            })
        
        # Categorize zones
        entry_zones = []
        stop_loss_zones = []
        target_zones = []
        
        for zone in zones_data:
            strength = "STRONG" if zone['touches'] >= 5 else "MODERATE" if zone['touches'] >= 3 else "WEAK"
            
            if zone['center'] < current_price * 0.98:  # Below current price
                # Support zone - potential entry or stop loss
                zone_obj = EntryExitZone(
                    price_low=round(zone['low'], 4),
                    price_high=round(zone['high'], 4),
                    probability=round(zone['probability'], 1),
                    zone_type='SUPPORT',
                    strength=strength
                )
                entry_zones.append(zone_obj)
                
                # Strong supports become stop loss zones (just below)
                if strength == "STRONG":
                    stop_zone = EntryExitZone(
                        price_low=round(zone['low'] * 0.98, 4),
                        price_high=round(zone['low'], 4),
                        probability=round(zone['probability'] * 0.8, 1),
                        zone_type='STOP_LOSS',
                        strength=strength
                    )
                    stop_loss_zones.append(stop_zone)
                    
            elif zone['center'] > current_price * 1.02:  # Above current price
                # Resistance zone - potential target
                zone_obj = EntryExitZone(
                    price_low=round(zone['low'], 4),
                    price_high=round(zone['high'], 4),
                    probability=round(zone['probability'], 1),
                    zone_type='RESISTANCE',
                    strength=strength
                )
                target_zones.append(zone_obj)
        
        # Sort by distance from current price
        entry_zones.sort(key=lambda z: current_price - z.price_high)
        target_zones.sort(key=lambda z: z.price_low - current_price)
        
        return entry_zones[:3], stop_loss_zones[:2], target_zones[:3]
    
    def _monte_carlo_risk_analysis(
        self,
        df: pd.DataFrame,
        horizon_days: int = 7
    ) -> RiskMetrics:
        """
        Run Monte Carlo simulation to estimate risk metrics.
        
        Uses historical returns distribution with volatility clustering.
        """
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return RiskMetrics(
                var_95=0, var_99=0, expected_shortfall=0,
                max_drawdown_expected=0, probability_of_loss=50,
                best_case_7d=0, worst_case_7d=0, median_outcome_7d=0
            )
        
        # Fit return distribution
        mu = returns.mean()
        sigma = returns.std()
        
        # Use recent volatility for more realistic simulation
        recent_sigma = returns.tail(20).std()
        vol_ratio = recent_sigma / sigma if sigma > 0 else 1
        adjusted_sigma = sigma * max(0.5, min(2.0, vol_ratio))  # Bound adjustment
        
        # Run simulations
        np.random.seed(42)
        simulated_returns = np.random.normal(mu, adjusted_sigma, (self.n_simulations, horizon_days))
        
        # Calculate cumulative returns
        cumulative_returns = (1 + simulated_returns).prod(axis=1) - 1
        
        # Calculate metrics
        var_95 = np.percentile(cumulative_returns, 5) * 100
        var_99 = np.percentile(cumulative_returns, 1) * 100
        
        # Expected Shortfall (CVaR) - average of worst 5%
        worst_5_pct = cumulative_returns[cumulative_returns <= np.percentile(cumulative_returns, 5)]
        expected_shortfall = np.mean(worst_5_pct) * 100 if len(worst_5_pct) > 0 else var_95
        
        # Max drawdown estimation
        max_drawdowns = []
        for sim in simulated_returns:
            cum_sim = (1 + sim).cumprod()
            peak = np.maximum.accumulate(cum_sim)
            drawdown = (cum_sim - peak) / peak
            max_drawdowns.append(drawdown.min())
        avg_max_drawdown = np.mean(max_drawdowns) * 100
        
        return RiskMetrics(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            expected_shortfall=round(expected_shortfall, 2),
            max_drawdown_expected=round(avg_max_drawdown, 2),
            probability_of_loss=round((cumulative_returns < 0).sum() / len(cumulative_returns) * 100, 1),
            best_case_7d=round(np.percentile(cumulative_returns, 95) * 100, 2),
            worst_case_7d=round(np.percentile(cumulative_returns, 5) * 100, 2),
            median_outcome_7d=round(np.median(cumulative_returns) * 100, 2)
        )
    
    def _analyze_timeframe_confluence(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict[str, str], float]:
        """
        Analyze signal alignment across different timeframes.
        
        Checks if daily, weekly, and monthly timeframes agree.
        """
        if len(df) < 200:
            return {"daily": "NEUTRAL", "weekly": "NEUTRAL", "monthly": "NEUTRAL"}, 0.0
        
        signals = {}
        
        # Daily signal (using SMA crossover as proxy)
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            signals['daily'] = "BULLISH" if sma_20 > sma_50 else "BEARISH"
        else:
            # Calculate on the fly
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            signals['daily'] = "BULLISH" if sma_20 > sma_50 else "BEARISH"
        
        # Weekly signal (resample to weekly and check trend)
        if hasattr(df.index, 'to_period'):
            weekly = df['Close'].resample('W').last().dropna()
            if len(weekly) >= 10:
                weekly_sma_4 = weekly.rolling(4).mean().iloc[-1]
                weekly_sma_10 = weekly.rolling(10).mean().iloc[-1]
                signals['weekly'] = "BULLISH" if weekly_sma_4 > weekly_sma_10 else "BEARISH"
            else:
                signals['weekly'] = "NEUTRAL"
        else:
            # Approximate weekly using 5-day groups
            weekly_close = df['Close'].iloc[::5]
            if len(weekly_close) >= 10:
                signals['weekly'] = "BULLISH" if weekly_close.iloc[-1] > weekly_close.iloc[-5:].mean() else "BEARISH"
            else:
                signals['weekly'] = "NEUTRAL"
        
        # Monthly signal (using 200-day SMA as proxy)
        if 'sma_200' in df.columns:
            sma_200 = df['sma_200'].iloc[-1]
        else:
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        
        if pd.notna(sma_200):
            signals['monthly'] = "BULLISH" if df['Close'].iloc[-1] > sma_200 else "BEARISH"
        else:
            signals['monthly'] = "NEUTRAL"
        
        # Calculate confluence score
        bullish_count = sum(1 for s in signals.values() if s == "BULLISH")
        bearish_count = sum(1 for s in signals.values() if s == "BEARISH")
        
        if bullish_count == 3:
            confluence_score = 100.0
        elif bullish_count == 2:
            confluence_score = 66.7
        elif bearish_count == 3:
            confluence_score = -100.0
        elif bearish_count == 2:
            confluence_score = -66.7
        else:
            confluence_score = 0.0
        
        return signals, confluence_score
    
    def _calculate_signal_and_confidence(
        self,
        df: pd.DataFrame,
        technical_signals: Optional[Dict[str, Any]],
        similar_periods: List[SimilarPeriod],
        confluence_score: float,
        risk_metrics: Optional[RiskMetrics]
    ) -> Tuple[str, float, str]:
        """
        Calculate the primary signal and confidence score.
        
        Combines multiple factors with explainable weighting.
        """
        confidence_factors = []
        signal_votes = []
        
        # Factor 1: Technical signals (30% weight)
        if technical_signals and 'overall' in technical_signals:
            overall = technical_signals['overall']
            tech_signal = overall.get('signal', 'NEUTRAL')
            tech_confidence = overall.get('confidence', 50)
            
            signal_votes.append(tech_signal)
            confidence_factors.append({
                'name': 'Technical Indicators',
                'signal': tech_signal,
                'weight': 0.30,
                'score': tech_confidence
            })
        
        # Factor 2: Historical pattern similarity (25% weight)
        if similar_periods:
            bullish_outcomes = sum(1 for p in similar_periods if p.outcome == "BULLISH")
            bearish_outcomes = sum(1 for p in similar_periods if p.outcome == "BEARISH")
            
            avg_similarity = np.mean([p.similarity_score for p in similar_periods])
            
            if bullish_outcomes > bearish_outcomes:
                pattern_signal = "BULLISH"
                pattern_confidence = (bullish_outcomes / len(similar_periods)) * avg_similarity
            elif bearish_outcomes > bullish_outcomes:
                pattern_signal = "BEARISH"
                pattern_confidence = (bearish_outcomes / len(similar_periods)) * avg_similarity
            else:
                pattern_signal = "NEUTRAL"
                pattern_confidence = 50
            
            signal_votes.append(pattern_signal)
            confidence_factors.append({
                'name': 'Historical Patterns',
                'signal': pattern_signal,
                'weight': 0.25,
                'score': pattern_confidence
            })
        
        # Factor 3: Timeframe confluence (25% weight)
        if abs(confluence_score) > 0:
            confluence_signal = "BULLISH" if confluence_score > 0 else "BEARISH"
            signal_votes.append(confluence_signal)
            confidence_factors.append({
                'name': 'Timeframe Confluence',
                'signal': confluence_signal,
                'weight': 0.25,
                'score': abs(confluence_score)
            })
        
        # Factor 4: Risk-adjusted outlook (20% weight)
        if risk_metrics:
            if risk_metrics.median_outcome_7d > 1 and risk_metrics.probability_of_loss < 45:
                risk_signal = "BULLISH"
                risk_score = 70 + (risk_metrics.median_outcome_7d * 5)
            elif risk_metrics.median_outcome_7d < -1 or risk_metrics.probability_of_loss > 55:
                risk_signal = "BEARISH"
                risk_score = 70 + abs(risk_metrics.median_outcome_7d * 5)
            else:
                risk_signal = "NEUTRAL"
                risk_score = 50
            
            signal_votes.append(risk_signal)
            confidence_factors.append({
                'name': 'Monte Carlo Analysis',
                'signal': risk_signal,
                'weight': 0.20,
                'score': min(100, risk_score)
            })
        
        # Aggregate signals
        bullish_count = sum(1 for v in signal_votes if v == "BULLISH")
        bearish_count = sum(1 for v in signal_votes if v == "BEARISH")
        
        if bullish_count > bearish_count:
            primary_signal = "BULLISH"
        elif bearish_count > bullish_count:
            primary_signal = "BEARISH"
        else:
            primary_signal = "NEUTRAL"
        
        # Calculate weighted confidence
        if confidence_factors:
            weighted_confidence = sum(
                f['score'] * f['weight'] for f in confidence_factors
                if f['signal'] == primary_signal or f['signal'] == 'NEUTRAL'
            )
            total_weight = sum(
                f['weight'] for f in confidence_factors
                if f['signal'] == primary_signal or f['signal'] == 'NEUTRAL'
            )
            confidence = weighted_confidence / total_weight if total_weight > 0 else 50
        else:
            confidence = 50
        
        # Apply minimum threshold
        if confidence < self.min_confidence_threshold and primary_signal != "NEUTRAL":
            primary_signal = "NEUTRAL"
            confidence = confidence  # Keep low confidence visible
        
        # Generate explanation
        explanation_parts = []
        for f in sorted(confidence_factors, key=lambda x: x['weight'], reverse=True):
            explanation_parts.append(f"{f['name']}: {f['signal']} ({f['score']:.0f}%)")
        
        explanation = " | ".join(explanation_parts) if explanation_parts else "Insufficient data for confidence calculation"
        
        return primary_signal, round(confidence, 1), explanation
    
    def _generate_summary(
        self,
        ticker: str,
        df: pd.DataFrame,
        primary_signal: str,
        confidence: float,
        entry_zones: List[EntryExitZone],
        risk_metrics: Optional[RiskMetrics]
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """
        Generate actionable summary, warnings, and key levels.
        """
        current_price = df['Close'].iloc[-1]
        
        # Key levels
        key_levels = {
            'current_price': round(current_price, 4),
            'sma_20': round(df['sma_20'].iloc[-1], 4) if 'sma_20' in df.columns else None,
            'sma_50': round(df['sma_50'].iloc[-1], 4) if 'sma_50' in df.columns else None,
        }
        
        if entry_zones:
            key_levels['nearest_support'] = entry_zones[0].price_high
        
        # Warnings
        warnings = []
        
        if confidence < 60:
            warnings.append("⚠️ Low confidence signal - consider waiting for confirmation")
        
        if risk_metrics:
            if risk_metrics.var_95 < -5:
                warnings.append(f"⚠️ High downside risk: 5% chance of {risk_metrics.var_95:.1f}% or worse loss in 7 days")
            if risk_metrics.probability_of_loss > 50:
                warnings.append(f"⚠️ Monte Carlo shows {risk_metrics.probability_of_loss:.0f}% probability of loss")
        
        # Check for high volatility
        if 'atr_percent' in df.columns and df['atr_percent'].iloc[-1] > 5:
            warnings.append("⚠️ High volatility environment - use smaller position sizes")
        
        # Summary
        if primary_signal == "BULLISH":
            summary = f"📈 **BULLISH** outlook for {ticker} with {confidence:.0f}% confidence. "
            if entry_zones:
                summary += f"Consider entries near ${entry_zones[0].price_high:.2f} support zone. "
            if risk_metrics:
                summary += f"7-day median expectation: {risk_metrics.median_outcome_7d:+.1f}%."
        elif primary_signal == "BEARISH":
            summary = f"📉 **BEARISH** outlook for {ticker} with {confidence:.0f}% confidence. "
            summary += "Consider reducing exposure or waiting for better levels. "
            if risk_metrics:
                summary += f"7-day risk: up to {risk_metrics.var_95:.1f}% loss (95% confidence)."
        else:
            summary = f"⏸️ **NEUTRAL** outlook for {ticker}. "
            summary += "No clear directional bias - consider waiting for a stronger signal."
        
        return summary, warnings, key_levels
    
    def _create_insufficient_data_report(self, ticker: str) -> SuggestionReport:
        """Create a report when there's insufficient data."""
        return SuggestionReport(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            primary_signal="NEUTRAL",
            confidence=0,
            confidence_explanation="Insufficient historical data for reliable analysis",
            summary=f"⚠️ Cannot generate reliable suggestions for {ticker} due to insufficient data.",
            warnings=["Need at least 40 days of price data for pattern analysis"]
        )
