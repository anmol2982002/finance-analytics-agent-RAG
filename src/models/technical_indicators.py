"""
Technical Analysis Indicators
Comprehensive technical analysis toolkit using the 'ta' library
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from src.utils import get_logger

logger = get_logger(__name__)


class TechnicalAnalyzer:
    """
    Calculates and analyzes technical indicators.
    
    Features:
    - Trend indicators (SMA, EMA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, Williams %R)
    - Volatility indicators (Bollinger Bands, ATR, Keltner)
    - Volume indicators (OBV, VWAP, MFI)
    - Signal generation
    """
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        
        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            include_volume: Whether to include volume-based indicators
        
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return df
        
        # === TREND INDICATORS ===
        
        # Simple Moving Averages
        df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # Exponential Moving Averages
        df['ema_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_21'] = ta.trend.ema_indicator(df['Close'], window=21)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(
            df['High'], df['Low'], df['Close']
        ).psar()
        
        # === MOMENTUM INDICATORS ===
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(
            df['High'], df['Low'], df['Close']
        )
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Rate of Change
        df['roc'] = ta.momentum.roc(df['Close'], window=12)
        
        # === VOLATILITY INDICATORS ===
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pband'] = bollinger.bollinger_pband()  # Price position in band
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close']
        )
        df['atr_percent'] = (df['atr'] / df['Close']) * 100
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(
            df['High'], df['Low'], df['Close']
        )
        df['keltner_upper'] = keltner.keltner_channel_hband()
        df['keltner_middle'] = keltner.keltner_channel_mband()
        df['keltner_lower'] = keltner.keltner_channel_lband()
        
        # === VOLUME INDICATORS ===
        if include_volume and 'Volume' in df.columns:
            # On-Balance Volume
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Volume SMA
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            
            # Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(
                df['High'], df['Low'], df['Close'], df['Volume']
            )
            
            # VWAP (approximate for daily data)
            df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Accumulation/Distribution
            df['ad'] = ta.volume.acc_dist_index(
                df['High'], df['Low'], df['Close'], df['Volume']
            )
        
        # === ADDITIONAL METRICS ===
        
        # Price changes
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility (rolling std of returns)
        df['volatility_20'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        logger.info(f"Calculated {len(df.columns)} indicators")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with indicators calculated
        
        Returns:
            Dict with signals and their interpretations
        """
        if df.empty:
            return {"error": "Empty DataFrame"}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        signals = {}
        
        # === TREND SIGNALS ===
        signals['trend'] = self._analyze_trend(latest, prev)
        
        # === MOMENTUM SIGNALS ===
        signals['momentum'] = self._analyze_momentum(latest, prev)
        
        # === VOLATILITY SIGNALS ===
        signals['volatility'] = self._analyze_volatility(latest, df)
        
        # === VOLUME SIGNALS ===
        if 'volume_ratio' in latest.index:
            signals['volume'] = self._analyze_volume(latest)
        
        # === OVERALL SIGNAL ===
        signals['overall'] = self._calculate_overall_signal(signals)
        
        return signals
    
    def _analyze_trend(self, latest: pd.Series, prev: pd.Series) -> Dict[str, Any]:
        """Analyze trend indicators"""
        result = {
            "moving_averages": {},
            "macd": {},
            "adx": {},
            "overall_trend": "NEUTRAL"
        }
        
        # Moving Average Analysis
        if 'sma_20' in latest.index and 'sma_50' in latest.index and 'sma_200' in latest.index:
            if latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
                result['moving_averages'] = {
                    "signal": "BULLISH",
                    "description": "Price above all major MAs - Strong uptrend"
                }
            elif latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
                result['moving_averages'] = {
                    "signal": "BEARISH",
                    "description": "Price below all major MAs - Strong downtrend"
                }
            else:
                result['moving_averages'] = {
                    "signal": "NEUTRAL",
                    "description": "Mixed MA signals - Consolidation"
                }
        
        # MACD Analysis
        if 'macd' in latest.index and 'macd_signal' in latest.index:
            macd_cross = (
                prev['macd'] <= prev['macd_signal'] and 
                latest['macd'] > latest['macd_signal']
            )
            macd_cross_down = (
                prev['macd'] >= prev['macd_signal'] and 
                latest['macd'] < latest['macd_signal']
            )
            
            if macd_cross:
                result['macd'] = {
                    "signal": "BULLISH",
                    "description": "MACD bullish crossover"
                }
            elif macd_cross_down:
                result['macd'] = {
                    "signal": "BEARISH",
                    "description": "MACD bearish crossover"
                }
            else:
                result['macd'] = {
                    "signal": "BULLISH" if latest['macd'] > latest['macd_signal'] else "BEARISH",
                    "description": f"MACD {'above' if latest['macd'] > latest['macd_signal'] else 'below'} signal line"
                }
        
        # ADX Analysis
        if 'adx' in latest.index:
            trend_strength = "Strong" if latest['adx'] > 25 else "Weak"
            result['adx'] = {
                "value": round(latest['adx'], 2),
                "strength": trend_strength,
                "direction": "BULLISH" if latest.get('adx_pos', 0) > latest.get('adx_neg', 0) else "BEARISH"
            }
        
        # Overall trend
        bullish_count = sum(1 for v in result.values() 
                          if isinstance(v, dict) and v.get('signal') == 'BULLISH')
        bearish_count = sum(1 for v in result.values() 
                          if isinstance(v, dict) and v.get('signal') == 'BEARISH')
        
        if bullish_count > bearish_count:
            result['overall_trend'] = "BULLISH"
        elif bearish_count > bullish_count:
            result['overall_trend'] = "BEARISH"
        
        return result
    
    def _analyze_momentum(self, latest: pd.Series, prev: pd.Series) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        result = {}
        
        # RSI Analysis
        if 'rsi' in latest.index:
            rsi = latest['rsi']
            if rsi > 70:
                result['rsi'] = {
                    "value": round(rsi, 2),
                    "signal": "OVERBOUGHT",
                    "description": "RSI above 70 - Consider taking profits"
                }
            elif rsi < 30:
                result['rsi'] = {
                    "value": round(rsi, 2),
                    "signal": "OVERSOLD",
                    "description": "RSI below 30 - Potential buying opportunity"
                }
            else:
                result['rsi'] = {
                    "value": round(rsi, 2),
                    "signal": "NEUTRAL",
                    "description": "RSI in neutral zone"
                }
        
        # Stochastic Analysis
        if 'stoch_k' in latest.index and 'stoch_d' in latest.index:
            stoch_k = latest['stoch_k']
            stoch_d = latest['stoch_d']
            
            cross_up = prev['stoch_k'] <= prev['stoch_d'] and stoch_k > stoch_d
            cross_down = prev['stoch_k'] >= prev['stoch_d'] and stoch_k < stoch_d
            
            if stoch_k > 80:
                signal = "OVERBOUGHT"
            elif stoch_k < 20:
                signal = "OVERSOLD"
            elif cross_up:
                signal = "BULLISH CROSSOVER"
            elif cross_down:
                signal = "BEARISH CROSSOVER"
            else:
                signal = "NEUTRAL"
            
            result['stochastic'] = {
                "k": round(stoch_k, 2),
                "d": round(stoch_d, 2),
                "signal": signal
            }
        
        # Williams %R
        if 'williams_r' in latest.index:
            wr = latest['williams_r']
            if wr > -20:
                signal = "OVERBOUGHT"
            elif wr < -80:
                signal = "OVERSOLD"
            else:
                signal = "NEUTRAL"
            result['williams_r'] = {"value": round(wr, 2), "signal": signal}
        
        return result
    
    def _analyze_volatility(self, latest: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility indicators"""
        result = {}
        
        # ATR Analysis
        if 'atr' in latest.index and 'atr_percent' in latest.index:
            result['atr'] = {
                "value": round(latest['atr'], 4),
                "percent": round(latest['atr_percent'], 2),
                "description": f"Average True Range is {latest['atr_percent']:.2f}% of price"
            }
        
        # Bollinger Band Analysis
        if 'bb_pband' in latest.index:
            pband = latest['bb_pband']
            if pband > 1:
                signal = "OVERBOUGHT - Price above upper band"
            elif pband < 0:
                signal = "OVERSOLD - Price below lower band"
            elif pband > 0.8:
                signal = "Near upper band - Potential resistance"
            elif pband < 0.2:
                signal = "Near lower band - Potential support"
            else:
                signal = "Within normal range"
            
            result['bollinger'] = {
                "position": round(pband, 2),
                "width": round(latest.get('bb_width', 0), 4),
                "signal": signal
            }
        
        # Rolling volatility
        if 'volatility_20' in latest.index:
            vol = latest['volatility_20'] * 100
            if vol > 40:
                level = "HIGH"
            elif vol > 20:
                level = "MODERATE"
            else:
                level = "LOW"
            
            result['volatility'] = {
                "annualized": round(vol, 2),
                "level": level
            }
        
        return result
    
    def _analyze_volume(self, latest: pd.Series) -> Dict[str, Any]:
        """Analyze volume indicators"""
        result = {}
        
        if 'volume_ratio' in latest.index:
            ratio = latest['volume_ratio']
            if ratio > 2:
                signal = "VERY HIGH - Significant interest"
            elif ratio > 1.5:
                signal = "HIGH - Above average interest"
            elif ratio < 0.5:
                signal = "LOW - Below average interest"
            else:
                signal = "NORMAL"
            
            result['volume_ratio'] = {
                "value": round(ratio, 2),
                "signal": signal
            }
        
        if 'mfi' in latest.index:
            mfi = latest['mfi']
            if mfi > 80:
                signal = "OVERBOUGHT"
            elif mfi < 20:
                signal = "OVERSOLD"
            else:
                signal = "NEUTRAL"
            
            result['mfi'] = {"value": round(mfi, 2), "signal": signal}
        
        return result
    
    def _calculate_overall_signal(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall trading signal from all indicators"""
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        def count_signals(obj):
            nonlocal bullish_signals, bearish_signals, neutral_signals
            if isinstance(obj, dict):
                sig = obj.get('signal', obj.get('overall_trend', ''))
                if 'BULLISH' in str(sig).upper():
                    bullish_signals += 1
                elif 'BEARISH' in str(sig).upper():
                    bearish_signals += 1
                elif 'OVERSOLD' in str(sig).upper():
                    bullish_signals += 0.5  # Oversold is bullish opportunity
                elif 'OVERBOUGHT' in str(sig).upper():
                    bearish_signals += 0.5  # Overbought is bearish warning
                elif 'NEUTRAL' in str(sig).upper():
                    neutral_signals += 1
                
                for v in obj.values():
                    count_signals(v)
        
        count_signals(signals)
        
        total = bullish_signals + bearish_signals + neutral_signals
        if total == 0:
            return {"signal": "NEUTRAL", "confidence": 0}
        
        if bullish_signals > bearish_signals:
            signal = "BULLISH"
            confidence = (bullish_signals / total) * 100
        elif bearish_signals > bullish_signals:
            signal = "BEARISH"
            confidence = (bearish_signals / total) * 100
        else:
            signal = "NEUTRAL"
            confidence = (neutral_signals / total) * 100
        
        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "bullish_count": bullish_signals,
            "bearish_count": bearish_signals,
            "neutral_count": neutral_signals
        }
    
    def get_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: Price DataFrame
            window: Lookback window for pivot points
            num_levels: Number of levels to return
        
        Returns:
            Dict with support and resistance levels
        """
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        # Find pivot highs and lows
        pivot_highs = df[df['High'] == highs]['High'].dropna()
        pivot_lows = df[df['Low'] == lows]['Low'].dropna()
        
        current_price = df['Close'].iloc[-1]
        
        # Resistance levels (above current price)
        resistance = sorted([p for p in pivot_highs.unique() if p > current_price])[:num_levels]
        
        # Support levels (below current price)
        support = sorted([p for p in pivot_lows.unique() if p < current_price], reverse=True)[:num_levels]
        
        return {
            "support": [round(s, 4) for s in support],
            "resistance": [round(r, 4) for r in resistance],
            "current_price": round(current_price, 4)
        }
