"""
Anomaly Detection for Financial Data
Detects unusual price movements, volume spikes, and pattern breaks
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in financial time series data.
    
    Methods:
    - Z-score based detection
    - IQR (Interquartile Range) method
    - Isolation Forest
    - Rolling window deviation
    - Volume spike detection
    """
    
    def __init__(
        self,
        zscore_threshold: float = 2.5,
        iqr_multiplier: float = 1.5,
        isolation_contamination: float = 0.05
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            zscore_threshold: Z-score threshold for anomaly detection
            iqr_multiplier: IQR multiplier for outlier detection
            isolation_contamination: Expected proportion of outliers
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.isolation_contamination = isolation_contamination
    
    def detect_all_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all anomaly detection methods on the data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with anomaly flags added
        """
        df = df.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['Close'].pct_change()
        
        # Z-score anomalies
        df = self._detect_zscore_anomalies(df)
        
        # IQR anomalies
        df = self._detect_iqr_anomalies(df)
        
        # Isolation Forest anomalies
        df = self._detect_isolation_forest_anomalies(df)
        
        # Volume spike detection
        if 'Volume' in df.columns:
            df = self._detect_volume_spikes(df)
        
        # Combined anomaly flag
        anomaly_cols = [col for col in df.columns if col.startswith('is_anomaly_')]
        if anomaly_cols:
            df['is_anomaly'] = df[anomaly_cols].any(axis=1)
        
        return df
    
    def _detect_zscore_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Z-score method"""
        df = df.copy()
        
        # Calculate Z-score of returns
        returns = df['returns'].dropna()
        if len(returns) < 10:
            df['zscore'] = np.nan
            df['is_anomaly_zscore'] = False
            return df
        
        df['zscore'] = stats.zscore(df['returns'].fillna(0))
        df['is_anomaly_zscore'] = abs(df['zscore']) > self.zscore_threshold
        
        return df
    
    def _detect_iqr_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using IQR method"""
        df = df.copy()
        
        returns = df['returns'].dropna()
        if len(returns) < 10:
            df['is_anomaly_iqr'] = False
            return df
        
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        
        df['is_anomaly_iqr'] = (
            (df['returns'] < lower_bound) | 
            (df['returns'] > upper_bound)
        )
        
        return df
    
    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest"""
        df = df.copy()
        
        # Prepare features
        feature_cols = ['returns']
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            feature_cols.append('volume_change')
        
        # Drop NaN values for training
        feature_df = df[feature_cols].dropna()
        
        if len(feature_df) < 20:
            df['is_anomaly_iforest'] = False
            return df
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_df)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(features_scaled)
        
        # Map predictions back to DataFrame
        df['is_anomaly_iforest'] = False
        df.loc[feature_df.index, 'is_anomaly_iforest'] = predictions == -1
        
        # Get anomaly scores
        scores = iso_forest.score_samples(features_scaled)
        df['anomaly_score'] = np.nan
        df.loc[feature_df.index, 'anomaly_score'] = scores
        
        return df
    
    def _detect_volume_spikes(
        self,
        df: pd.DataFrame,
        window: int = 20,
        threshold: float = 2.0
    ) -> pd.DataFrame:
        """Detect unusual volume activity"""
        df = df.copy()
        
        if 'Volume' not in df.columns:
            df['is_volume_spike'] = False
            return df
        
        # Calculate rolling statistics
        volume_mean = df['Volume'].rolling(window=window).mean()
        volume_std = df['Volume'].rolling(window=window).std()
        
        # Z-score of volume
        df['volume_zscore'] = (df['Volume'] - volume_mean) / volume_std
        df['is_volume_spike'] = df['volume_zscore'] > threshold
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of detected anomalies.
        
        Args:
            df: DataFrame with anomaly flags
        
        Returns:
            Summary dict with counts and recent anomalies
        """
        if 'is_anomaly' not in df.columns:
            df = self.detect_all_anomalies(df)
        
        anomalies = df[df['is_anomaly'] == True]
        
        return {
            "total_anomalies": len(anomalies),
            "anomaly_rate": f"{(len(anomalies) / len(df) * 100):.2f}%",
            "recent_anomalies": [
                {
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                    "return": f"{row['returns']*100:.2f}%",
                    "zscore": round(row.get('zscore', 0), 2),
                    "close": round(row['Close'], 2),
                    "volume_spike": row.get('is_volume_spike', False)
                }
                for idx, row in anomalies.tail(10).iterrows()
            ],
            "methods_triggered": {
                "zscore": int(df.get('is_anomaly_zscore', pd.Series([False])).sum()),
                "iqr": int(df.get('is_anomaly_iqr', pd.Series([False])).sum()),
                "isolation_forest": int(df.get('is_anomaly_iforest', pd.Series([False])).sum()),
                "volume_spike": int(df.get('is_volume_spike', pd.Series([False])).sum())
            }
        }
    
    def detect_flash_crash(
        self,
        df: pd.DataFrame,
        price_threshold: float = -0.05,
        time_window: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect potential flash crash events.
        
        Args:
            df: Price DataFrame
            price_threshold: Minimum price drop (negative, e.g., -0.05 for 5%)
            time_window: Number of periods for rapid decline
        
        Returns:
            List of flash crash events
        """
        df = df.copy()
        
        # Calculate rolling returns
        df['rolling_return'] = df['Close'].pct_change(periods=time_window)
        
        # Find flash crashes
        crashes = df[df['rolling_return'] <= price_threshold]
        
        events = []
        for idx, row in crashes.iterrows():
            events.append({
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                "drop_percent": f"{row['rolling_return']*100:.2f}%",
                "close_price": round(row['Close'], 2),
                "time_window": time_window,
                "severity": "HIGH" if row['rolling_return'] <= -0.10 else "MODERATE"
            })
        
        return events
    
    def detect_breakouts(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        volume_confirm: bool = True
    ) -> Dict[str, Any]:
        """
        Detect price breakouts from consolidation.
        
        Args:
            df: Price DataFrame with Volume
            lookback: Lookback period for range calculation
            volume_confirm: Require volume confirmation
        
        Returns:
            Breakout analysis
        """
        df = df.copy()
        
        # Calculate range
        df['range_high'] = df['High'].rolling(window=lookback).max()
        df['range_low'] = df['Low'].rolling(window=lookback).min()
        
        # Check for breakouts
        latest = df.iloc[-1]
        
        result = {
            "breakout_detected": False,
            "direction": None,
            "range_high": round(latest['range_high'], 4),
            "range_low": round(latest['range_low'], 4),
            "current_price": round(latest['Close'], 4)
        }
        
        # Upside breakout
        if latest['Close'] > latest['range_high']:
            result["breakout_detected"] = True
            result["direction"] = "BULLISH"
            result["description"] = f"Price broke above {lookback}-day high"
        
        # Downside breakout
        elif latest['Close'] < latest['range_low']:
            result["breakout_detected"] = True
            result["direction"] = "BEARISH"
            result["description"] = f"Price broke below {lookback}-day low"
        
        # Volume confirmation
        if volume_confirm and 'Volume' in df.columns and result["breakout_detected"]:
            avg_volume = df['Volume'].rolling(window=lookback).mean().iloc[-1]
            current_volume = latest['Volume']
            
            result["volume_confirmation"] = current_volume > avg_volume * 1.5
            result["volume_ratio"] = round(current_volume / avg_volume, 2)
        
        return result
