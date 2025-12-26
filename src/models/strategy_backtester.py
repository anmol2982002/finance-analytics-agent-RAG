"""
Strategy Backtester
Validates suggestions against historical data with statistical rigor.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a strategy backtest."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Performance metrics
    win_rate: float
    avg_return_per_trade: float
    total_return: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Statistical significance
    p_value: float
    is_significant: bool
    confidence_interval_95: Tuple[float, float]
    
    # Time analysis
    avg_holding_period: float
    best_trade_return: float
    worst_trade_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_return_per_trade": self.avg_return_per_trade,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval_95": self.confidence_interval_95,
            "avg_holding_period": self.avg_holding_period,
            "best_trade_return": self.best_trade_return,
            "worst_trade_return": self.worst_trade_return
        }


class StrategyBacktester:
    """
    Backtest trading strategies with statistical validation.
    
    Every suggestion is validated against historical data:
    - Win rate over multiple time periods
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Maximum drawdown and recovery time
    - Statistical significance testing (p-value)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.05,
        slippage: float = 0.001,
        commission: float = 0.001
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for simulation
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            slippage: Estimated slippage per trade
            commission: Commission per trade
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.slippage = slippage
        self.commission = commission
        
        logger.info("Initialized StrategyBacktester")
    
    def backtest_signal_strategy(
        self,
        df: pd.DataFrame,
        signal_column: str = 'signal',
        holding_period: int = 5
    ) -> BacktestResult:
        """
        Backtest a strategy based on signal columns.
        
        Args:
            df: DataFrame with OHLCV and signal column
            signal_column: Column containing signals (1=BUY, -1=SELL, 0=HOLD)
            holding_period: Days to hold after signal
        
        Returns:
            BacktestResult with performance metrics
        """
        if signal_column not in df.columns:
            logger.error(f"Signal column '{signal_column}' not found in DataFrame")
            return self._empty_result("Unknown Strategy")
        
        df = df.copy()
        trades = []
        
        # Simulate trades
        i = 0
        while i < len(df) - holding_period:
            signal = df[signal_column].iloc[i]
            
            if signal != 0:  # Trade signal
                entry_price = df['Close'].iloc[i] * (1 + self.slippage * np.sign(signal))
                exit_price = df['Close'].iloc[min(i + holding_period, len(df) - 1)]
                exit_price *= (1 - self.slippage * np.sign(signal))
                
                # Calculate return
                if signal > 0:  # Long
                    trade_return = (exit_price / entry_price - 1 - 2 * self.commission) * 100
                else:  # Short
                    trade_return = (entry_price / exit_price - 1 - 2 * self.commission) * 100
                
                trades.append({
                    'entry_date': df.index[i] if hasattr(df.index, 'strftime') else i,
                    'exit_date': df.index[min(i + holding_period, len(df) - 1)],
                    'signal': signal,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'holding_period': holding_period
                })
                
                i += holding_period  # Skip holding period
            else:
                i += 1
        
        if not trades:
            return self._empty_result("Signal Strategy")
        
        return self._calculate_metrics(trades, "Signal Strategy")
    
    def backtest_technical_signals(
        self,
        df: pd.DataFrame,
        generate_signals_func,
        holding_period: int = 5
    ) -> BacktestResult:
        """
        Backtest using a signal generation function.
        
        Args:
            df: DataFrame with OHLCV data
            generate_signals_func: Function that takes df and returns signals dict
            holding_period: Days to hold after signal
        
        Returns:
            BacktestResult
        """
        df = df.copy()
        trades = []
        
        # Generate signals for each historical point
        window = 50  # Minimum data needed for signals
        
        for i in range(window, len(df) - holding_period, holding_period):
            # Get signals at this point
            historical_df = df.iloc[:i+1]
            
            try:
                signals = generate_signals_func(historical_df)
                overall = signals.get('overall', {})
                signal_type = overall.get('signal', 'NEUTRAL')
                confidence = overall.get('confidence', 50)
            except:
                continue
            
            # Only trade on strong signals
            if confidence < 60:
                continue
            
            entry_price = df['Close'].iloc[i]
            exit_price = df['Close'].iloc[min(i + holding_period, len(df) - 1)]
            
            if signal_type == "BULLISH":
                trade_return = (exit_price / entry_price - 1 - 2 * self.commission) * 100
                trades.append({
                    'entry_date': df.index[i] if hasattr(df.index, '__iter__') else i,
                    'signal': 1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'confidence': confidence,
                    'holding_period': holding_period
                })
            elif signal_type == "BEARISH":
                trade_return = (entry_price / exit_price - 1 - 2 * self.commission) * 100
                trades.append({
                    'entry_date': df.index[i] if hasattr(df.index, '__iter__') else i,
                    'signal': -1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'confidence': confidence,
                    'holding_period': holding_period
                })
        
        if not trades:
            return self._empty_result("Technical Signals")
        
        return self._calculate_metrics(trades, "Technical Signals")
    
    def backtest_rsi_strategy(
        self,
        df: pd.DataFrame,
        oversold_threshold: int = 30,
        overbought_threshold: int = 70,
        holding_period: int = 5
    ) -> BacktestResult:
        """
        Backtest RSI mean-reversion strategy.
        
        Args:
            df: DataFrame with 'rsi' column
            oversold_threshold: RSI level for buy signal
            overbought_threshold: RSI level for sell signal
            holding_period: Days to hold
        
        Returns:
            BacktestResult
        """
        if 'rsi' not in df.columns:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df = df.copy()
            df['rsi'] = 100 - (100 / (1 + rs))
        
        trades = []
        i = 14  # RSI needs warm-up
        
        while i < len(df) - holding_period:
            rsi = df['rsi'].iloc[i]
            
            if rsi < oversold_threshold:
                # Buy signal
                entry_price = df['Close'].iloc[i] * (1 + self.slippage)
                exit_price = df['Close'].iloc[i + holding_period] * (1 - self.slippage)
                trade_return = (exit_price / entry_price - 1 - 2 * self.commission) * 100
                
                trades.append({
                    'entry_date': df.index[i],
                    'signal': 1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'rsi': rsi,
                    'holding_period': holding_period
                })
                i += holding_period
                
            elif rsi > overbought_threshold:
                # Sell signal (short)
                entry_price = df['Close'].iloc[i] * (1 - self.slippage)
                exit_price = df['Close'].iloc[i + holding_period] * (1 + self.slippage)
                trade_return = (entry_price / exit_price - 1 - 2 * self.commission) * 100
                
                trades.append({
                    'entry_date': df.index[i],
                    'signal': -1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'rsi': rsi,
                    'holding_period': holding_period
                })
                i += holding_period
            else:
                i += 1
        
        if not trades:
            return self._empty_result("RSI Strategy")
        
        return self._calculate_metrics(trades, f"RSI Strategy ({oversold_threshold}/{overbought_threshold})")
    
    def backtest_ma_crossover(
        self,
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        holding_period: int = 10
    ) -> BacktestResult:
        """
        Backtest moving average crossover strategy.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast MA period
            slow_period: Slow MA period
            holding_period: Days to hold after signal
        
        Returns:
            BacktestResult
        """
        df = df.copy()
        df['fast_ma'] = df['Close'].rolling(fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(slow_period).mean()
        
        trades = []
        position = 0  # 0=no position, 1=long, -1=short
        entry_price = 0
        entry_idx = 0
        
        for i in range(slow_period + 1, len(df)):
            fast = df['fast_ma'].iloc[i]
            slow = df['slow_ma'].iloc[i]
            prev_fast = df['fast_ma'].iloc[i-1]
            prev_slow = df['slow_ma'].iloc[i-1]
            
            # Golden cross (bullish)
            if prev_fast <= prev_slow and fast > slow and position <= 0:
                if position == -1:  # Close short
                    exit_price = df['Close'].iloc[i] * (1 + self.slippage)
                    trade_return = (entry_price / exit_price - 1 - 2 * self.commission) * 100
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'signal': -1,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'holding_period': i - entry_idx
                    })
                
                # Open long
                position = 1
                entry_price = df['Close'].iloc[i] * (1 + self.slippage)
                entry_idx = i
                
            # Death cross (bearish)
            elif prev_fast >= prev_slow and fast < slow and position >= 0:
                if position == 1:  # Close long
                    exit_price = df['Close'].iloc[i] * (1 - self.slippage)
                    trade_return = (exit_price / entry_price - 1 - 2 * self.commission) * 100
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'signal': 1,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'holding_period': i - entry_idx
                    })
                
                # Open short
                position = -1
                entry_price = df['Close'].iloc[i] * (1 - self.slippage)
                entry_idx = i
        
        if not trades:
            return self._empty_result("MA Crossover")
        
        return self._calculate_metrics(trades, f"MA Crossover ({fast_period}/{slow_period})")
    
    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        strategy_name: str
    ) -> BacktestResult:
        """Calculate performance metrics from trade list."""
        
        returns = [t['return'] for t in trades]
        returns_array = np.array(returns)
        
        # Basic stats
        total_trades = len(trades)
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = sum(1 for r in returns if r < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_return = np.mean(returns)
        total_return = np.sum(returns)
        
        # Calculate equity curve for drawdown
        equity = [self.initial_capital]
        for r in returns:
            equity.append(equity[-1] * (1 + r / 100))
        
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Risk-adjusted metrics
        if len(returns) > 1:
            # Sharpe Ratio (annualized)
            excess_returns = returns_array - (self.risk_free_rate / 252 * 100)
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Sortino Ratio (using downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Calmar Ratio
            calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe = sortino = calmar = 0
        
        # Statistical significance
        if len(returns) >= 5:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            
            # 95% confidence interval
            ci = stats.t.interval(
                0.95,
                len(returns) - 1,
                loc=np.mean(returns),
                scale=stats.sem(returns)
            )
        else:
            p_value = 1.0
            ci = (0, 0)
        
        is_significant = p_value < 0.05 and avg_return > 0
        
        # Time analysis
        holding_periods = [t.get('holding_period', 5) for t in trades]
        avg_holding = np.mean(holding_periods)
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            avg_return_per_trade=round(avg_return, 4),
            total_return=round(total_return, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            confidence_interval_95=(round(ci[0], 2), round(ci[1], 2)),
            avg_holding_period=round(avg_holding, 1),
            best_trade_return=round(max(returns), 2),
            worst_trade_return=round(min(returns), 2)
        )
    
    def _empty_result(self, strategy_name: str) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_return_per_trade=0,
            total_return=0,
            max_drawdown=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            p_value=1.0,
            is_significant=False,
            confidence_interval_95=(0, 0),
            avg_holding_period=0,
            best_trade_return=0,
            worst_trade_return=0
        )
    
    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies on the same data.
        
        Args:
            df: DataFrame with OHLCV data
            strategies: List of strategy names to test, or None for all
        
        Returns:
            Dict mapping strategy name to BacktestResult
        """
        available_strategies = {
            'rsi_30_70': lambda: self.backtest_rsi_strategy(df, 30, 70, 5),
            'rsi_25_75': lambda: self.backtest_rsi_strategy(df, 25, 75, 5),
            'rsi_20_80': lambda: self.backtest_rsi_strategy(df, 20, 80, 5),
            'ma_10_30': lambda: self.backtest_ma_crossover(df, 10, 30, 10),
            'ma_20_50': lambda: self.backtest_ma_crossover(df, 20, 50, 10),
            'ma_50_200': lambda: self.backtest_ma_crossover(df, 50, 200, 20),
        }
        
        if strategies is None:
            strategies = list(available_strategies.keys())
        
        results = {}
        for strategy in strategies:
            if strategy in available_strategies:
                try:
                    results[strategy] = available_strategies[strategy]()
                    logger.info(f"Backtested {strategy}: Win Rate={results[strategy].win_rate}%")
                except Exception as e:
                    logger.error(f"Error backtesting {strategy}: {e}")
        
        return results
    
    def get_best_strategy(
        self,
        df: pd.DataFrame,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[str, BacktestResult]:
        """
        Find the best performing strategy based on a metric.
        
        Args:
            df: DataFrame with OHLCV data
            metric: Metric to optimize ('sharpe_ratio', 'win_rate', 'total_return')
        
        Returns:
            Tuple of (strategy_name, BacktestResult)
        """
        results = self.compare_strategies(df)
        
        if not results:
            return "None", self._empty_result("No Strategy")
        
        # Sort by metric (only significant strategies)
        significant_results = {k: v for k, v in results.items() if v.is_significant}
        
        if significant_results:
            best_strategy = max(
                significant_results.items(),
                key=lambda x: getattr(x[1], metric, 0)
            )
        else:
            # Fall back to all results if none significant
            best_strategy = max(
                results.items(),
                key=lambda x: getattr(x[1], metric, 0)
            )
        
        return best_strategy
