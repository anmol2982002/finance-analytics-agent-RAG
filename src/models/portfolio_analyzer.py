"""
Portfolio Analyzer
Multi-asset portfolio analysis with correlation, risk parity, and stress testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationAnalysis:
    """Correlation matrix and insights."""
    correlation_matrix: Dict[str, Dict[str, float]]
    highly_correlated_pairs: List[Tuple[str, str, float]]
    diversification_score: float  # 0-100, higher = more diversified
    recommended_additions: List[str]
    warnings: List[str]


@dataclass
class RiskParityWeights:
    """Risk parity portfolio weights."""
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    portfolio_volatility: float
    sharpe_ratio: float
    equal_weight_comparison: Dict[str, float]


@dataclass
class StressTestResult:
    """Results from portfolio stress testing."""
    scenario_name: str
    portfolio_return: float
    max_drawdown: float
    worst_asset: str
    worst_asset_return: float
    best_asset: str
    best_asset_return: float
    recovery_estimate_days: int


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis."""
    assets: List[str]
    total_value: float
    
    # Risk metrics
    portfolio_volatility: float
    portfolio_beta: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    
    # Correlations
    correlation_analysis: CorrelationAnalysis
    
    # Weights
    current_weights: Dict[str, float]
    optimal_weights: RiskParityWeights
    
    # Stress tests
    stress_tests: List[StressTestResult]
    
    # Sector exposure
    sector_exposure: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": self.assets,
            "total_value": self.total_value,
            "risk_metrics": {
                "portfolio_volatility": self.portfolio_volatility,
                "portfolio_beta": self.portfolio_beta,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "var_95": self.var_95
            },
            "correlation_analysis": {
                "diversification_score": self.correlation_analysis.diversification_score,
                "highly_correlated_pairs": self.correlation_analysis.highly_correlated_pairs,
                "warnings": self.correlation_analysis.warnings
            },
            "current_weights": self.current_weights,
            "optimal_weights": self.optimal_weights.weights if self.optimal_weights else {},
            "stress_tests": [
                {"scenario": st.scenario_name, "return": st.portfolio_return, "max_drawdown": st.max_drawdown}
                for st in self.stress_tests
            ],
            "sector_exposure": self.sector_exposure
        }


class PortfolioAnalyzer:
    """
    Advanced portfolio analysis tool.
    
    Features:
    - Correlation Matrix: Dynamic correlation heatmaps
    - Risk Parity: Optimal position sizing by risk contribution
    - Diversification Score: How diversified is the portfolio?
    - Stress Testing: Historical scenario analysis (2008, COVID, etc.)
    - Sector Exposure: Concentration risk detection
    """
    
    # Historical stress scenarios (approximate returns)
    STRESS_SCENARIOS = {
        '2008_financial_crisis': {
            'name': '2008 Financial Crisis',
            'SPY': -0.37, 'QQQ': -0.42, 'IWM': -0.34,
            'BTC': 0.0, 'ETH': 0.0,  # Didn't exist
            'GLD': 0.05, 'TLT': 0.20,
            'default': -0.30  # Default for unknown assets
        },
        '2020_covid_crash': {
            'name': 'COVID-19 Crash (Mar 2020)',
            'SPY': -0.34, 'QQQ': -0.28, 'IWM': -0.41,
            'BTC': -0.40, 'ETH': -0.50,
            'GLD': -0.05, 'TLT': 0.15,
            'default': -0.25
        },
        '2022_crypto_winter': {
            'name': '2022 Crypto Winter',
            'SPY': -0.19, 'QQQ': -0.33, 'IWM': -0.21,
            'BTC': -0.65, 'ETH': -0.68,
            'GLD': -0.01, 'TLT': -0.31,
            'default': -0.20
        },
        'flash_crash': {
            'name': 'Flash Crash Scenario',
            'SPY': -0.10, 'QQQ': -0.12, 'IWM': -0.11,
            'BTC': -0.20, 'ETH': -0.25,
            'GLD': 0.02, 'TLT': 0.05,
            'default': -0.10
        },
        'rates_spike': {
            'name': 'Interest Rate Spike',
            'SPY': -0.15, 'QQQ': -0.25, 'IWM': -0.18,
            'BTC': -0.30, 'ETH': -0.35,
            'GLD': -0.08, 'TLT': -0.20,
            'default': -0.15
        }
    }
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        benchmark_ticker: str = "SPY"
    ):
        """
        Initialize the portfolio analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            benchmark_ticker: Benchmark for beta calculation
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker
        
        logger.info("Initialized PortfolioAnalyzer")
    
    def analyze_portfolio(
        self,
        price_data: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, float]] = None
    ) -> PortfolioAnalysis:
        """
        Perform comprehensive portfolio analysis.
        
        Args:
            price_data: Dict mapping ticker to DataFrame with 'Close' column
            weights: Optional portfolio weights (if not provided, calculated from values)
            values: Optional dict of {ticker: dollar_value}
        
        Returns:
            PortfolioAnalysis with all metrics
        """
        assets = list(price_data.keys())
        
        if not assets:
            logger.error("No assets provided for analysis")
            return self._empty_analysis()
        
        # Calculate weights
        if weights is None:
            if values:
                total = sum(values.values())
                weights = {k: v / total for k, v in values.items()}
                total_value = total
            else:
                weights = {k: 1 / len(assets) for k in assets}  # Equal weight
                total_value = 10000  # Default
        else:
            total_value = sum(values.values()) if values else 10000
        
        # Get returns matrix
        returns_df = self._align_returns(price_data)
        
        if returns_df.empty:
            return self._empty_analysis()
        
        # Calculate portfolio metrics
        portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
        
        # Risk metrics
        port_vol = portfolio_returns.std() * np.sqrt(252)
        port_mean = portfolio_returns.mean() * 252
        sharpe = (port_mean - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Sortino
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else port_vol
        sortino = (port_mean - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        # VaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252) * 100
        
        # Beta (if benchmark in data)
        if self.benchmark_ticker in returns_df.columns:
            benchmark_returns = returns_df[self.benchmark_ticker]
            cov = portfolio_returns.cov(benchmark_returns)
            var_benchmark = benchmark_returns.var()
            beta = cov / var_benchmark if var_benchmark > 0 else 1
        else:
            beta = 1.0
        
        # Correlation analysis
        correlation = self._analyze_correlations(returns_df)
        
        # Risk parity weights
        optimal = self._calculate_risk_parity(returns_df, weights)
        
        # Stress tests
        stress_tests = self._run_stress_tests(assets, weights)
        
        # Sector exposure (simplified)
        sector_exposure = self._estimate_sector_exposure(assets)
        
        return PortfolioAnalysis(
            assets=assets,
            total_value=total_value,
            portfolio_volatility=round(port_vol * 100, 2),
            portfolio_beta=round(beta, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown=round(max_dd * 100, 2),
            var_95=round(var_95, 2),
            correlation_analysis=correlation,
            current_weights=weights,
            optimal_weights=optimal,
            stress_tests=stress_tests,
            sector_exposure=sector_exposure
        )
    
    def _align_returns(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align price data and calculate returns."""
        returns_dict = {}
        
        for ticker, df in price_data.items():
            if 'Close' in df.columns:
                returns_dict[ticker] = df['Close'].pct_change()
            elif len(df.columns) == 1:
                returns_dict[ticker] = df.iloc[:, 0].pct_change()
        
        if not returns_dict:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(how='all')
        
        return returns_df
    
    def _calculate_portfolio_returns(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns from asset returns and weights."""
        portfolio_returns = pd.Series(0, index=returns_df.index)
        
        for ticker, weight in weights.items():
            if ticker in returns_df.columns:
                portfolio_returns += returns_df[ticker].fillna(0) * weight
        
        return portfolio_returns
    
    def _analyze_correlations(self, returns_df: pd.DataFrame) -> CorrelationAnalysis:
        """Analyze correlations between assets."""
        corr_matrix = returns_df.corr()
        
        # Convert to dict
        corr_dict = corr_matrix.to_dict()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        tickers = corr_matrix.columns.tolist()
        
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i < j:  # Avoid duplicates
                    corr = corr_matrix.loc[t1, t2]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append((t1, t2, round(corr, 3)))
        
        # Diversification score (lower average correlation = more diversified)
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        diversification_score = max(0, min(100, (1 - avg_corr) * 100))
        
        # Warnings
        warnings = []
        if diversification_score < 40:
            warnings.append("⚠️ Portfolio is highly concentrated - consider adding uncorrelated assets")
        
        for t1, t2, corr in high_corr_pairs:
            if corr > 0.85:
                warnings.append(f"⚠️ {t1} and {t2} are highly correlated ({corr:.0%}) - consider reducing one")
        
        # Recommendations
        recommendations = []
        if avg_corr > 0.5:
            recommendations = ["GLD", "TLT", "VXX"]  # Typical diversifiers
        
        return CorrelationAnalysis(
            correlation_matrix=corr_dict,
            highly_correlated_pairs=high_corr_pairs,
            diversification_score=round(diversification_score, 1),
            recommended_additions=recommendations,
            warnings=warnings
        )
    
    def _calculate_risk_parity(
        self,
        returns_df: pd.DataFrame,
        current_weights: Dict[str, float]
    ) -> RiskParityWeights:
        """Calculate risk parity weights."""
        assets = [a for a in current_weights.keys() if a in returns_df.columns]
        n = len(assets)
        
        if n < 2:
            return RiskParityWeights(
                weights=current_weights,
                risk_contributions={},
                portfolio_volatility=0,
                sharpe_ratio=0,
                equal_weight_comparison={}
            )
        
        # Covariance matrix
        cov_matrix = returns_df[assets].cov().values * 252
        
        # Risk parity optimization
        def risk_parity_objective(weights):
            weights = np.array(weights)
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Marginal risk contribution
            mrc = cov_matrix @ weights / port_vol
            
            # Total risk contribution
            trc = weights * mrc
            
            # Target: equal risk contribution
            target_rc = port_vol / n
            
            # Minimize deviation from target
            return np.sum((trc - target_rc) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        bounds = [(0.01, 0.5) for _ in range(n)]  # Min 1%, max 50% per asset
        
        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = dict(zip(assets, np.round(result.x, 4)))
        else:
            optimal_weights = dict(zip(assets, [1/n] * n))  # Fall back to equal weight
        
        # Calculate risk contributions
        final_weights = np.array(list(optimal_weights.values()))
        port_vol = np.sqrt(final_weights.T @ cov_matrix @ final_weights)
        mrc = cov_matrix @ final_weights / port_vol
        risk_contributions = dict(zip(assets, np.round(final_weights * mrc / port_vol * 100, 2)))
        
        # Compare to equal weight
        equal_weights = np.ones(n) / n
        equal_vol = np.sqrt(equal_weights.T @ cov_matrix @ equal_weights)
        
        return RiskParityWeights(
            weights=optimal_weights,
            risk_contributions=risk_contributions,
            portfolio_volatility=round(port_vol * 100, 2),
            sharpe_ratio=0,  # Would need returns data
            equal_weight_comparison={
                'equal_weight_volatility': round(equal_vol * 100, 2),
                'volatility_reduction': round((1 - port_vol / equal_vol) * 100, 2) if equal_vol > 0 else 0
            }
        )
    
    def _run_stress_tests(
        self,
        assets: List[str],
        weights: Dict[str, float]
    ) -> List[StressTestResult]:
        """Run historical stress tests on portfolio."""
        results = []
        
        for scenario_id, scenario in self.STRESS_SCENARIOS.items():
            portfolio_return = 0
            asset_returns = {}
            
            for asset in assets:
                weight = weights.get(asset, 0)
                asset_return = scenario.get(asset.upper(), scenario['default'])
                asset_returns[asset] = asset_return
                portfolio_return += weight * asset_return
            
            if asset_returns:
                worst_asset = min(asset_returns, key=asset_returns.get)
                best_asset = max(asset_returns, key=asset_returns.get)
            else:
                worst_asset = best_asset = "N/A"
            
            # Estimate recovery (crude approximation)
            if portfolio_return < -0.30:
                recovery_days = 365 * 2  # 2 years for severe crash
            elif portfolio_return < -0.20:
                recovery_days = 365  # 1 year
            elif portfolio_return < -0.10:
                recovery_days = 180  # 6 months
            else:
                recovery_days = 90  # 3 months
            
            results.append(StressTestResult(
                scenario_name=scenario['name'],
                portfolio_return=round(portfolio_return * 100, 2),
                max_drawdown=round(portfolio_return * 100, 2),  # Simplified
                worst_asset=worst_asset,
                worst_asset_return=round(asset_returns.get(worst_asset, 0) * 100, 2),
                best_asset=best_asset,
                best_asset_return=round(asset_returns.get(best_asset, 0) * 100, 2),
                recovery_estimate_days=recovery_days
            ))
        
        return results
    
    def _estimate_sector_exposure(self, assets: List[str]) -> Dict[str, float]:
        """Estimate sector exposure (simplified mapping)."""
        # Simple sector mapping
        sector_map = {
            # Tech
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'QQQ': 'Technology',
            # Finance
            'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
            'XLF': 'Financials',
            # Healthcare
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
            'XLV': 'Healthcare',
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'XLE': 'Energy',
            # Crypto
            'BTC': 'Crypto', 'ETH': 'Crypto', 'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto',
            # Bonds
            'TLT': 'Bonds', 'BND': 'Bonds', 'AGG': 'Bonds',
            # Commodities
            'GLD': 'Commodities', 'SLV': 'Commodities', 'USO': 'Commodities',
            # Market
            'SPY': 'Broad Market', 'IWM': 'Broad Market', 'VTI': 'Broad Market',
        }
        
        exposure = {}
        for asset in assets:
            sector = sector_map.get(asset.upper(), 'Other')
            exposure[sector] = exposure.get(sector, 0) + 1
        
        # Normalize to percentages
        total = sum(exposure.values())
        return {k: round(v / total * 100, 1) for k, v in exposure.items()}
    
    def _empty_analysis(self) -> PortfolioAnalysis:
        """Return empty analysis."""
        return PortfolioAnalysis(
            assets=[],
            total_value=0,
            portfolio_volatility=0,
            portfolio_beta=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            var_95=0,
            correlation_analysis=CorrelationAnalysis({}, [], 0, [], ["No data available"]),
            current_weights={},
            optimal_weights=RiskParityWeights({}, {}, 0, 0, {}),
            stress_tests=[],
            sector_exposure={}
        )
    
    def what_if_analysis(
        self,
        price_data: Dict[str, pd.DataFrame],
        current_weights: Dict[str, float],
        proposed_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze impact of proposed weight changes.
        
        Args:
            price_data: Historical price data
            current_weights: Current portfolio weights
            proposed_changes: Proposed weight changes (positive = add, negative = reduce)
        
        Returns:
            Comparison of current vs proposed portfolio
        """
        # Adjust weights
        new_weights = current_weights.copy()
        for asset, change in proposed_changes.items():
            if asset in new_weights:
                new_weights[asset] = max(0, new_weights[asset] + change)
            else:
                new_weights[asset] = max(0, change)
        
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        # Analyze both
        current_analysis = self.analyze_portfolio(price_data, current_weights)
        proposed_analysis = self.analyze_portfolio(price_data, new_weights)
        
        return {
            'current': {
                'weights': current_weights,
                'volatility': current_analysis.portfolio_volatility,
                'sharpe': current_analysis.sharpe_ratio,
                'diversification': current_analysis.correlation_analysis.diversification_score
            },
            'proposed': {
                'weights': new_weights,
                'volatility': proposed_analysis.portfolio_volatility,
                'sharpe': proposed_analysis.sharpe_ratio,
                'diversification': proposed_analysis.correlation_analysis.diversification_score
            },
            'changes': {
                'volatility_change': proposed_analysis.portfolio_volatility - current_analysis.portfolio_volatility,
                'sharpe_change': proposed_analysis.sharpe_ratio - current_analysis.sharpe_ratio,
                'diversification_change': proposed_analysis.correlation_analysis.diversification_score - current_analysis.correlation_analysis.diversification_score
            },
            'recommendation': 'PROCEED' if proposed_analysis.sharpe_ratio > current_analysis.sharpe_ratio else 'RECONSIDER'
        }
