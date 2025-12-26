"""
Finance Analytics Agent - Main Package
"""

from src.data import YahooFetcher, CoinGeckoFetcher, NewsFetcher, SentimentFetcher
from src.models import (
    TechnicalAnalyzer, 
    AnomalyDetector,
    SuggestionsEngine,
    SuggestionReport,
    StrategyBacktester,
    BacktestResult,
    PricePredictor,
    PredictionResult,
    RegimeDetector,
    RegimeState,
    PortfolioAnalyzer,
    PortfolioAnalysis
)
from src.rag import FinanceVectorStore, FinanceRAGChain
from src.visualization import ChartGenerator, SuggestionsChartGenerator
from src.utils import get_settings, setup_logger

__version__ = "2.0.0"

__all__ = [
    # Data Fetchers
    "YahooFetcher",
    "CoinGeckoFetcher",
    "NewsFetcher",
    "SentimentFetcher",
    
    # Models - Core
    "TechnicalAnalyzer",
    "AnomalyDetector",
    
    # Models - ML Suggestions Engine
    "SuggestionsEngine",
    "SuggestionReport",
    "StrategyBacktester",
    "BacktestResult",
    "PricePredictor",
    "PredictionResult",
    "RegimeDetector",
    "RegimeState",
    "PortfolioAnalyzer",
    "PortfolioAnalysis",
    
    # RAG
    "FinanceVectorStore",
    "FinanceRAGChain",
    
    # Visualization
    "ChartGenerator",
    "SuggestionsChartGenerator",
    
    # Utils
    "get_settings",
    "setup_logger"
]
