"""
Yahoo Finance Data Fetcher
Fetches stock and crypto data from Yahoo Finance (free, no API key required)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from src.utils import get_logger

logger = get_logger(__name__)


class YahooFetcher:
    """
    Fetches market data from Yahoo Finance.
    
    Features:
    - OHLCV historical data
    - Company fundamentals (P/E, Market Cap, etc.)
    - Analyst recommendations
    - News headlines
    - Dividends and splits
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a ticker object"""
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV historical data.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD')
            period: Data period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval - 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        try:
            ticker = self.get_ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.replace(" ", "_") for col in df.columns]
            
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current/latest price info.
        
        Returns:
            Dict with price, change, volume info
        """
        try:
            ticker = self.get_ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "open": info.get("open") or info.get("regularMarketOpen"),
                "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching realtime price for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get company fundamentals and financial metrics.
        
        Returns:
            Dict with P/E, EPS, revenue, and other metrics
        """
        try:
            ticker = self.get_ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                
                # Valuation
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                
                # Profitability
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                
                # Growth
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                
                # Dividends
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
                
                # Financial Health
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                
                # Other
                "beta": info.get("beta"),
                "fifty_day_average": info.get("fiftyDayAverage"),
                "two_hundred_day_average": info.get("twoHundredDayAverage"),
                
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations.
        
        Returns:
            DataFrame with analyst recommendations over time
        """
        try:
            ticker = self.get_ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                logger.warning(f"No recommendations found for {symbol}")
                return pd.DataFrame()
            
            return recommendations
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recent news articles for a ticker.
        
        Returns:
            List of news articles with title, link, publisher, date
        """
        try:
            ticker = self.get_ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return []
            
            # Process news items
            processed_news = []
            for item in news:
                processed_news.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "publish_time": datetime.fromtimestamp(
                        item.get("providerPublishTime", 0)
                    ).isoformat() if item.get("providerPublishTime") else None,
                    "type": item.get("type"),
                    "thumbnail": item.get("thumbnail", {}).get("resolutions", [{}])[0].get("url"),
                    "related_tickers": item.get("relatedTickers", [])
                })
            
            logger.info(f"Fetched {len(processed_news)} news items for {symbol}")
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_earnings(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get earnings history and upcoming dates.
        
        Returns:
            Dict with earnings_history and earnings_dates DataFrames
        """
        try:
            ticker = self.get_ticker(symbol)
            
            return {
                "earnings_history": ticker.earnings_history if hasattr(ticker, 'earnings_history') else pd.DataFrame(),
                "earnings_dates": ticker.earnings_dates if hasattr(ticker, 'earnings_dates') else pd.DataFrame()
            }
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return {"earnings_history": pd.DataFrame(), "earnings_dates": pd.DataFrame()}
    
    def get_multiple_tickers(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers efficiently.
        
        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        
        # Use yfinance's built-in multiple ticker download
        try:
            data = yf.download(
                symbols,
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            
            for symbol in symbols:
                if len(symbols) == 1:
                    results[symbol] = data
                else:
                    if symbol in data.columns.get_level_values(0):
                        results[symbol] = data[symbol].dropna()
                    else:
                        results[symbol] = pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error in batch download: {e}")
            # Fallback to individual fetches
            for symbol in symbols:
                results[symbol] = self.get_historical_data(symbol, period, interval)
        
        return results
    
    def clear_cache(self):
        """Clear the ticker cache"""
        self._cache.clear()
        logger.info("Ticker cache cleared")
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for tickers matching a query.
        
        Args:
            query: Search query (partial ticker or company name)
            limit: Maximum number of results
        
        Returns:
            List of dicts with 'symbol' and 'name' keys
        """
        if not query or len(query) < 1:
            return []
        
        try:
            import requests
            
            # Use Yahoo Finance search API
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                "q": query,
                "quotesCount": limit,
                "newsCount": 0,
                "enableFuzzyQuery": True,
                "quotesQueryId": "tss_match_phrase_query"
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=3)
            
            if response.status_code != 200:
                return self._get_fallback_suggestions(query)
            
            data = response.json()
            quotes = data.get("quotes", [])
            
            results = []
            for quote in quotes[:limit]:
                symbol = quote.get("symbol", "")
                name = quote.get("shortname", quote.get("longname", ""))
                exchange = quote.get("exchange", "")
                quote_type = quote.get("quoteType", "")
                
                # Filter to stocks and ETFs primarily
                if quote_type in ["EQUITY", "ETF", "CRYPTOCURRENCY"]:
                    results.append({
                        "symbol": symbol,
                        "name": name,
                        "exchange": exchange,
                        "type": quote_type
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Ticker search error: {e}")
            return self._get_fallback_suggestions(query)
    
    def _get_fallback_suggestions(self, query: str) -> List[Dict[str, str]]:
        """Fallback suggestions when API fails"""
        popular_tickers = [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ", "type": "EQUITY"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "type": "EQUITY"},
            {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE", "type": "EQUITY"},
            {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE", "type": "EQUITY"},
            {"symbol": "BTC-USD", "name": "Bitcoin USD", "exchange": "CCC", "type": "CRYPTOCURRENCY"},
            {"symbol": "ETH-USD", "name": "Ethereum USD", "exchange": "CCC", "type": "CRYPTOCURRENCY"},
        ]
        
        query_upper = query.upper()
        return [t for t in popular_tickers if query_upper in t["symbol"] or query.lower() in t["name"].lower()][:5]
