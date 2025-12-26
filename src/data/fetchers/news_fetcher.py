"""
News Data Fetcher
Fetches financial news from multiple free sources
"""

import requests
import feedparser
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from urllib.parse import quote
from src.utils import get_logger, get_settings

logger = get_logger(__name__)


class NewsFetcher:
    """
    Fetches financial news from multiple sources.
    
    Supported Sources:
    - NewsAPI (requires free API key)
    - Google News RSS (no key required)
    - Yahoo Finance News (via yfinance)
    - Seeking Alpha RSS (no key required)
    - Finnhub News (requires free API key)
    """
    
    NEWSAPI_URL = "https://newsapi.org/v2"
    FINNHUB_URL = "https://finnhub.io/api/v1"
    
    # Google News RSS endpoints
    GOOGLE_NEWS_RSS = {
        "top_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        "markets": "https://news.google.com/rss/search?q=stock+market",
        "crypto": "https://news.google.com/rss/search?q=cryptocurrency+bitcoin",
    }
    
    # Seeking Alpha RSS
    SEEKING_ALPHA_RSS = {
        "market_news": "https://seekingalpha.com/market_currents.xml",
        "stock_ideas": "https://seekingalpha.com/feed.xml"
    }
    
    def __init__(self):
        settings = get_settings()
        self.newsapi_key = settings.data_sources.newsapi_key
        self.finnhub_key = settings.data_sources.finnhub_api_key
    
    def _parse_rss(self, url: str, source: str) -> List[Dict[str, Any]]:
        """Parse RSS feed and return standardized news items"""
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo and not feed.entries:
                logger.warning(f"Failed to parse RSS from {source}: {feed.bozo_exception}")
                return []
            
            articles = []
            for entry in feed.entries[:20]:  # Limit to 20 most recent
                articles.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", entry.get("description", "")),
                    "url": entry.get("link", ""),
                    "source": source,
                    "published_at": entry.get("published", entry.get("updated", "")),
                    "author": entry.get("author", ""),
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RSS from {source}: {e}")
            return []
    
    def get_google_news(
        self,
        category: str = "markets",
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news from Google News RSS.
        
        Args:
            category: 'top_business', 'markets', 'crypto'
            query: Optional search query (overrides category)
        
        Returns:
            List of news articles
        """
        if query:
            # URL encode the query to handle spaces and special characters
            encoded_query = quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}"
        else:
            url = self.GOOGLE_NEWS_RSS.get(category, self.GOOGLE_NEWS_RSS["markets"])
        
        return self._parse_rss(url, "Google News")

    
    def get_seeking_alpha_news(
        self,
        feed_type: str = "market_news"
    ) -> List[Dict[str, Any]]:
        """
        Get news from Seeking Alpha RSS.
        
        Args:
            feed_type: 'market_news' or 'stock_ideas'
        
        Returns:
            List of news articles
        """
        url = self.SEEKING_ALPHA_RSS.get(feed_type, self.SEEKING_ALPHA_RSS["market_news"])
        return self._parse_rss(url, "Seeking Alpha")
    
    def get_newsapi_news(
        self,
        query: str = "stock market OR cryptocurrency",
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get news from NewsAPI (requires API key).
        
        Args:
            query: Search query
            language: Language code
            sort_by: 'publishedAt', 'relevancy', 'popularity'
            page_size: Number of results (max 100)
        
        Returns:
            List of news articles
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        try:
            params = {
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
                "apiKey": self.newsapi_key
            }
            
            response = requests.get(
                f"{self.NEWSAPI_URL}/everything",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message')}")
                return []
            
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "published_at": article.get("publishedAt"),
                    "author": article.get("author"),
                    "image_url": article.get("urlToImage")
                })
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def get_finnhub_news(
        self,
        category: str = "general",
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news from Finnhub (requires API key).
        
        Args:
            category: 'general', 'forex', 'crypto', 'merger'
            symbol: Optional stock symbol for company-specific news
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            List of news articles
        """
        if not self.finnhub_key:
            logger.warning("Finnhub API key not configured")
            return []
        
        try:
            if symbol:
                # Company-specific news
                endpoint = f"{self.FINNHUB_URL}/company-news"
                from_date = from_date or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                to_date = to_date or datetime.now().strftime("%Y-%m-%d")
                params = {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": self.finnhub_key
                }
            else:
                # General market news
                endpoint = f"{self.FINNHUB_URL}/news"
                params = {
                    "category": category,
                    "token": self.finnhub_key
                }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data[:20]:  # Limit to 20
                articles.append({
                    "title": article.get("headline"),
                    "description": article.get("summary"),
                    "url": article.get("url"),
                    "source": article.get("source", "Finnhub"),
                    "published_at": datetime.fromtimestamp(
                        article.get("datetime", 0)
                    ).isoformat() if article.get("datetime") else None,
                    "related": article.get("related"),
                    "image_url": article.get("image")
                })
            
            logger.info(f"Fetched {len(articles)} articles from Finnhub")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from Finnhub: {e}")
            return []
    
    def get_ticker_news(
        self,
        symbol: str,
        sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news for a specific ticker from all available sources.
        
        Args:
            symbol: Stock/crypto ticker symbol
            sources: List of sources to use (default: all)
        
        Returns:
            Combined list of news articles
        """
        sources = sources or ["google", "finnhub"]
        all_news = []
        
        if "google" in sources:
            google_news = self.get_google_news(query=f"{symbol} stock")
            all_news.extend(google_news)
        
        if "finnhub" in sources and self.finnhub_key:
            finnhub_news = self.get_finnhub_news(symbol=symbol)
            all_news.extend(finnhub_news)
        
        if "newsapi" in sources and self.newsapi_key:
            newsapi_news = self.get_newsapi_news(query=symbol)
            all_news.extend(newsapi_news)
        
        # Sort by date (newest first)
        all_news.sort(
            key=lambda x: x.get("published_at", ""),
            reverse=True
        )
        
        # Remove duplicates by title
        seen_titles = set()
        unique_news = []
        for article in all_news:
            title = article.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
        
        return unique_news
    
    def get_market_news(
        self,
        category: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Get general market news from all sources.
        
        Args:
            category: 'general', 'crypto', 'forex'
        
        Returns:
            Combined list of news articles
        """
        all_news = []
        
        # Google News
        if category == "crypto":
            all_news.extend(self.get_google_news(category="crypto"))
        else:
            all_news.extend(self.get_google_news(category="markets"))
        
        # Seeking Alpha
        all_news.extend(self.get_seeking_alpha_news())
        
        # Finnhub
        if self.finnhub_key:
            all_news.extend(self.get_finnhub_news(category=category))
        
        # NewsAPI
        if self.newsapi_key:
            query = "cryptocurrency bitcoin ethereum" if category == "crypto" else "stock market"
            all_news.extend(self.get_newsapi_news(query=query))
        
        # Sort and deduplicate
        all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        
        seen_titles = set()
        unique_news = []
        for article in all_news:
            title = article.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
        
        return unique_news[:50]  # Return top 50
