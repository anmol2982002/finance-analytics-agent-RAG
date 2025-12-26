"""
CoinGecko Crypto Data Fetcher
Fetches cryptocurrency data from CoinGecko API (free, no API key required for basic use)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from time import sleep
from src.utils import get_logger

logger = get_logger(__name__)


class CoinGeckoFetcher:
    """
    Fetches cryptocurrency data from CoinGecko.
    
    Features:
    - Market data (prices, volumes, market caps)
    - Historical OHLC data
    - Trending coins
    - Fear & Greed Index (via alternative.me)
    - Coin search and details
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    FEAR_GREED_URL = "https://api.alternative.me/fng"
    
    # Rate limiting: CoinGecko allows ~10-50 calls/min on free tier
    REQUEST_DELAY = 1.5  # seconds between requests
    
    def __init__(self):
        self._last_request_time = 0
        self._coin_list_cache: List[Dict] = []
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        now = datetime.now().timestamp()
        elapsed = now - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = datetime.now().timestamp()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited request to CoinGecko"""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            return {}
    
    def get_coin_list(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get list of all supported coins with their IDs.
        
        Returns:
            List of coins with id, symbol, name
        """
        if self._coin_list_cache and not force_refresh:
            return self._coin_list_cache
        
        data = self._make_request("coins/list")
        if data:
            self._coin_list_cache = data
            logger.info(f"Fetched {len(data)} coins from CoinGecko")
        return self._coin_list_cache
    
    def search_coin(self, query: str) -> List[Dict]:
        """
        Search for a coin by name or symbol.
        
        Args:
            query: Search term (e.g., 'bitcoin', 'btc')
        
        Returns:
            List of matching coins
        """
        coins = self.get_coin_list()
        query_lower = query.lower()
        
        matches = [
            coin for coin in coins
            if query_lower in coin["id"].lower() 
            or query_lower in coin["symbol"].lower()
            or query_lower in coin["name"].lower()
        ]
        
        return matches[:10]  # Return top 10 matches
    
    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get detailed data for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        
        Returns:
            Dict with price, market cap, volume, and more
        """
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        data = self._make_request(f"coins/{coin_id}", params)
        
        if not data:
            return {"coin_id": coin_id, "error": "No data returned"}
        
        market_data = data.get("market_data", {})
        
        return {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            
            # Current prices
            "current_price_usd": market_data.get("current_price", {}).get("usd"),
            "current_price_btc": market_data.get("current_price", {}).get("btc"),
            
            # Market metrics
            "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
            "market_cap_rank": data.get("market_cap_rank"),
            "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
            
            # Price changes
            "price_change_24h": market_data.get("price_change_percentage_24h"),
            "price_change_7d": market_data.get("price_change_percentage_7d"),
            "price_change_30d": market_data.get("price_change_percentage_30d"),
            "price_change_1y": market_data.get("price_change_percentage_1y"),
            
            # ATH/ATL
            "ath_usd": market_data.get("ath", {}).get("usd"),
            "ath_date": market_data.get("ath_date", {}).get("usd"),
            "ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd"),
            "atl_usd": market_data.get("atl", {}).get("usd"),
            "atl_date": market_data.get("atl_date", {}).get("usd"),
            
            # Supply
            "circulating_supply": market_data.get("circulating_supply"),
            "total_supply": market_data.get("total_supply"),
            "max_supply": market_data.get("max_supply"),
            
            # Community
            "twitter_followers": data.get("community_data", {}).get("twitter_followers"),
            "reddit_subscribers": data.get("community_data", {}).get("reddit_subscribers"),
            
            # Links
            "homepage": data.get("links", {}).get("homepage", [None])[0],
            "blockchain_site": data.get("links", {}).get("blockchain_site", [None])[0],
            
            "timestamp": datetime.now().isoformat()
        }
    
    def get_market_chart(
        self,
        coin_id: str,
        days: int = 365,
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get historical market data (prices, volumes, market caps).
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of data (1, 7, 14, 30, 90, 180, 365, max)
            vs_currency: Target currency (usd, btc, eth)
        
        Returns:
            DataFrame with timestamp, price, volume, market_cap columns
        """
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        data = self._make_request(f"coins/{coin_id}/market_chart", params)
        
        if not data or "prices" not in data:
            logger.warning(f"No market chart data for {coin_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "timestamp": [p[0] for p in data["prices"]],
            "price": [p[1] for p in data["prices"]],
            "volume": [v[1] for v in data.get("total_volumes", [])],
            "market_cap": [m[1] for m in data.get("market_caps", [])]
        })
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        logger.info(f"Fetched {len(df)} data points for {coin_id}")
        return df
    
    def get_ohlc(
        self,
        coin_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get OHLC (candlestick) data.
        
        Args:
            coin_id: CoinGecko coin ID
            days: 1, 7, 14, 30, 90, 180, 365
            vs_currency: Target currency
        
        Returns:
            DataFrame with Open, High, Low, Close columns
        """
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        data = self._make_request(f"coins/{coin_id}/ohlc", params)
        
        if not data:
            logger.warning(f"No OHLC data for {coin_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def get_trending(self) -> List[Dict[str, Any]]:
        """
        Get trending coins (top 7 by search in last 24h).
        
        Returns:
            List of trending coins with basic info
        """
        data = self._make_request("search/trending")
        
        if not data or "coins" not in data:
            return []
        
        trending = []
        for item in data["coins"]:
            coin = item.get("item", {})
            trending.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol"),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "price_btc": coin.get("price_btc"),
                "score": coin.get("score"),  # Trending rank (0-6)
                "thumb": coin.get("thumb")
            })
        
        return trending
    
    def get_market_overview(
        self,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 50,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get market overview with top coins.
        
        Args:
            vs_currency: Target currency
            order: Sort order (market_cap_desc, volume_desc, etc.)
            per_page: Results per page (max 250)
            page: Page number
        
        Returns:
            List of coins with market data
        """
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d"
        }
        
        data = self._make_request("coins/markets", params)
        
        if not data:
            return []
        
        return [
            {
                "id": coin.get("id"),
                "symbol": coin.get("symbol"),
                "name": coin.get("name"),
                "current_price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "total_volume": coin.get("total_volume"),
                "price_change_1h": coin.get("price_change_percentage_1h_in_currency"),
                "price_change_24h": coin.get("price_change_percentage_24h"),
                "price_change_7d": coin.get("price_change_percentage_7d_in_currency"),
                "ath": coin.get("ath"),
                "ath_change_percentage": coin.get("ath_change_percentage"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply")
            }
            for coin in data
        ]
    
    def get_fear_greed_index(self, limit: int = 30) -> pd.DataFrame:
        """
        Get Fear & Greed Index from Alternative.me.
        
        Args:
            limit: Number of days of history
        
        Returns:
            DataFrame with date, value, classification
        """
        try:
            response = requests.get(
                self.FEAR_GREED_URL,
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df["value"] = df["value"].astype(int)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            return df[["value", "value_classification"]]
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def get_global_market_data(self) -> Dict[str, Any]:
        """
        Get global cryptocurrency market data.
        
        Returns:
            Dict with total market cap, volume, BTC dominance, etc.
        """
        data = self._make_request("global")
        
        if not data or "data" not in data:
            return {}
        
        global_data = data["data"]
        
        return {
            "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd"),
            "bitcoin_dominance": global_data.get("market_cap_percentage", {}).get("btc"),
            "ethereum_dominance": global_data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
            "markets": global_data.get("markets"),
            "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd"),
            "timestamp": datetime.now().isoformat()
        }
