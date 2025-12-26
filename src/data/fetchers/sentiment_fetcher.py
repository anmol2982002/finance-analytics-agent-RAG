"""
Sentiment Data Fetcher
Fetches sentiment data from social media and alternative sources
"""

import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
from src.utils import get_logger, get_settings

logger = get_logger(__name__)


class SentimentFetcher:
    """
    Fetches sentiment data from various sources.
    
    Supported Sources:
    - Reddit (via PRAW)
    - Fear & Greed Index
    - Google Trends (basic)
    """
    
    FEAR_GREED_URL = "https://api.alternative.me/fng"
    
    def __init__(self):
        settings = get_settings()
        self.reddit_client_id = settings.data_sources.reddit_client_id
        self.reddit_client_secret = settings.data_sources.reddit_client_secret
        self.reddit_user_agent = settings.data_sources.reddit_user_agent
        self._reddit = None
    
    def _get_reddit(self):
        """Lazy initialization of Reddit client"""
        if self._reddit is None:
            if not self.reddit_client_id or not self.reddit_client_secret:
                logger.warning("Reddit credentials not configured")
                return None
            
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
            except ImportError:
                logger.error("praw not installed. Run: pip install praw")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                return None
        
        return self._reddit
    
    def get_fear_greed_index(self, limit: int = 1) -> Dict[str, Any]:
        """
        Get the latest Fear & Greed Index value.
        
        Args:
            limit: Number of days to fetch (default: 1 for latest)
        
        Returns:
            Dict with value, classification, and timestamp
        """
        try:
            response = requests.get(
                self.FEAR_GREED_URL,
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not data["data"]:
                return {"error": "No data returned"}
            
            latest = data["data"][0]
            
            return {
                "value": int(latest.get("value", 0)),
                "classification": latest.get("value_classification", "Unknown"),
                "timestamp": datetime.fromtimestamp(
                    int(latest.get("timestamp", 0))
                ).isoformat(),
                "time_until_update": latest.get("time_until_update"),
                "interpretation": self._interpret_fear_greed(int(latest.get("value", 50)))
            }
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {"error": str(e)}
    
    def _interpret_fear_greed(self, value: int) -> str:
        """Interpret Fear & Greed value"""
        if value <= 25:
            return "Extreme fear indicates potential buying opportunities as markets may be oversold."
        elif value <= 45:
            return "Fear suggests cautious sentiment. Market may present value opportunities."
        elif value <= 55:
            return "Neutral sentiment. Market is balanced between fear and greed."
        elif value <= 75:
            return "Greed indicates bullish sentiment. Consider taking some profits."
        else:
            return "Extreme greed suggests potential market top. High risk of correction."
    
    def get_reddit_sentiment(
        self,
        subreddit: str = "wallstreetbets",
        query: Optional[str] = None,
        time_filter: str = "day",
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Get sentiment from Reddit posts.
        
        Args:
            subreddit: Subreddit to search (wallstreetbets, cryptocurrency, etc.)
            query: Optional search query
            time_filter: 'hour', 'day', 'week', 'month'
            limit: Number of posts to fetch
        
        Returns:
            Dict with posts and basic sentiment metrics
        """
        reddit = self._get_reddit()
        if not reddit:
            return {"error": "Reddit client not available"}
        
        try:
            sub = reddit.subreddit(subreddit)
            
            if query:
                posts = list(sub.search(query, time_filter=time_filter, limit=limit))
            else:
                posts = list(sub.hot(limit=limit))
            
            post_data = []
            total_score = 0
            total_comments = 0
            
            for post in posts:
                post_data.append({
                    "title": post.title,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "url": f"https://reddit.com{post.permalink}",
                    "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "flair": post.link_flair_text
                })
                total_score += post.score
                total_comments += post.num_comments
            
            avg_score = total_score / len(posts) if posts else 0
            avg_comments = total_comments / len(posts) if posts else 0
            
            return {
                "subreddit": subreddit,
                "query": query,
                "post_count": len(posts),
                "avg_score": avg_score,
                "avg_comments": avg_comments,
                "total_engagement": total_score + total_comments,
                "posts": post_data[:10],  # Return top 10 posts
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return {"error": str(e)}
    
    def get_wsb_sentiment(
        self,
        ticker: Optional[str] = None,
        time_filter: str = "day"
    ) -> Dict[str, Any]:
        """
        Get sentiment from r/wallstreetbets.
        
        Args:
            ticker: Optional ticker to search for
            time_filter: Time filter for search
        
        Returns:
            Sentiment data from WSB
        """
        return self.get_reddit_sentiment(
            subreddit="wallstreetbets",
            query=ticker,
            time_filter=time_filter
        )
    
    def get_crypto_reddit_sentiment(
        self,
        coin: Optional[str] = None,
        time_filter: str = "day"
    ) -> Dict[str, Any]:
        """
        Get sentiment from r/cryptocurrency.
        
        Args:
            coin: Optional coin name to search for
            time_filter: Time filter for search
        
        Returns:
            Sentiment data from r/cryptocurrency
        """
        return self.get_reddit_sentiment(
            subreddit="cryptocurrency",
            query=coin,
            time_filter=time_filter
        )
    
    def get_combined_sentiment(
        self,
        ticker_or_coin: str,
        asset_type: str = "stock"
    ) -> Dict[str, Any]:
        """
        Get combined sentiment from all available sources.
        
        Args:
            ticker_or_coin: Ticker symbol or coin name
            asset_type: 'stock' or 'crypto'
        
        Returns:
            Combined sentiment data
        """
        result = {
            "asset": ticker_or_coin,
            "asset_type": asset_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Fear & Greed Index (always available)
        result["fear_greed"] = self.get_fear_greed_index()
        
        # Reddit sentiment based on asset type
        if asset_type == "crypto":
            result["reddit"] = self.get_crypto_reddit_sentiment(
                coin=ticker_or_coin
            )
        else:
            result["reddit"] = self.get_wsb_sentiment(
                ticker=ticker_or_coin
            )
        
        # Calculate overall sentiment score (simple weighted average)
        scores = []
        
        # Fear & Greed contributes
        if "value" in result["fear_greed"]:
            # Normalize to 0-100 scale (already is for F&G)
            scores.append(result["fear_greed"]["value"])
        
        # Reddit engagement score
        if "avg_score" in result.get("reddit", {}):
            reddit_data = result["reddit"]
            if reddit_data["post_count"] > 0:
                # Normalize based on typical engagement
                engagement_score = min(
                    100,
                    (reddit_data["avg_score"] / 100) * 50 + 
                    (reddit_data["avg_comments"] / 50) * 50
                )
                scores.append(engagement_score)
        
        if scores:
            result["overall_sentiment_score"] = sum(scores) / len(scores)
            result["sentiment_label"] = self._label_sentiment(
                result["overall_sentiment_score"]
            )
        
        return result
    
    def _label_sentiment(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score <= 25:
            return "Very Bearish"
        elif score <= 40:
            return "Bearish"
        elif score <= 60:
            return "Neutral"
        elif score <= 75:
            return "Bullish"
        else:
            return "Very Bullish"
