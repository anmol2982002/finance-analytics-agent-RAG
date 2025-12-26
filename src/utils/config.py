"""
Finance Analytics Agent - Configuration Management
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMSettings(BaseSettings):
    """LLM API Configuration"""
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    # Model settings
    default_model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 4096
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class DataSourceSettings(BaseSettings):
    """Data Source API Configuration"""
    alpha_vantage_api_key: Optional[str] = Field(default=None, alias="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, alias="FINNHUB_API_KEY")
    newsapi_key: Optional[str] = Field(default=None, alias="NEWSAPI_KEY")
    
    # Reddit
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="FinanceAnalyticsAgent/1.0", alias="REDDIT_USER_AGENT")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class StorageSettings(BaseSettings):
    """Storage Configuration"""
    vector_db_path: Path = Field(default=Path("./data/vectordb"), alias="VECTOR_DB_PATH")
    raw_data_path: Path = Path("./data/raw")
    processed_data_path: Path = Path("./data/processed")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def ensure_directories(self):
        """Create storage directories if they don't exist"""
        for path in [self.vector_db_path, self.raw_data_path, self.processed_data_path]:
            path.mkdir(parents=True, exist_ok=True)


class AppSettings(BaseSettings):
    """Application Settings"""
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data_sources: DataSourceSettings = Field(default_factory=DataSourceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get or create the global settings instance"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        _settings.storage.ensure_directories()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload settings from environment"""
    global _settings
    _settings = None
    return get_settings()
