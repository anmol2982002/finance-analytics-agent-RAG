"""
Finance Analytics Agent - Logging Configuration
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Path = Path("./logs")
):
    """
    Configure the application logger.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
        log_dir: Directory for log files
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler - all logs
        logger.add(
            log_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Error-only file handler
        logger.add(
            log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    return logger


# Create a contextualized logger for different modules
def get_logger(name: str):
    """Get a logger with context for a specific module"""
    return logger.bind(name=name)
