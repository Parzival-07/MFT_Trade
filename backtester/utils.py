"""
Utility functions.

Common helpers used across the backtesting engine.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path = None) -> None:
    """Configure logging for the backtesting engine.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Optional directory to write log files.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    # Force UTF-8 on Windows to avoid cp1252 encoding errors
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "backtest.log", mode="w", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
