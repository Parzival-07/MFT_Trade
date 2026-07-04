"""
Main entry point for the backtesting engine.

Usage:
    python main.py [--config config.json]
"""

import argparse
import sys
import time
from pathlib import Path

from backtester.alignment import get_alignment_policy
from backtester.atm_straddle_strategy import ATMStraddleStrategy
from backtester.config import Config
from backtester.data_handler import DataHandler
from backtester.engine import BacktestEngine
from backtester.execution import ExecutionHandler
from backtester.performance import PerformanceAnalyzer
from backtester.portfolio import Portfolio
from backtester.utils import setup_logging


def main():
    """Run the backtest."""
    parser = argparse.ArgumentParser(
        description="Event-Driven Backtesting Engine for NSE Options"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = Config.from_json(config_path)

    # Setup logging
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level=config.logging_level,
        log_dir=results_dir / "logs",
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Data root: {config.data_root}")
    logger.info(f"Underliers: {config.underliers}")
    logger.info(f"Date range: {config.date_range.start} to {config.date_range.end}")
    logger.info(f"Initial capital: Rs.{config.initial_capital:,.0f}")

    # Create alignment policy (pluggable)
    alignment_policy = get_alignment_policy(config.alignment_policy)
    logger.info(f"Alignment policy: {config.alignment_policy}")

    # Create data handler
    data_handler = DataHandler(config, alignment_policy)

    # Create portfolio
    portfolio = Portfolio(config.initial_capital)

    # Create execution handler
    execution_handler = ExecutionHandler(config)

    # Create strategies (one per underlier)
    strategies = {}
    for underlier in config.underliers:
        strategies[underlier] = ATMStraddleStrategy(
            quantity=config.quantity,
        )
    logger.info(f"Strategies created: {list(strategies.keys())}")

    # Create performance analyzer
    performance = PerformanceAnalyzer(
        initial_capital=config.initial_capital,
        results_dir=results_dir,
    )

    # Create and run the engine
    engine = BacktestEngine(
        config=config,
        strategies=strategies,
        portfolio=portfolio,
        execution_handler=execution_handler,
        data_handler=data_handler,
        performance=performance,
    )

    start_time = time.time()
    engine.run()
    elapsed = time.time() - start_time

    logger.info(f"Backtest completed in {elapsed:.1f} seconds")
    logger.info(f"Results saved to: {results_dir.absolute()}")


if __name__ == "__main__":
    main()
