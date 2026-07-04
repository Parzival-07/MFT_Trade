"""
Configuration loader for the backtesting engine.

Loads settings from a JSON config file and provides typed access
to all configurable parameters.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DateRange:
    """Date range for the backtest."""
    start: str
    end: str


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the backtesting engine.

    All parameters that control the backtest are centralized here.
    """
    data_root: Path
    results_dir: Path
    underliers: List[str]
    initial_capital: float
    quantity: int
    transaction_cost: float
    slippage: float
    alignment_policy: str
    trading_start_time: str
    trading_end_time: str
    date_range: DateRange
    futures_folder_name: str
    options_folder_name: str
    logging_level: str

    @staticmethod
    def from_json(config_path: Path) -> "Config":
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the config.json file.

        Returns:
            A Config instance with all parameters loaded.
        """
        with open(config_path, "r") as f:
            raw = json.load(f)

        base_dir = config_path.parent

        return Config(
            data_root=base_dir / raw["data_root"],
            results_dir=base_dir / raw["results_dir"],
            underliers=raw["underliers"],
            initial_capital=float(raw["initial_capital"]),
            quantity=int(raw["quantity"]),
            transaction_cost=float(raw["transaction_cost"]),
            slippage=float(raw["slippage"]),
            alignment_policy=raw["alignment_policy"],
            trading_start_time=raw["trading_start_time"],
            trading_end_time=raw["trading_end_time"],
            date_range=DateRange(
                start=raw["date_range"]["start"],
                end=raw["date_range"]["end"],
            ),
            futures_folder_name=raw["futures_folder_name"],
            options_folder_name=raw["options_folder_name"],
            logging_level=raw.get("logging_level", "INFO"),
        )
