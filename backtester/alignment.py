"""
Price alignment policies.

Handles the temporal alignment of tick data across different instruments.
The data is tick-level (trade-level), meaning:
  - Multiple ticks can occur in the same second
  - Seconds can be skipped entirely

Alignment policies define how to create a uniform time grid and handle
missing data points.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class AlignmentPolicy(ABC):
    """Abstract interface for price alignment strategies.

    Different alignment policies can be plugged in to handle
    missing data differently.
    """

    @abstractmethod
    def align(
        self,
        data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Align tick data to a uniform 1-second time grid.

        Args:
            data: DataFrame with columns [timestamp, price] at minimum.
                  May have multiple rows per second or missing seconds.
            start_time: Start of the trading session.
            end_time: End of the trading session.

        Returns:
            DataFrame reindexed to a uniform 1-second grid from
            start_time to end_time (inclusive), with prices filled
            according to the policy.
        """
        ...


class ForwardFillAlignmentPolicy(AlignmentPolicy):
    """Forward-fill alignment: carry the last known price forward.

    For seconds with multiple ticks, the last tick price is used.
    For seconds with no ticks, the most recent previous price is carried forward.
    """

    def align(
        self,
        data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Align tick data using forward-fill.

        Steps:
        1. For each second with multiple ticks, keep only the last tick.
        2. Create a uniform 1-second grid.
        3. Reindex and forward-fill missing prices.

        Returns:
            DataFrame indexed by second-resolution timestamps with
            columns [price, volume, open_interest]. Volume and OI
            are summed/last per second respectively.
        """
        if data.empty:
            return pd.DataFrame(
                columns=["price", "volume", "open_interest"],
                index=pd.DatetimeIndex([], name="timestamp"),
            )

        # Ensure timestamp is the index
        df = data.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Floor to seconds, take last tick per second
        df.index = df.index.floor("s")
        df = df.groupby(level=0).agg({
            "price": "last",
            "volume": "sum",
            "open_interest": "last",
        })

        # Create uniform 1-second grid
        full_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq="s",
            name="timestamp",
        )

        # Reindex and forward-fill
        df = df.reindex(full_index)
        df["price"] = df["price"].ffill()
        df["volume"] = df["volume"].fillna(0)
        df["open_interest"] = df["open_interest"].ffill().fillna(0)

        return df


def get_alignment_policy(name: str) -> AlignmentPolicy:
    """Factory function to create alignment policies by name.

    Args:
        name: Policy name (e.g. "forward_fill").

    Returns:
        An AlignmentPolicy instance.

    Raises:
        ValueError: If the policy name is unknown.
    """
    policies = {
        "forward_fill": ForwardFillAlignmentPolicy,
    }
    if name not in policies:
        raise ValueError(
            f"Unknown alignment policy '{name}'. "
            f"Available: {list(policies.keys())}"
        )
    return policies[name]()
