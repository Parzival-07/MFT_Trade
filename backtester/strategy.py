"""
Abstract strategy interface.

Defines the contract that all trading strategies must implement.
Strategies receive market data and produce orders — they never
modify the portfolio directly.
"""

from abc import ABC, abstractmethod
from typing import List

from backtester.models import MarketSnapshot, Order
from backtester.portfolio import Portfolio


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Strategies are stateful objects that:
      1. Receive market snapshots via on_tick()
      2. Decide what trades to make
      3. Return a list of Order objects

    Strategies must NOT modify the portfolio directly.
    They must NOT execute trades — only signal intent via orders.
    """

    @abstractmethod
    def on_tick(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
    ) -> List[Order]:
        """Process a new market snapshot and generate orders.

        Args:
            snapshot: The current market state.
            portfolio: Read-only view of current positions (for decision-making).

        Returns:
            List of Order objects to be executed. Empty list means no action.
        """
        ...

    @abstractmethod
    def on_day_end(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
    ) -> List[Order]:
        """Generate end-of-day close orders.

        Called at the last tick of each trading day.

        Args:
            snapshot: The last market snapshot of the day.
            portfolio: Current portfolio state.

        Returns:
            List of orders to close all remaining positions.
        """
        ...

    @abstractmethod
    def on_day_start(self, trading_date, underlier: str) -> None:
        """Reset strategy state for a new trading day.

        Args:
            trading_date: The new trading date.
            underlier: The underlier being traded.
        """
        ...
