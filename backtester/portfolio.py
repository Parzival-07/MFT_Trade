"""
Portfolio management.

Responsible for:
  - Tracking current positions
  - Cash management
  - Equity calculation (Cash + Market Value of Open Positions)
  - Realized and unrealized PnL tracking
  - Trade recording

The portfolio is a pure accounting module.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backtester.models import InstrumentMetadata, Order, Position, Trade

logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks positions, cash, and PnL.

    Positions are keyed by instrument name (string).
    The portfolio provides methods for the execution handler to update
    state after fills, and for the strategy to query current state.
    """

    def __init__(self, initial_capital: float):
        """Initialize portfolio with starting cash.

        Args:
            initial_capital: Starting cash balance.
        """
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._total_transaction_costs = 0.0

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def initial_capital(self) -> float:
        """Starting capital."""
        return self._initial_capital

    @property
    def positions(self) -> Dict[str, Position]:
        """Current open positions (read-only view)."""
        return dict(self._positions)

    @property
    def trades(self) -> List[Trade]:
        """All completed trades."""
        return list(self._trades)

    @property
    def total_transaction_costs(self) -> float:
        """Total transaction costs incurred."""
        return self._total_transaction_costs

    def has_position(self, instrument: InstrumentMetadata) -> bool:
        """Check if we currently hold a position in this instrument."""
        return instrument.name in self._positions

    def get_position(self, instrument: InstrumentMetadata) -> Optional[Position]:
        """Get current position for an instrument, if any."""
        return self._positions.get(instrument.name)

    def get_positions_for_underlier(self, underlier: str) -> List[Position]:
        """Get all open positions for a given underlier."""
        return [
            pos for pos in self._positions.values()
            if pos.instrument.underlier == underlier
        ]

    @property
    def market_value(self) -> float:
        """Total market value of all open positions."""
        return sum(pos.market_value for pos in self._positions.values())

    @property
    def equity(self) -> float:
        """Current equity = Cash + Market Value of Open Positions."""
        return self._cash + self.market_value

    @property
    def total_pnl(self) -> float:
        """Total PnL = Current Equity - Initial Capital."""
        return self.equity - self._initial_capital

    @property
    def realized_pnl(self) -> float:
        """Sum of all realized trade PnLs."""
        return sum(t.net_pnl for t in self._trades)

    @property
    def unrealized_pnl(self) -> float:
        """Sum of unrealized PnL across all open positions."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def open_position(
        self,
        instrument: InstrumentMetadata,
        quantity: int,
        price: float,
        timestamp: datetime,
        transaction_cost: float,
    ) -> None:
        """Open a new position (BUY).

        Args:
            instrument: The instrument to buy.
            quantity: Number of units.
            price: Execution price per unit.
            timestamp: Time of execution.
            transaction_cost: Cost of this transaction.
        """
        key = instrument.name

        if key in self._positions:
            logger.warning(f"Position already exists for {key}, skipping open")
            return

        # Deduct cost: price * quantity + transaction cost
        cost = price * quantity + transaction_cost
        self._cash -= cost
        self._total_transaction_costs += transaction_cost

        self._positions[key] = Position(
            instrument=instrument,
            quantity=quantity,
            average_price=price,
            current_price=price,
            entry_time=timestamp,
        )

        logger.debug(
            f"OPENED {key}: qty={quantity} @ {price:.2f}, "
            f"cost={cost:.2f}, cash={self._cash:.2f}"
        )

    def close_position(
        self,
        instrument: InstrumentMetadata,
        price: float,
        timestamp: datetime,
        transaction_cost: float,
    ) -> Optional[Trade]:
        """Close an existing position (SELL).

        Args:
            instrument: The instrument to sell.
            price: Execution price per unit.
            timestamp: Time of execution.
            transaction_cost: Cost of this transaction.

        Returns:
            The completed Trade, or None if no position exists.
        """
        key = instrument.name
        pos = self._positions.get(key)

        if pos is None:
            logger.warning(f"No position to close for {key}")
            return None

        # Add proceeds: price * quantity - transaction cost
        proceeds = price * pos.quantity - transaction_cost
        self._cash += proceeds
        self._total_transaction_costs += transaction_cost

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            instrument=instrument,
            entry_price=pos.average_price,
            exit_price=price,
            quantity=pos.quantity,
            transaction_costs=transaction_cost * 2,  # entry + exit
        )
        self._trades.append(trade)

        del self._positions[key]

        logger.debug(
            f"CLOSED {key}: qty={pos.quantity} @ {price:.2f}, "
            f"pnl={trade.net_pnl:.2f}, cash={self._cash:.2f}"
        )
        return trade

    def update_market_prices(
        self, price_map: Dict[str, float]
    ) -> None:
        """Update current prices for all held positions (MTM).

        Args:
            price_map: Mapping of instrument_name → latest price.
        """
        for key, pos in self._positions.items():
            if key in price_map:
                pos.current_price = price_map[key]

    def reset_for_new_day(self) -> None:
        """Verify all positions are closed at start of new day.

        Raises:
            RuntimeError: If positions remain open (indicates a bug).
        """
        if self._positions:
            open_names = list(self._positions.keys())
            logger.error(
                f"Positions still open at day start: {open_names}"
            )
            raise RuntimeError(
                f"Cannot start new day with open positions: {open_names}"
            )
