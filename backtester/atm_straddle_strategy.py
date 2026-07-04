"""
ATM Straddle Trading Strategy.

Implements the specific trading logic:
  - At every second, find the ATM strike (closest to futures price)
  - Hold exactly 1 CE + 1 PE at that strike (long straddle)
  - If ATM strike changes → roll (sell old, buy new)
  - Close all at end of day

"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from backtester.models import InstrumentMetadata, MarketSnapshot, Order
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

logger = logging.getLogger(__name__)


class ATMStraddleStrategy(Strategy):
    """ATM Straddle: always hold 1 CE + 1 PE at the nearest strike.

    State tracked per day:
      - current_strike: The strike we are currently holding
      - current_expiry: The expiry of instruments we hold
    """

    def __init__(self, quantity: int = 1):
        """Initialize the ATM Straddle strategy.

        Args:
            quantity: Number of lots per leg. Default = 1.
        """
        self._quantity = quantity
        self._current_strike: Optional[int] = None
        self._current_expiry: Optional[date] = None
        self._underlier: Optional[str] = None
        self._roll_count = 0

    @property
    def roll_count(self) -> int:
        """Number of rolls performed on current day."""
        return self._roll_count

    def on_day_start(self, trading_date, underlier: str) -> None:
        """Reset state for a new trading day."""
        self._current_strike = None
        self._current_expiry = None
        self._underlier = underlier
        self._roll_count = 0
        logger.info(
            f"Strategy reset for {underlier} on {trading_date}"
        )

    def _find_atm_strike(self, snapshot: MarketSnapshot) -> int:
        """Find the ATM strike closest to the current futures price.

        ATM = argmin(|strike - futures_price|)

        If equidistant, picks the lower strike.
        """
        futures_price = snapshot.futures_price
        strikes = snapshot.available_strikes

        atm = min(strikes, key=lambda s: (abs(s - futures_price), s))
        return atm

    def _make_instrument(
        self, strike: int, opt_type: str, expiry: date, underlier: str
    ) -> InstrumentMetadata:
        """Create an InstrumentMetadata for the given parameters."""
        return InstrumentMetadata(
            underlier=underlier,
            expiry=expiry,
            strike=strike,
            option_type=opt_type,
        )

    def on_tick(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
    ) -> List[Order]:
        """Process a market tick and generate orders if ATM strike changed.

        Logic:
          1. Compute ATM strike from futures price.
          2. If we have no position → buy the straddle.
          3. If ATM strike changed → sell old straddle, buy new one (roll).
          4. If ATM strike unchanged → do nothing.
        """
        orders: List[Order] = []
        underlier = snapshot.underlier
        expiry = snapshot.expiry
        timestamp = snapshot.timestamp

        atm_strike = self._find_atm_strike(snapshot)

        # Case 1: No current position — initial entry
        if self._current_strike is None:
            orders.extend(
                self._generate_entry_orders(
                    atm_strike, expiry, underlier, timestamp, snapshot
                )
            )
            self._current_strike = atm_strike
            self._current_expiry = expiry
            logger.info(
                f"[{timestamp}] {underlier} ENTRY at strike {atm_strike} "
                f"(futures={snapshot.futures_price:.2f})"
            )
            return orders

        # Case 2: ATM strike has changed — roll
        if atm_strike != self._current_strike:
            # 1. Attempt to generate the orders
            exit_orders = self._generate_exit_orders(
                self._current_strike,
                self._current_expiry,
                underlier,
                timestamp,
                snapshot,
                reason="roll",
            )
            entry_orders = self._generate_entry_orders(
                atm_strike, expiry, underlier, timestamp, snapshot
            )
            
            # 2. ONLY update state if BOTH exit and entry orders were successfully generated
            if exit_orders and entry_orders:
                orders.extend(exit_orders)
                orders.extend(entry_orders)
                self._roll_count += 1
                logger.info(
                    f"[{timestamp}] {underlier} ROLL {self._current_strike} -> "
                    f"{atm_strike} (futures={snapshot.futures_price:.2f}, "
                    f"roll #{self._roll_count})"
                )
                self._current_strike = atm_strike
                self._current_expiry = expiry
            else:
                logger.warning(
                    f"[{timestamp}] {underlier} ROLL ABORTED: Missing option prices."
                )
                
            return orders

        # Case 3: Same strike — do nothing
        return orders

    def on_day_end(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
    ) -> List[Order]:
        """Close all positions at end of day."""
        orders: List[Order] = []

        if self._current_strike is not None:
            orders.extend(
                self._generate_exit_orders(
                    self._current_strike,
                    self._current_expiry,
                    snapshot.underlier,
                    snapshot.timestamp,
                    snapshot,
                    reason="eod_close",
                )
            )
            logger.info(
                f"[{snapshot.timestamp}] {snapshot.underlier} EOD CLOSE "
                f"at strike {self._current_strike}"
            )
            self._current_strike = None
            self._current_expiry = None

        return orders

    def _generate_entry_orders(
        self,
        strike: int,
        expiry: date,
        underlier: str,
        timestamp: datetime,
        snapshot: MarketSnapshot,
    ) -> List[Order]:
        """Generate BUY orders for a CE+PE straddle."""
        orders = []
        for opt_type in ["CE", "PE"]:
            inst = self._make_instrument(strike, opt_type, expiry, underlier)
            price_key = (strike, opt_type)

            if price_key not in snapshot.option_prices:
                logger.warning(
                    f"No price for {inst.name} at {timestamp}, skipping"
                )
                continue

            price = snapshot.option_prices[price_key]
            orders.append(Order(
                timestamp=timestamp,
                instrument=inst,
                side="BUY",
                quantity=self._quantity,
                execution_price=price,
                reason="entry",
            ))
        return orders

    def _generate_exit_orders(
        self,
        strike: int,
        expiry: date,
        underlier: str,
        timestamp: datetime,
        snapshot: MarketSnapshot,
        reason: str = "exit",
    ) -> List[Order]:
        """Generate SELL orders for a CE+PE straddle."""
        orders = []
        for opt_type in ["CE", "PE"]:
            inst = self._make_instrument(strike, opt_type, expiry, underlier)
            price_key = (strike, opt_type)

            if price_key not in snapshot.option_prices:
                logger.warning(
                    f"No price for {inst.name} at {timestamp} for exit, "
                    f"using last known price"
                )
                continue

            price = snapshot.option_prices[price_key]
            orders.append(Order(
                timestamp=timestamp,
                instrument=inst,
                side="SELL",
                quantity=self._quantity,
                execution_price=price,
                reason=reason,
            ))
        return orders
