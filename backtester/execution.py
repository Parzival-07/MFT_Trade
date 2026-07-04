"""
Execution handler.

Responsible for:
  - Executing orders (immediate fill assumed)
  - Applying transaction costs and slippage
  - Updating the portfolio

This module contains NO trading logic — it only processes fills.
"""

import logging
from typing import List, Optional

from backtester.config import Config
from backtester.events import FillEvent
from backtester.models import Order, Trade
from backtester.portfolio import Portfolio

logger = logging.getLogger(__name__)


class ExecutionHandler:
    """Handles order execution with transaction costs.

    Assumes immediate fills at the specified execution price.
    No partial fills, no order queue, no market impact modeling.
    """

    def __init__(self, config: Config):
        """Initialize the execution handler.

        Args:
            config: Application configuration (for transaction costs, slippage).
        """
        self._transaction_cost = config.transaction_cost
        self._slippage = config.slippage

    def execute_orders(
        self,
        orders: List[Order],
        portfolio: Portfolio,
    ) -> List[FillEvent]:
        """Execute a list of orders against the portfolio.

        BUY orders open positions.
        SELL orders close positions.

        Args:
            orders: Orders to execute.
            portfolio: Portfolio to update.

        Returns:
            List of FillEvent objects documenting each execution.
        """
        fills: List[FillEvent] = []

        for order in orders:
            fill_price = self._apply_slippage(order)
            fill = self._execute_single(order, fill_price, portfolio)
            if fill:
                fills.append(fill)

        return fills

    def _apply_slippage(self, order: Order) -> float:
        """Apply slippage to the execution price.

        BUY: price increases by slippage (worse fill)
        SELL: price decreases by slippage (worse fill)
        """
        if order.side == "BUY":
            return order.execution_price + self._slippage
        else:
            return order.execution_price - self._slippage

    def _execute_single(
        self,
        order: Order,
        fill_price: float,
        portfolio: Portfolio,
    ) -> Optional[FillEvent]:
        """Execute a single order.

        Args:
            order: The order to execute.
            fill_price: Price after slippage.
            portfolio: Portfolio to update.

        Returns:
            FillEvent if execution succeeded, None otherwise.
        """
        if order.side == "BUY":
            portfolio.open_position(
                instrument=order.instrument,
                quantity=order.quantity,
                price=fill_price,
                timestamp=order.timestamp,
                transaction_cost=self._transaction_cost,
            )
        elif order.side == "SELL":
            trade = portfolio.close_position(
                instrument=order.instrument,
                price=fill_price,
                timestamp=order.timestamp,
                transaction_cost=self._transaction_cost,
            )
            if trade is None:
                logger.warning(
                    f"Failed to close position for {order.instrument.name}"
                )
                return None
        else:
            logger.error(f"Unknown order side: {order.side}")
            return None

        fill = FillEvent(
            timestamp=order.timestamp,
            order=order,
            fill_price=fill_price,
            transaction_cost=self._transaction_cost,
            slippage=self._slippage,
        )

        logger.debug(
            f"FILL: {order.side} {order.instrument.name} "
            f"qty={order.quantity} @ {fill_price:.2f} "
            f"txn_cost={self._transaction_cost:.2f} "
            f"reason={order.reason}"
        )

        return fill
