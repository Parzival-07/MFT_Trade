"""
Lightweight event classes for the event-driven backtesting engine.

Events flow through the system in this sequence:
  MarketEvent → SignalEvent → OrderEvent → FillEvent

"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from backtester.models import MarketSnapshot, Order


@dataclass
class MarketEvent:
    """A new market data tick has arrived.

    Produced by: DataHandler
    Consumed by: Strategy
    """
    snapshot: MarketSnapshot


@dataclass
class SignalEvent:
    """The strategy has produced trading signals (orders).

    Produced by: Strategy
    Consumed by: ExecutionHandler
    """
    timestamp: datetime
    orders: List[Order]


@dataclass
class OrderEvent:
    """An order is ready for execution.

    Produced by: Strategy (wrapped by engine)
    Consumed by: ExecutionHandler
    """
    order: Order


@dataclass
class FillEvent:
    """An order has been filled (executed).

    Produced by: ExecutionHandler
    Consumed by: Portfolio
    """
    timestamp: datetime
    order: Order
    fill_price: float
    transaction_cost: float
    slippage: float
