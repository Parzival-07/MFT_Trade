"""
Unit tests for the backtesting engine.

Tests:
  1. Filename parser
  2. Strategy logic
  3. Portfolio accounting
  4. Price alignment
"""

import unittest
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

from backtester.parser import parse_instrument_name, is_relevant_option
from backtester.models import InstrumentMetadata, MarketSnapshot, Order, Position, Trade
from backtester.portfolio import Portfolio
from backtester.atm_straddle_strategy import ATMStraddleStrategy
from backtester.alignment import ForwardFillAlignmentPolicy


class TestParser(unittest.TestCase):
    """Tests for filename parsing."""

    def test_parse_nifty_pe(self):
        result = parse_instrument_name("NIFTY22110314550PE")
        self.assertIsNotNone(result)
        self.assertEqual(result.underlier, "NIFTY")
        self.assertEqual(result.expiry, date(2022, 11, 3))
        self.assertEqual(result.strike, 14550)
        self.assertEqual(result.option_type, "PE")

    def test_parse_banknifty_ce(self):
        result = parse_instrument_name("BANKNIFTY22112443200CE")
        self.assertIsNotNone(result)
        self.assertEqual(result.underlier, "BANKNIFTY")
        self.assertEqual(result.expiry, date(2022, 11, 24))
        self.assertEqual(result.strike, 43200)
        self.assertEqual(result.option_type, "CE")

    def test_parse_finnifty(self):
        result = parse_instrument_name("FINNIFTY22110719500CE")
        self.assertIsNotNone(result)
        self.assertEqual(result.underlier, "FINNIFTY")
        self.assertEqual(result.expiry, date(2022, 11, 7))
        self.assertEqual(result.strike, 19500)
        self.assertEqual(result.option_type, "CE")

    def test_parse_invalid(self):
        self.assertIsNone(parse_instrument_name("invalid"))
        self.assertIsNone(parse_instrument_name(""))
        self.assertIsNone(parse_instrument_name("NIFTY-I"))

    def test_instrument_name_roundtrip(self):
        result = parse_instrument_name("NIFTY22110318100CE")
        self.assertEqual(result.name, "NIFTY22110318100CE")

    def test_is_relevant_option(self):
        inst = parse_instrument_name("NIFTY22110318100CE")
        self.assertTrue(is_relevant_option(inst, "NIFTY", date(2022, 11, 3)))
        self.assertFalse(is_relevant_option(inst, "BANKNIFTY", date(2022, 11, 3)))
        self.assertFalse(is_relevant_option(inst, "NIFTY", date(2022, 11, 10)))


class TestPortfolio(unittest.TestCase):
    """Tests for portfolio accounting."""

    def setUp(self):
        self.portfolio = Portfolio(initial_capital=1000000.0)
        self.instrument = InstrumentMetadata(
            underlier="NIFTY",
            expiry=date(2022, 11, 3),
            strike=18100,
            option_type="CE",
        )
        self.timestamp = datetime(2022, 11, 1, 9, 15, 0)

    def test_initial_state(self):
        self.assertEqual(self.portfolio.cash, 1000000.0)
        self.assertEqual(self.portfolio.equity, 1000000.0)
        self.assertEqual(self.portfolio.total_pnl, 0.0)
        self.assertEqual(len(self.portfolio.positions), 0)

    def test_open_position(self):
        self.portfolio.open_position(
            self.instrument, quantity=1, price=100.0,
            timestamp=self.timestamp, transaction_cost=20.0,
        )
        self.assertEqual(len(self.portfolio.positions), 1)
        # Cash should be reduced by price * qty + txn_cost
        self.assertEqual(self.portfolio.cash, 1000000.0 - 100.0 - 20.0)
        self.assertTrue(self.portfolio.has_position(self.instrument))

    def test_close_position(self):
        self.portfolio.open_position(
            self.instrument, quantity=1, price=100.0,
            timestamp=self.timestamp, transaction_cost=20.0,
        )
        trade = self.portfolio.close_position(
            self.instrument, price=110.0,
            timestamp=self.timestamp + timedelta(seconds=60),
            transaction_cost=20.0,
        )
        self.assertIsNotNone(trade)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertFalse(self.portfolio.has_position(self.instrument))

        # PnL: (110 - 100) * 1 - 40 (entry + exit txn cost) = -30
        self.assertEqual(trade.gross_pnl, 10.0)
        self.assertEqual(trade.net_pnl, -30.0)

    def test_equity_with_position(self):
        self.portfolio.open_position(
            self.instrument, quantity=1, price=100.0,
            timestamp=self.timestamp, transaction_cost=20.0,
        )
        # Equity = Cash + MarketValue
        # Cash = 1000000 - 120 = 999880
        # MarketValue = 100 * 1 = 100
        # Equity = 999980
        self.assertAlmostEqual(self.portfolio.equity, 999980.0)

    def test_mtm_update(self):
        self.portfolio.open_position(
            self.instrument, quantity=1, price=100.0,
            timestamp=self.timestamp, transaction_cost=20.0,
        )
        # Update price to 150
        self.portfolio.update_market_prices({self.instrument.name: 150.0})
        pos = self.portfolio.get_position(self.instrument)
        self.assertEqual(pos.current_price, 150.0)
        self.assertEqual(pos.unrealized_pnl, 50.0)

    def test_close_nonexistent_position(self):
        trade = self.portfolio.close_position(
            self.instrument, price=100.0,
            timestamp=self.timestamp, transaction_cost=20.0,
        )
        self.assertIsNone(trade)


class TestATMStraddleStrategy(unittest.TestCase):
    """Tests for the ATM straddle strategy logic."""

    def setUp(self):
        self.strategy = ATMStraddleStrategy(quantity=1)
        self.portfolio = Portfolio(initial_capital=1000000.0)
        self.trading_date = date(2022, 11, 1)
        self.strategy.on_day_start(self.trading_date, "NIFTY")

    def _make_snapshot(
        self, futures_price, strikes, timestamp=None
    ) -> MarketSnapshot:
        if timestamp is None:
            timestamp = datetime(2022, 11, 1, 9, 15, 0)

        option_prices = {}
        for s in strikes:
            option_prices[(s, "CE")] = 100.0  # dummy price
            option_prices[(s, "PE")] = 100.0

        return MarketSnapshot(
            timestamp=timestamp,
            underlier="NIFTY",
            futures_price=futures_price,
            expiry=date(2022, 11, 3),
            available_strikes=strikes,
            option_prices=option_prices,
        )

    def test_initial_entry(self):
        snapshot = self._make_snapshot(
            futures_price=18100.0,
            strikes=[18000, 18050, 18100, 18150, 18200],
        )
        orders = self.strategy.on_tick(snapshot, self.portfolio)
        # Should generate 2 BUY orders (CE + PE)
        self.assertEqual(len(orders), 2)
        self.assertTrue(all(o.side == "BUY" for o in orders))
        self.assertTrue(
            any(o.instrument.option_type == "CE" for o in orders)
        )
        self.assertTrue(
            any(o.instrument.option_type == "PE" for o in orders)
        )

    def test_no_action_same_strike(self):
        snapshot = self._make_snapshot(
            futures_price=18100.0,
            strikes=[18000, 18050, 18100, 18150, 18200],
        )
        # First tick — entry
        self.strategy.on_tick(snapshot, self.portfolio)

        # Second tick — same strike, no action
        snapshot2 = self._make_snapshot(
            futures_price=18110.0,  # Still closest to 18100
            strikes=[18000, 18050, 18100, 18150, 18200],
            timestamp=datetime(2022, 11, 1, 9, 15, 1),
        )
        orders = self.strategy.on_tick(snapshot2, self.portfolio)
        self.assertEqual(len(orders), 0)

    def test_roll_on_strike_change(self):
        snapshot1 = self._make_snapshot(
            futures_price=18100.0,
            strikes=[18000, 18050, 18100, 18150, 18200],
        )
        self.strategy.on_tick(snapshot1, self.portfolio)

        # Futures moves to 18160 → ATM should be 18150
        snapshot2 = self._make_snapshot(
            futures_price=18160.0,
            strikes=[18000, 18050, 18100, 18150, 18200],
            timestamp=datetime(2022, 11, 1, 9, 15, 1),
        )
        orders = self.strategy.on_tick(snapshot2, self.portfolio)
        # Should have 2 SELL (old) + 2 BUY (new) = 4 orders
        self.assertEqual(len(orders), 4)
        sells = [o for o in orders if o.side == "SELL"]
        buys = [o for o in orders if o.side == "BUY"]
        self.assertEqual(len(sells), 2)
        self.assertEqual(len(buys), 2)
        self.assertEqual(sells[0].instrument.strike, 18100)
        self.assertEqual(buys[0].instrument.strike, 18150)

    def test_eod_close(self):
        snapshot = self._make_snapshot(
            futures_price=18100.0,
            strikes=[18000, 18050, 18100, 18150, 18200],
        )
        self.strategy.on_tick(snapshot, self.portfolio)

        eod_orders = self.strategy.on_day_end(snapshot, self.portfolio)
        self.assertEqual(len(eod_orders), 2)
        self.assertTrue(all(o.side == "SELL" for o in eod_orders))

    def test_atm_equidistant_picks_lower(self):
        snapshot = self._make_snapshot(
            futures_price=18125.0,  # Exactly between 18100 and 18150
            strikes=[18000, 18050, 18100, 18150, 18200],
        )
        orders = self.strategy.on_tick(snapshot, self.portfolio)
        # Should pick 18100 (lower strike when equidistant)
        buy_strikes = {o.instrument.strike for o in orders if o.side == "BUY"}
        self.assertEqual(buy_strikes, {18100})


class TestForwardFillAlignment(unittest.TestCase):
    """Tests for the forward-fill alignment policy."""

    def setUp(self):
        self.policy = ForwardFillAlignmentPolicy()

    def test_basic_alignment(self):
        data = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2022-11-01 09:15:00",
                "2022-11-01 09:15:02",  # gap at 09:15:01
                "2022-11-01 09:15:04",
            ]),
            "price": [100.0, 102.0, 104.0],
            "volume": [10, 20, 30],
            "open_interest": [1000, 1000, 1000],
        })

        start = datetime(2022, 11, 1, 9, 15, 0)
        end = datetime(2022, 11, 1, 9, 15, 4)

        result = self.policy.align(data, start, end)

        # Should have 5 rows (09:15:00 to 09:15:04)
        self.assertEqual(len(result), 5)

        # Forward-fill: 09:15:01 should have price 100.0
        self.assertEqual(result.loc[pd.Timestamp("2022-11-01 09:15:01"), "price"], 100.0)
        # 09:15:03 should have price 102.0
        self.assertEqual(result.loc[pd.Timestamp("2022-11-01 09:15:03"), "price"], 102.0)

    def test_multiple_ticks_per_second(self):
        data = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2022-11-01 09:15:00",
                "2022-11-01 09:15:00",
                "2022-11-01 09:15:00",
            ]),
            "price": [100.0, 101.0, 102.0],
            "volume": [10, 20, 30],
            "open_interest": [1000, 1000, 1000],
        })

        start = datetime(2022, 11, 1, 9, 15, 0)
        end = datetime(2022, 11, 1, 9, 15, 0)

        result = self.policy.align(data, start, end)
        # Should take the last price (102.0)
        self.assertEqual(result.iloc[0]["price"], 102.0)
        # Volume should be summed
        self.assertEqual(result.iloc[0]["volume"], 60)

    def test_empty_data(self):
        data = pd.DataFrame(
            columns=["timestamp", "price", "volume", "open_interest"]
        )
        start = datetime(2022, 11, 1, 9, 15, 0)
        end = datetime(2022, 11, 1, 9, 15, 5)

        result = self.policy.align(data, start, end)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
