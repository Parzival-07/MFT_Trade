"""
Backtest Engine — the main orchestrator.

Responsible ONLY for orchestration. It does NOT contain:
  - Trading logic (that's in Strategy)
  - Execution logic (that's in ExecutionHandler)
  - Portfolio accounting (that's in Portfolio)
  - Performance computation (that's in PerformanceAnalyzer)

Loop structure:
  For each day →
    For each underlier →
      For each MarketSnapshot →
        Strategy → Orders → Execution → Portfolio → Performance Recording
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from backtester.alignment import get_alignment_policy
from backtester.config import Config
from backtester.data_handler import DataHandler
from backtester.events import FillEvent, MarketEvent
from backtester.execution import ExecutionHandler
from backtester.models import MarketSnapshot, Order
from backtester.performance import DailySummary, PerformanceAnalyzer
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtesting engine.

    Orchestrates the flow of data through strategy, execution,
    portfolio, and performance recording.

    Supports multiple underliers running sequentially per day
    with a shared portfolio.
    """

    def __init__(
        self,
        config: Config,
        strategies: Dict[str, Strategy],
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        data_handler: DataHandler,
        performance: PerformanceAnalyzer,
    ):
        """Initialize the backtest engine.

        Args:
            config: Application configuration.
            strategies: Mapping of underlier → Strategy instance.
            portfolio: Shared portfolio across all underliers.
            execution_handler: Order execution handler.
            data_handler: Market data provider.
            performance: Performance recorder and analyzer.
        """
        self._config = config
        self._strategies = strategies
        self._portfolio = portfolio
        self._execution = execution_handler
        self._data_handler = data_handler
        self._performance = performance

    def run(self) -> None:
        """Execute the full backtest.

        Main loop:
          1. Discover trading dates.
          2. For each date, process each underlier.
          3. For each underlier, iterate through snapshots.
          4. At each snapshot: strategy → orders → execution → portfolio → record.
          5. At day end: close positions, record daily summary.
        """
        trading_dates = self._data_handler.discover_trading_dates()

        logger.info(
            f"Starting backtest: {len(trading_dates)} days, "
            f"underliers={list(self._strategies.keys())}, "
            f"initial_capital={self._config.initial_capital}"
        )
        print(
            f"Starting backtest: {len(trading_dates)} days, "
            f"underliers={list(self._strategies.keys())}"
        )

        for day_idx, trading_date in enumerate(trading_dates):
            logger.info(
                f"{'='*60}\n"
                f"DAY {day_idx + 1}/{len(trading_dates)}: {trading_date}\n"
                f"{'='*60}"
            )
            print(f"Processing day {day_idx + 1}/{len(trading_dates)}: {trading_date} ...", flush=True)

            for underlier in self._config.underliers:
                self._process_day(trading_date, underlier)

            # Clear data cache to manage memory
            self._data_handler.clear_cache()

        # Generate all outputs
        print("Backtest complete. Generating outputs...", flush=True)
        logger.info("Backtest complete. Generating outputs...")
        self._performance.generate_outputs()

        # Log final metrics
        metrics = self._performance.compute_metrics()
        logger.info(
            f"\n{'='*60}\n"
            f"FINAL RESULTS\n"
            f"{'='*60}\n"
            f"Final Equity:   Rs.{metrics.get('final_equity', 'N/A'):>12}\n"
            f"Total PnL:      Rs.{metrics.get('total_pnl', 'N/A'):>12}\n"
            f"Total Return:     {metrics.get('total_return_pct', 'N/A'):>11}%\n"
            f"Max Drawdown:     {metrics.get('max_drawdown_pct', 'N/A'):>11}%\n"
            f"Total Trades:     {metrics.get('total_trades', 'N/A'):>12}\n"
            f"Total Rolls:      {metrics.get('total_rolls', 'N/A'):>12}\n"
            f"Win Rate:         {metrics.get('win_rate_pct', 'N/A'):>11}%\n"
            f"Txn Costs:      Rs.{metrics.get('total_transaction_costs', 'N/A'):>12}\n"
            f"{'='*60}"
        )

    def _process_day(self, trading_date: date, underlier: str) -> None:
        """Process a single day for a single underlier.

        Args:
            trading_date: The trading date.
            underlier: The underlier to process.
        """
        strategy = self._strategies.get(underlier)
        if strategy is None:
            logger.warning(f"No strategy for {underlier}, skipping")
            return

        # Reset strategy state for new day
        strategy.on_day_start(trading_date, underlier)

        # Track daily stats
        start_equity = self._portfolio.equity
        max_equity = start_equity
        min_equity = start_equity
        day_trades_before = len(self._portfolio.trades)
        tick_count = 0
        last_snapshot: Optional[MarketSnapshot] = None
        prev_atm_strike: Optional[int] = None

        # Process each snapshot
        for snapshot in self._data_handler.generate_snapshots(trading_date, underlier):
            tick_count += 1
            last_snapshot = snapshot

            # Update MTM for existing positions
            price_map = {}
            for pos_key, pos in self._portfolio.positions.items():
                key = (pos.instrument.strike, pos.instrument.option_type)
                if key in snapshot.option_prices:
                    price_map[pos_key] = snapshot.option_prices[key]
            self._portfolio.update_market_prices(price_map)

            # Strategy generates orders
            orders = strategy.on_tick(snapshot, self._portfolio)

            # Detect rolls for performance recording
            if orders:
                sells = [o for o in orders if o.side == "SELL"]
                buys = [o for o in orders if o.side == "BUY"]
                if sells and buys:
                    old_strike = sells[0].instrument.strike
                    new_strike = buys[0].instrument.strike
                    if old_strike != new_strike:
                        self._performance.record_roll(
                            snapshot.timestamp, underlier,
                            old_strike, new_strike,
                            snapshot.futures_price,
                        )

            # Execute orders
            if orders:
                fills = self._execution.execute_orders(orders, self._portfolio)

            # Record performance data
            current_equity = self._portfolio.equity
            max_equity = max(max_equity, current_equity)
            min_equity = min(min_equity, current_equity)

            # Record equity point (sample every 10 seconds to reduce output size)
            if tick_count % 10 == 0 or tick_count <= 5 or orders:
                self._performance.record_equity_point(
                    timestamp=snapshot.timestamp,
                    equity=current_equity,
                    cash=self._portfolio.cash,
                    market_value=self._portfolio.market_value,
                    pnl=self._portfolio.total_pnl,
                    underlier=underlier,
                )

            # Record futures vs ATM for plotting (sample every 30 seconds)
            if tick_count % 30 == 0 or orders:
                atm_strike = min(
                    snapshot.available_strikes,
                    key=lambda s: (abs(s - snapshot.futures_price), s),
                )
                self._performance.record_futures_vs_atm(
                    snapshot.timestamp, underlier,
                    snapshot.futures_price, atm_strike,
                )

        # End of day — close all positions
        if last_snapshot is not None:
            eod_orders = strategy.on_day_end(last_snapshot, self._portfolio)
            if eod_orders:
                self._execution.execute_orders(eod_orders, self._portfolio)

            # Record final equity point
            self._performance.record_equity_point(
                timestamp=last_snapshot.timestamp,
                equity=self._portfolio.equity,
                cash=self._portfolio.cash,
                market_value=self._portfolio.market_value,
                pnl=self._portfolio.total_pnl,
                underlier=underlier,
            )

        # Record daily summary
        end_equity = self._portfolio.equity
        day_trades = self._portfolio.trades[day_trades_before:]

        summary = DailySummary(
            date=trading_date,
            underlier=underlier,
            start_equity=start_equity,
            end_equity=end_equity,
            daily_pnl=end_equity - start_equity,
            num_trades=len(day_trades),
            num_rolls=strategy.roll_count if hasattr(strategy, 'roll_count') else 0,
            max_equity=max_equity,
            min_equity=min_equity,
        )
        self._performance.record_daily_summary(summary)
        self._performance.record_trades(day_trades)

        logger.info(
            f"{underlier} on {trading_date}: "
            f"{tick_count} ticks, "
            f"{len(day_trades)} trades, "
            f"PnL=Rs.{summary.daily_pnl:.2f}, "
            f"equity=Rs.{end_equity:.2f}"
        )
