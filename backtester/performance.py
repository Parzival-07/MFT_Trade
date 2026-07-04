"""
Performance analyzer and metrics computation.

Separated from the engine — this module only computes metrics
from recorded equity curves, trades, and positions.

Generates:
  - metrics.json
  - equity_curve.csv
  - daily_summary.csv
  - trades.csv
  - positions.csv
  - Plots (equity curve, drawdown, daily PnL, futures vs ATM strike)
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtester.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    """A single point on the equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    market_value: float
    pnl: float
    underlier: str


@dataclass
class DailySummary:
    """Summary of one trading day."""
    date: date
    underlier: str
    start_equity: float
    end_equity: float
    daily_pnl: float
    num_trades: int
    num_rolls: int
    max_equity: float
    min_equity: float


class PerformanceAnalyzer:
    """Computes and records performance metrics.

    Usage:
      1. During backtest: call record_equity_point() on every tick.
      2. After backtest: call compute_metrics() and generate_outputs().
    """

    def __init__(self, initial_capital: float, results_dir: Path):
        """Initialize the performance analyzer.

        Args:
            initial_capital: Starting capital.
            results_dir: Directory to write output files.
        """
        self._initial_capital = initial_capital
        self._results_dir = results_dir
        self._equity_points: List[EquityPoint] = []
        self._daily_summaries: List[DailySummary] = []
        self._trades: List[Trade] = []
        self._roll_log: List[Dict[str, Any]] = []
        self._futures_vs_atm: List[Dict[str, Any]] = []

    def record_equity_point(
        self,
        timestamp: datetime,
        equity: float,
        cash: float,
        market_value: float,
        pnl: float,
        underlier: str,
    ) -> None:
        """Record a single equity curve point."""
        self._equity_points.append(EquityPoint(
            timestamp=timestamp,
            equity=equity,
            cash=cash,
            market_value=market_value,
            pnl=pnl,
            underlier=underlier,
        ))

    def record_daily_summary(self, summary: DailySummary) -> None:
        """Record a daily summary."""
        self._daily_summaries.append(summary)

    def record_trades(self, trades: List[Trade]) -> None:
        """Record completed trades."""
        self._trades.extend(trades)

    def record_roll(
        self,
        timestamp: datetime,
        underlier: str,
        old_strike: int,
        new_strike: int,
        futures_price: float,
    ) -> None:
        """Record a roll event for visualization."""
        self._roll_log.append({
            "timestamp": timestamp,
            "underlier": underlier,
            "old_strike": old_strike,
            "new_strike": new_strike,
            "futures_price": futures_price,
        })

    def record_futures_vs_atm(
        self,
        timestamp: datetime,
        underlier: str,
        futures_price: float,
        atm_strike: int,
    ) -> None:
        """Record futures price vs ATM strike for plotting."""
        self._futures_vs_atm.append({
            "timestamp": timestamp,
            "underlier": underlier,
            "futures_price": futures_price,
            "atm_strike": atm_strike,
        })

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate performance metrics.

        Returns:
            Dictionary of metrics suitable for JSON serialization.
        """
        if not self._equity_points:
            return {"error": "No equity data recorded"}

        equity_series = pd.Series(
            [ep.equity for ep in self._equity_points],
            index=[ep.timestamp for ep in self._equity_points],
        )

        # Cumulative PnL
        pnl_series = equity_series - self._initial_capital

        # Maximum drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100

        # Total return
        final_equity = equity_series.iloc[-1]
        total_return = (final_equity - self._initial_capital) / self._initial_capital
        total_return_pct = total_return * 100

        # Trade statistics
        trade_pnls = [t.net_pnl for t in self._trades]
        winning = [p for p in trade_pnls if p > 0]
        losing = [p for p in trade_pnls if p <= 0]
        holding_times = [t.holding_period_seconds for t in self._trades]

        # Roll statistics
        total_rolls = len(self._roll_log)
        roll_pnls = []
        # Pair sell+buy trades as rolls
        for i in range(0, len(self._trades) - 1, 2):
            if i + 1 < len(self._trades):
                roll_pnls.append(
                    self._trades[i].net_pnl + self._trades[i + 1].net_pnl
                )

        metrics = {
            "initial_capital": self._initial_capital,
            "final_equity": round(final_equity, 2),
            "total_pnl": round(float(pnl_series.iloc[-1]), 2),
            "total_return_pct": round(total_return_pct, 4),
            "max_drawdown_pct": round(float(max_drawdown_pct), 4),
            "total_trades": len(self._trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": round(
                len(winning) / len(trade_pnls) * 100, 2
            ) if trade_pnls else 0,
            "avg_trade_pnl": round(
                np.mean(trade_pnls), 2
            ) if trade_pnls else 0,
            "avg_winning_pnl": round(
                np.mean(winning), 2
            ) if winning else 0,
            "avg_losing_pnl": round(
                np.mean(losing), 2
            ) if losing else 0,
            "total_rolls": total_rolls,
            "avg_holding_time_seconds": round(
                np.mean(holding_times), 2
            ) if holding_times else 0,
            "max_holding_time_seconds": round(
                max(holding_times), 2
            ) if holding_times else 0,
            "total_transaction_costs": round(
                sum(t.transaction_costs for t in self._trades), 2
            ),
            "trading_days": len(self._daily_summaries),
        }

        # Per-underlier breakdown
        underlier_metrics = {}
        for underlier in set(ep.underlier for ep in self._equity_points):
            ul_trades = [
                t for t in self._trades if t.instrument.underlier == underlier
            ]
            ul_pnls = [t.net_pnl for t in ul_trades]
            ul_rolls = [
                r for r in self._roll_log if r["underlier"] == underlier
            ]
            underlier_metrics[underlier] = {
                "total_trades": len(ul_trades),
                "total_pnl": round(sum(ul_pnls), 2) if ul_pnls else 0,
                "total_rolls": len(ul_rolls),
                "avg_trade_pnl": round(
                    np.mean(ul_pnls), 2
                ) if ul_pnls else 0,
            }

        metrics["per_underlier"] = underlier_metrics
        return metrics

    def generate_outputs(self) -> None:
        """Generate all output files and plots."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        (self._results_dir / "logs").mkdir(exist_ok=True)
        (self._results_dir / "plots").mkdir(exist_ok=True)

        self._write_equity_curve()
        self._write_trades()
        self._write_daily_summary()
        self._write_metrics()
        self._write_positions_log()
        self._plot_equity_curve()
        self._plot_drawdown()
        self._plot_daily_pnl()
        self._plot_futures_vs_atm()
        self._plot_trade_timeline()

        logger.info(f"All outputs written to {self._results_dir}")

    def _write_equity_curve(self) -> None:
        """Write equity_curve.csv."""
        if not self._equity_points:
            return

        rows = [
            {
                "timestamp": ep.timestamp.isoformat(),
                "equity": round(ep.equity, 2),
                "cash": round(ep.cash, 2),
                "market_value": round(ep.market_value, 2),
                "pnl": round(ep.pnl, 2),
                "underlier": ep.underlier,
            }
            for ep in self._equity_points
        ]
        df = pd.DataFrame(rows)
        path = self._results_dir / "equity_curve.csv"
        df.to_csv(path, index=False)
        logger.info(f"Equity curve written: {path} ({len(rows)} points)")

    def _write_trades(self) -> None:
        """Write trades.csv."""
        if not self._trades:
            return

        rows = [
            {
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "instrument": t.instrument.name,
                "underlier": t.instrument.underlier,
                "expiry": t.instrument.expiry.isoformat(),
                "strike": t.instrument.strike,
                "option_type": t.instrument.option_type,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "quantity": t.quantity,
                "gross_pnl": round(t.gross_pnl, 2),
                "net_pnl": round(t.net_pnl, 2),
                "transaction_costs": round(t.transaction_costs, 2),
                "holding_period_seconds": round(t.holding_period_seconds, 2),
            }
            for t in self._trades
        ]
        df = pd.DataFrame(rows)
        path = self._results_dir / "trades.csv"
        df.to_csv(path, index=False)
        logger.info(f"Trades written: {path} ({len(rows)} trades)")

    def _write_daily_summary(self) -> None:
        """Write daily_summary.csv."""
        if not self._daily_summaries:
            return

        rows = [
            {
                "date": ds.date.isoformat(),
                "underlier": ds.underlier,
                "start_equity": round(ds.start_equity, 2),
                "end_equity": round(ds.end_equity, 2),
                "daily_pnl": round(ds.daily_pnl, 2),
                "num_trades": ds.num_trades,
                "num_rolls": ds.num_rolls,
                "max_equity": round(ds.max_equity, 2),
                "min_equity": round(ds.min_equity, 2),
            }
            for ds in self._daily_summaries
        ]
        df = pd.DataFrame(rows)
        path = self._results_dir / "daily_summary.csv"
        df.to_csv(path, index=False)
        logger.info(f"Daily summary written: {path}")

    def _write_metrics(self) -> None:
        """Write metrics.json."""
        metrics = self.compute_metrics()
        path = self._results_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics written: {path}")

    def _write_positions_log(self) -> None:
        """Write positions.csv from roll log."""
        if not self._roll_log:
            return

        df = pd.DataFrame(self._roll_log)
        df["timestamp"] = df["timestamp"].apply(
            lambda x: x.isoformat() if isinstance(x, datetime) else x
        )
        path = self._results_dir / "positions.csv"
        df.to_csv(path, index=False)
        logger.info(f"Positions log written: {path}")

    def _plot_equity_curve(self) -> None:
        """Plot 1: Cumulative MTM PnL (equity curve)."""
        if not self._equity_points:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        for underlier in sorted(set(ep.underlier for ep in self._equity_points)):
            ul_points = [ep for ep in self._equity_points if ep.underlier == underlier]
            timestamps = [ep.timestamp for ep in ul_points]
            pnls = [ep.pnl for ep in ul_points]
            ax.plot(timestamps, pnls, label=underlier, alpha=0.8, linewidth=0.5)

        # Combined PnL
        df = pd.DataFrame([
            {"timestamp": ep.timestamp, "pnl": ep.pnl, "underlier": ep.underlier}
            for ep in self._equity_points
        ])
        combined = df.groupby("timestamp")["pnl"].sum().reset_index()
        ax.plot(
            combined["timestamp"], combined["pnl"],
            label="Combined", color="black", linewidth=1.5, alpha=0.9,
        )

        ax.set_title("Cumulative MTM PnL", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("PnL (Rs.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self._results_dir / "plots" / "equity_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Equity curve plot saved")

    def _plot_drawdown(self) -> None:
        """Plot 2: Underwater drawdown plot."""
        if not self._equity_points:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Combined equity
        df = pd.DataFrame([
            {"timestamp": ep.timestamp, "equity": ep.equity, "underlier": ep.underlier}
            for ep in self._equity_points
        ])
        combined = df.groupby("timestamp")["equity"].sum().reset_index()
        equity = combined["equity"].values
        cummax = np.maximum.accumulate(equity)
        drawdown_pct = (equity - cummax) / cummax * 100

        ax.fill_between(
            combined["timestamp"], drawdown_pct, 0,
            color="red", alpha=0.3, label="Drawdown",
        )
        ax.plot(combined["timestamp"], drawdown_pct, color="red", linewidth=0.5)

        ax.set_title("Drawdown (Underwater Plot)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._results_dir / "plots" / "drawdown.png", dpi=150)
        plt.close(fig)
        logger.info("Drawdown plot saved")

    def _plot_daily_pnl(self) -> None:
        """Plot 4: Daily PnL bar chart."""
        if not self._daily_summaries:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Group by date, sum PnL across underliers
        daily_data: Dict[date, float] = {}
        for ds in self._daily_summaries:
            daily_data[ds.date] = daily_data.get(ds.date, 0) + ds.daily_pnl

        dates = sorted(daily_data.keys())
        pnls = [daily_data[d] for d in dates]
        colors = ["green" if p >= 0 else "red" for p in pnls]

        ax.bar(
            [d.isoformat() for d in dates], pnls,
            color=colors, alpha=0.7, edgecolor="black", linewidth=0.5,
        )

        ax.set_title("Daily PnL", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL (Rs.)")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(self._results_dir / "plots" / "daily_pnl.png", dpi=150)
        plt.close(fig)
        logger.info("Daily PnL plot saved")

    def _plot_futures_vs_atm(self) -> None:
        """Plot 3: Futures price vs ATM strike for selected days."""
        if not self._futures_vs_atm:
            return

        df = pd.DataFrame(self._futures_vs_atm)

        # Plot for each underlier, pick first day with data
        for underlier in df["underlier"].unique():
            ul_df = df[df["underlier"] == underlier]
            dates = ul_df["timestamp"].dt.date.unique()

            # Plot first, middle, and last day
            selected_dates = []
            if len(dates) >= 1:
                selected_dates.append(dates[0])
            if len(dates) >= 3:
                selected_dates.append(dates[len(dates) // 2])
            if len(dates) >= 2:
                selected_dates.append(dates[-1])

            for sel_date in selected_dates:
                day_df = ul_df[ul_df["timestamp"].dt.date == sel_date]
                if day_df.empty:
                    continue

                fig, ax = plt.subplots(figsize=(14, 6))

                ax.plot(
                    day_df["timestamp"], day_df["futures_price"],
                    label="Futures Price", color="blue", linewidth=1, alpha=0.8,
                )
                ax.step(
                    day_df["timestamp"], day_df["atm_strike"],
                    label="ATM Strike", color="red", linewidth=1.5,
                    where="post", alpha=0.8,
                )

                ax.set_title(
                    f"{underlier} — Futures vs ATM Strike ({sel_date})",
                    fontsize=14, fontweight="bold",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                fname = f"futures_vs_atm_{underlier}_{sel_date}.png"
                fig.savefig(self._results_dir / "plots" / fname, dpi=150)
                plt.close(fig)

        logger.info("Futures vs ATM plots saved")

    def _plot_trade_timeline(self) -> None:
        """Plot 5: Trade timeline showing rolls."""
        if not self._roll_log:
            return

        fig, axes = plt.subplots(
            len(set(r["underlier"] for r in self._roll_log)), 1,
            figsize=(14, 4 * len(set(r["underlier"] for r in self._roll_log))),
            squeeze=False,
        )

        for idx, underlier in enumerate(
            sorted(set(r["underlier"] for r in self._roll_log))
        ):
            ax = axes[idx, 0]
            ul_rolls = [r for r in self._roll_log if r["underlier"] == underlier]

            timestamps = [r["timestamp"] for r in ul_rolls]
            old_strikes = [r["old_strike"] for r in ul_rolls]
            new_strikes = [r["new_strike"] for r in ul_rolls]

            ax.scatter(
                timestamps, old_strikes,
                marker="v", color="red", s=10, alpha=0.6, label="Rolled From",
            )
            ax.scatter(
                timestamps, new_strikes,
                marker="^", color="green", s=10, alpha=0.6, label="Rolled To",
            )

            ax.set_title(
                f"{underlier} — Roll Timeline",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Strike")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._results_dir / "plots" / "trade_timeline.png", dpi=150)
        plt.close(fig)
        logger.info("Trade timeline plot saved")
