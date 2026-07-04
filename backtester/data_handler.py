"""
Data handler for the backtesting engine.

Responsible for:
  - Discovering available trading dates
  - Discovering and filtering instruments
  - Loading and caching CSV data
  - Determining nearest expiry per underlier per date
  - Building chronological MarketSnapshot sequences
  - Yielding one MarketSnapshot at a time (generator pattern)
"""

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import pandas as pd

from backtester.alignment import AlignmentPolicy, get_alignment_policy
from backtester.config import Config
from backtester.models import InstrumentMetadata, MarketSnapshot
from backtester.parser import parse_instrument_name, is_relevant_option

logger = logging.getLogger(__name__)


class DataHandler:
    """Manages all data loading, caching, and snapshot generation.

    Design:
      - Loads only required files (lazy loading per day/underlier).
      - Caches loaded DataFrames to avoid re-reading.
      - Builds aligned 1-second snapshots combining futures + options.
    """

    def __init__(self, config: Config, alignment_policy: AlignmentPolicy):
        """Initialize the DataHandler.

        Args:
            config: Application configuration.
            alignment_policy: Policy for aligning tick data to 1-second grid.
        """
        self._config = config
        self._alignment = alignment_policy
        self._csv_cache: Dict[str, pd.DataFrame] = {}
        self._instrument_cache: Dict[str, List[InstrumentMetadata]] = {}

    def discover_trading_dates(self) -> List[date]:
        """Find all available trading dates within the configured range.

        Returns:
            Sorted list of trading dates found on disk.
        """
        data_root = self._config.data_root
        start = datetime.strptime(self._config.date_range.start, "%Y%m%d").date()
        end = datetime.strptime(self._config.date_range.end, "%Y%m%d").date()

        dates = []
        for folder in sorted(data_root.iterdir()):
            if folder.is_dir() and folder.name.startswith("NSE_"):
                date_str = folder.name.replace("NSE_", "")
                try:
                    d = datetime.strptime(date_str, "%Y%m%d").date()
                    if start <= d <= end:
                        dates.append(d)
                except ValueError:
                    continue

        logger.info(f"Discovered {len(dates)} trading dates: {dates[0]} to {dates[-1]}")
        return dates

    def _day_folder(self, trading_date: date) -> Path:
        """Get the folder path for a trading date."""
        return self._config.data_root / f"NSE_{trading_date.strftime('%Y%m%d')}"

    def _futures_path(self, trading_date: date, underlier: str) -> Path:
        """Get the futures CSV path for a given date and underlier."""
        return (
            self._day_folder(trading_date)
            / self._config.futures_folder_name
            / f"{underlier}-I.csv"
        )

    def _options_folder(self, trading_date: date) -> Path:
        """Get the options folder path for a given date."""
        return self._day_folder(trading_date) / self._config.options_folder_name

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load a CSV file with caching.

        The CSVs have no header. Columns are:
        Date, Time, Price, Volume, OpenInterest
        """
        cache_key = str(path)
        if cache_key in self._csv_cache:
            return self._csv_cache[cache_key]

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame(columns=["timestamp", "price", "volume", "open_interest"])

        df = pd.read_csv(
            path,
            header=None,
            names=["date", "time", "price", "volume", "open_interest"],
        )
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y%m%d %H:%M:%S",
        )
        df = df[["timestamp", "price", "volume", "open_interest"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._csv_cache[cache_key] = df
        return df

    def discover_instruments(
        self, trading_date: date, underlier: str
    ) -> List[InstrumentMetadata]:
        """Discover all option instruments for a given date and underlier.

        Returns:
            List of InstrumentMetadata for all matching option files.
        """
        cache_key = f"{trading_date}_{underlier}"
        if cache_key in self._instrument_cache:
            return self._instrument_cache[cache_key]

        options_dir = self._options_folder(trading_date)
        instruments = []

        for csv_file in options_dir.glob(f"{underlier}*.csv"):
            name = csv_file.stem
            metadata = parse_instrument_name(name)
            if metadata and metadata.underlier == underlier:
                instruments.append(metadata)

        self._instrument_cache[cache_key] = instruments
        logger.debug(
            f"Discovered {len(instruments)} instruments for "
            f"{underlier} on {trading_date}"
        )
        return instruments

    def find_nearest_expiry(
        self, trading_date: date, underlier: str
    ) -> Optional[date]:
        """Find the nearest expiry >= trading_date for the given underlier.

        Args:
            trading_date: Current trading date.
            underlier: The underlier name.

        Returns:
            The nearest expiry date, or None if none found.
        """
        instruments = self.discover_instruments(trading_date, underlier)
        expiries = sorted(set(
            inst.expiry for inst in instruments if inst.expiry >= trading_date
        ))

        if not expiries:
            logger.warning(
                f"No expiry found >= {trading_date} for {underlier}"
            )
            return None

        nearest = expiries[0]
        logger.info(
            f"{underlier} on {trading_date}: nearest expiry = {nearest}"
        )
        return nearest

    def get_available_strikes(
        self,
        trading_date: date,
        underlier: str,
        expiry: date,
    ) -> List[int]:
        """Get all available strikes for a given underlier and expiry.

        Only returns strikes where BOTH CE and PE files exist.
        """
        instruments = self.discover_instruments(trading_date, underlier)
        relevant = [
            inst for inst in instruments
            if is_relevant_option(inst, underlier, expiry)
        ]

        # Find strikes with both CE and PE
        ce_strikes = {inst.strike for inst in relevant if inst.option_type == "CE"}
        pe_strikes = {inst.strike for inst in relevant if inst.option_type == "PE"}
        both = sorted(ce_strikes & pe_strikes)

        logger.debug(
            f"{underlier} expiry {expiry}: {len(both)} strikes with both CE+PE "
            f"(CE-only: {len(ce_strikes - pe_strikes)}, "
            f"PE-only: {len(pe_strikes - ce_strikes)})"
        )
        return both

    def generate_snapshots(
        self,
        trading_date: date,
        underlier: str,
    ) -> Generator[MarketSnapshot, None, None]:
        """Generate chronological MarketSnapshots for a full trading day.

        This is the main data-feeding interface. It:
        1. Loads and aligns futures data to a 1-second grid.
        2. Determines the nearest expiry.
        3. Finds all available strikes (both CE+PE required).
        4. Loads and aligns option data for those strikes.
        5. Yields one MarketSnapshot per second.

        Args:
            trading_date: The trading date.
            underlier: The underlier to process.

        Yields:
            MarketSnapshot objects, one per second.
        """
        # Determine session bounds
        session_start = datetime.combine(
            trading_date,
            datetime.strptime(self._config.trading_start_time, "%H:%M:%S").time(),
        )
        session_end = datetime.combine(
            trading_date,
            datetime.strptime(self._config.trading_end_time, "%H:%M:%S").time(),
        )

        # Load futures
        futures_path = self._futures_path(trading_date, underlier)
        futures_raw = self._load_csv(futures_path)
        if futures_raw.empty:
            logger.warning(f"No futures data for {underlier} on {trading_date}")
            return

        futures_aligned = self._alignment.align(futures_raw, session_start, session_end)

        # Find nearest expiry
        nearest_expiry = self.find_nearest_expiry(trading_date, underlier)
        if nearest_expiry is None:
            return

        # Find available strikes with both CE + PE
        available_strikes = self.get_available_strikes(
            trading_date, underlier, nearest_expiry
        )
        if not available_strikes:
            logger.warning(
                f"No strikes with both CE+PE for {underlier} "
                f"expiry {nearest_expiry} on {trading_date}"
            )
            return

        # Load and align option data for all relevant strikes
        option_data: Dict[Tuple[int, str], pd.DataFrame] = {}
        for strike in available_strikes:
            for opt_type in ["CE", "PE"]:
                inst = InstrumentMetadata(
                    underlier=underlier,
                    expiry=nearest_expiry,
                    strike=strike,
                    option_type=opt_type,
                )
                option_path = (
                    self._options_folder(trading_date) / f"{inst.name}.csv"
                )
                raw = self._load_csv(option_path)
                if not raw.empty:
                    aligned = self._alignment.align(raw, session_start, session_end)
                    option_data[(strike, opt_type)] = aligned

        logger.info(
            f"Generating snapshots for {underlier} on {trading_date}: "
            f"{len(available_strikes)} strikes, expiry={nearest_expiry}"
        )

        futures_prices = futures_aligned["price"].values
        timestamps = futures_aligned.index.to_pydatetime()
        
        option_prices_arrays = {}
        expected_len = len(timestamps)
        
        for key, df in option_data.items():
            if len(df) == expected_len:
                option_prices_arrays[key] = df["price"].values

        # Yield one snapshot per second using fast array indexing
        for i in range(expected_len):
            ts = timestamps[i]
            futures_price = futures_prices[i]
            
            if pd.isna(futures_price):
                continue  # No futures price yet (before first tick)

            # Collect option prices at this timestamp
            opt_prices: Dict[Tuple[int, str], float] = {}
            valid_strikes = []
            
            for strike in available_strikes:
                ce_key = (strike, "CE")
                pe_key = (strike, "PE")
                
                ce_price = None
                pe_price = None

                # Look up price instantly via integer index 'i'
                if ce_key in option_prices_arrays:
                    p = option_prices_arrays[ce_key][i]
                    if not pd.isna(p):
                        ce_price = p

                if pe_key in option_prices_arrays:
                    p = option_prices_arrays[pe_key][i]
                    if not pd.isna(p):
                        pe_price = p

                if ce_price is not None and pe_price is not None:
                    opt_prices[ce_key] = ce_price
                    opt_prices[pe_key] = pe_price
                    valid_strikes.append(strike)

            if not valid_strikes:
                continue

            yield MarketSnapshot(
                timestamp=ts,
                underlier=underlier,
                futures_price=futures_price,
                expiry=nearest_expiry,
                available_strikes=sorted(valid_strikes),
                option_prices=opt_prices,
            )

    def clear_cache(self) -> None:
        """Clear all cached data. Call between days to manage memory."""
        self._csv_cache.clear()
        self._instrument_cache.clear()
        logger.debug("DataHandler cache cleared")
