# Event-Driven Backtesting Engine for NSE Options

A professional, production-quality, event-driven backtesting engine for intraday options trading strategies on NSE data.

## Project Architecture

```
allData/
├── main.py                              # Entry point
├── config.json                          # Configuration file
├── backtester/
│   ├── __init__.py
│   ├── config.py                        # Configuration loader
│   ├── models.py                        # Core domain models (dataclasses)
│   ├── parser.py                        # Instrument filename parser
│   ├── alignment.py                     # Pluggable price alignment policies
│   ├── data_handler.py                  # Data loading, caching, snapshot generation
│   ├── events.py                        # Event classes (MarketEvent, FillEvent, etc.)
│   ├── strategy.py                      # Abstract Strategy interface
│   ├── atm_straddle_strategy.py         # ATM Straddle strategy implementation
│   ├── portfolio.py                     # Portfolio management (positions, cash, MTM)
│   ├── execution.py                     # Order execution handler
│   ├── performance.py                   # Performance metrics and visualization
│   ├── engine.py                        # Backtest engine (orchestrator)
│   └── utils.py                         # Logging setup and utilities
├── tests/
│   ├── __init__.py
│   └── test_backtester.py               # Unit tests
├── results/                             # Generated output (after running)
│   ├── equity_curve.csv
│   ├── trades.csv
│   ├── daily_summary.csv
│   ├── positions.csv
│   ├── metrics.json
│   ├── logs/
│   │   └── backtest.log
│   └── plots/
│       ├── equity_curve.png
│       ├── drawdown.png
│       ├── daily_pnl.png
│       ├── futures_vs_atm_*.png
│       └── trade_timeline.png
└── allData/                             # Dataset (NSE_20221101, ...)
    └── NSE_YYYYMMDD/
        ├── Futures (Continuous)/
        │   ├── NIFTY-I.csv
        │   └── BANKNIFTY-I.csv
        └── Options/
            ├── NIFTY22110318100CE.csv
            └── ...
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Required packages: `pandas`, `numpy`, `matplotlib`

### Installation

```bash
pip install pandas numpy matplotlib
```

### Running the Backtest

```bash
python main.py --config config.json
```

### Running Tests

```bash
python -m pytest tests/test_backtester.py -v
```

## Design Decisions

### 1. Event-Driven Architecture
The engine processes events sequentially: `MarketEvent → Strategy → Orders → Execution → Portfolio → Performance`. Each component has a single responsibility and communicates only through well-defined interfaces.

### 2. Pluggable Strategy Interface
The `Strategy` abstract class defines `on_tick()`, `on_day_end()`, and `on_day_start()`. New strategies can be implemented by subclassing without modifying any other module.

### 3. Pluggable Alignment Policy
Tick data is irregular (trade-level). The `AlignmentPolicy` interface allows different methods for creating a uniform 1-second grid. Default: `ForwardFillAlignmentPolicy`.

### 4. Lazy Data Loading with Caching
The `DataHandler` only loads CSV files when needed and caches them. It also clears cache between days to manage memory.

### 5. Separation of Concerns
- **Strategy** produces orders (never modifies portfolio)
- **ExecutionHandler** processes orders (applies costs, updates portfolio)
- **Portfolio** tracks positions and cash (no strategy logic)
- **PerformanceAnalyzer** computes metrics (no trading logic)
- **Engine** orchestrates everything (no business logic)

### 6. Configuration-Driven
All parameters (underliers, capital, costs, dates, alignment policy) are in `config.json`. No hardcoded constants.

## Assumptions

1. **Tick data, not 1-second bars**: CSVs contain trade-level data with multiple rows per second and gaps. Forward-fill alignment creates a uniform grid.
2. **Immediate fills**: Orders execute at the quoted price with no partial fills.
3. **Both CE + PE required**: A strike is only considered "available" if both CE and PE files exist and have prices at that timestamp.
4. **Nearest expiry ≥ current date**: The strategy only trades the closest upcoming expiry.
5. **Folder naming**: Futures folder is `"Futures (Continuous)"`, Options folder is `"Options"`.
6. **Position accounting**: Buy = pay `price × qty + txn_cost`, Sell = receive `price × qty - txn_cost`.

## Class Diagram

![Class Diagram](Class%20Diagram.png)

## Sequence Diagram

![Sequence Diagram](Sequence%20Diagram.png)

