"""
Instrument filename parser.

Responsible for parsing option instrument filenames into structured
InstrumentMetadata objects.
"""

import re
from datetime import date, datetime
from typing import Optional

from backtester.models import InstrumentMetadata


# Known underliers ordered longest-first so greedy match works correctly.
_KNOWN_UNDERLIERS = ["BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTY"]

# Regex: <UNDERLIER><6-digit-expiry><digits-strike><CE|PE>
_OPTION_PATTERN = re.compile(
    r"^(?P<underlier>[A-Z]+)"
    r"(?P<expiry>\d{6})"
    r"(?P<strike>\d+)"
    r"(?P<option_type>CE|PE)$"
)


def parse_instrument_name(name: str) -> Optional[InstrumentMetadata]:
    """Parse an instrument name (filename without .csv) into metadata.

    Args:
        name: The instrument name, e.g. "NIFTY22110318100CE".

    Returns:
        InstrumentMetadata if parsing succeeds, None otherwise.

    Examples:
        >>> parse_instrument_name("NIFTY22110314550PE")
        InstrumentMetadata(underlier='NIFTY', expiry=date(2022,11,3),
                          strike=14550, option_type='PE')

        >>> parse_instrument_name("BANKNIFTY22112443200CE")
        InstrumentMetadata(underlier='BANKNIFTY', expiry=date(2022,11,24),
                          strike=43200, option_type='CE')
    """
    match = _OPTION_PATTERN.match(name)
    if not match:
        return None

    underlier = match.group("underlier")
    expiry_str = match.group("expiry")
    strike_str = match.group("strike")
    option_type = match.group("option_type")

    try:
        expiry_date = datetime.strptime(expiry_str, "%y%m%d").date()
    except ValueError:
        return None

    try:
        strike = int(strike_str)
    except ValueError:
        return None

    return InstrumentMetadata(
        underlier=underlier,
        expiry=expiry_date,
        strike=strike,
        option_type=option_type,
    )


def is_relevant_option(
    metadata: InstrumentMetadata,
    underlier: str,
    nearest_expiry: date,
) -> bool:
    """Check if an option belongs to the required underlier and expiry.

    Args:
        metadata: Parsed instrument metadata.
        underlier: The underlier we are trading (e.g. "NIFTY").
        nearest_expiry: The nearest expiry date >= current trading date.

    Returns:
        True if this option should be considered for trading.
    """
    return metadata.underlier == underlier and metadata.expiry == nearest_expiry
