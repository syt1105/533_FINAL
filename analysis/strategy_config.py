import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MARKET_HISTORY_DIR = DATA_DIR / "market_history"
DOCS_DIR = ROOT / "docs"
DOWNLOADS_DIR = DOCS_DIR / "downloads"

PROJECT_TITLE = "Volatility Breakout Final Project"
ASSET_UNIVERSE = ["SPY", "QQQ", "IWM", "XLE", "GLD", "TLT"]
DEFAULT_SELECTED_SYMBOL = "XLE"

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "9999"))

HISTORY_DURATION = os.getenv("IBKR_HISTORY_DURATION", "5 Y")
BAR_SIZE = os.getenv("IBKR_BAR_SIZE", "1 day")
WHAT_TO_SHOW = os.getenv("IBKR_WHAT_TO_SHOW", "TRADES")
USE_RTH = os.getenv("IBKR_USE_RTH", "true").lower() in {"1", "true", "yes", "y"}

MIN_HISTORY_ROWS = 126
LOOKBACK_RETURN_DAYS = 63
