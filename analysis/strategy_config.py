from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
DOWNLOADS_DIR = DOCS_DIR / "downloads"

PROJECT_TITLE = "Volatility Breakout Final Project"
ASSET_UNIVERSE = ["SPY", "QQQ", "IWM", "XLE", "GLD", "TLT"]
SELECTED_SYMBOL = "XLE"
