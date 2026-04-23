from __future__ import annotations

import json

import pandas as pd

from analysis.strategy_config import ASSET_UNIVERSE, DATA_DIR, DOCS_DIR, DOWNLOADS_DIR, SELECTED_SYMBOL


def _save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    asset_screen = pd.DataFrame(
        [{"symbol": symbol, "status": "planned"} for symbol in ASSET_UNIVERSE]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": SELECTED_SYMBOL, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1_000_000},
            {"date": "2024-01-03", "symbol": SELECTED_SYMBOL, "open": 100.5, "high": 101.4, "low": 100.1, "close": 101.0, "volume": 1_050_000},
        ]
    )
    selected_asset = {
        "selected_symbol": SELECTED_SYMBOL,
        "selection_status": "placeholder",
        "selection_note": "Replace with shinybroker-driven screening results.",
    }
    blotter = pd.DataFrame(
        columns=["entry_timestamp", "exit_timestamp", "direction", "qty", "entry_price", "exit_price", "exit_reason", "trade_return", "pnl"]
    )
    ledger = pd.DataFrame(columns=["date", "cash", "market_value", "equity", "drawdown"])

    _save_csv(asset_screen, DATA_DIR / "asset_screening.csv")
    _save_csv(prices, DATA_DIR / "historical_prices.csv")
    _save_csv(blotter, DOWNLOADS_DIR / "trade_blotter.csv")
    _save_csv(ledger, DOWNLOADS_DIR / "ledger.csv")
    (DATA_DIR / "selected_asset.json").write_text(json.dumps(selected_asset, indent=2), encoding="utf-8")

    print("Scaffold pipeline outputs written.")


if __name__ == "__main__":
    main()
