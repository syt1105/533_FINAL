from __future__ import annotations

import json

import pandas as pd

from analysis.breakout_strategy import run_analysis
from analysis.data_pipeline import build_asset_screen, fetch_universe_history, select_asset_from_screen
from analysis.strategy_config import ASSET_UNIVERSE, DATA_DIR, DOCS_DIR, DOWNLOADS_DIR, MARKET_HISTORY_DIR


def _save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    histories = fetch_universe_history(ASSET_UNIVERSE)
    asset_screen = build_asset_screen(histories)
    selected_asset = select_asset_from_screen(asset_screen)

    _save_csv(asset_screen, DATA_DIR / "asset_screening.csv")
    for symbol, history in histories.items():
        history_out = history.copy()
        history_out["date"] = pd.to_datetime(history_out["date"]).dt.strftime("%Y-%m-%d")
        _save_csv(history_out, MARKET_HISTORY_DIR / f"{symbol}.csv")
    (DATA_DIR / "selected_asset.json").write_text(json.dumps(selected_asset, indent=2), encoding="utf-8")
    run_analysis(save_outputs=True)

    print(
        f"Fetched history for {len(histories)} assets and selected "
        f"{selected_asset['selected_symbol']} for downstream analysis."
    )


if __name__ == "__main__":
    main()
