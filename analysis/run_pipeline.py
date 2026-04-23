from __future__ import annotations

import json

import pandas as pd

from analysis.breakout_strategy import run_analysis
from analysis.data_pipeline import build_asset_screen, fetch_project_histories, select_asset_from_screen
from analysis.features import build_selected_asset_feature_dataset, build_universe_feature_dataset, save_feature_dataset
from analysis.filtered_backtest import run_filtered_backtest, save_filtered_backtest_artifacts
from analysis.labels import build_selected_asset_labeled_dataset, build_universe_labeled_dataset, save_labeled_breakout_dataset
from analysis.model import save_model_artifacts, train_universe_model
from analysis.strategy_config import ASSET_UNIVERSE, DATA_DIR, DOCS_DIR, DOWNLOADS_DIR, MARKET_HISTORY_DIR


def _save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    histories = fetch_project_histories(ASSET_UNIVERSE)
    asset_histories = {symbol: histories[symbol] for symbol in ASSET_UNIVERSE}
    asset_screen = build_asset_screen(asset_histories)
    selected_asset = select_asset_from_screen(asset_screen)

    _save_csv(asset_screen, DATA_DIR / "asset_screening.csv")
    for symbol, history in histories.items():
        history_out = history.copy()
        history_out["date"] = pd.to_datetime(history_out["date"]).dt.strftime("%Y-%m-%d")
        _save_csv(history_out, MARKET_HISTORY_DIR / f"{symbol}.csv")
    (DATA_DIR / "selected_asset.json").write_text(json.dumps(selected_asset, indent=2), encoding="utf-8")
    analysis = run_analysis(save_outputs=True)
    labeled_breakouts = build_selected_asset_labeled_dataset()
    save_labeled_breakout_dataset(labeled_breakouts)
    universe_labeled_breakouts = build_universe_labeled_dataset()
    save_labeled_breakout_dataset(universe_labeled_breakouts, filename="labeled_breakouts_universe.csv")
    feature_dataset = build_selected_asset_feature_dataset()
    save_feature_dataset(feature_dataset)
    universe_feature_dataset = build_universe_feature_dataset()
    save_feature_dataset(universe_feature_dataset, filename="ml_features_universe.csv")
    model_results = train_universe_model()
    save_model_artifacts(model_results)
    filtered_backtest_results = run_filtered_backtest()
    save_filtered_backtest_artifacts(filtered_backtest_results)

    print(
        f"Fetched history for {len(histories)} assets and selected "
        f"{analysis['overview'].loc[analysis['overview']['Field'] == 'Selected symbol', 'Value'].iloc[0]} "
        "for downstream analysis."
    )


if __name__ == "__main__":
    main()
