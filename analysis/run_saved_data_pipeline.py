from __future__ import annotations

import json

from analysis.breakout_strategy import run_analysis
from analysis.features import build_selected_asset_feature_dataset, build_universe_feature_dataset, save_feature_dataset
from analysis.filtered_backtest import (
    compare_selected_asset_ml_overlays,
    run_filtered_backtest,
    run_threshold_sweep,
    save_filtered_backtest_artifacts,
    save_threshold_sweep,
)
from analysis.labels import build_selected_asset_labeled_dataset, build_universe_labeled_dataset, save_labeled_breakout_dataset
from analysis.model import save_model_artifacts, train_universe_model
from analysis.strategy_config import DOWNLOADS_DIR


def main() -> None:
    """Regenerate all report artifacts from checked-in CSV data without calling IBKR."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

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

    hard_filter_results = run_filtered_backtest(filter_mode="hard_filter_selected_asset")
    save_filtered_backtest_artifacts(hard_filter_results, filename_prefix="hard_filter_selected_asset_backtest")

    scaled_results = run_filtered_backtest(filter_mode="scaled_selected_asset")
    save_filtered_backtest_artifacts(scaled_results, filename_prefix="scaled_selected_asset_backtest")

    overlay_comparison = compare_selected_asset_ml_overlays()
    overlay_comparison.to_csv(DOWNLOADS_DIR / "selected_asset_overlay_comparison.csv", index=False)

    threshold_sweep = run_threshold_sweep(filter_mode="scaled_selected_asset")
    save_threshold_sweep(threshold_sweep, filename="scaled_selected_asset_threshold_sweep.csv")

    summary = {
        "selected_symbol": analysis["overview"].loc[
            analysis["overview"]["Field"] == "Selected symbol", "Value"
        ].iloc[0],
        "baseline_trade_count": int(analysis["metrics_raw"]["Trade Count"]),
        "saved_data_pipeline": "completed",
    }
    (DOWNLOADS_DIR / "saved_data_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        "Regenerated saved-data artifacts for "
        f"{summary['selected_symbol']} with {summary['baseline_trade_count']} baseline trades."
    )


if __name__ == "__main__":
    main()
