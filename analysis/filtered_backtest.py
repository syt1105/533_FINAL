from __future__ import annotations

import json

import pandas as pd

from analysis.breakout_strategy import (
    BASELINE_PARAMS,
    _build_features,
    _build_ledger_with_initial_capital,
    _compute_metrics,
    _load_selected_asset,
    _load_history,
    _simulate_trades_with_filters,
)
from analysis.model import DEFAULT_MODEL_PARAMS, train_universe_model
from analysis.strategy_config import DEFAULT_SELECTED_SYMBOL, DOWNLOADS_DIR


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _selected_symbol() -> str:
    selected = _load_selected_asset()
    return str(selected.get("selected_symbol", DEFAULT_SELECTED_SYMBOL))


def run_filtered_backtest(probability_threshold: float | None = None) -> dict[str, object]:
    symbol = _selected_symbol()
    model_results = train_universe_model()
    threshold = probability_threshold or DEFAULT_MODEL_PARAMS.probability_threshold

    predictions = model_results["test_predictions"].copy()
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"]).dt.normalize()
    symbol_predictions = predictions.loc[predictions["symbol"] == symbol].copy()
    if symbol_predictions.empty:
        raise RuntimeError(f"No out-of-sample model predictions are available for {symbol}.")

    test_start = pd.Timestamp(symbol_predictions["signal_date"].min()).normalize()
    test_end = pd.Timestamp(symbol_predictions["signal_date"].max()).normalize()
    allowed_signal_dates = set(
        symbol_predictions.loc[
            symbol_predictions["predicted_probability"] >= threshold, "signal_date"
        ].tolist()
    )

    history = _load_history(symbol)
    featured = _build_features(history, BASELINE_PARAMS)

    baseline_trades = _simulate_trades_with_filters(
        featured,
        symbol,
        BASELINE_PARAMS,
        signal_start_date=test_start,
        signal_end_date=test_end,
    )
    filtered_trades = _simulate_trades_with_filters(
        featured,
        symbol,
        BASELINE_PARAMS,
        allowed_signal_dates=allowed_signal_dates,
        signal_start_date=test_start,
        signal_end_date=test_end,
    )

    ledger_history = history.loc[history["date"] >= test_start].copy().reset_index(drop=True)
    baseline_ledger = _build_ledger_with_initial_capital(
        ledger_history, baseline_trades, BASELINE_PARAMS, initial_capital=100_000.0
    )
    filtered_ledger = _build_ledger_with_initial_capital(
        ledger_history, filtered_trades, BASELINE_PARAMS, initial_capital=100_000.0
    )

    _, baseline_metrics = _compute_metrics(baseline_trades, baseline_ledger)
    _, filtered_metrics = _compute_metrics(filtered_trades, filtered_ledger)

    comparison = pd.DataFrame(
        [
            {
                "strategy": "baseline_test_period",
                "trade_count": int(baseline_metrics["Trade Count"]),
                "sharpe_ratio": float(baseline_metrics["Sharpe Ratio"]),
                "total_return": float(baseline_metrics["Total Return"]),
                "win_rate": float(baseline_metrics["Win Rate"]),
                "max_drawdown": float(baseline_metrics["Max Drawdown"]),
                "expected_return_per_trade": float(baseline_metrics["Expected Return Per Trade"]),
            },
            {
                "strategy": "filtered_test_period",
                "trade_count": int(filtered_metrics["Trade Count"]),
                "sharpe_ratio": float(filtered_metrics["Sharpe Ratio"]),
                "total_return": float(filtered_metrics["Total Return"]),
                "win_rate": float(filtered_metrics["Win Rate"]),
                "max_drawdown": float(filtered_metrics["Max Drawdown"]),
                "expected_return_per_trade": float(filtered_metrics["Expected Return Per Trade"]),
            },
        ]
    )

    summary = {
        "selected_symbol": symbol,
        "test_start": test_start.strftime("%Y-%m-%d"),
        "test_end": test_end.strftime("%Y-%m-%d"),
        "probability_threshold": threshold,
        "predicted_trade_dates": len(allowed_signal_dates),
        "baseline_trade_count": int(len(baseline_trades)),
        "filtered_trade_count": int(len(filtered_trades)),
        "baseline_metrics": {k: _to_native(v) for k, v in baseline_metrics.items()},
        "filtered_metrics": {k: _to_native(v) for k, v in filtered_metrics.items()},
    }

    return {
        "summary": summary,
        "comparison": comparison,
        "baseline_trades": baseline_trades,
        "filtered_trades": filtered_trades,
        "baseline_ledger": baseline_ledger,
        "filtered_ledger": filtered_ledger,
        "test_predictions": symbol_predictions,
    }


def save_filtered_backtest_artifacts(results: dict[str, object]) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = results["comparison"].copy()
    comparison.to_csv(DOWNLOADS_DIR / "filtered_backtest_comparison.csv", index=False)

    for name in ["baseline_trades", "filtered_trades", "baseline_ledger", "filtered_ledger", "test_predictions"]:
        df = results[name].copy()
        for col in ["signal_date", "entry_timestamp", "exit_timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        df.to_csv(DOWNLOADS_DIR / f"{name}.csv", index=False)

    with (DOWNLOADS_DIR / "filtered_backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)


def run_threshold_sweep(
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    rows: list[dict[str, object]] = []

    baseline_recorded = False
    for threshold in thresholds:
        results = run_filtered_backtest(probability_threshold=threshold)
        summary = results["summary"]
        baseline_metrics = summary["baseline_metrics"]
        filtered_metrics = summary["filtered_metrics"]

        if not baseline_recorded:
            rows.append(
                {
                    "threshold": "baseline",
                    "trade_count": int(summary["baseline_trade_count"]),
                    "sharpe_ratio": float(baseline_metrics["Sharpe Ratio"]),
                    "total_return": float(baseline_metrics["Total Return"]),
                    "win_rate": float(baseline_metrics["Win Rate"]),
                    "max_drawdown": float(baseline_metrics["Max Drawdown"]),
                    "expected_return_per_trade": float(baseline_metrics["Expected Return Per Trade"]),
                    "test_start": summary["test_start"],
                    "test_end": summary["test_end"],
                }
            )
            baseline_recorded = True

        rows.append(
            {
                "threshold": float(threshold),
                "trade_count": int(summary["filtered_trade_count"]),
                "sharpe_ratio": float(filtered_metrics["Sharpe Ratio"]),
                "total_return": float(filtered_metrics["Total Return"]),
                "win_rate": float(filtered_metrics["Win Rate"]),
                "max_drawdown": float(filtered_metrics["Max Drawdown"]),
                "expected_return_per_trade": float(filtered_metrics["Expected Return Per Trade"]),
                "test_start": summary["test_start"],
                "test_end": summary["test_end"],
            }
        )

    return pd.DataFrame(rows)


def save_threshold_sweep(df: pd.DataFrame) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOWNLOADS_DIR / "threshold_sweep.csv", index=False)
