from __future__ import annotations

import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.breakout_strategy import (
    BASELINE_PARAMS,
    _build_features,
    _build_ledger_with_initial_capital,
    _compute_metrics,
    _load_selected_asset,
    _load_history,
    _simulate_trades_with_filters,
)
from analysis.features import build_selected_asset_feature_dataset
from analysis.model import (
    DEFAULT_MODEL_PARAMS,
    ModelParams,
    train_downside_universe_model,
    train_selected_asset_model,
    train_universe_model,
)
from analysis.strategy_config import DEFAULT_SELECTED_SYMBOL, DOWNLOADS_DIR

SELECTED_ASSET_CORE_IV_FEATURES = [
    "breakout_strength",
    "distance_from_trend",
    "distance_from_upper_channel",
    "channel_width_pct",
    "atr_pct",
    "iv_hv_spread",
    "iv_hv_ratio",
    "iv_zscore_60d",
    "iv_trend_20d",
]


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _selected_symbol() -> str:
    selected = _load_selected_asset()
    return str(selected.get("selected_symbol", DEFAULT_SELECTED_SYMBOL))


def _allowed_signal_dates(
    symbol_predictions: pd.DataFrame,
    threshold: float,
    filter_mode: str,
) -> tuple[set[pd.Timestamp], int]:
    if filter_mode == "top_pick":
        allowed = set(
            symbol_predictions.loc[
                symbol_predictions["predicted_probability"] >= threshold, "signal_date"
            ].tolist()
        )
        removed_count = int(len(symbol_predictions) - len(allowed))
        return allowed, removed_count

    if filter_mode == "veto":
        allowed = set(
            symbol_predictions.loc[
                symbol_predictions["predicted_probability"] > threshold, "signal_date"
            ].tolist()
        )
        removed_count = int(len(symbol_predictions) - len(allowed))
        return allowed, removed_count

    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def _scaled_signal_size_multipliers(symbol_predictions: pd.DataFrame) -> dict[pd.Timestamp, float]:
    multipliers: dict[pd.Timestamp, float] = {}
    for _, row in symbol_predictions.iterrows():
        downside_probability = float(row["predicted_probability"])
        if downside_probability >= 0.70:
            size = 0.25
        elif downside_probability >= 0.55:
            size = 0.50
        elif downside_probability >= 0.40:
            size = 0.75
        else:
            size = 1.00
        multipliers[pd.Timestamp(row["signal_date"]).normalize()] = size
    return multipliers


def _scaled_upside_signal_size_multipliers(symbol_predictions: pd.DataFrame) -> dict[pd.Timestamp, float]:
    multipliers: dict[pd.Timestamp, float] = {}
    for _, row in symbol_predictions.iterrows():
        upside_probability = float(row["predicted_probability"])
        if upside_probability >= 0.70:
            size = 1.30
        elif upside_probability >= 0.55:
            size = 1.15
        elif upside_probability >= 0.40:
            size = 1.05
        elif upside_probability >= 0.25:
            size = 0.97
        else:
            size = 0.93
        multipliers[pd.Timestamp(row["signal_date"]).normalize()] = size
    return multipliers


def _dedupe_selected_asset_events(df: pd.DataFrame, min_gap_days: int = 3) -> pd.DataFrame:
    kept_rows: list[pd.Series] = []
    last_signal_date: pd.Timestamp | None = None
    for _, row in df.sort_values("signal_date").iterrows():
        signal_date = pd.Timestamp(row["signal_date"]).normalize()
        if last_signal_date is None or (signal_date - last_signal_date).days >= min_gap_days:
            kept_rows.append(row)
            last_signal_date = signal_date
    if not kept_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _train_selected_asset_core_iv_model() -> dict[str, object]:
    dataset = build_selected_asset_feature_dataset().copy()
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    required_columns = ["signal_date", "label"] + SELECTED_ASSET_CORE_IV_FEATURES
    dataset = dataset.loc[:, required_columns].dropna().copy()
    dataset = _dedupe_selected_asset_events(dataset, min_gap_days=3)

    unique_dates = sorted(dataset["signal_date"].dt.normalize().unique())
    split_idx = max(1, int(len(unique_dates) * 0.7))
    split_idx = min(split_idx, len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx]).normalize()

    train_df = dataset.loc[dataset["signal_date"].dt.normalize() < split_date].copy()
    test_df = dataset.loc[dataset["signal_date"].dt.normalize() >= split_date].copy()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")),
        ]
    )
    pipeline.fit(train_df[SELECTED_ASSET_CORE_IV_FEATURES], train_df["label"].astype(int))

    test_predictions = test_df.copy()
    test_predictions["predicted_probability"] = pipeline.predict_proba(test_df[SELECTED_ASSET_CORE_IV_FEATURES])[:, 1]
    test_predictions["predicted_label"] = (test_predictions["predicted_probability"] >= 0.30).astype(int)
    test_predictions["sample_split"] = "test"

    return {
        "summary": {"test_start": split_date.strftime("%Y-%m-%d")},
        "test_predictions": test_predictions,
    }


def _train_selected_asset_core_iv_model_with_holdout(
    holdout_start: str = "2025-10-01",
    holdout_end: str | None = None,
) -> dict[str, object]:
    dataset = build_selected_asset_feature_dataset().copy()
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"]).dt.normalize()
    required_columns = ["signal_date", "label", "label_event"] + SELECTED_ASSET_CORE_IV_FEATURES
    dataset = dataset.loc[:, required_columns].dropna().copy()
    dataset = _dedupe_selected_asset_events(dataset, min_gap_days=3)

    holdout_start_ts = pd.Timestamp(holdout_start).normalize()
    holdout_end_ts = pd.Timestamp(holdout_end).normalize() if holdout_end else None
    train_df = dataset.loc[dataset["signal_date"] < holdout_start_ts].copy()
    test_mask = dataset["signal_date"] >= holdout_start_ts
    if holdout_end_ts is not None:
        test_mask &= dataset["signal_date"] <= holdout_end_ts
    test_df = dataset.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Fixed holdout split produced an empty train or test partition.")
    if train_df["label"].nunique() < 2:
        raise RuntimeError("Fixed holdout training data has only one label class.")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")),
        ]
    )
    pipeline.fit(train_df[SELECTED_ASSET_CORE_IV_FEATURES], train_df["label"].astype(int))

    train_predictions = train_df.copy()
    train_predictions["predicted_probability"] = pipeline.predict_proba(train_df[SELECTED_ASSET_CORE_IV_FEATURES])[:, 1]
    train_predictions["predicted_label"] = (train_predictions["predicted_probability"] >= 0.30).astype(int)
    train_predictions["sample_split"] = "pre_holdout_train"

    test_predictions = test_df.copy()
    test_predictions["predicted_probability"] = pipeline.predict_proba(test_df[SELECTED_ASSET_CORE_IV_FEATURES])[:, 1]
    test_predictions["predicted_label"] = (test_predictions["predicted_probability"] >= 0.30).astype(int)
    test_predictions["sample_split"] = "fixed_live_holdout"

    return {
        "summary": {
            "test_start": holdout_start_ts.strftime("%Y-%m-%d"),
            "test_end": holdout_end_ts.strftime("%Y-%m-%d") if holdout_end_ts is not None else "",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "predictions": pd.concat([train_predictions, test_predictions], ignore_index=True),
    }


def run_filtered_backtest(
    probability_threshold: float | None = None,
    filter_mode: str = "top_pick",
) -> dict[str, object]:
    symbol = _selected_symbol()
    if filter_mode == "scaled":
        model_results = train_downside_universe_model()
    elif filter_mode in {"scaled_selected_asset", "hard_filter_selected_asset"}:
        model_results = _train_selected_asset_core_iv_model()
    else:
        model_results = train_universe_model()
    threshold = probability_threshold or DEFAULT_MODEL_PARAMS.probability_threshold

    predictions = model_results["test_predictions"].copy()
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"]).dt.normalize()
    symbol_predictions = predictions.copy()
    if "symbol" in symbol_predictions.columns:
        symbol_predictions = symbol_predictions.loc[symbol_predictions["symbol"] == symbol].copy()
    else:
        symbol_predictions["symbol"] = symbol
    if symbol_predictions.empty:
        raise RuntimeError(f"No out-of-sample model predictions are available for {symbol}.")

    test_start = pd.Timestamp(model_results["summary"]["test_start"]).normalize()
    test_end = pd.Timestamp(symbol_predictions["signal_date"].max()).normalize()
    allowed_signal_dates = None
    signal_size_multipliers = None
    removed_signal_count = 0
    if filter_mode == "scaled":
        signal_size_multipliers = _scaled_signal_size_multipliers(symbol_predictions)
    elif filter_mode == "scaled_selected_asset":
        signal_size_multipliers = _scaled_upside_signal_size_multipliers(symbol_predictions)
    elif filter_mode == "hard_filter_selected_asset":
        allowed_signal_dates, removed_signal_count = _allowed_signal_dates(symbol_predictions, threshold, "top_pick")
    else:
        allowed_signal_dates, removed_signal_count = _allowed_signal_dates(symbol_predictions, threshold, filter_mode)

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
        signal_size_multipliers=signal_size_multipliers,
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
        "filter_mode": filter_mode,
        "probability_threshold": threshold,
        "predicted_trade_dates": len(allowed_signal_dates) if allowed_signal_dates is not None else int(len(symbol_predictions)),
        "removed_signal_count": removed_signal_count,
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


def compare_selected_asset_ml_overlays(probability_threshold: float | None = None) -> pd.DataFrame:
    """Compare the professor's hard-filter idea with the final sizing overlay."""
    hard_filter = run_filtered_backtest(
        probability_threshold=probability_threshold,
        filter_mode="hard_filter_selected_asset",
    )
    scaled = run_filtered_backtest(filter_mode="scaled_selected_asset")

    rows: list[dict[str, object]] = []
    baseline_metrics = hard_filter["summary"]["baseline_metrics"]
    rows.append(
        {
            "strategy": "baseline",
            "trade_count": int(hard_filter["summary"]["baseline_trade_count"]),
            "removed_signal_count": 0,
            "sharpe_ratio": float(baseline_metrics["Sharpe Ratio"]),
            "total_return": float(baseline_metrics["Total Return"]),
            "win_rate": float(baseline_metrics["Win Rate"]),
            "max_drawdown": float(baseline_metrics["Max Drawdown"]),
            "expected_return_per_trade": float(baseline_metrics["Expected Return Per Trade"]),
        }
    )

    for name, results in [
        ("ml_hard_filter", hard_filter),
        ("ml_position_scaling", scaled),
    ]:
        summary = results["summary"]
        metrics = summary["filtered_metrics"]
        rows.append(
            {
                "strategy": name,
                "trade_count": int(summary["filtered_trade_count"]),
                "removed_signal_count": int(summary["removed_signal_count"]),
                "sharpe_ratio": float(metrics["Sharpe Ratio"]),
                "total_return": float(metrics["Total Return"]),
                "win_rate": float(metrics["Win Rate"]),
                "max_drawdown": float(metrics["Max Drawdown"]),
                "expected_return_per_trade": float(metrics["Expected Return Per Trade"]),
            }
        )

    return pd.DataFrame(rows)


def run_fixed_live_holdout_backtest(
    holdout_start: str = "2025-10-01",
    holdout_end: str | None = None,
) -> dict[str, object]:
    """Train before the holdout start and evaluate the later period as unseen data."""
    symbol = _selected_symbol()
    history = _load_history(symbol)
    history_end = pd.Timestamp(history["date"].max()).normalize()
    holdout_start_ts = pd.Timestamp(holdout_start).normalize()
    holdout_end_ts = pd.Timestamp(holdout_end).normalize() if holdout_end else history_end

    model_results = _train_selected_asset_core_iv_model_with_holdout(
        holdout_start=holdout_start_ts.strftime("%Y-%m-%d"),
        holdout_end=holdout_end_ts.strftime("%Y-%m-%d"),
    )
    predictions = model_results["test_predictions"].copy()
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"]).dt.normalize()
    predictions["symbol"] = symbol

    signal_size_multipliers = _scaled_upside_signal_size_multipliers(predictions)
    allowed_signal_dates, removed_signal_count = _allowed_signal_dates(
        predictions,
        DEFAULT_MODEL_PARAMS.probability_threshold,
        "top_pick",
    )

    featured = _build_features(history, BASELINE_PARAMS)
    baseline_trades = _simulate_trades_with_filters(
        featured,
        symbol,
        BASELINE_PARAMS,
        signal_start_date=holdout_start_ts,
        signal_end_date=holdout_end_ts,
    )
    scaled_trades = _simulate_trades_with_filters(
        featured,
        symbol,
        BASELINE_PARAMS,
        signal_size_multipliers=signal_size_multipliers,
        signal_start_date=holdout_start_ts,
        signal_end_date=holdout_end_ts,
    )
    hard_filter_trades = _simulate_trades_with_filters(
        featured,
        symbol,
        BASELINE_PARAMS,
        allowed_signal_dates=allowed_signal_dates,
        signal_start_date=holdout_start_ts,
        signal_end_date=holdout_end_ts,
    )

    ledger_history = history.loc[
        (history["date"] >= holdout_start_ts) & (history["date"] <= holdout_end_ts)
    ].copy().reset_index(drop=True)
    baseline_ledger = _build_ledger_with_initial_capital(
        ledger_history, baseline_trades, BASELINE_PARAMS, initial_capital=100_000.0
    )
    scaled_ledger = _build_ledger_with_initial_capital(
        ledger_history, scaled_trades, BASELINE_PARAMS, initial_capital=100_000.0
    )
    hard_filter_ledger = _build_ledger_with_initial_capital(
        ledger_history, hard_filter_trades, BASELINE_PARAMS, initial_capital=100_000.0
    )

    _, baseline_metrics = _compute_metrics(baseline_trades, baseline_ledger)
    _, scaled_metrics = _compute_metrics(scaled_trades, scaled_ledger)
    _, hard_filter_metrics = _compute_metrics(hard_filter_trades, hard_filter_ledger)

    comparison_rows = []
    for strategy, trades, metrics, removed_count in [
        ("baseline_live_holdout", baseline_trades, baseline_metrics, 0),
        ("ml_hard_filter_live_holdout", hard_filter_trades, hard_filter_metrics, removed_signal_count),
        ("ml_position_scaling_live_holdout", scaled_trades, scaled_metrics, 0),
    ]:
        comparison_rows.append(
            {
                "strategy": strategy,
                "trade_count": int(len(trades)),
                "removed_signal_count": int(removed_count),
                "sharpe_ratio": float(metrics["Sharpe Ratio"]),
                "total_return": float(metrics["Total Return"]),
                "win_rate": float(metrics["Win Rate"]),
                "max_drawdown": float(metrics["Max Drawdown"]),
                "expected_return_per_trade": float(metrics["Expected Return Per Trade"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    equity_curves = pd.concat(
        [
            baseline_ledger.assign(strategy="baseline_live_holdout"),
            hard_filter_ledger.assign(strategy="ml_hard_filter_live_holdout"),
            scaled_ledger.assign(strategy="ml_position_scaling_live_holdout"),
        ],
        ignore_index=True,
    )

    summary = {
        "selected_symbol": symbol,
        "holdout_start": holdout_start_ts.strftime("%Y-%m-%d"),
        "holdout_end": holdout_end_ts.strftime("%Y-%m-%d"),
        "training_window": f"all labeled breakouts before {holdout_start_ts.strftime('%Y-%m-%d')}",
        "model_features": SELECTED_ASSET_CORE_IV_FEATURES,
        "test_prediction_count": int(len(predictions)),
        "baseline_trade_count": int(len(baseline_trades)),
        "scaled_trade_count": int(len(scaled_trades)),
        "hard_filter_trade_count": int(len(hard_filter_trades)),
        "hard_filter_removed_signal_count": int(removed_signal_count),
        "baseline_metrics": {k: _to_native(v) for k, v in baseline_metrics.items()},
        "scaled_metrics": {k: _to_native(v) for k, v in scaled_metrics.items()},
        "hard_filter_metrics": {k: _to_native(v) for k, v in hard_filter_metrics.items()},
    }

    return {
        "summary": summary,
        "comparison": comparison,
        "baseline_trades": baseline_trades,
        "scaled_trades": scaled_trades,
        "hard_filter_trades": hard_filter_trades,
        "baseline_ledger": baseline_ledger,
        "scaled_ledger": scaled_ledger,
        "hard_filter_ledger": hard_filter_ledger,
        "equity_curves": equity_curves,
        "test_predictions": predictions,
        "all_predictions": model_results["predictions"],
    }


def save_fixed_live_holdout_artifacts(
    results: dict[str, object],
    filename_prefix: str = "fixed_live_holdout",
) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    results["comparison"].to_csv(DOWNLOADS_DIR / f"{filename_prefix}_comparison.csv", index=False)

    for name in [
        "baseline_trades",
        "scaled_trades",
        "hard_filter_trades",
        "baseline_ledger",
        "scaled_ledger",
        "hard_filter_ledger",
        "equity_curves",
        "test_predictions",
        "all_predictions",
    ]:
        df = results[name].copy()
        for col in ["signal_date", "entry_date", "label_event_date", "entry_timestamp", "exit_timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        df.to_csv(DOWNLOADS_DIR / f"{filename_prefix}_{name}.csv", index=False)

    with (DOWNLOADS_DIR / f"{filename_prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)


def save_filtered_backtest_artifacts(results: dict[str, object], filename_prefix: str = "filtered_backtest") -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = results["comparison"].copy()
    comparison.to_csv(DOWNLOADS_DIR / f"{filename_prefix}_comparison.csv", index=False)

    for name in ["baseline_trades", "filtered_trades", "baseline_ledger", "filtered_ledger", "test_predictions"]:
        df = results[name].copy()
        for col in ["signal_date", "entry_timestamp", "exit_timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        df.to_csv(DOWNLOADS_DIR / f"{filename_prefix}_{name}.csv", index=False)

    with (DOWNLOADS_DIR / f"{filename_prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)


def run_threshold_sweep(
    thresholds: list[float] | None = None,
    filter_mode: str = "top_pick",
) -> pd.DataFrame:
    if filter_mode in {"scaled", "scaled_selected_asset"}:
        results = run_filtered_backtest(filter_mode=filter_mode)
        summary = results["summary"]
        baseline_metrics = summary["baseline_metrics"]
        filtered_metrics = summary["filtered_metrics"]
        return pd.DataFrame(
            [
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
                    "removed_signal_count": 0,
                },
                {
                    "threshold": "scaled",
                    "trade_count": int(summary["filtered_trade_count"]),
                    "sharpe_ratio": float(filtered_metrics["Sharpe Ratio"]),
                    "total_return": float(filtered_metrics["Total Return"]),
                    "win_rate": float(filtered_metrics["Win Rate"]),
                    "max_drawdown": float(filtered_metrics["Max Drawdown"]),
                    "expected_return_per_trade": float(filtered_metrics["Expected Return Per Trade"]),
                    "test_start": summary["test_start"],
                    "test_end": summary["test_end"],
                    "removed_signal_count": 0,
                },
            ]
        )

    thresholds = thresholds or [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows: list[dict[str, object]] = []

    baseline_recorded = False
    for threshold in thresholds:
        results = run_filtered_backtest(probability_threshold=threshold, filter_mode=filter_mode)
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
                    "removed_signal_count": int(summary["removed_signal_count"]),
                    "test_start": summary["test_start"],
                    "test_end": summary["test_end"],
                }
        )

    return pd.DataFrame(rows)


def save_threshold_sweep(df: pd.DataFrame, filename: str = "threshold_sweep.csv") -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOWNLOADS_DIR / filename, index=False)
