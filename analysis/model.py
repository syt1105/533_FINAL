from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.features import build_selected_asset_feature_dataset, build_universe_feature_dataset
from analysis.strategy_config import DOWNLOADS_DIR

FEATURE_COLUMNS = [
    "breakout_strength",
    "channel_width",
    "channel_width_pct",
    "atr_at_signal",
    "atr_pct",
    "atr_ratio_20d",
    "atr_zscore_20d",
    "realized_vol_ratio",
    "realized_vol_regime",
    "range_expansion_1d",
    "close_location_in_bar",
    "breakout_vol_confirmation",
    "breakout_volume_confirmation",
    "iv_30d",
    "hv_30d",
    "iv_hv_spread",
    "iv_hv_ratio",
    "iv_change_1d",
    "iv_change_5d",
    "iv_change_20d",
    "hv_change_20d",
    "iv_zscore_20d",
    "iv_zscore_60d",
    "iv_percentile_60d",
    "iv_trend_5d",
    "iv_trend_20d",
    "iv_spike_flag",
    "iv_rich_vs_hv_flag",
    "iv_breakout_confirmation",
    "iv_regime_confirmation",
    "distance_from_trend",
    "distance_from_upper_channel",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "volume_ratio_20d",
    "volume_zscore_20d",
    "dollar_volume_ratio_20d",
    "trend_slope_5d",
    "trend_slope_10d",
    "shy_return_5d",
    "shy_return_20d",
    "shy_trend_20d",
    "vix_return_1d",
    "vix_return_5d",
    "vix_return_20d",
    "vix_level_zscore_20d",
    "vix_level_zscore_60d",
    "vix_level_pct_to_20d_mean",
    "vix_percentile_60d",
    "vix_trend_5d",
    "vix_trend_20d",
    "vix_spike_flag",
    "vix_above_20d_mean_flag",
]


@dataclass(frozen=True)
class ModelParams:
    train_fraction: float = 0.7
    probability_threshold: float = 0.30
    max_iter: int = 3000
    regularization_c: float = 0.5
    min_signal_gap_days: int = 7
    use_symbol_feature: bool = True
    class_weight_balanced: bool = True


DEFAULT_MODEL_PARAMS = ModelParams()


def _build_logistic_pipeline(params: ModelParams, feature_columns: list[str] | None = None) -> Pipeline:
    selected_features = feature_columns or FEATURE_COLUMNS
    transformers: list[tuple[str, object, list[str]]] = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            selected_features,
        )
    ]
    if params.use_symbol_feature:
        transformers.append(("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]))

    class_weight = "balanced" if params.class_weight_balanced else None
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "model",
                LogisticRegression(
                    max_iter=params.max_iter,
                    C=params.regularization_c,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def _feature_input_columns(
    use_symbol_feature: bool,
    feature_columns: list[str] | None = None,
) -> list[str]:
    selected_features = feature_columns or FEATURE_COLUMNS
    return selected_features + (["symbol"] if use_symbol_feature else [])


def _deduplicate_breakout_events(df: pd.DataFrame, min_signal_gap_days: int) -> pd.DataFrame:
    if df.empty or min_signal_gap_days <= 1:
        return df.copy()

    deduped_frames: list[pd.DataFrame] = []
    for _, symbol_df in df.sort_values(["symbol", "signal_date"]).groupby("symbol", sort=False):
        kept_rows: list[pd.Series] = []
        last_signal_date: pd.Timestamp | None = None
        for _, row in symbol_df.iterrows():
            signal_date = pd.Timestamp(row["signal_date"]).normalize()
            if last_signal_date is None or (signal_date - last_signal_date).days >= min_signal_gap_days:
                kept_rows.append(row)
                last_signal_date = signal_date
        if kept_rows:
            deduped_frames.append(pd.DataFrame(kept_rows))

    if not deduped_frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(deduped_frames, ignore_index=True).sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def _time_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(df["signal_date"]).dt.normalize().unique())
    if len(unique_dates) < 2:
        raise RuntimeError("Not enough unique signal dates to create a train/test split.")

    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx]).normalize()

    train = df.loc[pd.to_datetime(df["signal_date"]).dt.normalize() < split_date].copy()
    test = df.loc[pd.to_datetime(df["signal_date"]).dt.normalize() >= split_date].copy()
    if train.empty or test.empty:
        raise RuntimeError("Train/test split produced an empty partition.")
    return train, test, split_date


def _evaluate_predictions(y_true: pd.Series, probas: pd.Series, threshold: float) -> dict[str, float]:
    preds = (probas >= threshold).astype(int)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "positive_prediction_rate": float(preds.mean()),
        "base_rate": float(y_true.mean()),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probas))
        metrics["log_loss"] = float(log_loss(y_true, probas, labels=[0, 1]))
    else:
        metrics["roc_auc"] = 0.0
        metrics["log_loss"] = 0.0
    return metrics


def _build_coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: LogisticRegression = pipeline.named_steps["model"]
    coef = model.coef_[0]
    feature_names = preprocessor.get_feature_names_out()
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": pd.Series(coef).abs(),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return table.reset_index(drop=True)


def _filtered_trade_summary(predictions: pd.DataFrame, threshold: float, target_col: str) -> dict[str, float]:
    filtered = predictions.loc[predictions["predicted_probability"] >= threshold].copy()
    if filtered.empty:
        return {
            "filtered_trade_count": 0,
            "filtered_hit_rate": 0.0,
            "filtered_average_label": 0.0,
        }

    return {
        "filtered_trade_count": int(len(filtered)),
        "filtered_hit_rate": float((filtered[target_col] == 1).mean()),
        "filtered_average_label": float(filtered[target_col].mean()),
    }


def train_selected_asset_model(
    params: ModelParams = DEFAULT_MODEL_PARAMS,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    dataset = build_selected_asset_feature_dataset().copy()
    return train_model_from_dataset(
        dataset,
        params=params,
        target_col="label",
        model_name="upside",
        feature_columns=feature_columns,
    )


def train_universe_model(
    params: ModelParams = DEFAULT_MODEL_PARAMS,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    dataset = build_universe_feature_dataset().copy()
    return train_model_from_dataset(
        dataset,
        params=params,
        target_col="label",
        model_name="upside",
        feature_columns=feature_columns,
    )


def train_downside_universe_model(
    params: ModelParams = DEFAULT_MODEL_PARAMS,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    dataset = build_universe_feature_dataset().copy()
    return train_model_from_dataset(
        dataset,
        params=params,
        target_col="downside_label",
        model_name="downside",
        feature_columns=feature_columns,
    )


def train_model_from_dataset(
    dataset: pd.DataFrame,
    params: ModelParams = DEFAULT_MODEL_PARAMS,
    target_col: str = "label",
    model_name: str = "upside",
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    if dataset.empty:
        raise RuntimeError("Feature dataset is empty; cannot train the logistic model.")

    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    dataset = dataset.sort_values("signal_date").reset_index(drop=True)
    dataset = _deduplicate_breakout_events(dataset, params.min_signal_gap_days)
    if target_col not in dataset.columns:
        raise RuntimeError(f"Target column '{target_col}' is missing from the feature dataset.")
    selected_features = feature_columns or FEATURE_COLUMNS
    missing_features = [col for col in selected_features if col not in dataset.columns]
    if missing_features:
        raise RuntimeError(f"Missing feature columns: {missing_features}")

    train_df, test_df, split_date = _time_split(dataset, params.train_fraction)
    feature_input_columns = _feature_input_columns(params.use_symbol_feature, selected_features)
    X_train = train_df.loc[:, feature_input_columns]
    y_train = train_df[target_col].astype(int)
    X_test = test_df.loc[:, feature_input_columns]
    y_test = test_df[target_col].astype(int)

    pipeline = _build_logistic_pipeline(params, selected_features)
    pipeline.fit(X_train, y_train)

    train_proba = pd.Series(pipeline.predict_proba(X_train)[:, 1], index=train_df.index)
    test_proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=test_df.index)

    train_metrics = _evaluate_predictions(y_train, train_proba, params.probability_threshold)
    test_metrics = _evaluate_predictions(y_test, test_proba, params.probability_threshold)

    train_predictions = train_df.copy()
    train_predictions["predicted_probability"] = train_proba.values
    train_predictions["predicted_label"] = (
        train_predictions["predicted_probability"] >= params.probability_threshold
    ).astype(int)
    train_predictions["sample_split"] = "train"

    test_predictions = test_df.copy()
    test_predictions["predicted_probability"] = test_proba.values
    test_predictions["predicted_label"] = (
        test_predictions["predicted_probability"] >= params.probability_threshold
    ).astype(int)
    test_predictions["sample_split"] = "test"

    predictions = pd.concat([train_predictions, test_predictions], ignore_index=True)
    coefficient_table = _build_coefficient_table(pipeline)
    filtered_summary = _filtered_trade_summary(test_predictions, params.probability_threshold, target_col)

    summary = {
        "model_name": model_name,
        "target_col": target_col,
        "feature_columns": selected_features,
        "symbols": sorted(dataset["symbol"].astype(str).unique().tolist()),
        "post_dedup_rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start": split_date.strftime("%Y-%m-%d"),
        "probability_threshold": params.probability_threshold,
        "min_signal_gap_days": params.min_signal_gap_days,
        "use_symbol_feature": params.use_symbol_feature,
        "class_weight_balanced": params.class_weight_balanced,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "filtered_summary": filtered_summary,
    }

    return {
        "dataset": dataset,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "predictions": predictions,
        "coefficient_table": coefficient_table,
        "summary": summary,
    }


def save_model_artifacts(results: dict[str, object]) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = results["predictions"].copy()
    for col in ["signal_date", "entry_date", "label_event_date"]:
        if col in predictions.columns:
            predictions[col] = pd.to_datetime(predictions[col]).dt.strftime("%Y-%m-%d")
    predictions.to_csv(DOWNLOADS_DIR / "ml_predictions.csv", index=False)

    coefficient_table = results["coefficient_table"].copy()
    coefficient_table.to_csv(DOWNLOADS_DIR / "logistic_coefficients.csv", index=False)

    with (DOWNLOADS_DIR / "model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)
