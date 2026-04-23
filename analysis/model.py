from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.features import build_selected_asset_feature_dataset, build_universe_feature_dataset
from analysis.strategy_config import DOWNLOADS_DIR

FEATURE_COLUMNS = [
    "breakout_strength",
    "channel_width",
    "channel_width_pct",
    "atr_at_signal",
    "atr_pct",
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
    "vix_return_1d",
    "vix_return_5d",
    "vix_level_zscore_20d",
    "vix_level_pct_to_20d_mean",
]


@dataclass(frozen=True)
class ModelParams:
    train_fraction: float = 0.7
    probability_threshold: float = 0.55
    max_iter: int = 2000
    regularization_c: float = 1.0


DEFAULT_MODEL_PARAMS = ModelParams()


def _build_logistic_pipeline(params: ModelParams) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=params.max_iter,
                    C=params.regularization_c,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _time_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * train_fraction))
    split_idx = min(split_idx, len(df) - 1)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


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
    model: LogisticRegression = pipeline.named_steps["model"]
    coef = model.coef_[0]
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coef,
            "abs_coefficient": pd.Series(coef).abs(),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return table.reset_index(drop=True)


def _filtered_trade_summary(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:
    filtered = predictions.loc[predictions["predicted_probability"] >= threshold].copy()
    if filtered.empty:
        return {
            "filtered_trade_count": 0,
            "filtered_hit_rate": 0.0,
            "filtered_average_label": 0.0,
        }

    return {
        "filtered_trade_count": int(len(filtered)),
        "filtered_hit_rate": float((filtered["label"] == 1).mean()),
        "filtered_average_label": float(filtered["label"].mean()),
    }


def train_selected_asset_model(
    params: ModelParams = DEFAULT_MODEL_PARAMS,
) -> dict[str, object]:
    dataset = build_selected_asset_feature_dataset().copy()
    return train_model_from_dataset(dataset, params=params)


def train_universe_model(
    params: ModelParams = DEFAULT_MODEL_PARAMS,
) -> dict[str, object]:
    dataset = build_universe_feature_dataset().copy()
    return train_model_from_dataset(dataset, params=params)


def train_model_from_dataset(
    dataset: pd.DataFrame,
    params: ModelParams = DEFAULT_MODEL_PARAMS,
) -> dict[str, object]:
    if dataset.empty:
        raise RuntimeError("Feature dataset is empty; cannot train the logistic model.")

    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    dataset = dataset.sort_values("signal_date").reset_index(drop=True)

    train_df, test_df = _time_split(dataset, params.train_fraction)
    X_train = train_df.loc[:, FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)
    X_test = test_df.loc[:, FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)

    pipeline = _build_logistic_pipeline(params)
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
    filtered_summary = _filtered_trade_summary(test_predictions, params.probability_threshold)

    summary = {
        "symbols": sorted(dataset["symbol"].astype(str).unique().tolist()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "probability_threshold": params.probability_threshold,
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
