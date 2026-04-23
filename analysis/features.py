from __future__ import annotations

import pandas as pd

from analysis.breakout_strategy import BASELINE_PARAMS, BaselineParams, _build_features, _load_history
from analysis.labels import (
    DEFAULT_LABEL_PARAMS,
    LabelParams,
    _load_selected_symbol,
    build_labeled_breakout_dataset,
    build_universe_labeled_dataset,
)
from analysis.strategy_config import ASSET_UNIVERSE, DOWNLOADS_DIR


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, pd.NA)


def _build_ml_feature_frame(
    symbol: str,
    strategy_params: BaselineParams,
) -> pd.DataFrame:
    history = _load_history(symbol)
    featured = _build_features(history, strategy_params).copy()

    featured["daily_return"] = featured["close"].pct_change()
    featured["return_3d"] = featured["close"].pct_change(3)
    featured["return_5d"] = featured["close"].pct_change(5)
    featured["return_10d"] = featured["close"].pct_change(10)
    featured["return_20d"] = featured["close"].pct_change(20)

    featured["vol_5d"] = featured["daily_return"].rolling(5, min_periods=5).std(ddof=0) * (252**0.5)
    featured["vol_10d"] = featured["daily_return"].rolling(10, min_periods=10).std(ddof=0) * (252**0.5)
    featured["vol_20d"] = featured["daily_return"].rolling(20, min_periods=20).std(ddof=0) * (252**0.5)

    featured["volume_avg_20d"] = featured["volume"].rolling(20, min_periods=20).mean()
    featured["volume_ratio_20d"] = featured["volume"] / featured["volume_avg_20d"]
    featured["volume_zscore_20d"] = _rolling_zscore(featured["volume"], 20)
    featured["dollar_volume"] = featured["close"] * featured["volume"]
    featured["dollar_volume_avg_20d"] = featured["dollar_volume"].rolling(20, min_periods=20).mean()
    featured["dollar_volume_ratio_20d"] = featured["dollar_volume"] / featured["dollar_volume_avg_20d"]

    featured["atr_pct"] = featured["atr"] / featured["close"]
    featured["channel_width_pct"] = featured["channel_width"] / featured["close"]
    featured["distance_from_trend"] = (featured["close"] - featured["trend_filter"]) / featured["trend_filter"]
    featured["distance_from_upper_channel"] = (featured["close"] - featured["upper_channel"]) / featured["close"]
    featured["trend_slope_5d"] = featured["trend_filter"].pct_change(5)
    featured["trend_slope_10d"] = featured["trend_filter"].pct_change(10)

    return featured


def _build_context_feature_frame() -> pd.DataFrame:
    shy = _load_history("SHY").loc[:, ["date", "close"]].rename(columns={"close": "shy_close"})
    vix = _load_history("VIX").loc[:, ["date", "close"]].rename(columns={"close": "vix_close"})

    context = shy.merge(vix, on="date", how="outer").sort_values("date").reset_index(drop=True)
    context["shy_return_5d"] = context["shy_close"].pct_change(5)
    context["shy_return_20d"] = context["shy_close"].pct_change(20)
    context["vix_return_1d"] = context["vix_close"].pct_change(1)
    context["vix_return_5d"] = context["vix_close"].pct_change(5)
    context["vix_level_zscore_20d"] = _rolling_zscore(context["vix_close"], 20)
    context["vix_level_pct_to_20d_mean"] = (
        context["vix_close"] / context["vix_close"].rolling(20, min_periods=20).mean() - 1.0
    )
    return context


def build_feature_dataset(
    symbol: str,
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    labeled = build_labeled_breakout_dataset(symbol, strategy_params=strategy_params, label_params=label_params)
    if labeled.empty:
        return labeled

    feature_frame = _build_ml_feature_frame(symbol, strategy_params)
    context_frame = _build_context_feature_frame()
    feature_columns = [
        "date",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "volume_avg_20d",
        "volume_ratio_20d",
        "volume_zscore_20d",
        "dollar_volume",
        "dollar_volume_avg_20d",
        "dollar_volume_ratio_20d",
        "atr_pct",
        "channel_width_pct",
        "distance_from_trend",
        "distance_from_upper_channel",
        "trend_slope_5d",
        "trend_slope_10d",
    ]
    feature_slice = feature_frame.loc[:, feature_columns].rename(columns={"date": "signal_date"})
    dataset = labeled.merge(feature_slice, on="signal_date", how="left")
    context_slice = context_frame.rename(columns={"date": "signal_date"})
    dataset = dataset.merge(context_slice, on="signal_date", how="left")

    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce").astype("Int64")
    dataset = dataset.sort_values("signal_date").reset_index(drop=True)
    return dataset


def build_selected_asset_feature_dataset(
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    symbol = _load_selected_symbol()
    return build_feature_dataset(symbol, strategy_params=strategy_params, label_params=label_params)


def build_universe_feature_dataset(
    symbols: list[str] | None = None,
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    symbols = symbols or ASSET_UNIVERSE
    labeled = build_universe_labeled_dataset(symbols=symbols, strategy_params=strategy_params, label_params=label_params)
    if labeled.empty:
        return labeled

    context_frame = _build_context_feature_frame().rename(columns={"date": "signal_date"})
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol_labeled = labeled.loc[labeled["symbol"] == symbol].copy()
        if symbol_labeled.empty:
            continue
        symbol_features = _build_ml_feature_frame(symbol, strategy_params)
        feature_columns = [
            "date",
            "return_3d",
            "return_5d",
            "return_10d",
            "return_20d",
            "vol_5d",
            "vol_10d",
            "vol_20d",
            "volume_avg_20d",
            "volume_ratio_20d",
            "volume_zscore_20d",
            "dollar_volume",
            "dollar_volume_avg_20d",
            "dollar_volume_ratio_20d",
            "atr_pct",
            "channel_width_pct",
            "distance_from_trend",
            "distance_from_upper_channel",
            "trend_slope_5d",
            "trend_slope_10d",
        ]
        feature_slice = symbol_features.loc[:, feature_columns].rename(columns={"date": "signal_date"})
        symbol_dataset = symbol_labeled.merge(feature_slice, on="signal_date", how="left")
        symbol_dataset = symbol_dataset.merge(context_frame, on="signal_date", how="left")
        frames.append(symbol_dataset)

    dataset = pd.concat(frames, ignore_index=True).sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce").astype("Int64")
    return dataset


def save_feature_dataset(
    df: pd.DataFrame,
    filename: str = "ml_features.csv",
) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    for col in ["signal_date", "entry_date", "label_event_date"]:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col]).dt.strftime("%Y-%m-%d")
    output.to_csv(DOWNLOADS_DIR / filename, index=False)
