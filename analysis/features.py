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
from analysis.strategy_config import ASSET_UNIVERSE, DOWNLOADS_DIR, OPTIONS_HISTORY_DIR


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, pd.NA)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).rank(pct=True)


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
    featured["vol_60d"] = featured["daily_return"].rolling(60, min_periods=60).std(ddof=0) * (252**0.5)

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
    featured["atr_ratio_20d"] = featured["atr"] / featured["atr"].rolling(20, min_periods=20).mean()
    featured["atr_zscore_20d"] = _rolling_zscore(featured["atr"], 20)
    featured["realized_vol_ratio"] = featured["vol_10d"] / featured["vol_20d"]
    featured["realized_vol_regime"] = featured["vol_20d"] / featured["vol_60d"]
    featured["range_expansion_1d"] = (featured["high"] - featured["low"]) / featured["atr"]
    featured["close_location_in_bar"] = (
        (featured["close"] - featured["low"]) / (featured["high"] - featured["low"]).replace(0, pd.NA)
    )
    featured["breakout_vol_confirmation"] = featured["breakout_strength"] * featured["atr_ratio_20d"]
    featured["breakout_volume_confirmation"] = featured["breakout_strength"] * featured["volume_ratio_20d"]

    return featured


def _load_option_derived_volatility_frame(symbol: str) -> pd.DataFrame:
    iv_path = OPTIONS_HISTORY_DIR / f"{symbol}_OPTION_IMPLIED_VOLATILITY.csv"
    hv_path = OPTIONS_HISTORY_DIR / f"{symbol}_HISTORICAL_VOLATILITY.csv"
    if not iv_path.exists() or not hv_path.exists():
        return pd.DataFrame(columns=["date"])

    iv = pd.read_csv(iv_path)
    hv = pd.read_csv(hv_path)
    iv["date"] = pd.to_datetime(iv["date"])
    hv["date"] = pd.to_datetime(hv["date"])

    iv = iv.loc[:, ["date", "close"]].rename(columns={"close": "iv_30d"})
    hv = hv.loc[:, ["date", "close"]].rename(columns={"close": "hv_30d"})
    option_vol = iv.merge(hv, on="date", how="outer").sort_values("date").reset_index(drop=True)

    option_vol["iv_hv_spread"] = option_vol["iv_30d"] - option_vol["hv_30d"]
    option_vol["iv_hv_ratio"] = option_vol["iv_30d"] / option_vol["hv_30d"].replace(0, pd.NA)
    option_vol["iv_change_1d"] = option_vol["iv_30d"].pct_change(1, fill_method=None)
    option_vol["iv_change_5d"] = option_vol["iv_30d"].pct_change(5, fill_method=None)
    option_vol["iv_change_20d"] = option_vol["iv_30d"].pct_change(20, fill_method=None)
    option_vol["hv_change_20d"] = option_vol["hv_30d"].pct_change(20, fill_method=None)
    option_vol["iv_zscore_20d"] = _rolling_zscore(option_vol["iv_30d"], 20)
    option_vol["iv_zscore_60d"] = _rolling_zscore(option_vol["iv_30d"], 60)
    option_vol["iv_percentile_60d"] = _rolling_percentile(option_vol["iv_30d"], 60)
    option_vol["iv_trend_5d"] = option_vol["iv_30d"].rolling(5, min_periods=5).mean().pct_change(5)
    option_vol["iv_trend_20d"] = option_vol["iv_30d"].rolling(20, min_periods=20).mean().pct_change(20)
    option_vol["iv_spike_flag"] = (option_vol["iv_change_1d"] > 0.03).astype(float)
    option_vol["iv_rich_vs_hv_flag"] = (option_vol["iv_hv_spread"] > 0).astype(float)
    return option_vol


def _build_context_feature_frame() -> pd.DataFrame:
    shy = _load_history("SHY").loc[:, ["date", "close"]].rename(columns={"close": "shy_close"})
    vix = _load_history("VIX").loc[:, ["date", "close"]].rename(columns={"close": "vix_close"})

    context = shy.merge(vix, on="date", how="outer").sort_values("date").reset_index(drop=True)
    context["shy_return_5d"] = context["shy_close"].pct_change(5)
    context["shy_return_20d"] = context["shy_close"].pct_change(20)
    context["vix_return_1d"] = context["vix_close"].pct_change(1)
    context["vix_return_5d"] = context["vix_close"].pct_change(5)
    context["vix_return_20d"] = context["vix_close"].pct_change(20)
    context["vix_level_zscore_20d"] = _rolling_zscore(context["vix_close"], 20)
    context["vix_level_pct_to_20d_mean"] = (
        context["vix_close"] / context["vix_close"].rolling(20, min_periods=20).mean() - 1.0
    )
    context["vix_level_zscore_60d"] = _rolling_zscore(context["vix_close"], 60)
    context["vix_percentile_60d"] = _rolling_percentile(context["vix_close"], 60)
    context["vix_ma_5"] = context["vix_close"].rolling(5, min_periods=5).mean()
    context["vix_ma_20"] = context["vix_close"].rolling(20, min_periods=20).mean()
    context["vix_trend_5d"] = context["vix_ma_5"].pct_change(5)
    context["vix_trend_20d"] = context["vix_ma_20"].pct_change(20)
    context["vix_spike_flag"] = (context["vix_return_1d"] > 0.05).astype(float)
    context["vix_above_20d_mean_flag"] = (context["vix_close"] > context["vix_ma_20"]).astype(float)
    context["shy_trend_20d"] = context["shy_close"].pct_change(20)
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
    option_vol_frame = _load_option_derived_volatility_frame(symbol)
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
        "atr_ratio_20d",
        "atr_zscore_20d",
        "realized_vol_ratio",
        "realized_vol_regime",
        "range_expansion_1d",
        "close_location_in_bar",
        "breakout_vol_confirmation",
        "breakout_volume_confirmation",
        "distance_from_trend",
        "distance_from_upper_channel",
        "trend_slope_5d",
        "trend_slope_10d",
    ]
    feature_slice = feature_frame.loc[:, feature_columns].rename(columns={"date": "signal_date"})
    dataset = labeled.merge(feature_slice, on="signal_date", how="left")
    context_slice = context_frame.rename(columns={"date": "signal_date"})
    dataset = dataset.merge(context_slice, on="signal_date", how="left")
    if not option_vol_frame.empty:
        option_vol_slice = option_vol_frame.rename(columns={"date": "signal_date"})
        dataset = dataset.merge(option_vol_slice, on="signal_date", how="left")

    dataset["iv_breakout_confirmation"] = dataset["breakout_strength"] * dataset.get("iv_hv_spread", pd.NA)
    dataset["iv_regime_confirmation"] = dataset.get("iv_percentile_60d", pd.NA) * dataset["breakout_strength"]

    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce").astype("Int64")
    dataset["downside_label"] = (dataset["label_event"].astype(str) == "stop_first").astype("Int64")
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
        option_vol_frame = _load_option_derived_volatility_frame(symbol)
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
        "atr_ratio_20d",
        "atr_zscore_20d",
        "realized_vol_ratio",
        "realized_vol_regime",
        "range_expansion_1d",
        "close_location_in_bar",
            "breakout_vol_confirmation",
            "breakout_volume_confirmation",
            "distance_from_trend",
            "distance_from_upper_channel",
            "trend_slope_5d",
            "trend_slope_10d",
        ]
        feature_slice = symbol_features.loc[:, feature_columns].rename(columns={"date": "signal_date"})
        symbol_dataset = symbol_labeled.merge(feature_slice, on="signal_date", how="left")
        symbol_dataset = symbol_dataset.merge(context_frame, on="signal_date", how="left")
        if not option_vol_frame.empty:
            option_vol_slice = option_vol_frame.rename(columns={"date": "signal_date"})
            symbol_dataset = symbol_dataset.merge(option_vol_slice, on="signal_date", how="left")
        symbol_dataset["iv_breakout_confirmation"] = (
            symbol_dataset["breakout_strength"] * symbol_dataset.get("iv_hv_spread", pd.NA)
        )
        symbol_dataset["iv_regime_confirmation"] = (
            symbol_dataset.get("iv_percentile_60d", pd.NA) * symbol_dataset["breakout_strength"]
        )
        symbol_dataset["downside_label"] = (symbol_dataset["label_event"].astype(str) == "stop_first").astype("Int64")
        frames.append(symbol_dataset)

    dataset = pd.concat(frames, ignore_index=True).sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce").astype("Int64")
    dataset["downside_label"] = pd.to_numeric(dataset["downside_label"], errors="coerce").astype("Int64")
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
