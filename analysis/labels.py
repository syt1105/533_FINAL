from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.breakout_strategy import BASELINE_PARAMS, BaselineParams, _build_features, _load_history
from analysis.strategy_config import ASSET_UNIVERSE, DATA_DIR, DEFAULT_SELECTED_SYMBOL, DOWNLOADS_DIR


@dataclass(frozen=True)
class LabelParams:
    horizon_days: int = 15
    target_atr_multiple: float = 1.5
    stop_atr_multiple: float = 1.0


DEFAULT_LABEL_PARAMS = LabelParams()


def _load_selected_symbol() -> str:
    path = DATA_DIR / "selected_asset.json"
    if not path.exists():
        return DEFAULT_SELECTED_SYMBOL
    selected = pd.read_json(path, typ="series")
    return str(selected.get("selected_symbol", DEFAULT_SELECTED_SYMBOL))


def _future_label_for_long(
    featured: pd.DataFrame,
    signal_idx: int,
    label_params: LabelParams,
) -> dict[str, object] | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(featured):
        return None

    signal_row = featured.iloc[signal_idx]
    entry_row = featured.iloc[entry_idx]
    atr_value = float(signal_row["atr"]) if pd.notna(signal_row["atr"]) else float("nan")
    if pd.isna(atr_value) or atr_value <= 0:
        return None

    entry_price = float(entry_row["open"])
    target_price = entry_price + label_params.target_atr_multiple * atr_value
    stop_price = entry_price - label_params.stop_atr_multiple * atr_value
    horizon_end_idx = min(len(featured) - 1, entry_idx + label_params.horizon_days - 1)

    first_event = "timeout"
    first_event_date = pd.Timestamp(featured.iloc[horizon_end_idx]["date"])
    first_event_price = float(featured.iloc[horizon_end_idx]["close"])

    for j in range(entry_idx, horizon_end_idx + 1):
        current = featured.iloc[j]
        hit_target = float(current["high"]) >= target_price
        hit_stop = float(current["low"]) <= stop_price

        if hit_target and hit_stop:
            # Conservative tie-break: assume the stop is hit first when the bar contains both.
            first_event = "stop_first"
            first_event_date = pd.Timestamp(current["date"])
            first_event_price = stop_price
            break
        if hit_stop:
            first_event = "stop_first"
            first_event_date = pd.Timestamp(current["date"])
            first_event_price = stop_price
            break
        if hit_target:
            first_event = "target_first"
            first_event_date = pd.Timestamp(current["date"])
            first_event_price = target_price
            break

    label = 1 if first_event == "target_first" else 0
    return {
        "symbol": str(entry_row["symbol"]),
        "signal_date": pd.Timestamp(signal_row["date"]),
        "entry_date": pd.Timestamp(entry_row["date"]),
        "entry_price": entry_price,
        "direction": "LONG",
        "atr_at_signal": atr_value,
        "upper_channel": float(signal_row["upper_channel"]),
        "lower_channel": float(signal_row["lower_channel"]),
        "channel_width": float(signal_row["channel_width"]),
        "trend_filter": float(signal_row["trend_filter"]) if pd.notna(signal_row["trend_filter"]) else pd.NA,
        "breakout_strength": float(signal_row["breakout_strength"]),
        "volume": float(signal_row["volume"]),
        "target_price": target_price,
        "stop_price": stop_price,
        "label_horizon_days": label_params.horizon_days,
        "label_target_atr": label_params.target_atr_multiple,
        "label_stop_atr": label_params.stop_atr_multiple,
        "label_event": first_event,
        "label_event_date": first_event_date,
        "label_event_price": first_event_price,
        "label": label,
        "days_to_event": int((first_event_date - pd.Timestamp(entry_row["date"])).days) + 1,
    }


def build_labeled_breakout_dataset(
    symbol: str,
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    history = _load_history(symbol)
    featured = _build_features(history, strategy_params)

    events: list[dict[str, object]] = []
    for i in range(len(featured) - 1):
        if bool(featured.iloc[i]["long_signal"]):
            labeled_event = _future_label_for_long(featured, i, label_params)
            if labeled_event is not None:
                events.append(labeled_event)

    return pd.DataFrame(events)


def build_selected_asset_labeled_dataset(
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    symbol = _load_selected_symbol()
    return build_labeled_breakout_dataset(symbol, strategy_params=strategy_params, label_params=label_params)


def build_universe_labeled_dataset(
    symbols: list[str] | None = None,
    strategy_params: BaselineParams = BASELINE_PARAMS,
    label_params: LabelParams = DEFAULT_LABEL_PARAMS,
) -> pd.DataFrame:
    symbols = symbols or ASSET_UNIVERSE
    frames = [
        build_labeled_breakout_dataset(symbol, strategy_params=strategy_params, label_params=label_params)
        for symbol in symbols
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def save_labeled_breakout_dataset(
    df: pd.DataFrame,
    filename: str = "labeled_breakouts.csv",
) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    for col in ["signal_date", "entry_date", "label_event_date"]:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col]).dt.strftime("%Y-%m-%d")
    output.to_csv(DOWNLOADS_DIR / filename, index=False)
