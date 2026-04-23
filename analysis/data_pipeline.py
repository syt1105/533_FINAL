from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.strategy_config import (
    BAR_SIZE,
    HISTORY_DURATION,
    IBKR_CLIENT_ID,
    IBKR_HOST,
    IBKR_PORT,
    LOOKBACK_RETURN_DAYS,
    MIN_HISTORY_ROWS,
    USE_RTH,
    WHAT_TO_SHOW,
)


@dataclass(frozen=True)
class FetchConfig:
    host: str = IBKR_HOST
    port: int = IBKR_PORT
    client_id: int = IBKR_CLIENT_ID
    duration: str = HISTORY_DURATION
    bar_size: str = BAR_SIZE
    what_to_show: str = WHAT_TO_SHOW
    use_rth: bool = USE_RTH
    timeout: int = 15


def _load_shinybroker() -> tuple[Any, Any]:
    try:
        from shinybroker import Contract, fetch_historical_data
    except ImportError as exc:
        raise RuntimeError(
            "shinybroker is not installed in the active Python environment. "
            "Install project requirements before running the data pipeline."
        ) from exc

    return Contract, fetch_historical_data


def _normalize_history(raw_data: Any, symbol: str) -> pd.DataFrame:
    if isinstance(raw_data, dict) and "hst_dta" in raw_data:
        df = pd.DataFrame(raw_data["hst_dta"]).copy()
    else:
        df = pd.DataFrame(raw_data).copy()

    if df.empty:
        raise RuntimeError(f"No historical data returned for {symbol}.")

    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    elif "date" not in df.columns:
        raise RuntimeError(f"Historical data for {symbol} is missing a timestamp/date column.")

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Historical data for {symbol} is missing required columns: {missing}.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[required_cols]
        .dropna()
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["symbol"] = symbol
    return df


def fetch_symbol_history(symbol: str, config: FetchConfig | None = None) -> pd.DataFrame:
    config = config or FetchConfig()
    Contract, fetch_historical_data = _load_shinybroker()

    contract = Contract(
        {
            "symbol": symbol,
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    )

    raw_data = fetch_historical_data(
        contract=contract,
        durationStr=config.duration,
        barSizeSetting=config.bar_size,
        whatToShow=config.what_to_show,
        useRTH=config.use_rth,
        host=config.host,
        port=config.port,
        client_id=config.client_id,
        timeout=config.timeout,
    )

    return _normalize_history(raw_data, symbol)


def fetch_universe_history(symbols: list[str], config: FetchConfig | None = None) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        histories[symbol] = fetch_symbol_history(symbol, config=config)
    return histories


def screen_asset_history(history: pd.DataFrame) -> dict[str, object]:
    if len(history) < MIN_HISTORY_ROWS:
        raise RuntimeError(
            f"{history['symbol'].iloc[0]} has only {len(history)} rows, below the minimum {MIN_HISTORY_ROWS}."
        )

    daily_returns = history["close"].pct_change().dropna()
    recent_window = history.tail(min(len(history), LOOKBACK_RETURN_DAYS))
    recent_return = 0.0
    if len(recent_window) >= 2:
        recent_return = recent_window["close"].iloc[-1] / recent_window["close"].iloc[0] - 1.0

    avg_dollar_volume = float((history["close"] * history["volume"]).tail(LOOKBACK_RETURN_DAYS).mean())
    realized_vol = float(daily_returns.std(ddof=0) * (252 ** 0.5)) if not daily_returns.empty else 0.0
    score = recent_return / realized_vol if realized_vol > 0 else 0.0

    return {
        "symbol": str(history["symbol"].iloc[0]),
        "rows": int(len(history)),
        "start_date": history["date"].min().strftime("%Y-%m-%d"),
        "end_date": history["date"].max().strftime("%Y-%m-%d"),
        "recent_return_63d": recent_return,
        "annualized_volatility": realized_vol,
        "avg_dollar_volume_63d": avg_dollar_volume,
        "selection_score": score,
        "selection_status": "eligible",
    }


def build_asset_screen(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, history in histories.items():
        try:
            rows.append(screen_asset_history(history))
        except RuntimeError as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "rows": int(len(history)),
                    "start_date": "",
                    "end_date": "",
                    "recent_return_63d": pd.NA,
                    "annualized_volatility": pd.NA,
                    "avg_dollar_volume_63d": pd.NA,
                    "selection_score": pd.NA,
                    "selection_status": str(exc),
                }
            )

    screen = pd.DataFrame(rows)
    if not screen.empty and "selection_score" in screen.columns:
        screen = screen.sort_values(
            by=["selection_score", "avg_dollar_volume_63d"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
    return screen


def select_asset_from_screen(screen: pd.DataFrame) -> dict[str, object]:
    eligible = screen.loc[screen["selection_status"] == "eligible"].copy()
    if eligible.empty:
        raise RuntimeError("No eligible assets were available after screening.")

    best = eligible.iloc[0]
    return {
        "selected_symbol": str(best["symbol"]),
        "selection_metric": "63-day return divided by annualized volatility",
        "selection_score": float(best["selection_score"]),
        "start_date": str(best["start_date"]),
        "end_date": str(best["end_date"]),
        "selection_note": (
            f"{best['symbol']} ranked highest in the initial shinybroker screen using a simple "
            "risk-adjusted momentum score. This is a temporary selection rule that will be refined "
            "once the breakout backtest and walk-forward filter are in place."
        ),
    }
