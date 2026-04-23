from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt

import pandas as pd

from analysis.strategy_config import (
    ASSET_UNIVERSE,
    DATA_DIR,
    DEFAULT_SELECTED_SYMBOL,
    DOWNLOADS_DIR,
    MARKET_HISTORY_DIR,
)

INITIAL_CAPITAL = 100_000.0
TRADING_DAYS_PER_YEAR = 252
MIN_TRADES_FOR_SELECTION = 15


@dataclass(frozen=True)
class BaselineParams:
    channel_lookback: int = 20
    atr_window: int = 14
    stop_atr: float = 2.25
    trailing_stop_atr: float = 3.0
    max_hold_days: int = 15
    risk_per_trade: float = 0.01
    max_capital_fraction: float = 0.95
    transaction_cost_bps: float = 10.0
    allow_shorts: bool = False
    min_breakout_strength: float = 0.18
    trend_filter_window: int = 50
    breakout_failure_wait_days: int = 4
    min_bars_before_trailing_exit: int = 2


BASELINE_PARAMS = BaselineParams()


def _load_selected_asset() -> dict:
    path = DATA_DIR / "selected_asset.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"selected_symbol": DEFAULT_SELECTED_SYMBOL, "selection_status": "placeholder"}


def _load_asset_screen() -> pd.DataFrame:
    path = DATA_DIR / "asset_screening.csv"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "selection_status"])
    return pd.read_csv(path)


def _load_history(symbol: str) -> pd.DataFrame:
    path = MARKET_HISTORY_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing market history file for {symbol}: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").reset_index(drop=True)
    return df


def _average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def _build_features(df: pd.DataFrame, params: BaselineParams) -> pd.DataFrame:
    featured = df.copy()
    featured["upper_channel"] = featured["high"].shift(1).rolling(
        params.channel_lookback, min_periods=params.channel_lookback
    ).max()
    featured["lower_channel"] = featured["low"].shift(1).rolling(
        params.channel_lookback, min_periods=params.channel_lookback
    ).min()
    featured["atr"] = _average_true_range(featured, params.atr_window)
    featured["trend_filter"] = featured["close"].rolling(
        params.trend_filter_window, min_periods=params.trend_filter_window
    ).mean()
    featured["long_signal"] = featured["close"] > featured["upper_channel"]
    featured["short_signal"] = featured["close"] < featured["lower_channel"]
    featured["channel_width"] = featured["upper_channel"] - featured["lower_channel"]
    featured["breakout_strength"] = 0.0

    long_mask = featured["long_signal"] & featured["upper_channel"].notna()
    short_mask = featured["short_signal"] & featured["lower_channel"].notna()
    featured.loc[long_mask, "breakout_strength"] = (
        (featured.loc[long_mask, "close"] - featured.loc[long_mask, "upper_channel"]) / featured.loc[long_mask, "atr"]
    )
    featured.loc[short_mask, "breakout_strength"] = (
        (featured.loc[short_mask, "lower_channel"] - featured.loc[short_mask, "close"]) / featured.loc[short_mask, "atr"]
    )
    featured["long_signal"] = (
        featured["long_signal"]
        & (featured["breakout_strength"] >= params.min_breakout_strength)
        & (featured["close"] > featured["trend_filter"])
    )
    featured["short_signal"] = (
        featured["short_signal"]
        & params.allow_shorts
        & (featured["breakout_strength"] >= params.min_breakout_strength)
        & (featured["close"] < featured["trend_filter"])
    )
    return featured


def _per_side_cost(notional: float, params: BaselineParams) -> float:
    return abs(notional) * params.transaction_cost_bps / 10_000.0


def _position_size(entry_price: float, atr_value: float, equity: float, params: BaselineParams) -> int:
    if atr_value <= 0 or entry_price <= 0 or equity <= 0:
        return 0

    risk_budget = equity * params.risk_per_trade
    stop_distance = params.stop_atr * atr_value
    risk_size = int(risk_budget // stop_distance) if stop_distance > 0 else 0
    notional_cap = equity * params.max_capital_fraction
    max_size = int(notional_cap // entry_price)
    return max(0, min(risk_size, max_size))


def _simulate_trades(featured: pd.DataFrame, symbol: str, params: BaselineParams) -> pd.DataFrame:
    return _simulate_trades_with_filters(featured, symbol, params)


def _simulate_trades_with_filters(
    featured: pd.DataFrame,
    symbol: str,
    params: BaselineParams,
    allowed_signal_dates: set[pd.Timestamp] | None = None,
    signal_start_date: pd.Timestamp | None = None,
    signal_end_date: pd.Timestamp | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    equity = initial_capital
    i = params.channel_lookback

    while i < len(featured) - 1:
        signal_row = featured.iloc[i]
        signal_date = pd.Timestamp(signal_row["date"]).normalize()
        if signal_start_date is not None and signal_date < pd.Timestamp(signal_start_date).normalize():
            i += 1
            continue
        if signal_end_date is not None and signal_date > pd.Timestamp(signal_end_date).normalize():
            i += 1
            continue
        if allowed_signal_dates is not None and signal_date not in allowed_signal_dates:
            i += 1
            continue

        direction = 0
        if bool(signal_row["long_signal"]):
            direction = 1
        elif params.allow_shorts and bool(signal_row["short_signal"]):
            direction = -1

        if direction == 0 or pd.isna(signal_row["atr"]) or signal_row["atr"] <= 0:
            i += 1
            continue

        entry_idx = i + 1
        entry_row = featured.iloc[entry_idx]
        entry_price = float(entry_row["open"])
        atr_value = float(signal_row["atr"])
        qty = _position_size(entry_price, atr_value, equity, params)
        if qty <= 0:
            i += 1
            continue

        if direction == 1:
            stop_price = entry_price - params.stop_atr * atr_value
            trailing_stop = stop_price
        else:
            stop_price = entry_price + params.stop_atr * atr_value
            trailing_stop = stop_price

        exit_price = float(featured.iloc[-1]["close"])
        exit_date = pd.Timestamp(featured.iloc[-1]["date"])
        exit_reason = "end_of_sample"

        hold_limit = min(len(featured) - 1, entry_idx + params.max_hold_days - 1)

        for j in range(entry_idx, hold_limit + 1):
            current = featured.iloc[j]
            current_atr = float(current["atr"]) if pd.notna(current["atr"]) else atr_value
            bars_open = j - entry_idx + 1

            if direction == 1:
                candidate_trail = float(current["close"]) - params.trailing_stop_atr * current_atr
                trailing_stop = max(trailing_stop, candidate_trail)
                if float(current["low"]) <= stop_price:
                    exit_price = stop_price
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "stop_loss"
                    break
                if (
                    bars_open >= params.min_bars_before_trailing_exit
                    and float(current["low"]) <= trailing_stop
                    and trailing_stop > entry_price
                ):
                    exit_price = trailing_stop
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "success"
                    break
                if (
                    bars_open >= params.breakout_failure_wait_days
                    and float(current["close"]) < float(current["upper_channel"])
                ):
                    exit_price = float(current["close"])
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "breakout_failure"
                    break
            else:
                candidate_trail = float(current["close"]) + params.trailing_stop_atr * current_atr
                trailing_stop = min(trailing_stop, candidate_trail)
                if float(current["high"]) >= stop_price:
                    exit_price = stop_price
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "stop_loss"
                    break
                if (
                    bars_open >= params.min_bars_before_trailing_exit
                    and float(current["high"]) >= trailing_stop
                    and trailing_stop < entry_price
                ):
                    exit_price = trailing_stop
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "success"
                    break
                if (
                    bars_open >= params.breakout_failure_wait_days
                    and float(current["close"]) > float(current["lower_channel"])
                ):
                    exit_price = float(current["close"])
                    exit_date = pd.Timestamp(current["date"])
                    exit_reason = "breakout_failure"
                    break

            if j == hold_limit:
                exit_price = float(current["close"])
                exit_date = pd.Timestamp(current["date"])
                exit_reason = "timeout"
                break

        gross_pnl = (exit_price - entry_price) * qty * direction
        total_cost = _per_side_cost(entry_price * qty, params) + _per_side_cost(exit_price * qty, params)
        net_pnl = gross_pnl - total_cost
        trade_return = net_pnl / (entry_price * qty) if entry_price * qty else 0.0
        holding_periods = int((exit_date - pd.Timestamp(entry_row["date"])).days) + 1
        outcome = "Successful" if exit_reason == "success" or net_pnl > 0 else {
            "stop_loss": "Stop-loss triggered",
            "timeout": "Timed out",
            "breakout_failure": "Breakout failure",
            "end_of_sample": "End of sample",
        }.get(exit_reason, "Timed out")

        trades.append(
            {
                "symbol": symbol,
                "entry_timestamp": pd.Timestamp(entry_row["date"]),
                "exit_timestamp": exit_date,
                "direction": "LONG" if direction == 1 else "SHORT",
                "side": direction,
                "qty": int(qty),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "trailing_stop_exit": trailing_stop,
                "atr_at_entry": atr_value,
                "entry_channel_high": float(signal_row["upper_channel"]),
                "entry_channel_low": float(signal_row["lower_channel"]),
                "breakout_strength": float(signal_row["breakout_strength"]),
                "holding_periods": holding_periods,
                "exit_reason": exit_reason,
                "trade_outcome": outcome,
                "gross_pnl": gross_pnl,
                "transaction_cost": total_cost,
                "pnl": net_pnl,
                "trade_return": trade_return,
            }
        )
        equity += net_pnl
        i = hold_limit + 1 if exit_reason == "timeout" else j + 1

    return pd.DataFrame(trades)


def _build_ledger(df: pd.DataFrame, trades: pd.DataFrame, params: BaselineParams) -> pd.DataFrame:
    return _build_ledger_with_initial_capital(df, trades, params, INITIAL_CAPITAL)


def _build_ledger_with_initial_capital(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    params: BaselineParams,
    initial_capital: float,
) -> pd.DataFrame:
    ledger = df.loc[:, ["date", "close"]].copy()
    ledger["cash_flow"] = 0.0
    ledger["position_delta"] = 0

    for trade in trades.to_dict("records"):
        entry_mask = ledger["date"] == pd.Timestamp(trade["entry_timestamp"])
        exit_mask = ledger["date"] == pd.Timestamp(trade["exit_timestamp"])
        direction = int(trade["side"])
        qty = int(trade["qty"])
        entry_notional = float(trade["entry_price"]) * qty
        exit_notional = float(trade["exit_price"]) * qty
        entry_cost = _per_side_cost(entry_notional, params)
        exit_cost = _per_side_cost(exit_notional, params)

        if direction == 1:
            ledger.loc[entry_mask, "cash_flow"] -= entry_notional + entry_cost
            ledger.loc[exit_mask, "cash_flow"] += exit_notional - exit_cost
            ledger.loc[entry_mask, "position_delta"] += qty
            ledger.loc[exit_mask, "position_delta"] -= qty
        else:
            ledger.loc[entry_mask, "cash_flow"] += entry_notional - entry_cost
            ledger.loc[exit_mask, "cash_flow"] -= exit_notional + exit_cost
            ledger.loc[entry_mask, "position_delta"] -= qty
            ledger.loc[exit_mask, "position_delta"] += qty

    ledger["cash"] = initial_capital + ledger["cash_flow"].cumsum()
    ledger["position_shares"] = ledger["position_delta"].cumsum()
    ledger["market_value"] = ledger["position_shares"] * ledger["close"]
    ledger["equity"] = ledger["cash"] + ledger["market_value"]
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["running_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["running_peak"] - 1.0
    ledger["open_positions"] = (ledger["position_shares"] != 0).astype(int)
    return ledger


def _compute_metrics(trades: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    initial_equity = float(ledger["equity"].iloc[0]) if not ledger.empty else INITIAL_CAPITAL
    total_return = ledger["equity"].iloc[-1] / initial_equity - 1.0 if not ledger.empty else 0.0
    benchmark_return = ledger["close"].iloc[-1] / ledger["close"].iloc[0] - 1.0 if len(ledger) > 1 else 0.0
    daily_returns = ledger["daily_return"] if "daily_return" in ledger else pd.Series(dtype=float)
    daily_std = daily_returns.std(ddof=0) if not daily_returns.empty else 0.0
    sharpe = (daily_returns.mean() / daily_std * sqrt(TRADING_DAYS_PER_YEAR)) if daily_std and daily_std > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=0) if not downside.empty else 0.0
    sortino = (daily_returns.mean() / downside_std * sqrt(TRADING_DAYS_PER_YEAR)) if downside_std and downside_std > 0 else 0.0

    trade_count = int(len(trades))
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    win_rate = len(wins) / trade_count if trade_count else 0.0
    avg_trade_return = trades["trade_return"].mean() if trade_count else 0.0
    expected_pnl = trades["pnl"].mean() if trade_count else 0.0
    avg_holding = trades["holding_periods"].mean() if trade_count else 0.0
    profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if not losses.empty and losses["pnl"].sum() != 0 else 0.0
    max_drawdown = ledger["drawdown"].min() if not ledger.empty else 0.0

    metrics = {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Total Return": total_return,
        "Benchmark Return": benchmark_return,
        "Expected Return Per Trade": avg_trade_return,
        "Expected PnL Per Trade": expected_pnl,
        "Average Trade Lifetime": avg_holding,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_drawdown,
        "Trade Count": trade_count,
    }

    metrics_display = pd.DataFrame(
        [
            {"Metric": "Sharpe Ratio", "Value": f"{sharpe:.2f}"},
            {"Metric": "Expected Return Per Trade", "Value": f"{avg_trade_return:.2%}"},
            {"Metric": "Average Trade Lifetime", "Value": f"{avg_holding:.1f} days"},
            {"Metric": "Max Drawdown", "Value": f"{max_drawdown:.2%}"},
            {"Metric": "Win Rate", "Value": f"{win_rate:.2%}"},
            {"Metric": "Trade Count", "Value": f"{trade_count}"},
            {"Metric": "Total Return", "Value": f"{total_return:.2%}"},
        ]
    )
    return metrics_display, metrics


def _evaluate_symbol(symbol: str, params: BaselineParams) -> dict[str, object]:
    history = _load_history(symbol)
    featured = _build_features(history, params)
    trades = _simulate_trades(featured, symbol, params)
    ledger = _build_ledger(history, trades, params)
    _, metrics = _compute_metrics(trades, ledger)
    return {
        "symbol": symbol,
        "trades": int(metrics["Trade Count"]),
        "sharpe": float(metrics["Sharpe Ratio"]),
        "total_return": float(metrics["Total Return"]),
        "win_rate": float(metrics["Win Rate"]),
        "max_drawdown": float(metrics["Max Drawdown"]),
        "selection_status": "eligible" if int(metrics["Trade Count"]) >= MIN_TRADES_FOR_SELECTION else "too_few_trades",
    }


def _screen_baseline_universe(params: BaselineParams) -> pd.DataFrame:
    rows = [_evaluate_symbol(symbol, params) for symbol in ASSET_UNIVERSE]
    return pd.DataFrame(rows).sort_values(
        by=["selection_status", "sharpe", "total_return", "win_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _select_symbol_for_baseline(params: BaselineParams) -> dict[str, object]:
    screen = _screen_baseline_universe(params)
    eligible = screen.loc[screen["selection_status"] == "eligible"].copy()
    if eligible.empty:
        eligible = screen.copy()
    best = eligible.iloc[0]
    return {
        "selected_symbol": str(best["symbol"]),
        "selection_metric": "baseline breakout Sharpe ratio",
        "selection_score": float(best["sharpe"]),
        "selection_note": (
            f"{best['symbol']} is the current baseline showcase asset because it produced the best "
            "risk-adjusted result among the screened ETFs under the tuned breakout rules."
        ),
        "strategy_params": asdict(params),
        "screen_table": screen,
    }


def _metric_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Sharpe Ratio", "Explanation": "Risk-adjusted return using daily strategy equity changes."},
            {"Metric": "Expected Return Per Trade", "Explanation": "Average net return after costs across completed trades."},
            {"Metric": "Average Trade Lifetime", "Explanation": "Average number of calendar days each trade stayed open."},
            {"Metric": "Max Drawdown", "Explanation": "Largest peak-to-trough decline in strategy equity."},
            {"Metric": "Win Rate", "Explanation": "Share of completed trades that finished with positive net PnL."},
            {"Metric": "Trade Count", "Explanation": "Number of completed breakout trades in the backtest."},
            {"Metric": "Total Return", "Explanation": "Cumulative strategy return from the starting capital of $100,000."},
        ]
    )


def _parameter_table(selected_symbol: str, params: BaselineParams) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Parameter": "Selected Symbol", "Value": selected_symbol},
            {"Parameter": "Universe", "Value": ", ".join(ASSET_UNIVERSE)},
            {"Parameter": "Entry Rule", "Value": f"Close breaks above prior {params.channel_lookback}-day channel"},
            {"Parameter": "ATR Window", "Value": str(params.atr_window)},
            {"Parameter": "Initial Stop", "Value": f"{params.stop_atr:.1f} x ATR"},
            {"Parameter": "Trailing Stop", "Value": f"{params.trailing_stop_atr:.1f} x ATR"},
            {"Parameter": "Timeout", "Value": f"{params.max_hold_days} trading days"},
            {"Parameter": "Direction", "Value": "Long only" if not params.allow_shorts else "Long and short"},
            {"Parameter": "Breakout Strength Filter", "Value": f"At least {params.min_breakout_strength:.2f} ATR past the channel"},
            {"Parameter": "Trend Filter", "Value": f"Close must be above the {params.trend_filter_window}-day moving average"},
            {"Parameter": "Breakout Failure Delay", "Value": f"Do not classify breakout failure until day {params.breakout_failure_wait_days}"},
            {"Parameter": "Transaction Cost", "Value": f"{params.transaction_cost_bps:.0f} bps round-turn split across entry and exit"},
            {"Parameter": "Position Sizing", "Value": "1% equity risk budget, capped at 95% notional deployment"},
        ]
    )


def _outcome_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {"Outcome": "Successful", "Trades": 0},
                {"Outcome": "Stop-loss triggered", "Trades": 0},
                {"Outcome": "Timed out", "Trades": 0},
                {"Outcome": "Breakout failure", "Trades": 0},
            ]
        )
    return (
        trades.groupby("trade_outcome")
        .size()
        .reset_index(name="Trades")
        .rename(columns={"trade_outcome": "Outcome"})
        .sort_values("Trades", ascending=False)
        .reset_index(drop=True)
    )


def _save_outputs(trades: pd.DataFrame, ledger: pd.DataFrame) -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    blotter_out = trades.copy()
    ledger_out = ledger.copy()
    for col in ["entry_timestamp", "exit_timestamp"]:
        if col in blotter_out.columns:
            blotter_out[col] = pd.to_datetime(blotter_out[col]).dt.strftime("%Y-%m-%d")
    if "date" in ledger_out.columns:
        ledger_out["date"] = pd.to_datetime(ledger_out["date"]).dt.strftime("%Y-%m-%d")

    blotter_out.to_csv(DOWNLOADS_DIR / "trade_blotter.csv", index=False)
    ledger_out.to_csv(DOWNLOADS_DIR / "ledger.csv", index=False)


def run_analysis(save_outputs: bool = True, params: BaselineParams = BASELINE_PARAMS) -> dict[str, object]:
    selected = _load_selected_asset()
    strategy_selection = _select_symbol_for_baseline(params)
    selected_symbol = strategy_selection.get("selected_symbol") or selected.get("selected_symbol", DEFAULT_SELECTED_SYMBOL)
    history = _load_history(selected_symbol)
    featured = _build_features(history, params)
    trades = _simulate_trades(featured, selected_symbol, params)
    ledger = _build_ledger(history, trades, params)
    metrics_display, metrics = _compute_metrics(trades, ledger)
    asset_screen_display = strategy_selection["screen_table"]

    if save_outputs:
        _save_outputs(trades, ledger)
        selected_path = DATA_DIR / "selected_asset.json"
        selected_path.write_text(
            json.dumps(
                {
                    "selected_symbol": selected_symbol,
                    "selection_metric": strategy_selection["selection_metric"],
                    "selection_score": strategy_selection["selection_score"],
                    "selection_note": strategy_selection["selection_note"],
                    "strategy_params": strategy_selection["strategy_params"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    overview = pd.DataFrame(
        [
            {"Field": "Project", "Value": "FINTECH 533 Final Project"},
            {"Field": "Strategy", "Value": "Tuned baseline volatility and channel breakout"},
            {"Field": "Data source", "Value": "shinybroker / Interactive Brokers historical daily bars"},
            {"Field": "Selected symbol", "Value": selected_symbol},
            {"Field": "Sample window", "Value": f"{history['date'].min().strftime('%Y-%m-%d')} to {history['date'].max().strftime('%Y-%m-%d')}"},
            {"Field": "Status", "Value": "Baseline strategy tuned and implemented"},
        ]
    )

    monitoring_display = pd.DataFrame(
        [
            {"Metric": "Rolling expectancy", "Threshold": "< 0 over the latest 50 trades"},
            {"Metric": "Rolling hit rate", "Threshold": "Below the backtest range for 3 consecutive months"},
            {"Metric": "Drawdown", "Threshold": "Worse than 1.5x historical max drawdown"},
        ]
    )

    export_paths = {
        "blotter_csv_href": "downloads/trade_blotter.csv",
        "ledger_csv_href": "downloads/ledger.csv",
    }

    return {
        "overview": overview,
        "metrics_display": metrics_display,
        "metrics_raw": metrics,
        "metric_explanations": _metric_explanations(),
        "parameter_table": _parameter_table(selected_symbol, params),
        "asset_screen_display": asset_screen_display,
        "outcome_summary": _outcome_summary(trades),
        "blotter": trades,
        "blotter_display": trades.loc[
            :,
            [
                "entry_timestamp",
                "exit_timestamp",
                "direction",
                "qty",
                "entry_price",
                "exit_price",
                "exit_reason",
                "trade_return",
                "pnl",
            ],
        ],
        "ledger": ledger,
        "ledger_display": ledger.loc[:, ["date", "cash", "market_value", "equity", "drawdown"]].tail(20),
        "equity_curve": ledger.loc[:, ["date", "equity", "drawdown", "close", "daily_return"]],
        "monitoring_display": monitoring_display,
        "export_paths": export_paths,
        "plain_english_rules": [
            f"Enter long when the close breaks above the previous {params.channel_lookback}-day high.",
            "Only take trades that are aligned with the medium-term trend and clear the breakout-strength filter.",
            f"Place an initial stop {params.stop_atr:.1f} ATR away from entry and trail winners by {params.trailing_stop_atr:.1f} ATR.",
            f"Exit failed breakouts when price closes back inside the channel or after {params.max_hold_days} trading days.",
        ],
    }
