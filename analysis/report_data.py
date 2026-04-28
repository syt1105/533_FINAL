from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = ROOT / "downloads"
DATA_DIR = ROOT / "data"


def _read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DOWNLOADS_DIR / name, **kwargs)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(metric: str, value: float) -> str:
    if metric in {
        "Total Return",
        "Benchmark Return",
        "Expected Return Per Trade",
        "Win Rate",
        "Max Drawdown",
    }:
        return f"{value:.2%}"
    if metric == "Average Trade Lifetime":
        return f"{value:.1f} days"
    if metric == "Trade Count":
        return f"{int(value)}"
    if metric == "Expected PnL Per Trade":
        return f"${value:,.0f}"
    return f"{value:.2f}"


def compute_metrics(trades: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    daily_returns = ledger["daily_return"].fillna(0.0)
    sharpe = 0.0
    if daily_returns.std(ddof=0) != 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=0) * (252**0.5))

    downside = daily_returns.loc[daily_returns < 0]
    sortino = 0.0
    if not downside.empty and downside.std(ddof=0) != 0:
        sortino = float(daily_returns.mean() / downside.std(ddof=0) * (252**0.5))

    total_return = float(ledger["equity"].iloc[-1] / ledger["equity"].iloc[0] - 1)
    benchmark_return = float(ledger["close"].iloc[-1] / ledger["close"].iloc[0] - 1)
    trade_count = int(len(trades))
    avg_trade_return = float(trades["trade_return"].mean()) if trade_count else 0.0
    expected_pnl = float(trades["pnl"].mean()) if trade_count else 0.0
    avg_holding = float(trades["holding_periods"].mean()) if trade_count else 0.0
    win_rate = float((trades["pnl"] > 0).mean()) if trade_count else 0.0
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if trade_count else 0.0
    gross_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].sum()) if trade_count else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else 0.0
    max_drawdown = float(ledger["drawdown"].min()) if "drawdown" in ledger else 0.0

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
    display_order = [
        "Sharpe Ratio",
        "Expected Return Per Trade",
        "Average Trade Lifetime",
        "Max Drawdown",
        "Win Rate",
        "Trade Count",
        "Total Return",
    ]
    metrics_display = pd.DataFrame(
        [{"Metric": metric, "Value": _format_metric(metric, metrics[metric])} for metric in display_order]
    )
    return metrics_display, metrics


def outcome_rate_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["Outcome", "Trades", "Rate"])
    summary = (
        trades.groupby("trade_outcome")
        .size()
        .reset_index(name="Trades")
        .rename(columns={"trade_outcome": "Outcome"})
        .sort_values("Trades", ascending=False)
        .reset_index(drop=True)
    )
    total = int(summary["Trades"].sum())
    summary["Rate"] = summary["Trades"].apply(lambda value: f"{value / total:.2%}" if total else "0.00%")
    return summary


def trade_quality_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        rows = [
            ("Average winner", 0.0, "percent"),
            ("Average loser", 0.0, "percent"),
            ("Payoff ratio", 0.0, "number"),
            ("Largest winner", 0.0, "percent"),
            ("Largest loser", 0.0, "percent"),
            ("Top 3 trade PnL share", 0.0, "percent"),
        ]
    else:
        winners = trades.loc[trades["trade_return"] > 0, "trade_return"]
        losers = trades.loc[trades["trade_return"] < 0, "trade_return"]
        avg_winner = float(winners.mean()) if not winners.empty else 0.0
        avg_loser = float(losers.mean()) if not losers.empty else 0.0
        payoff_ratio = avg_winner / abs(avg_loser) if avg_loser < 0 else 0.0
        total_pnl = float(trades["pnl"].sum())
        top3_pnl = float(trades.sort_values("pnl", ascending=False).head(3)["pnl"].sum())
        top3_share = top3_pnl / total_pnl if total_pnl > 0 else 0.0
        rows = [
            ("Average winner", avg_winner, "percent"),
            ("Average loser", avg_loser, "percent"),
            ("Payoff ratio", payoff_ratio, "number"),
            ("Largest winner", float(trades["trade_return"].max()), "percent"),
            ("Largest loser", float(trades["trade_return"].min()), "percent"),
            ("Top 3 trade PnL share", top3_share, "percent"),
        ]

    return pd.DataFrame(
        [
            {"Metric": metric, "Value": f"{value:.2%}" if value_type == "percent" else f"{value:.2f}"}
            for metric, value, value_type in rows
        ]
    )


def parameter_table(selected: dict) -> pd.DataFrame:
    params = selected["strategy_params"]
    universe = ", ".join(["SPY", "QQQ", "IWM", "XLE", "GLD", "TLT"])
    return pd.DataFrame(
        [
            {"Parameter": "Selected Symbol", "Value": selected["selected_symbol"]},
            {"Parameter": "Universe", "Value": universe},
            {"Parameter": "Entry Rule", "Value": f"Close breaks above prior {params['channel_lookback']}-day channel"},
            {"Parameter": "ATR Window", "Value": str(params["atr_window"])},
            {"Parameter": "Initial Stop", "Value": f"{params['stop_atr']:.1f} x ATR"},
            {"Parameter": "Trailing Stop", "Value": f"{params['trailing_stop_atr']:.1f} x ATR"},
            {"Parameter": "Timeout", "Value": f"{params['max_hold_days']} trading days"},
            {"Parameter": "Direction", "Value": "Long only" if not params["allow_shorts"] else "Long and short"},
            {
                "Parameter": "Breakout Strength Filter",
                "Value": f"At least {params['min_breakout_strength']:.2f} ATR past the channel",
            },
            {
                "Parameter": "Trend Filter",
                "Value": f"Close must be above the {params['trend_filter_window']}-day moving average",
            },
            {
                "Parameter": "Breakout Failure Delay",
                "Value": f"Do not classify breakout failure until day {params['breakout_failure_wait_days']}",
            },
            {
                "Parameter": "Transaction Cost",
                "Value": f"{params['transaction_cost_bps']:.0f} bps round-turn split across entry and exit",
            },
            {"Parameter": "Position Sizing", "Value": "1% equity risk budget, capped at 95% notional deployment"},
        ]
    )


def parameter_comparison(selected: dict, metrics: dict[str, float]) -> pd.DataFrame:
    current = selected["strategy_params"]
    sweep = pd.read_csv(ROOT / "parameter_sweep_top100.csv").iloc[0]
    return pd.DataFrame(
        [
            {
                "parameter_set": "final_current_params",
                "selected_symbol": selected["selected_symbol"],
                "channel_lookback": current["channel_lookback"],
                "stop_atr": current["stop_atr"],
                "trailing_stop_atr": current["trailing_stop_atr"],
                "trade_count": int(metrics["Trade Count"]),
                "sharpe_ratio": float(metrics["Sharpe Ratio"]),
                "total_return": float(metrics["Total Return"]),
                "max_drawdown": float(metrics["Max Drawdown"]),
            },
            {
                "parameter_set": "sweep_top_universe_params",
                "selected_symbol": sweep["top_symbol"],
                "channel_lookback": int(sweep["lookback"]),
                "stop_atr": float(sweep["stop_atr"]),
                "trailing_stop_atr": float(sweep["trail_atr"]),
                "trade_count": int(round(sweep["avg_trades"])),
                "sharpe_ratio": float(sweep["best_sharpe"]),
                "total_return": float(sweep["avg_return"]),
                "max_drawdown": float(sweep["avg_max_dd"]),
            },
        ]
    )


def load_baseline_report() -> dict[str, object]:
    selected = _read_json(DATA_DIR / "selected_asset.json")
    trades = _read_csv("trade_blotter.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    ledger = _read_csv("ledger.csv", parse_dates=["date"])
    metrics_display, metrics = compute_metrics(trades, ledger)
    asset_screen = pd.read_csv(DATA_DIR / "asset_screening.csv")
    overview = pd.DataFrame(
        [
            {"Field": "Strategy", "Value": "Tuned baseline volatility and channel breakout"},
            {"Field": "Data source", "Value": "shinybroker / Interactive Brokers historical daily bars"},
            {"Field": "Selected symbol", "Value": selected["selected_symbol"]},
            {
                "Field": "Sample window",
                "Value": f"{ledger['date'].min().strftime('%Y-%m-%d')} to {ledger['date'].max().strftime('%Y-%m-%d')}",
            },
            {"Field": "Status", "Value": "Baseline strategy tuned and implemented"},
        ]
    )
    return {
        "selected": selected,
        "overview": overview,
        "metrics_display": metrics_display,
        "metrics_raw": metrics,
        "parameter_table": parameter_table(selected),
        "parameter_comparison": parameter_comparison(selected, metrics),
        "asset_screen_display": asset_screen,
        "outcome_rate_summary": outcome_rate_summary(trades),
        "trade_quality_summary": trade_quality_summary(trades),
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
        "monitoring_display": pd.DataFrame(
            [
                {"Metric": "Rolling expectancy", "Threshold": "< 0 over the latest 50 trades"},
                {"Metric": "Rolling hit rate", "Threshold": "Below the backtest range for 3 consecutive months"},
                {"Metric": "Drawdown", "Threshold": "Worse than 1.5x historical max drawdown"},
            ]
        ),
        "export_paths": {
            "blotter_csv_href": "downloads/trade_blotter.csv",
            "ledger_csv_href": "downloads/ledger.csv",
        },
    }


def load_holdout_report() -> dict[str, object]:
    summary = _read_json(DOWNLOADS_DIR / "fixed_live_holdout_summary.json")
    comparison = _read_csv("fixed_live_holdout_comparison.csv")
    equity = _read_csv("fixed_live_holdout_equity_curves.csv", parse_dates=["date"])
    trades = _read_csv("fixed_live_holdout_scaled_trades.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    predictions = _read_csv("fixed_live_holdout_test_predictions.csv", parse_dates=["signal_date"])
    return {
        "summary": summary,
        "comparison": comparison,
        "equity_curves": equity,
        "scaled_trades": trades,
        "test_predictions": predictions,
    }


def load_overlay_comparison() -> pd.DataFrame:
    return _read_csv("selected_asset_overlay_comparison.csv")
