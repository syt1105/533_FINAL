from __future__ import annotations

import json

import pandas as pd

from analysis.strategy_config import ASSET_UNIVERSE, DATA_DIR


def _load_selected_asset() -> dict:
    path = DATA_DIR / "selected_asset.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"selected_symbol": "XLE", "selection_status": "placeholder"}


def run_analysis() -> dict[str, object]:
    selected = _load_selected_asset()

    overview = pd.DataFrame(
        [
            {"Field": "Project", "Value": "FINTECH 533 Final Project"},
            {"Field": "Strategy", "Value": "Volatility and channel breakout"},
            {"Field": "Data source", "Value": "shinybroker / Interactive Brokers planned"},
            {"Field": "Selected symbol", "Value": selected.get("selected_symbol", "XLE")},
            {"Field": "Status", "Value": "Scaffold only"},
        ]
    )

    metrics_display = pd.DataFrame(
        [
            {"Metric": "Sharpe Ratio", "Value": "TBD"},
            {"Metric": "Expected Return Per Trade", "Value": "TBD"},
            {"Metric": "Average Trade Lifetime", "Value": "TBD"},
            {"Metric": "Max Drawdown", "Value": "TBD"},
        ]
    )

    metric_explanations = pd.DataFrame(
        [
            {"Metric": "Sharpe Ratio", "Explanation": "Risk-adjusted return once the backtest is implemented."},
            {"Metric": "Expected Return Per Trade", "Explanation": "Average edge per completed trade."},
            {"Metric": "Average Trade Lifetime", "Explanation": "Average number of days the strategy holds a position."},
            {"Metric": "Max Drawdown", "Explanation": "Largest historical peak-to-trough equity decline."},
        ]
    )

    parameter_table = pd.DataFrame(
        [
            {"Parameter": "Universe", "Value": ", ".join(ASSET_UNIVERSE)},
            {"Parameter": "Breakout family", "Value": "Channel breakout / volatility breakout"},
            {"Parameter": "Base lookback", "Value": "Planned around 20 days"},
            {"Parameter": "Exit styles", "Value": "Stop-loss, timeout, trailing stop, breakout failure"},
            {"Parameter": "ML extension", "Value": "Supervised filter on breakout quality"},
        ]
    )

    asset_screen_display = pd.DataFrame(
        [{"Symbol": symbol, "Status": "Planned"} for symbol in ASSET_UNIVERSE]
    )

    outcome_summary = pd.DataFrame(
        [
            {"Outcome": "success", "Trades": 0},
            {"Outcome": "stop_loss", "Trades": 0},
            {"Outcome": "timeout", "Trades": 0},
            {"Outcome": "breakout_failure", "Trades": 0},
        ]
    )

    blotter_display = pd.DataFrame(
        columns=[
            "entry_timestamp",
            "exit_timestamp",
            "direction",
            "qty",
            "entry_price",
            "exit_price",
            "exit_reason",
            "trade_return",
            "pnl",
        ]
    )

    ledger_display = pd.DataFrame(
        columns=["date", "cash", "market_value", "equity", "drawdown"]
    )

    monitoring_display = pd.DataFrame(
        [
            {"Metric": "Rolling expectancy", "Threshold": "< 0"},
            {"Metric": "Rolling hit rate", "Threshold": "Below out-of-sample confidence band"},
            {"Metric": "Drawdown", "Threshold": "Above historical stress threshold"},
        ]
    )

    export_paths = {
        "blotter_csv_href": "downloads/trade_blotter.csv",
        "ledger_csv_href": "downloads/ledger.csv",
    }

    return {
        "overview": overview,
        "metrics_display": metrics_display,
        "metric_explanations": metric_explanations,
        "parameter_table": parameter_table,
        "asset_screen_display": asset_screen_display,
        "outcome_summary": outcome_summary,
        "blotter_display": blotter_display,
        "ledger_display": ledger_display,
        "monitoring_display": monitoring_display,
        "export_paths": export_paths,
    }
