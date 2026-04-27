# Volatility Breakout Strategy Lab

This project builds, backtests, and documents a volatility and channel breakout trading strategy in Quarto, using `shinybroker` as the bridge to Interactive Brokers historical data and options-derived implied-volatility series.

The workflow does four things:

1. Screen a compact universe of liquid ETFs.
2. Select a showcase asset based on strategy-level performance.
3. Run the breakout backtest to create the trade blotter, ledger, and performance metrics.
4. Add a supervised-learning overlay that uses breakout and implied-volatility features to adjust position sizing.

## Project Structure

- `analysis/`: Python modules for configuration, asset screening, breakout analysis, and pipeline execution
- `data/`: market-history files, options-volatility files, and selected-asset metadata
- `docs/`: rendered website output for GitHub Pages plus downloadable artifacts
- `*.qmd`: Quarto pages for the website

## Current Status

The repository now contains a working end-to-end research pipeline:

- separate market-history CSVs in `data/market_history/`
- separate IBKR implied-volatility and historical-volatility CSVs in `data/options_history/`
- a tuned baseline `QQQ` breakout strategy
- an ML overlay based on compact `core_iv` features
- downloadable blotter, ledger, model, and overlay comparison files in `docs/downloads/`

## Reproducible Saved-Data Workflow

Use this workflow when grading or rendering the report from the checked-in CSV files. It does not call Interactive Brokers.

From the project directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m analysis.run_saved_data_pipeline
quarto render
```

The saved-data pipeline regenerates the baseline blotter and ledger, labeled breakout datasets, ML feature files, model artifacts, the hard-filter overlay backtest, the position-scaling overlay backtest, and the final comparison tables in `docs/downloads/`.

## Full IBKR Refresh Workflow

Use this only when Interactive Brokers TWS or Gateway is running and API access is enabled:

```bash
python3 -m analysis.run_pipeline
quarto render
```

The full pipeline currently:

- fetches market history,
- fetches options-derived volatility history,
- generate breakout, volatility, and implied-volatility features,
- create a selected asset file,
- run the baseline and ML-augmented backtests,
- and export the artifacts used by the Quarto site.

## Data Layout

- `data/market_history/`: one CSV per symbol fetched through `shinybroker`
- `data/options_history/`: one CSV per symbol and volatility series fetched through `shinybroker`
- `data/asset_screening.csv`: cross-asset screening summary
- `data/selected_asset.json`: the currently chosen symbol and selection metadata
