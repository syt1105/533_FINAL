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

## Planned Workflow

From the project directory:

```bash
python3 -m analysis.run_pipeline
quarto render
```

The pipeline currently:

- fetch or load market history,
- fetch or load options-derived volatility history,
- generate breakout, volatility, and implied-volatility features,
- create a selected asset file,
- run the baseline and ML-augmented backtests,
- and export the artifacts used by the Quarto site.

## Data Layout

- `data/market_history/`: one CSV per symbol fetched through `shinybroker`
- `data/options_history/`: one CSV per symbol and volatility series fetched through `shinybroker`
- `data/asset_screening.csv`: cross-asset screening summary
- `data/selected_asset.json`: the currently chosen symbol and selection metadata
