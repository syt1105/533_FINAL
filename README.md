# Whale Hunters

This project builds, backtests, and documents a volatility and channel breakout trading strategy in Quarto. Whale Hunters uses `shinybroker` as the bridge to Interactive Brokers historical data and options-derived implied-volatility series.

Website: https://www.whale-hunters.com

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
quarto preview
```

The saved-data pipeline regenerates the baseline blotter and ledger, labeled breakout datasets, ML feature files, model artifacts, the hard-filter overlay backtest, the position-scaling overlay backtest, and the final comparison tables in `docs/downloads/`.

The project is configured so `quarto render` runs the saved-data pipeline before rendering and writes the static website into `docs/`. The `quarto preview` command serves the rendered `docs/` site locally.

## How To Run

For a clean local run from the checked-in data:

```bash
python3 -m pip install -r requirements.txt
quarto render
quarto preview
```

Open the local URL printed by Quarto. The rendered website is also stored in `docs/` for GitHub Pages.

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

## What We Would Do Next

- Extend the fixed out-of-sample paper-trading window as more data becomes available.
- Test the same breakout and ML overlay framework across more liquid ETFs and futures.
- Add richer options-chain features such as implied-volatility skew and term structure.
- Compare the current logistic model against tree-based classifiers, using walk-forward testing rather than in-sample fit.
- Paper-trade the strategy before live deployment and monitor whether the ML overlay continues to improve the baseline.
