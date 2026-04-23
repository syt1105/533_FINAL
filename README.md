# Volatility Breakout Strategy Lab

This project builds, backtests, and documents a volatility and channel breakout trading strategy in Quarto, using `shinybroker` as the intended bridge to Interactive Brokers historical data.

The workflow is being organized to do four things:

1. Screen a compact universe of liquid ETFs.
2. Select a candidate asset or ranked basket for the breakout strategy.
3. Run the breakout backtest to create the trade blotter, ledger, and performance metrics.
4. Publish the results as a Quarto website through the `docs/` folder.

## Project Structure

- `analysis/`: Python modules for configuration, asset screening, breakout analysis, and pipeline execution
- `data/`: selected asset metadata and placeholder research inputs
- `docs/`: rendered website output for GitHub Pages plus downloadable artifacts
- `*.qmd`: Quarto pages for the website

## Current Status

This repository is currently a scaffold aligned to the example project structure you shared. The Quarto pages, analysis package, and placeholder data files are in place, but the full volatility-breakout backtest and ML filter are still to be implemented.

## Planned Workflow

From the project directory:

```bash
python3 -m analysis.run_pipeline
quarto render
```

The pipeline will eventually:

- fetch or load market history,
- generate breakout and volatility features,
- create a selected asset file,
- run the backtest and trade logging,
- and export the artifacts used by the Quarto site.
