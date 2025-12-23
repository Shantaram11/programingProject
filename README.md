# programingProject

This repo contains:

- **Go code** (original project files)
- **A PyQt5 time-series model training platform** under `ts_platform/`

## Time-series model training platform (PyQt5 + qt-material)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ts_platform/requirements.txt
```

### Run

```bash
python3 -m ts_platform
```

### What it supports (UI features)

- **Upload data** from local PC (CSV / Excel / Parquet)
- **Select multiple target variables** (checkbox list)
- **Select multiple feature columns** (checkbox list)
- **Data cleaning choices** (missing values, outlier clipping, scaling, target transform) with **defaults + allowed ranges**
- **Multiple models** selectable at once:
  - MA, WMA
  - ARIMA (statsmodels)
  - Prophet
  - XGBoost (lag-feature regression)
  - DeepAR (PyTorch lightweight implementation)
- **Visualization**: actual vs model predictions, with **checkboxes to show/hide** each model line
- **Save results locally**, plus a **Saved Results** page to view/export/delete runs

### Where results are saved

Runs are stored under:

- `~/.ts_model_platform/runs/<run_id>/`

Each run includes `info.json`, `predictions.csv`, and `plot.png`.
