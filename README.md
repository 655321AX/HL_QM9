# HL_QM9 — HOMO/LUMO prediction and functional-group analysis on QM9

Predicts HOMO and LUMO energies from molecular structure with collision-free Morgan
fingerprints (CF-MF), then uses SHAP to measure how much each functional group shifts them.

## 1. Setup

Python 3.13 or newer.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the two QM9 data files here:

```
HL_QM9/
  qm9.csv
  data/
    uncharacterized.txt
```

Both come from the QM9 figshare collection:
https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

## 2. Run

```
python run_all.py
```

This runs every script in order and stops at the first error. Each script can also be run on
its own (`python 04_groups.py`); it reads the files the earlier scripts saved.

| script | what it does | expected time |
|---|---|---|
| `01_data.py` | clean QM9, random 80/10/10 split | ~1 min |
| `02_fingerprints.py` | CF-MF fingerprint matrix | <1 min |
| `02b_descriptors.py` | RDKit descriptors | ~5–10 min |
| `03b_tune.py` | hyperparameter search, LightGBM + RF, 30 candidates each, per target | **longest: many hours** |
| `03_train.py` | final models, test-set metrics | ~30–60 min |
| `04_groups.py` | SHAP per functional group, Ridge + LightGBM | **long: hours (LightGBM TreeSHAP)** |
| `05_validate.py` | stability, Ridge-vs-LightGBM agreement, removal test | ~30 min |
| `06_figures.py` | parity plot, group-contribution figure | <1 min |

Times depend on the number of CPU cores. `03b_tune.py` prints the time of every candidate:
multiply the first one by ~120 for a rough total. `04_groups.py` prints progress every 200 molecules.

**If something crashes:** fix it and rerun from that script; earlier outputs are kept.
`03b_tune.py` saves `results/best_params.json` after each finished search, but a restart
begins the search again from the start.

Memory: tested to fit in 48 GB. If `04_groups.py` runs out of memory, lower `CHUNK` at the top.

## 3. What to send back

- `results/` — all numerical results as CSV, plus `best_params.json`
- `figures/` — the figures
- optionally `data/` and `models/` (large) — only needed to rerun steps 04–06 elsewhere

## Files

| file | contents |
|---|---|
| `lib.py` | functional group definitions (SMARTS) and SHAP-to-group assignment |
| `results/test_metrics.csv` | MAE, RMSE, R² of every model on the test set |
| `results/ridge_alpha.csv`, `lgbm_trees.csv` | Ridge alpha search, LightGBM tree counts |
| `results/tuning_*.csv` | every hyperparameter candidate and its validation MAE |
| `results/groups_*.csv` | per functional group: count, mean, median, quartiles (eV) |
| `results/validation_*.csv` | stability, model agreement and removal test per group |
