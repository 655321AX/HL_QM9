# 04_groups.py — how much does each functional group shift HOMO / LUMO?
# Per-column SHAP values are computed for every test molecule and then summed
# per functional group (lib.group_totals). Done for Ridge (exact SHAP) and for
# LightGBM (TreeSHAP; slow — this is the longest step of the pipeline).
# Output: data/group_shap_{model}_{target}.parquet   (rows = molecules, columns = groups)
#         results/groups_{model}_{target}.csv       (one row per group: count, median, quartiles)
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from lib import group_totals

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TARGETS = ["homo", "lumo"]
CHUNK = 200          # molecules processed at once (200 x 166k columns ~ 270 MB)
MIN_MOLECULES = 50   # report a group only if it occurs in at least this many molecules

df = pd.read_parquet(DATA / "qm9_clean.parquet")
X = sparse.load_npz(DATA / "cfmf.npz").tocsr()
vocabulary = joblib.load(DATA / "cfmf_vectorizer.joblib").vocabulary_
train_rows = np.flatnonzero(df["split"] == "train")
test_rows = np.flatnonzero(df["split"] == "test")
x_mean = np.asarray(X[train_rows].mean(axis=0)).ravel()    # average count of each column


def ridge_shap(model, rows):
    """Exact SHAP for a linear model: weight * (count - average count)."""
    return (X[rows].toarray() - x_mean) * model.coef_


def lgbm_shap(model, rows):
    """TreeSHAP, computed by LightGBM itself. The last column is the base value."""
    contrib = model.predict(X[rows], pred_contrib=True)
    if sparse.issparse(contrib):
        contrib = contrib.toarray()
    return contrib[:, :-1]


explainers = {"ridge_cfmf": ridge_shap, "lgbm_cfmf": lgbm_shap}

for target in TARGETS:
    for name, shap_function in explainers.items():
        model = joblib.load(MODELS / f"{name}_{target}.joblib")
        print(f"\n===== {target.upper()} — {name} ({len(test_rows)} test molecules) =====", flush=True)

        # ---- per-molecule group totals, CHUNK molecules at a time ----
        records = []
        for start in range(0, len(test_rows), CHUNK):
            chunk = test_rows[start:start + CHUNK]
            phi = shap_function(model, chunk)
            for smiles, phi_row in zip(df["smiles"].iloc[chunk], phi):
                records.append(group_totals(smiles, phi_row, vocabulary))
            print(f"  {start + len(chunk)}/{len(test_rows)} molecules", flush=True)
        totals = pd.DataFrame(records, index=df["mol_id"].iloc[test_rows])
        totals.to_parquet(DATA / f"group_shap_{name}_{target}.parquet")

        # ---- sanity check (Ridge): group totals must add up to prediction - average prediction ----
        if name == "ridge_cfmf":
            expected = model.predict(X[test_rows]) - (model.intercept_ + x_mean @ model.coef_)
            error = np.abs(totals.sum(axis=1).to_numpy() - expected).max()
            print(f"check: largest difference between sum of groups and prediction = {error:.1e} eV")

        # ---- summary: median and spread over molecules that contain the group ----
        summary = totals.describe().T
        summary = summary[summary["count"] >= MIN_MOLECULES].drop(index=["rest", "baseline"])
        summary = summary[["count", "mean", "25%", "50%", "75%"]].sort_values("50%")
        summary.index.name = "group"
        print(summary.round(3))
        summary.to_csv(RESULTS / f"groups_{name}_{target}.csv")