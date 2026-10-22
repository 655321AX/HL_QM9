# 05_validate.py — can the group contributions from 04_groups.py be trusted?
#   1. stability:  refit Ridge on resampled training sets; do group medians stay put?
#   2. agreement:  do Ridge and LightGBM give the same group contributions?
#   3. removal:    remove a group's fingerprint occurrences from a molecule; does the
#                  LightGBM prediction change by about the group's SHAP value?
# Output: results/validation_{target}.csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.utils import resample

from lib import group_columns, group_totals

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TARGETS = ["homo", "lumo"]
N_BOOTSTRAP = 10
N_EVAL = 2000          # test molecules used for the stability test
MIN_MOLECULES = 50

df = pd.read_parquet(DATA / "qm9_clean.parquet")
X = sparse.load_npz(DATA / "cfmf.npz").tocsr()
vocabulary = joblib.load(DATA / "cfmf_vectorizer.joblib").vocabulary_
train_rows = np.flatnonzero(df["split"] == "train")
test_rows = np.flatnonzero(df["split"] == "test")
eval_rows = test_rows[:N_EVAL]


def group_medians(totals):
    """Median contribution of each group over the molecules that contain it."""
    counts = totals.count()
    medians = totals.median()
    keep = (counts >= MIN_MOLECULES) & ~medians.index.isin(["rest", "baseline"])
    return medians[keep]


for target in TARGETS:
    print(f"\n==================== {target.upper()} ====================")
    y = df[target].to_numpy()
    alpha = joblib.load(MODELS / f"ridge_cfmf_{target}.joblib").alpha

    # ---------- 1. stability across bootstrap refits of Ridge ----------
    runs = []
    for b in range(N_BOOTSTRAP):
        rows = resample(train_rows, random_state=b)          # draw training rows with replacement
        model = Ridge(alpha=alpha).fit(X[rows], y[rows])
        x_mean = np.asarray(X[rows].mean(axis=0)).ravel()
        phi = (X[eval_rows].toarray() - x_mean) * model.coef_
        totals = pd.DataFrame([group_totals(s, p, vocabulary)
                               for s, p in zip(df["smiles"].iloc[eval_rows], phi)])
        runs.append(group_medians(totals))
        print(f"bootstrap {b + 1}/{N_BOOTSTRAP} done", flush=True)

    runs = pd.DataFrame(runs)                                  # rows = refits, columns = groups
    stability = pd.DataFrame({
        "median": runs.mean(),
        "sd": runs.std(),
        "same_sign": (np.sign(runs) == np.sign(runs.mean())).all(),
    })

    # ---------- 2. agreement between Ridge and LightGBM (all test molecules) ----------
    ridge_totals = pd.read_parquet(DATA / f"group_shap_ridge_cfmf_{target}.parquet")
    lgbm_totals = pd.read_parquet(DATA / f"group_shap_lgbm_cfmf_{target}.parquet")
    agreement = pd.DataFrame({
        "ridge": group_medians(ridge_totals),
        "lgbm": group_medians(lgbm_totals),
    })

    # ---------- 3. removal test on LightGBM (all test molecules) ----------
    lgbm = joblib.load(MODELS / f"lgbm_cfmf_{target}.joblib")
    full_prediction = lgbm.predict(X[test_rows])

    modified_rows, records = [], []
    for i, row in enumerate(test_rows):
        mol_id = df["mol_id"].iloc[row]
        for group, columns in group_columns(df["smiles"].iloc[row], vocabulary).items():
            if group == "rest":
                continue
            x = X[row].copy()
            for col, n in columns.items():
                x[0, col] -= n                                 # take the group's occurrences out
            modified_rows.append(x)
            records.append({"group": group, "molecule": i,
                            "shap": lgbm_totals.loc[mol_id, group]})

    removed_prediction = lgbm.predict(sparse.vstack(modified_rows))
    removal = pd.DataFrame(records)
    removal["removal"] = full_prediction[removal["molecule"]] - removed_prediction

    removal_summary = removal.groupby("group").agg(
        n=("shap", "size"),
        shap_median=("shap", "median"),
        removal_median=("removal", "median"),
    )
    removal_summary["r"] = removal.groupby("group").apply(lambda g: g["shap"].corr(g["removal"]))
    removal_summary = removal_summary[removal_summary["n"] >= MIN_MOLECULES]

    # ---------- report and save ----------
    table = stability.join(agreement, how="outer").join(removal_summary, how="outer")
    table = table.sort_values("median")
    table.index.name = "group"
    print(f"\n{target.upper()}: group contributions (eV) and validation")
    print(table.round(3).to_string())
    table.to_csv(RESULTS / f"validation_{target}.csv")