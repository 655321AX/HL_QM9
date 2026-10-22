# 03_train.py — train three models per target and evaluate them once on the test set.
#   Ridge     on CF-MF        (the model we interpret with SHAP)
#   LightGBM  on CF-MF        (the most accurate model, used as a cross-check)
#   RF        on descriptors  (reference: standard descriptors)
# Output: models/*.joblib, data/results_test.csv, data/predictions_test.parquet
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42
TARGETS = ["homo", "lumo"]

RIDGE_ALPHAS = [0.1, 0.3, 1, 3, 10]
LGBM_PARAMS = dict(n_estimators=5000, learning_rate=0.1, num_leaves=255,
                   min_child_samples=10, subsample=0.85, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=1, random_state=SEED, verbose=-1)
RF_PARAMS = dict(n_estimators=300, max_features=0.7, max_depth=25,
                 random_state=SEED, n_jobs=-1)

# ---- load data ----
df = pd.read_parquet(DATA / "qm9_clean.parquet")
X = sparse.load_npz(DATA / "cfmf.npz").tocsr()
D = pd.read_parquet(DATA / "descriptors.parquet")
tr = (df["split"] == "train").to_numpy()
va = (df["split"] == "val").to_numpy()
te = (df["split"] == "test").to_numpy()

results = []
predictions = df.loc[te, ["smiles", "homo", "lumo"]].reset_index(drop=True)

for target in TARGETS:
    y = df[target].to_numpy()
    print(f"\n========== {target.upper()} ==========")

    # ---- Ridge: try each alpha, keep the one with the lowest validation MAE ----
    best_mae = np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(X[tr], y[tr])
        mae = mean_absolute_error(y[va], model.predict(X[va]))
        print(f"Ridge alpha={alpha:<4} val MAE = {mae:.4f} eV")
        if mae < best_mae:
            best_mae, ridge = mae, model
    print(f"-> chosen alpha = {ridge.alpha}")

    # ---- LightGBM: add trees until the validation MAE stops improving ----
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X[tr], y[tr], eval_X=X[va], eval_y=y[va], eval_metric="l1",
             callbacks=[early_stopping(100, verbose=False), log_evaluation(500)])
    print(f"LightGBM: {lgbm.best_iteration_} trees")

    # ---- Random Forest on descriptors ----
    rf = RandomForestRegressor(**RF_PARAMS).fit(D[tr], y[tr])

    # ---- evaluate all models once on the test set ----
    models = {"ridge_cfmf": (ridge, X), "lgbm_cfmf": (lgbm, X), "rf_desc": (rf, D)}
    baseline = np.full(te.sum(), y[tr].mean())       # always predict the training mean
    candidates = {"mean": baseline}
    for name, (model, features) in models.items():
        candidates[name] = model.predict(features[te])
        joblib.dump(model, MODELS / f"{name}_{target}.joblib")

    for name, pred in candidates.items():
        predictions[f"{name}_{target}"] = pred
        results.append({
            "target": target, "model": name,
            "MAE": mean_absolute_error(y[te], pred),
            "RMSE": np.sqrt(mean_squared_error(y[te], pred)),
            "R2": r2_score(y[te], pred),
        })

# ---- report and save ----
results = pd.DataFrame(results)
print("\n========== TEST SET ==========")
print(results.pivot(index="model", columns="target").round(4))
results.to_csv(DATA / "results_test.csv", index=False)
predictions.to_parquet(DATA / "predictions_test.parquet", index=False)
print(f"\nsaved models -> {MODELS}, results -> {DATA}")