# 03b_tune.py — hyperparameter search for LightGBM (on CF-MF) and Random Forest (on descriptors).
# Each target (HOMO, LUMO) is tuned separately. Every candidate is trained on the training
# set and scored on the validation set (PredefinedSplit = "use my fixed split, no cross-validation").
# Output: results/best_params.json  (complete settings, read by 03_train.py)
#         results/tuning_{model}_{target}.csv  (every candidate and its validation MAE)
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42
TARGETS = ["homo", "lumo"]
N_CANDIDATES = 30          # random settings tried per model and target

# settings that are NOT searched
LGBM_FIXED = dict(learning_rate=0.05, n_estimators=15000, subsample_freq=1,
                  random_state=SEED, verbose=-1)
RF_FIXED = dict(n_estimators=300, random_state=SEED, n_jobs=-1)

# settings that ARE searched: each candidate picks one value from every list at random
LGBM_SPACE = {
    "num_leaves":        [31, 63, 127, 255],
    "min_child_samples": [5, 10, 20, 50],
    "colsample_bytree":  [0.3, 0.5, 0.7, 1.0],
    "subsample":         [0.7, 0.85, 1.0],
    "reg_lambda":        [0, 1, 5, 10],
}
RF_SPACE = {
    "max_features":     [0.2, 0.33, 0.5, 0.7, 1.0],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_depth":        [None, 15, 25],
}

# ---- data: train + val together; PredefinedSplit marks which rows are validation ----
df = pd.read_parquet(DATA / "qm9_clean.parquet")
X = sparse.load_npz(DATA / "cfmf.npz").tocsr()
D = pd.read_parquet(DATA / "descriptors.parquet")
use = (df["split"] != "test").to_numpy()            # the test set is never touched here
va = (df["split"] == "val").to_numpy()
fold = np.where(df.loc[use, "split"] == "val", 0, -1)  # -1 = always train, 0 = validation
cv = PredefinedSplit(fold)

best = {"lgbm": {}, "rf": {}}

for target in TARGETS:
    y = df[target].to_numpy()

    for name, estimator, space, features, fit_args in [
        ("lgbm", LGBMRegressor(**LGBM_FIXED), LGBM_SPACE, X,
         # early stopping watches the validation set, exactly as in 03_train.py
         dict(eval_X=X[va], eval_y=y[va], eval_metric="l1",
              callbacks=[early_stopping(200, verbose=False)])),
        ("rf", RandomForestRegressor(**RF_FIXED), RF_SPACE, D, {}),
    ]:
        print(f"\n===== {target.upper()}: tuning {name} ({N_CANDIDATES} candidates) =====", flush=True)
        search = RandomizedSearchCV(
            estimator, space, n_iter=N_CANDIDATES, cv=cv,
            scoring="neg_mean_absolute_error", refit=False,
            random_state=SEED, verbose=2,
        )
        search.fit(features[use], y[use], **fit_args)

        # save every candidate, best first
        table = pd.DataFrame(search.cv_results_["params"])
        table["val_MAE"] = -search.cv_results_["mean_test_score"]
        table["seconds"] = search.cv_results_["mean_fit_time"]
        table = table.sort_values("val_MAE")
        table.to_csv(RESULTS / f"tuning_{name}_{target}.csv", index=False)
        print(table.head(5).to_string(index=False))

        fixed = LGBM_FIXED if name == "lgbm" else RF_FIXED
        best[name][target] = {**fixed, **search.best_params_}

        # write after every search, so a crash later does not lose finished results
        (RESULTS / "best_params.json").write_text(json.dumps(best, indent=2))

print(f"\nsaved -> {RESULTS / 'best_params.json'}")
