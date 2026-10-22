# 06_figures.py — the two result figures.
#   figures/parity.png         predicted vs DFT for every model and target (test set)
#   figures/group_effects.png  median group contribution to HOMO and LUMO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
TARGETS = ["homo", "lumo"]
MODELS = {"ridge_cfmf": "Ridge, CF-MF", "lgbm_cfmf": "LightGBM, CF-MF", "rf_desc": "RF, descriptors"}
MIN_MOLECULES = 50

# ==================== 1. parity plots ====================
pred = pd.read_parquet(DATA / "predictions_test.parquet")
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

for i, target in enumerate(TARGETS):
    true = pred[target]
    lo, hi = true.min() - 0.3, true.max() + 0.3              # same axis range in every panel
    for j, (model, label) in enumerate(MODELS.items()):
        ax = axes[i, j]
        p = pred[f"{model}_{target}"]
        # hexbin = 2D histogram; colour = number of molecules per hexagon (log scale)
        ax.hexbin(true, p, gridsize=60, extent=(lo, hi, lo, hi), norm=LogNorm(), mincnt=1)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)              # perfect-prediction line
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.text(0.05, 0.95,
                f"MAE = {mean_absolute_error(true, p):.3f} eV\nR² = {r2_score(true, p):.3f}",
                transform=ax.transAxes, va="top", bbox=dict(fc="white", alpha=0.8))
        ax.set_title(f"{label}: {target.upper()}")
        ax.set_xlabel(f"DFT {target.upper()} (eV)")
        ax.set_ylabel(f"predicted {target.upper()} (eV)")

fig.tight_layout()
fig.savefig(FIGURES / "parity.png", dpi=200)
print(f"saved -> {FIGURES / 'parity.png'}")

# ==================== 2. group contributions ====================
def group_stats(path):
    """Median and quartiles per group, over the molecules that contain it."""
    stats = pd.read_parquet(path).describe().T
    stats = stats[stats["count"] >= MIN_MOLECULES]
    return stats.drop(index=["rest", "baseline"])


fig, axes = plt.subplots(1, 2, figsize=(13, 7))

for ax, target in zip(axes, TARGETS):
    ridge = group_stats(DATA / f"group_shap_ridge_cfmf_{target}.parquet").sort_values("50%")
    lgbm = group_stats(DATA / f"group_shap_lgbm_cfmf_{target}.parquet")
    y = np.arange(len(ridge))

    # Ridge: bar = median, whiskers = 25 % to 75 % of molecules containing the group
    ax.barh(y, ridge["50%"],
            xerr=[ridge["50%"] - ridge["25%"], ridge["75%"] - ridge["50%"]],
            color="#4a7ab5", capsize=3, label="Ridge (median, middle 50 %)")
    # LightGBM: black dot = median, for comparison
    lgbm = lgbm.reindex(ridge.index)                          # same group order as Ridge
    ax.scatter(lgbm["50%"], y, color="black", zorder=3, label="LightGBM (median)")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ridge.index)
    ax.set_xlabel(f"contribution to {target.upper()} (eV)")
    ax.set_title(target.upper())
    ax.grid(axis="x", alpha=0.3)

axes[0].legend(loc="lower right")
fig.suptitle("Functional group contributions, QM9 test set (CF-MF + SHAP)")
fig.tight_layout()
fig.savefig(FIGURES / "group_effects.png", dpi=200)
print(f"saved -> {FIGURES / 'group_effects.png'}")