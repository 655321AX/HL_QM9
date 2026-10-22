# 02b_descriptors.py — all RDKit 2D descriptors, for the Random Forest baseline.
# Output: data/descriptors.parquet, rows in qm9_clean.parquet order
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
DATA = ROOT / "data"

df = pd.read_parquet(DATA / "qm9_clean.parquet")

# ---- 1. compute every RDKit descriptor (~210) for every molecule ----
rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in df["smiles"]]
D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
print(f"computed: {D.shape[1]} descriptors")

# ---- 2. keep descriptors that are defined and not constant in the training set ----
train = D[(df["split"] == "train").to_numpy()]
keep = train.columns[train.notna().all() & (train.std() > 0)]
D = D[keep]
print(f"kept:     {D.shape[1]} descriptors")

D.to_parquet(DATA / "descriptors.parquet", index=False)
print(f"saved -> {DATA / 'descriptors.parquet'}")