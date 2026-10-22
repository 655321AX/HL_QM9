# 01_data.py — clean QM9 and assign every molecule to train/val/test
# (random 80/10/10 split). Output: data/qm9_clean.parquet
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
SEED = 42
HARTREE_TO_EV = 27.211386245988

# ---- 1. load ----
df = pd.read_csv(ROOT / "qm9.csv", usecols=["mol_id", "smiles", "homo", "lumo"])
print(f"loaded:                    {len(df)}")

# ---- 2. remove the 3054 molecules QM9 flags as uncharacterized ----
bad_file = DATA / "uncharacterized.txt"
if bad_file.exists():
    bad = set()
    for line in bad_file.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            bad.add(f"gdb_{int(parts[0])}")
    df = df[~df["mol_id"].isin(bad)]
    print(f"after uncharacterized:     {len(df)}")
else:
    print("WARNING: data/uncharacterized.txt not found, keeping all molecules")

# ---- 3. canonical SMILES, drop molecules with < 2 heavy atoms ----
df["smiles"] = df["smiles"].apply(Chem.CanonSmiles)
n_heavy = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
df = df[n_heavy >= 2]
print(f"after tiny molecules:      {len(df)}")

# ---- 4. merge duplicate structures (average their labels) ----
df = df.groupby("smiles", as_index=False).agg(
    mol_id=("mol_id", "first"), homo=("homo", "mean"), lumo=("lumo", "mean")
)
print(f"after merging duplicates:  {len(df)}")

# ---- 5. Hartree -> eV ----
df["homo"] = df["homo"] * HARTREE_TO_EV
df["lumo"] = df["lumo"] * HARTREE_TO_EV

# ---- 6. random split 80 / 10 / 10 ----
train, rest = train_test_split(df.index, test_size=0.2, random_state=SEED)
val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
df.loc[train, "split"] = "train"
df.loc[val, "split"] = "val"
df.loc[test, "split"] = "test"

# ---- 7. report and save ----
print(df.groupby("split")[["homo", "lumo"]].agg(["count", "mean", "std"]).round(3))
df.to_parquet(DATA / "qm9_clean.parquet", index=False)
print(f"\nsaved {len(df)} molecules -> {DATA / 'qm9_clean.parquet'}")