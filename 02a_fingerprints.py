# 02a_fingerprints.py — collision-free Morgan fingerprints (CF-MF).
# Every distinct Morgan substructure identifier found in the TRAINING molecules
# gets its own column, so two different substructures never share a column.
# Output: data/cfmf.npz             sparse matrix, rows in qm9_clean.parquet order
#         data/cfmf_vectorizer.joblib  remembers which identifier is which column
from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RADIUS = 2

df = pd.read_parquet(DATA / "qm9_clean.parquet")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)


def morgan_counts(smiles):
    """{identifier: count} for every circular substructure in one molecule (unfolded)."""
    mol = Chem.MolFromSmiles(smiles)
    fp = generator.GetSparseCountFingerprint(mol)
    return {str(k): v for k, v in fp.GetNonzeroElements().items()}


# ---- 1. fingerprint every molecule ----
fps = df["smiles"].apply(morgan_counts)

# ---- 2. one column per identifier, vocabulary taken from training molecules only ----
is_train = (df["split"] == "train").to_numpy()
vectorizer = DictVectorizer()
vectorizer.fit(fps[is_train])
X = vectorizer.transform(fps)        # identifiers never seen in training are ignored

# ---- 3. report ----
n_before = fps[~is_train].apply(len).sum()
n_after = X[~is_train].nnz
print(f"matrix: {X.shape[0]} molecules x {X.shape[1]} columns")
print(f"substructures per molecule: {X.nnz / X.shape[0]:.1f}")
print(f"val/test substructures unknown to training (ignored): "
      f"{n_before - n_after} of {n_before} ({1 - n_after / n_before:.1%})")

# ---- 4. save ----
sparse.save_npz(DATA / "cfmf.npz", X)
joblib.dump(vectorizer, DATA / "cfmf_vectorizer.joblib")
print(f"saved -> {DATA / 'cfmf.npz'}")