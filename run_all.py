# run_all.py — run the whole pipeline in order. Stops at the first script that fails.
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = [
    "01_data.py",
    "02_fingerprints.py",
    "02b_descriptors.py",
    "03b_tune.py",
    "03a_train.py",
    "04_groups.py",
    "05_validate.py",
    "06_figures.py",
]

start_all = time.time()
for script in SCRIPTS:
    print(f"\n{'#' * 60}\n# {script}\n{'#' * 60}", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, ROOT / script], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{script} FAILED — stopping here.")
        sys.exit(1)
    print(f"\n{script} finished in {(time.time() - start) / 60:.1f} min")

print(f"\nAll done in {(time.time() - start_all) / 60:.1f} min")
