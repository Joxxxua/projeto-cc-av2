import pandas as pd
import sys

checks = [
    ("data/processed/f1_processed.parquet",    (69155, 29)),
    ("data/processed/f1_race_results.parquet", (1247, 13)),
    ("data/final/f1_dataset_final.parquet",    (1247, 13)),
]
ok = True
for path, expected in checks:
    df = pd.read_parquet(path)
    status = "PASS" if df.shape == expected else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"  [{status}] {path}: {df.shape} (esperado {expected})")
sys.exit(0 if ok else 1)
