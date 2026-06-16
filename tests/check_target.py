import pandas as pd
import sys

df = pd.read_parquet("data/final/f1_dataset_final.parquet")
dist = df["top10"].value_counts(normalize=True)
p0, p1 = dist[0], dist[1]
ok = abs(p0 - 0.5) < 0.05 and abs(p1 - 0.5) < 0.05
status = "PASS" if ok else "FAIL"
print(f"  [{status}] top10=0: {p0:.1%}  top10=1: {p1:.1%}  (esperado ~50/50)")
sys.exit(0 if ok else 1)
