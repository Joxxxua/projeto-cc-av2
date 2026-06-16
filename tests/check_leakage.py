import pandas as pd
import sys

LEAKAGE = {
    "position_x", "position_y", "position_final", "status", "statusId",
    "milliseconds", "lap", "LapNumber", "time", "TyreLife", "Stint",
    "Compound", "FreshTyre",
}
df = pd.read_parquet("data/final/f1_dataset_final.parquet")
found = set(df.columns) & LEAKAGE
if not found:
    print("  [PASS] Nenhuma coluna de leakage encontrada")
    sys.exit(0)
else:
    print(f"  [FAIL] Colunas de leakage presentes: {found}")
    sys.exit(1)
