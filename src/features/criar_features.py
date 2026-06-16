import pandas as pd


FEATURES_NUMERICAS = ["grid", "round", "year", "TrackTemp", "AirTemp", "Humidity", "Rainfall", "WindSpeed"]
FEATURES_CATEGORICAS = ["constructorId", "circuitId"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def criar_features():
    df = pd.read_parquet("data/processed/f1_race_results.parquet")
    print(f"[criar_features] Shape entrada: {df.shape}")

    df["top10"] = (df["position_final"] <= 10).astype(int)

    # Descarta position_final — seria leakage se mantida
    df = df.drop(columns=["position_final"])

    # Mantém apenas as features seguras + target + chaves (raceId, driverId para rastreabilidade)
    cols_finais = ["raceId", "driverId"] + FEATURES + ["top10"]
    df = df[cols_finais]

    print(f"[criar_features] Shape final: {df.shape}")
    print(f"[criar_features] Features: {FEATURES}")
    print()

    dist = df["top10"].value_counts(normalize=True).sort_index()
    print("[criar_features] Distribuição do target top10:")
    print(f"  0 (fora do top10): {df['top10'].value_counts()[0]:>4d}  ({dist[0]:.1%})")
    print(f"  1 (top10):         {df['top10'].value_counts()[1]:>4d}  ({dist[1]:.1%})")

    df.to_parquet("data/final/f1_dataset_final.parquet", index=False)
    print(f"\n[criar_features] Salvo: data/final/f1_dataset_final.parquet")


if __name__ == "__main__":
    criar_features()
