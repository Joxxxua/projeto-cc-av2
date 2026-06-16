"""Agrega os dados volta-a-volta para uma linha por resultado de corrida."""
import pandas as pd


# Colunas de leakage — nunca devem entrar no dataset agregado
LEAKAGE_COLS = {
    "position_x", "status", "statusId", "milliseconds", "lap", "LapNumber",
    "time", "TyreLife", "Stint", "Compound", "FreshTyre",
}


def build_dataset():
    df = pd.read_parquet("data/processed/f1_processed.parquet")
    print(f"[build_dataset] Shape entrada: {df.shape}")

    agg = (
        df.groupby(["raceId", "driverId"])
        .agg(
            position_final=("position_y", "first"),
            grid=("grid", "first"),
            year=("year", "first"),
            round=("round", "first"),
            circuitId=("circuitId", "first"),
            constructorId=("constructorId", "first"),
            TrackTemp=("TrackTemp", "mean"),
            AirTemp=("AirTemp", "mean"),
            Humidity=("Humidity", "mean"),
            Rainfall=("Rainfall", "mean"),
            WindSpeed=("WindSpeed", "mean"),
        )
        .reset_index()
    )

    # Garantia: nenhuma coluna de leakage pode estar presente
    leaked = set(agg.columns) & LEAKAGE_COLS
    assert not leaked, f"LEAKAGE DETECTADO: {leaked}"

    agg.to_parquet("data/processed/f1_race_results.parquet", index=False)
    print(f"[build_dataset] Salvo: data/processed/f1_race_results.parquet — shape: {agg.shape}")


if __name__ == "__main__":
    build_dataset()
