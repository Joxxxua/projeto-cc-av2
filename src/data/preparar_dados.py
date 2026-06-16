import pandas as pd


def preparar_dados():
    df = pd.read_parquet("data/raw/f1_all.parquet")
    print(f"[preparar_dados] Shape bruto: {df.shape}")

    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

    antes = len(df)
    df = df.dropna(subset=["position_y"])
    print(f"[preparar_dados] Removidas {antes - len(df)} linhas com position_y nulo")

    df.to_parquet("data/processed/f1_processed.parquet", index=False)
    print(f"[preparar_dados] Salvo: data/processed/f1_processed.parquet — shape: {df.shape}")


if __name__ == "__main__":
    preparar_dados()
