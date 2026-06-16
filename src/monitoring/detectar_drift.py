import pandas as pd


FEATURES_NUMERICAS = ["grid", "TrackTemp", "AirTemp", "Humidity", "Rainfall", "WindSpeed"]
LIMIAR_DRIFT = 0.20


def calcular_drift():
    df = pd.read_parquet("data/final/f1_dataset_final.parquet")

    referencia = df[df["year"].isin([2021, 2022])]
    atual = df[df["year"] == 2023]

    print("=" * 58)
    print("  RELATORIO DE DRIFT -- F1 Top-10 Predictor")
    print("=" * 58)
    print(f"  Referência : 2021 + 2022  ({len(referencia)} corridas)")
    print(f"  Atual      : 2023         ({len(atual)} corridas)")
    print("=" * 58)
    print(f"  {'Feature':<12}  {'Ref (media)':>12}  {'Atual (media)':>13}  {'Delta%':>7}  Status")
    print("-" * 58)

    drifts = []
    for feature in FEATURES_NUMERICAS:
        media_ref = referencia[feature].mean()
        media_atual = atual[feature].mean()

        if media_ref == 0:
            delta_pct = float("inf")
        else:
            delta_pct = (media_atual - media_ref) / abs(media_ref)

        drift = abs(delta_pct) > LIMIAR_DRIFT
        status = "!! DRIFT DETECTADO" if drift else "OK"
        drifts.append(drift)

        print(f"  {feature:<12}  {media_ref:>12.4f}  {media_atual:>13.4f}  {delta_pct:>+6.1%}  {status}")

    print("=" * 58)
    total_drift = sum(drifts)
    if total_drift == 0:
        print("  Conclusao: nenhum drift significativo detectado.")
    else:
        print(f"  Conclusao: {total_drift}/{len(FEATURES_NUMERICAS)} feature(s) com drift > {LIMIAR_DRIFT:.0%}.")
    print("=" * 58)


if __name__ == "__main__":
    calcular_drift()
