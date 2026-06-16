"""Avalia o modelo salvo com validação cruzada e métricas de classificação."""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


FEATURES_NUMERICAS = ["grid", "round", "year", "TrackTemp", "AirTemp", "Humidity", "Rainfall", "WindSpeed"]
FEATURES_CATEGORICAS = ["constructorId", "circuitId"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS
TARGET = "top10"


def avaliar_modelo():
    pipeline = joblib.load("models/modelo_top10.pkl")
    df = pd.read_parquet("data/final/f1_dataset_final.parquet")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Validação cruzada no treino
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    cv_media = scores.mean()
    cv_desvio = scores.std()

    # Métricas no teste
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fora top10", "Top10"])

    # Exibição
    linhas = [
        "=== Avaliação do Modelo F1 Top-10 ===",
        "",
        f"Modelo carregado: models/modelo_top10.pkl",
        f"Dataset: {df.shape[0]} corridas | Treino: {len(X_train)} | Teste: {len(X_test)}",
        "",
        "--- Validação Cruzada (StratifiedKFold, 10 folds, scoring=f1) ---",
        f"  F1 por fold: {[round(s, 4) for s in scores]}",
        f"  Média F1:    {cv_media:.4f}",
        f"  Desvio-pad:  {cv_desvio:.4f}",
        "",
        "--- Matriz de Confusão (conjunto de teste) ---",
        f"  {'':15s}  Pred: Fora  Pred: Top10",
        f"  Real: Fora    {cm[0,0]:>10d}  {cm[0,1]:>10d}",
        f"  Real: Top10   {cm[1,0]:>10d}  {cm[1,1]:>10d}",
        "",
        "--- Classification Report (conjunto de teste) ---",
        report,
    ]

    saida = "\n".join(linhas)
    print(saida)

    with open("experiments/avaliacao.txt", "w", encoding="utf-8") as f:
        f.write(saida)

    print("[avaliar_modelo] Resumo salvo em experiments/avaliacao.txt")


if __name__ == "__main__":
    avaliar_modelo()
