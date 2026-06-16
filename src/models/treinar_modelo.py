import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


FEATURES_NUMERICAS = ["grid", "round", "year", "TrackTemp", "AirTemp", "Humidity", "Rainfall", "WindSpeed"]
FEATURES_CATEGORICAS = ["constructorId", "circuitId"]
TARGET = "top10"


def treinar_modelo():
    df = pd.read_parquet("data/final/f1_dataset_final.parquet")
    print(f"[treinar_modelo] Dataset: {df.shape}")

    X = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[treinar_modelo] Treino: {X_train.shape} | Teste: {X_test.shape}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CATEGORICAS),
        ]
    )

    candidatos = [
        (
            "RandomForest_100",
            RandomForestClassifier(n_estimators=100, random_state=42),
            {"n_estimators": 100, "random_state": 42},
        ),
        (
            "RandomForest_200",
            RandomForestClassifier(n_estimators=200, random_state=42),
            {"n_estimators": 200, "random_state": 42},
        ),
        (
            "LogisticRegression_default",
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": 1.0, "max_iter": 1000, "random_state": 42},
        ),
        (
            "LogisticRegression_C05",
            LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            {"C": 0.5, "max_iter": 1000, "random_state": 42},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        ),
    ]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("F1Top10")

    resultados = []
    melhor_f1 = -1
    melhor_pipeline = None
    melhor_nome = None

    for nome, clf, params in candidatos:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        }

        with mlflow.start_run(run_name=nome):
            mlflow.log_params(params)
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, "modelo")

        resultados.append({"modelo": nome, **metricas})

        if metricas["f1"] > melhor_f1:
            melhor_f1 = metricas["f1"]
            melhor_pipeline = pipeline
            melhor_nome = nome

        print(f"  [{nome}] f1={metricas['f1']} | auc={metricas['roc_auc']} | acc={metricas['accuracy']}")

    joblib.dump(melhor_pipeline, "models/modelo_top10.pkl")
    print(f"\n[treinar_modelo] Melhor modelo: {melhor_nome} (f1={melhor_f1})")
    print("[treinar_modelo] Salvo em models/modelo_top10.pkl")

    print("\n--- Comparativo dos 5 modelos ---")
    tabela = pd.DataFrame(resultados).set_index("modelo")
    print(tabela.to_string())


if __name__ == "__main__":
    treinar_modelo()
