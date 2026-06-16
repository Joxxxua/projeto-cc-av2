"""FastAPI app para servir o modelo F1 Top-10 Predictor."""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

app = FastAPI(
    title="F1 Top-10 Predictor",
    description="Prevê se um piloto de F1 termina a corrida no top 10 (zona de pontos).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_modelo = None


def get_modelo():
    global _modelo
    if _modelo is None:
        try:
            _modelo = joblib.load("models/modelo_top10.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Modelo não encontrado. Execute treinar_modelo() primeiro.")
    return _modelo


class EntradaPredicao(BaseModel):
    grid: int
    round: int
    year: int
    track_temp: float
    air_temp: float
    humidity: float
    rainfall: float
    wind_speed: float
    constructor_id: int
    circuit_id: int


@app.get("/")
def root():
    return {
        "api": "F1 Top-10 Predictor",
        "versao": "1.0.0",
        "descricao": "Prevê se um piloto termina a corrida no top 10 usando dados pré-corrida.",
        "endpoints": {
            "POST /predict": "Retorna previsão top10 e probabilidade",
            "GET /health": "Status do modelo carregado",
            "GET /mlflow/experimentos": "Lista runs do experimento F1Top10",
        },
    }


@app.get("/health")
def health():
    try:
        modelo = get_modelo()
        tipo = type(modelo.named_steps["classifier"]).__name__
        return {"status": "ok", "modelo_carregado": True, "tipo_classificador": tipo}
    except HTTPException:
        return {"status": "degraded", "modelo_carregado": False}


@app.post("/predict")
def predict(entrada: EntradaPredicao):
    modelo = get_modelo()

    # Nomes de coluna EXATOS do treino — capitalização obrigatória
    X = pd.DataFrame([{
        "grid":          entrada.grid,
        "round":         entrada.round,
        "year":          entrada.year,
        "TrackTemp":     entrada.track_temp,
        "AirTemp":       entrada.air_temp,
        "Humidity":      entrada.humidity,
        "Rainfall":      entrada.rainfall,
        "WindSpeed":     entrada.wind_speed,
        "constructorId": entrada.constructor_id,
        "circuitId":     entrada.circuit_id,
    }])

    top10 = int(modelo.predict(X)[0])
    probabilidade = round(float(modelo.predict_proba(X)[0][1]), 4)

    if top10 == 1:
        explicacao = (
            f"O modelo prevê que o piloto TERMINA no top 10 "
            f"(probabilidade: {probabilidade:.1%}). "
            f"Largando da posição {entrada.grid}, equipe {entrada.constructor_id}, "
            f"etapa {entrada.round} de {entrada.year}."
        )
    else:
        explicacao = (
            f"O modelo prevê que o piloto NÃO termina no top 10 "
            f"(probabilidade de pontuar: {probabilidade:.1%}). "
            f"Largando da posição {entrada.grid}, equipe {entrada.constructor_id}, "
            f"etapa {entrada.round} de {entrada.year}."
        )

    return {"top10": top10, "probabilidade": probabilidade, "explicacao": explicacao}


@app.get("/mlflow/experimentos")
def mlflow_experimentos():
    try:
        client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
        experimento = client.get_experiment_by_name("F1Top10")
        if experimento is None:
            return {"erro": "Experimento F1Top10 não encontrado."}

        runs = client.search_runs(
            experiment_ids=[experimento.experiment_id],
            order_by=["metrics.f1 DESC"],
        )

        resultado = []
        for run in runs:
            resultado.append({
                "run_id": run.info.run_id[:8],
                "nome": run.info.run_name,
                "status": run.info.status,
                "metricas": {k: round(v, 4) for k, v in run.data.metrics.items()},
                "params": run.data.params,
            })

        return {"experimento": "F1Top10", "total_runs": len(resultado), "runs": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
