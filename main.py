import sys
import time

sys.path.insert(0, ".")

from src.data.preparar_dados import preparar_dados
from src.data.build_dataset import build_dataset
from src.features.criar_features import criar_features
from src.models.treinar_modelo import treinar_modelo
from src.models.avaliar_modelo import avaliar_modelo


def _step(n, total, descricao):
    print(f"\n{'='*58}")
    print(f"  Etapa {n}/{total}: {descricao}")
    print(f"{'='*58}")


if __name__ == "__main__":
    inicio = time.time()
    print("F1 Top-10 Predictor — Pipeline completo")

    _step(1, 5, "Limpeza dos dados brutos")
    preparar_dados()

    _step(2, 5, "Agregacao volta -> resultado de corrida")
    build_dataset()

    _step(3, 5, "Engenharia de features e target top10")
    criar_features()

    _step(4, 5, "Treino dos modelos + MLflow")
    treinar_modelo()

    _step(5, 5, "Avaliacao do melhor modelo")
    avaliar_modelo()

    elapsed = time.time() - inicio
    print(f"\n{'='*58}")
    print(f"  Pipeline concluido em {elapsed:.1f}s")
    print(f"  Modelo salvo em: models/modelo_top10.pkl")
    print(f"  Avaliacao em:    experiments/avaliacao.txt")
    print(f"  MLflow:          mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  API:             uvicorn src.api.main:app --port 8000")
    print(f"{'='*58}\n")
