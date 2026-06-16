PYTHON  = py -3.11
UVICORN = $(PYTHON) -m uvicorn
MLFLOW  = $(PYTHON) -m mlflow

.PHONY: install pipeline train api mlflow drift docker-build docker-run test help

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       Instala dependências (pip install .)"
	@echo "  make pipeline      Roda o pipeline completo (main.py)"
	@echo "  make train         Roda só o treino dos modelos"
	@echo "  make api           Sobe a API FastAPI na porta 8000"
	@echo "  make mlflow        Abre o MLflow UI (http://localhost:5000)"
	@echo "  make drift         Roda o detector de drift entre temporadas"
	@echo "  make docker-build  Faz o build da imagem Docker"
	@echo "  make docker-run    Roda o container na porta 8000"
	@echo "  make test          Roda o smoke test completo"

install:
	$(PYTHON) -m pip install .

pipeline:
	$(PYTHON) main.py

train:
	$(PYTHON) -c "import sys; sys.path.insert(0,'.'); from src.models.treinar_modelo import treinar_modelo; treinar_modelo()"

api:
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	$(MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --port 5000

drift:
	$(PYTHON) -c "import sys; sys.path.insert(0,'.'); from src.monitoring.detectar_drift import calcular_drift; calcular_drift()"

docker-build:
	docker build -t f1-top10-predictor .

docker-run:
	docker run -d --name f1api -p 8000:8000 f1-top10-predictor

test:
	powershell -ExecutionPolicy Bypass -File smoke_test.ps1
