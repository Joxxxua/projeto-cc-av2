# SPECS.md — Especificações por componente (F1 Top-10)

Cada spec descreve um módulo. Implementadas em ordem nos PROMPTS.md. A arquitetura espelha o AgroPredict, adaptada para classificação de top 10 em F1.

---

## SPEC-1 — preparar_dados.py (limpeza)

### Objetivo
Ler o parquet bruto, limpar e padronizar, salvar versão processada.

### Onde
`src/data/preparar_dados.py`

### Comportamento
1. Lê `data/raw/f1_all.parquet`
2. Converte `grid` de string para numérico (`pd.to_numeric`, errors='coerce')
3. Substitui `Compound == 'None'` por 'UNKNOWN' (embora Compound não vá pro modelo, manter consistência)
4. Remove linhas onde `position_y` é nulo (sem posição final, sem target)
5. Salva em `data/processed/f1_processed.parquet`

### Função
`preparar_dados()` → None (salva arquivo)

---

## SPEC-2 — build_dataset.py (agregação)

### Objetivo
Transformar os dados volta-a-volta em **uma linha por (raceId, driverId)** — granularidade de resultado de corrida.

### Onde
`src/data/build_dataset.py`

### Comportamento
1. Lê `data/processed/f1_processed.parquet`
2. Agrupa por `(raceId, driverId)` e agrega:
   - `position_final = position_y.first()` (constante na corrida)
   - `grid = grid.first()`
   - `year, round, circuitId, constructorId = first()`
   - Clima — **média da corrida**: `TrackTemp.mean()`, `AirTemp.mean()`, `Humidity.mean()`, `Rainfall.mean()`, `WindSpeed.mean()`
3. **NÃO incluir** nenhuma coluna de leakage (position_x, status, milliseconds, lap, TyreLife, Stint, Compound, time)
4. Salva em `data/processed/f1_race_results.parquet`

### Função
`build_dataset()` → None (salva arquivo)

### Resultado esperado
~1.248 linhas, balanceadas ~50/50 quando o target for criado.

---

## SPEC-3 — criar_features.py (features + target)

### Objetivo
Criar o target `top10` e as features finais de modelagem.

### Onde
`src/features/criar_features.py`

### Comportamento
1. Lê `data/processed/f1_race_results.parquet`
2. Cria o target: `top10 = (position_final <= 10).astype(int)`
3. Descarta `position_final` (não pode ser feature — vira leakage)
4. Features finais:
   - Numéricas: `grid`, `round`, `year`, `TrackTemp`, `AirTemp`, `Humidity`, `Rainfall`, `WindSpeed`
   - Categóricas: `constructorId`, `circuitId` (tratar como categóricas, não numéricas — IDs não têm ordem)
5. Opcional (feature engineering extra, valorizado): criar `grid_penalizado` (flag grid > 15), ou agregar histórico do construtor
6. Salva em `data/final/f1_dataset_final.parquet`

### Função
`criar_features()` → None (salva arquivo)

---

## SPEC-4 — treinar_modelo.py (treino + MLflow)

### Objetivo
Treinar 3+ modelos de classificação, logar tudo no MLflow, salvar o melhor.

### Onde
`src/models/treinar_modelo.py`

### Comportamento
1. Lê `data/final/f1_dataset_final.parquet`
2. Separa X (features) e y (`top10`)
3. **Split:** train_test_split com `random_state=42, stratify=y` (ou split temporal: 2021-2022 treino, 2023 teste — preferível, documentar a escolha)
4. Pré-processamento dentro de um Pipeline/ColumnTransformer:
   - Numéricas: StandardScaler
   - Categóricas (constructorId, circuitId): OneHotEncoder(handle_unknown='ignore')
   - **fit só no treino** (sem leakage)
5. Treina 3 modelos: `RandomForestClassifier`, `LogisticRegression(max_iter=1000)`, `GradientBoostingClassifier`
6. Para cada modelo, abre `mlflow.start_run(run_name=...)` e loga:
   - params (hiperparâmetros)
   - métricas: accuracy, f1, precision, recall, roc_auc (no teste)
   - o modelo: `mlflow.sklearn.log_model(pipeline, "modelo")`
7. MLflow config: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, `set_experiment("F1Top10")`
8. Salva o melhor (maior F1) em `models/modelo_top10.pkl` via joblib

### Função
`treinar_modelo()` → None

### Para 5+ experimentos
Adicionar variações: `RandomForest(n_estimators=200)`, `LogisticRegression(C=0.5)` — totalizando 5 runs.

---

## SPEC-5 — avaliar_modelo.py (diagnóstico)

### Objetivo
Validação cruzada, matriz de confusão e classification report.

### Onde
`src/models/avaliar_modelo.py`

### Comportamento
1. Carrega `models/modelo_top10.pkl` e o dataset final
2. Validação cruzada (StratifiedKFold, 10 folds) no conjunto de treino → média e desvio do F1
3. No teste: matriz de confusão + classification_report (precision/recall/f1 por classe)
4. Imprime tudo no stdout e salva um resumo em `experiments/avaliacao.txt`

### Função
`avaliar_modelo()` → None

---

## SPEC-6 — detectar_drift.py (monitoramento)

### Objetivo
Detectar drift de features entre temporadas (referência vs. atual).

### Onde
`src/monitoring/detectar_drift.py`

### Comportamento
1. Lê `data/final/f1_dataset_final.parquet`
2. Referência = 2021+2022; atual = 2023
3. Para cada feature numérica, calcula a diferença percentual da média entre os dois períodos
4. Alerta "DRIFT DETECTADO" se |Δ| > 20% (mesma lógica do AgroPredict)
5. Imprime relatório no stdout
6. (Opcional, valorizado) gerar um HTML com Evidently se preferir uma versão mais robusta

### Função
`calcular_drift()` → None

---

## SPEC-7 — api/main.py (FastAPI)

### Objetivo
Servir o modelo via API REST.

### Onde
`src/api/main.py`

### Comportamento
1. No startup, carregar `models/modelo_top10.pkl` (cached)
2. Modelo Pydantic de entrada com as features seguras:
   - `grid: int`, `round: int`, `year: int`
   - `track_temp: float`, `air_temp: float`, `humidity: float`, `rainfall: float`, `wind_speed: float`
   - `constructor_id: int`, `circuit_id: int`
3. Endpoints (espelhando o AgroPredict):
   - `GET /` — status da API
   - `GET /health` — status do modelo e dataset
   - `POST /predict` — recebe as features, monta DataFrame com nomes EXATOS de coluna, chama `pipeline.predict()` e `predict_proba()`, retorna `{"top10": 0/1, "probabilidade": float, "explicacao": str}`
   - `GET /mlflow/experimentos` — lista runs do experimento F1Top10 (via MlflowClient)
   - `GET /drift` — recalcula drift em memória
4. CORS liberado para localhost (caso queira frontend depois)

### Detalhe crítico
Os nomes das colunas no DataFrame de predição devem ser EXATAMENTE os mesmos usados no treino (grid, round, year, TrackTemp, AirTemp, Humidity, Rainfall, WindSpeed, constructorId, circuitId). Se divergir, o pipeline quebra.

### Exemplo de uso (README)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"grid": 3, "round": 5, "year": 2023, "track_temp": 35.0, "air_temp": 28.0, "humidity": 45.0, "rainfall": 0.0, "wind_speed": 12.0, "constructor_id": 9, "circuit_id": 14}'
```

---

## SPEC-8 — pyproject.toml + Dockerfile

### pyproject.toml
- `[project]`: name="f1-top10-predictor", version="1.0.0", requires-python=">=3.11"
- Deps: pandas, numpy, scikit-learn, joblib, mlflow, fastapi, uvicorn[standard], pydantic, matplotlib, seaborn, pyarrow
- Versões maduras e compatíveis (pandas>=2.0,<2.3; scikit-learn>=1.3,<1.6; numpy>=1.24,<2.0)

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY src/ ./src/
COPY models/ ./models/
COPY mlflow.db .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Atenção
- Treinar o modelo (gera modelo_top10.pkl e mlflow.db) ANTES de buildar.
- Versão do scikit-learn no Docker = versão do treino, senão o .pkl não carrega.

---

## Ordem de implementação

1. SPEC-8 parte 1 (pyproject.toml) — ambiente
2. SPEC-1, SPEC-2, SPEC-3 — pipeline de dados (limpa → agrega → features+target)
3. SPEC-4 — treino + MLflow
4. SPEC-5 — avaliação
5. SPEC-7 — API
6. SPEC-8 parte 2 (Dockerfile)
7. SPEC-6 — drift
8. EDA notebook + README