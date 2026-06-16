# F1 Top-10 Predictor

Prevê se um piloto de Fórmula 1 termina a corrida no **top 10 (zona de pontos)**, usando apenas informações conhecidas **antes da largada** — sem leakage de dados de corrida.

Projeto de Ciência de Dados (AV2) com pipeline completo: limpeza → features → treino → MLflow → API FastAPI → Docker → monitoramento de drift.

---

## Problema e stakeholder

**Objetivo:** classificação binária — `top10 = 1` se posição final ≤ 10, `top10 = 0` caso contrário.

**Stakeholder hipotético:** analista de estratégia de equipe de F1 ou analista de apostas esportivas que precisa estimar a probabilidade de pontuar *antes* da corrida começar.

**Métricas principais:** F1-score e AUC-ROC. Dataset balanceado (~50/50), então acurácia também é informativa.

---

## Dataset

Fonte: [Kaggle — Formula 1 Dataset with Weather and Tyre Features](https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features)

Baixe o arquivo `f1_all.parquet` e coloque em:
```
data/raw/f1_all.parquet
```

**Dimensões brutas:** 69.230 linhas × 29 colunas (uma linha = uma volta de um piloto).
**Após agregação:** 1.247 resultados de corrida (2021–2023).

---

## Instalação

Requer Python 3.11.

```bash
pip install .
```

---

## Como rodar o pipeline completo

```bash
python main.py
```

Executa em ordem:
1. `preparar_dados()` — limpeza do parquet bruto
2. `build_dataset()` — agrega volta → resultado de corrida
3. `criar_features()` — cria target `top10` e features finais
4. `treinar_modelo()` — treina 5 modelos, loga no MLflow, salva o melhor
5. `avaliar_modelo()` — validação cruzada + matriz de confusão

Artefatos gerados:
- `models/modelo_top10.pkl` — melhor modelo serializado
- `mlflow.db` — tracking de experimentos
- `experiments/avaliacao.txt` — relatório de avaliação

---

## Ver experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse `http://localhost:5000` → experimento **F1Top10** → 5 runs comparados.

---

## Subir a API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa: `http://localhost:8000/docs`

### Exemplo de predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "grid": 3,
    "round": 5,
    "year": 2023,
    "track_temp": 35.0,
    "air_temp": 28.0,
    "humidity": 45.0,
    "rainfall": 0.0,
    "wind_speed": 12.0,
    "constructor_id": 9,
    "circuit_id": 14
  }'
```

Resposta:
```json
{
  "top10": 1,
  "probabilidade": 0.923,
  "explicacao": "O modelo prevê que o piloto TERMINA no top 10 (probabilidade: 92.3%)..."
}
```

### Outros endpoints

| Endpoint | Descrição |
|----------|-----------|
| `GET /` | Descrição da API |
| `GET /health` | Status do modelo carregado |
| `GET /mlflow/experimentos` | Lista runs do experimento F1Top10 |

---

## Rodar com Docker

> O modelo deve ser treinado antes do build (`python main.py`).

```bash
# Build
docker build -t f1-top10-predictor .

# Run
docker run -d --name f1api -p 8000:8000 f1-top10-predictor

# Testar
curl http://localhost:8000/health

# Parar
docker stop f1api && docker rm f1api
```

---

## Monitoramento de drift

Compara distribuição das features entre 2021–2022 (referência) e 2023 (atual):

```bash
python -c "from src.monitoring.detectar_drift import calcular_drift; calcular_drift()"
```

Alerta se |Δ média| > 20%.

---

## Resultado principal

Modelos avaliados no conjunto de teste (20% estratificado, `random_state=42`):

| Modelo | Accuracy | F1 | Precision | Recall | AUC-ROC |
|--------|----------|----|-----------|--------|---------|
| RandomForest (100 árvores) | 0.948 | 0.949 | 0.938 | 0.960 | **0.994** |
| RandomForest (200 árvores) | 0.948 | 0.949 | 0.931 | 0.968 | 0.993 |
| LogisticRegression (C=1.0) | 0.880 | 0.882 | 0.868 | 0.896 | 0.930 |
| LogisticRegression (C=0.5) | 0.884 | 0.886 | 0.869 | 0.904 | 0.932 |
| **GradientBoosting** ✓ | **0.952** | **0.952** | **0.945** | **0.960** | 0.985 |

**Melhor modelo:** `GradientBoostingClassifier` (F1 = 0.952, salvo em `models/modelo_top10.pkl`).

Validação cruzada (StratifiedKFold, 10 folds): F1 médio = **0.947 ± 0.021**.

---

## Limitações

- **Features durante a corrida excluídas intencionalmente (anti-leakage):** posição durante a corrida (`position_x`), status de finalização (`status`), tempos de volta (`milliseconds`, `time`), estratégia de pneus (`Compound`, `TyreLife`, `Stint`, `FreshTyre`) e número de voltas (`lap`, `LapNumber`) foram descartados porque só existem *após* ou *durante* a corrida — usá-los seria data leakage.
- **Clima como média da corrida:** as variáveis climáticas são a média de todas as voltas, o que é uma aproximação. Na prática, a previsão do tempo pré-corrida seria usada.
- **IDs sem nome:** `constructorId` e `circuitId` são inteiros do dataset original; um enriquecimento com tabela de nomes melhoraria a interpretabilidade.
- **Drift detectado em 2023:** `WindSpeed` (+44,5%) e `Rainfall` (-20,3%) apresentaram drift significativo vs. 2021-2022. Recomenda-se re-treinar periodicamente.
- **Sem dados de 2024+:** o modelo foi treinado em 2021–2023. Desempenho em temporadas futuras deve ser monitorado.
