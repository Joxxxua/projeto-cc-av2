# CLAUDE.md — F1 Top-10 Predictor (Projeto AV2 Ciência de Dados)

## Contexto do projeto

Projeto de **Ciência de Dados** que prevê se um piloto de Fórmula 1 termina a corrida no **top 10** (zona de pontos). A arquitetura segue o mesmo padrão de um projeto de referência (AgroPredict): pipeline de dados em etapas, rastreamento de experimentos com MLflow (SQLite), API de serving com FastAPI, containerização com Docker, e monitoramento de drift. Este projeto NÃO copia o AgroPredict — usa a mesma estrutura técnica para resolver um problema diferente, com dados, features, modelo e análise próprios.

## Problema de negócio (Business Understanding)

- **Objetivo:** prever se um piloto termina a corrida no top 10 (posição final ≤ 10), usando apenas informações conhecidas ANTES da corrida (nunca o resultado).
- **Variável-alvo:** `top10` — binária (1 = terminou em posição ≤ 10; 0 = fora). Problema de **classificação binária**.
- **Stakeholder hipotético:** analista de estratégia de uma equipe de F1 (ou casa de apostas esportivas), que quer estimar a probabilidade de pontuar antes da corrida começar.
- **Métrica primária:** F1-score e AUC-ROC. Como o dataset é balanceado (~50/50), a acurácia também é informativa, mas F1 e AUC contam a história completa.

## Dataset

- **Arquivo:** `data/raw/f1_all.parquet` (não versionado — baixar do Kaggle)
- **Fonte:** https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
- **Dimensões brutas:** 69.230 linhas × 29 colunas (uma linha = uma volta de um piloto)
- **Granularidade do modelo:** agregar para **uma linha por (raceId, driverId)** = 1.248 resultados de corrida
- **Temporadas:** 2021, 2022, 2023

### Construção do target (sem leakage)
- `position_y` é a **posição final** (constante por piloto/corrida). Use-a APENAS para derivar o target: `top10 = (position_y <= 10).astype(int)`.
- Resultado: ~50% top10, dataset balanceado.

## ⚠️ REGRA CRÍTICA — Prevenção de leakage

Estas colunas só existem DEPOIS ou DURANTE a corrida. **NUNCA usar como feature:**

| Coluna | Por quê é leakage |
|--------|-------------------|
| `position_y` | É o próprio target |
| `position_x` | Posição durante a corrida (muda volta a volta) |
| `status` | "Finished/Retired" só se sabe no fim |
| `milliseconds`, `lap`, `LapNumber` | Tempo e voltas são resultado da corrida |
| `time` | Tempo da volta |
| `TyreLife`, `Stint`, `Compound`, `FreshTyre` | Dependem do desenrolar da corrida |

**Features seguras (conhecidas antes da corrida):**

| Feature | Descrição | Tipo |
|---------|-----------|------|
| `grid` | Posição de largada (resultado da classificação) | numérico |
| `constructorId` | Equipe/construtor | categórico |
| `circuitId` | Circuito | categórico |
| `round` | Etapa do campeonato | numérico |
| `year` | Temporada | numérico |
| Clima agregado (`TrackTemp`, `AirTemp`, `Humidity`, `Rainfall`, `WindSpeed`) | Condições do dia — média da corrida | numérico |

> Atenção: o clima muda durante a corrida, mas as condições gerais do dia (média/previsão) são conhecidas antes. Use a **média por corrida** como aproximação defensável, e documente isso como decisão técnica.

## Stack tecnológica

- Python 3.11 (fixar por causa do Docker)
- ML: pandas, numpy, scikit-learn, joblib
- Produção: mlflow, fastapi, uvicorn, pydantic
- Visualização: matplotlib, seaborn
- Dados: pyarrow

## Estrutura de pastas alvo (espelha o AgroPredict)

```
f1-top10-predictor/
├── README.md
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── main.py                     # orquestra o pipeline
├── data/
│   ├── raw/                    # f1_all.parquet (não versionado)
│   ├── processed/              # parquet limpo
│   └── final/                  # parquet com features + target
├── models/
│   └── modelo_top10.pkl        # modelo serializado
├── notebooks/
│   └── 01_eda.ipynb            # EDA com 5+ visualizações
├── src/
│   ├── data/
│   │   ├── preparar_dados.py   # limpa o parquet bruto
│   │   └── build_dataset.py    # agrega para 1 linha por piloto-corrida
│   ├── features/
│   │   └── criar_features.py   # features seguras + target top10
│   ├── models/
│   │   ├── treinar_modelo.py   # treino + MLflow
│   │   └── avaliar_modelo.py   # matriz de confusão + classification report
│   ├── monitoring/
│   │   └── detectar_drift.py   # drift entre temporadas
│   └── api/
│       └── main.py             # FastAPI
├── mlflow.db                   # tracking SQLite (não versionar)
└── mlruns/                     # artifacts MLflow (não versionar)
```

## Modelo

- **Principal:** `RandomForestClassifier` (igual ao padrão do AgroPredict)
- **Para comparação (mínimo 3 modelos, requisito do CD):** RandomForest, LogisticRegression, e um terceiro (GradientBoosting ou KNN)
- Serializar o melhor com joblib em `models/modelo_top10.pkl`
- Logar todos no MLflow (experimento `F1Top10`)

## Regras de implementação

1. **Leakage é o risco número 1.** Só usar as features seguras listadas acima. Qualquer feature que dependa do resultado da corrida está proibida.
2. **Split temporal recomendado:** treinar em 2021-2022, testar em 2023. Isso evita que o modelo "veja o futuro" e conecta com o monitoramento de drift. Alternativamente, split aleatório com `random_state=42` (mais simples, aceitável).
3. **Métricas de classificação:** F1, Precision, Recall, AUC-ROC, matriz de confusão. Não reportar só acurácia.
4. **Reprodutibilidade:** `random_state=42` em tudo. Versões fixas no pyproject.
5. **MLflow em SQLite:** `mlflow.set_tracking_uri("sqlite:///mlflow.db")` e `set_experiment("F1Top10")`.
6. **Docker funcional:** build e run sem ajuste manual. Treinar o modelo antes de buildar.

## Critérios de aceite

- [ ] Pipeline roda de ponta a ponta via main.py
- [ ] Target top10 construído sem leakage (só features pré-corrida)
- [ ] EDA com 5+ visualizações
- [ ] Mínimo 3 modelos comparados no MLflow
- [ ] Validação cruzada com média e desvio-padrão
- [ ] Métricas de classificação (F1, AUC, matriz de confusão)
- [ ] API FastAPI com /predict funcional
- [ ] Dockerfile que builda e roda
- [ ] Relatório de drift entre temporadas
- [ ] README com instruções e resultado principal