# Série: Previsão de Preços de Imóveis
# 2. Otimização do Modelo — Busca de Hiperparâmetros e Seleção de Variáveis

Extensão da solução base para a competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) do Kaggle, adicionando otimização bayesiana de hiperparâmetros (Optuna) e otimização multiobjetivo de seleção de features (NSGA-II).

---

## Resultados

### Parte A — Otimização Bayesiana

| Método | RMSLE CV (5-fold) | RMSLE Kaggle | Melhoria Kaggle |
|---|---|---|---|
| LightGBM + GridSearchCV (baseline) | 0.13067 | 0.12949 | — |
| LightGBM + Optuna TPE (80 trials) | **0.12640** | **0.12436** | **−3.96%** |

A melhoria de −3.96% no score público foi obtida apenas substituindo o tuning de hiperparâmetros — sem nenhuma mudança no modelo ou nas features. O Optuna explorou um espaço contínuo de 8 hiperparâmetros com 80 trials, enquanto o GridSearchCV avaliou 16 combinações fixas. O alinhamento entre CV (−3.27%) e Kaggle (−3.96%) confirma que o modelo não sofreu overfitting ao processo de tuning.

### Parte B — Otimização Multiobjetivo

| Métrica | Valor |
|---|---|
| Features totais (após encoding) | 240 |
| Features no ponto de joelho (NSGA-II) | **83** |
| Features eliminadas | 157 (65% do total) |
| Custo de performance na redução | +0.00270 de RMSLE |

O NSGA-II revelou que 65% das features podem ser eliminadas com impacto mínimo de +0.00270 de RMSLE — evidência de que o sinal preditivo está concentrado em ~83 variáveis e que a maioria das colunas geradas pelo one-hot encoding carrega informação redundante.

---

## Estrutura

```
house-prices-bayesian-multiobjective/
├── house_prices_bayesian_multiobjective.ipynb
├── README.md
├── submission_optuna.csv
```

---

## Etapas do Projeto

### Parte A — Otimização Bayesiana com Optuna

#### Por que substituir o GridSearchCV?

O GridSearchCV avalia uma grade fixa de forma cega — não aprende com os resultados anteriores e só encontra o ótimo se ele estiver dentro da grade definida. O Optuna com **TPE (Tree-structured Parzen Estimator)** constrói um modelo probabilístico a cada trial, concentrando a busca em regiões promissoras do espaço de hiperparâmetros.

```
Trial 1  →  avalia  →  atualiza P(bom | hiperparâmetros)
Trial 2  →  amostra de P(bom)  →  avalia  →  atualiza
   ...
Trial N  →  concentra busca em regiões de alta probabilidade
```

#### Hiperparâmetros otimizados

| Hiperparâmetro | Range | Escala |
|---|---|---|
| `n_estimators` | 300 – 2000 | linear |
| `learning_rate` | 0.005 – 0.2 | **log** |
| `num_leaves` | 20 – 150 | linear |
| `min_child_samples` | 5 – 60 | linear |
| `subsample` | 0.5 – 1.0 | linear |
| `colsample_bytree` | 0.5 – 1.0 | linear |
| `reg_alpha` | 1e-4 – 10.0 | **log** |
| `reg_lambda` | 1e-4 – 10.0 | **log** |

`learning_rate`, `reg_alpha` e `reg_lambda` usam escala logarítmica porque variam em ordens de magnitude — a diferença entre 0.005 e 0.01 é tão relevante quanto entre 0.1 e 0.2.

#### Melhores parâmetros encontrados

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 1402 |
| `learning_rate` | 0.00534 |
| `num_leaves` | 96 |
| `min_child_samples` | 20 |
| `subsample` | 0.529 |
| `colsample_bytree` | 0.516 |

#### Resultado

- RMSLE CV (5-fold): **0.12640** vs. baseline 0.13067 (−3.27%)
- RMSLE Kaggle público: **0.12436** vs. baseline 0.12949 (−3.96%)
- Trials podados pelo `MedianPruner`: 5/80 (6%)

---

### Parte B — Otimização Multiobjetivo com NSGA-II

#### Motivação

A otimização monoobjetivo escolhe um único ponto no espaço de trade-offs sem revelar seu custo. Com 240 features após encoding, a pergunta natural é: *quantas são realmente necessárias?*

O **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) otimiza dois objetivos simultaneamente, mapeando a **Fronteira de Pareto** — o conjunto de soluções onde não é possível melhorar um objetivo sem piorar o outro.

#### Configuração do problema

| Elemento | Definição |
|---|---|
| Variáveis de decisão | Vetor binário de 240 posições (0 = exclui feature, 1 = inclui) |
| Objetivo f1 | Minimizar RMSLE (CV 3-fold) |
| Objetivo f2 | Minimizar número de features selecionadas |
| Hiperparâmetros | Fixados no ótimo da Parte A |
| Operadores | `BinaryRandomSampling` + `TwoPointCrossover` + `BitflipMutation` |
| População / Gerações | 40 × 20 (800 avaliações) |

#### Fronteira de Pareto

> RMSLE avaliado com CV 3-fold (reduzido para viabilidade computacional dentro do NSGA-II).
> Coluna "vs. Optuna completo" compara com o modelo Optuna usando todas as 240 features (RMSLE CV = 0.12640).

| Solução | Features | RMSLE (CV 3-fold) | vs. Optuna completo |
|---|---|---|---|
| Mais enxuta | 75 | 0.13150 | +0.00510 |
| 2 | 77 | 0.13042 | +0.00402 |
| 3 | 79 | 0.13036 | +0.00396 |
| 4 | 80 | 0.12972 | +0.00331 |
| **Ponto de joelho** | **83** | **0.12910** | **+0.00270** |
| 6 | 92 | 0.12886 | +0.00246 |
| Mais precisa | 93 | 0.12855 | +0.00215 |

#### Quando usar cada solução

| Cenário | Solução | Justificativa |
|---|---|---|
| Produção com restrição de latência | 75 features | Inferência mais rápida e menor custo de serving |
| Modelo sujeito a auditoria ou regulação | **83 features** | Interpretável sem grande perda de precisão |
| Pesquisa ou competição | 93 features | RMSLE máxima é o único critério |

O ponto de joelho elimina **157 features (65% do total)** com custo de apenas **+0.00270 de RMSLE** — evidência de que o sinal preditivo está concentrado em ~83 variáveis, e que a maioria das colunas geradas pelo one-hot encoding carrega informação redundante.

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-4C9BE8?style=flat)
![pymoo](https://img.shields.io/badge/pymoo-4CA86B?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

*Juliana Burato — 2026*
