# Monitoramento de Modelo de Credit Scoring — Tech Challenge Fase 4

Camada de sustentação e confiabilidade para um modelo de risco de crédito em produção: contratos de dados que impedem a entrada de dados inválidos, simulação e detecção estatística de drift, monitoramento de desempenho com rótulos atrasados, auditoria de viés e documentação de governança (LGPD).


## Sumário

- [Contexto do problema](#contexto-do-problema)
- [Arquitetura](#arquitetura)
- [Dataset e análise exploratória](#dataset-e-análise-exploratória)
- [Etapa 1 — Contratos de dados e modelo baseline](#etapa-1--contratos-de-dados-e-modelo-baseline)
- [Etapa 2 — Simulação e detecção de drift](#etapa-2--simulação-e-detecção-de-drift)
- [Etapa 3 — Observabilidade](#etapa-3--observabilidade)
- [Etapa 4 — Governança e LGPD](#etapa-4--governança-e-lgpd)
- [Análise causal da degradação](#análise-causal-da-degradação)
- [Testes e qualidade de código](#testes-e-qualidade-de-código)
- [Como executar](#como-executar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Decisões de design registradas](#decisões-de-design-registradas)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Vídeo](#vídeo)

---

## Contexto do problema

Uma fintech de concessão de crédito tem um modelo de Credit Scoring em produção. Após mudanças econômicas (inflação, novos perfis de clientes), a diretoria suspeita que o modelo esteja tomando decisões erradas e sofrendo **degradação silenciosa**: o modelo continua respondendo normalmente, sem erros técnicos, mas suas decisões deixam de refletir o risco real.

O foco deste projeto não é obter o melhor modelo, e sim construir a camada que **detecta, mede e explica** essa degradação:

1. Impedir que dados inválidos cheguem ao modelo (contratos de dados).
2. Simular mudanças econômicas de forma controlada e reproduzível.
3. Detectar estatisticamente mudanças nas entradas (Data Drift).
4. Detectar, com rótulos, mudanças na relação entre dados e risco (Concept Drift).
5. Documentar privacidade, base legal, vieses e a causa da degradação.

---

## Arquitetura

```
application_train.csv (Kaggle)
          │
          ▼
   data/split.py ──────────► reference_raw.csv (80%)      production_raw.csv (20%)
          │                         │                              │
          │                         ▼                              ▼
          │                 data/contract.py ◄──────── monitoring/simulate.py
          │               (bloqueio + alertas)          ├─ production_drifted.csv (Data Drift)
          │                         │                   └─ production_concept.csv (Concept Drift)
          │                         ▼                              │
          │                 data/features.py                       │
          │                         │                              │
          │                         ▼                              ▼
          │               modeling/train.py ──► modelo + limiar ──► monitoring/predict.py
          │               baseline_metrics.json   congelado        (contrato → features → score → decisão)
          │                         │                               │
          │                         ▼               ┌───────────────┴───────────────────┐
          │           governance/bias_audit.py      ▼                                   ▼
          │           (gênero e faixa etária)  monitoring/drift.py          monitoring/performance.py
          │                                   PSI/KS + Evidently            calibração e desempenho
          │                                   (sem rótulos: "no dia")       ("meses depois")
          │                                                 │
          │                                                 ▼
          │                              observability/  logs JSONL + MLflow
          │                              dashboard HTML + métricas Prometheus
          ▼
     reports/  (somente resultados agregados, versionados)
```

O mesmo caminho **contrato → features → modelo → decisão** é aplicado à referência e a todos os lotes de produção. Toda estatística aprendida (limites de faixas, limiar de decisão, faixas do PSI, limite de comprometimento) é calculada **uma vez na referência e congelada**: recalculá-la na produção esconderia exatamente o drift que se quer detectar.

---

## Dataset e análise exploratória

**Fonte:** Home Credit Default Risk (Kaggle), tabela `application_train.csv`.

| Característica | Valor |
|---|---|
| Linhas × colunas | 307.511 × 122 |
| Alvo | `TARGET` (1 = inadimplente) |
| Inadimplência | 8,07% (classes desbalanceadas) |

**Por que este dataset:** atende ao mínimo de 5.000 amostras, tem alvo binário pronto, uma variável que permite calcular a idade (`DAYS_BIRTH`, exigida pela regra de maioridade) e variáveis de renda e emprego, naturais para simular uma crise econômica. O German Credit Data, também sugerido no enunciado, foi descartado por ter apenas 1.000 amostras. O Lending Club tem datas reais de emissão, mas não tem idade e exigiria binarizar e filtrar o alvo.

A EDA (`eda/eda_reference.ipynb`) foi feita **somente na referência**, para que nenhuma decisão de limpeza usasse informação da produção. Principais decisões, todas com evidência no notebook:

| Tema | Achado | Decisão |
|---|---|---|
| **Imóvel** (47 colunas `_AVG`, `_MODE`, `_MEDI`) | 59% a 70% de nulos; versões redundantes (correlação > 0,97, medida no grupo `COMMONAREA`); valor com sinal fraco (≈ −0,016). Mas a **ausência** é informativa: 9,25% de inadimplência sem informação × 7,01% com informação | As 47 colunas viram um indicador, `HOUSING_INFO_MISSING` |
| **Carro** | Nulos estruturais: todos os 162.413 clientes sem carro têm idade do carro nula. 5 clientes com carro e idade nula (inconsistência). A idade do carro separa mais o risco que a posse (5,7% → 9,4% entre faixas × 7,3% vs. 8,5% pela posse). Picos suspeitos em 64 anos (1.942) e 65 anos (694), com apenas 2 em 63 | `CAR_AGE_GROUP` com faixas fixas e categoria `NO_CAR`; regra de coerência no contrato; alerta para carro acima de 60 anos |
| **Emprego** | O valor 365.243 (~1.000 anos) é um código sentinela: 44.143 clientes (17,9%), todos aposentados (44.126) ou desempregados (17). Inadimplência cai com o tempo de emprego: 11,0% (0–1 ano) → 5,2% (10+ anos); `NOT_EMPLOYED` tem 5,4% | `EMPLOYMENT_GROUP` com faixas fixas e categoria `NOT_EMPLOYED`; regra de coerência no contrato |
| **Idade** | `DAYS_BIRTH` conta dias antes do pedido (valores negativos). Idade de 20,5 a 69,1 anos (mediana 43,1) | `AGE_YEARS = −DAYS_BIRTH / 365,25`; a coluna bruta é mantida para o contrato |
| **Renda** | Sem nulos; mediana 147.600; máximo 117 milhões (~800× a mediana), com contexto incoerente (trabalhador braçal, parcela de 0,02% da renda, inadimplente) | Não é bloqueado (é implausível, mas não impossível): vira **alerta** acima do p99,9 da referência (900.000) |
| **Gênero** | `CODE_GENDER` | Fora do modelo (não discriminação); mantido só para auditoria |

---

## Etapa 1 — Contratos de dados e modelo baseline

### Separação entre referência e produção

`data/split.py` divide o dataset em **80% referência** e **20% produção**, estratificado pelo `TARGET`, com `random_state=42`. A divisão é feita **antes** da EDA, para evitar vazamento de informação.

| Evidência | Resultado |
|---|---|
| Linhas | 246.008 + 61.503 = 307.511 |
| Inadimplência | 8,0729% × 8,0728% |
| IDs em comum | 0 |

A referência é o "passado" usado no treino; a produção é o "futuro", onde o drift é simulado. O conjunto de **teste** do modelo é uma divisão separada, dentro da referência.

### Contrato de dados (Pandera)

O contrato (`data/contract.py`) separa dois níveis de severidade, conforme a certeza sobre o erro:

- **Bloqueio** — o dado é certamente inválido: a linha vai para quarentena.
- **Alerta** — o dado é suspeito, mas pode ser real: a linha passa e é registrada para revisão.

E dois tipos de falha:

- **Falha de linha** (ex.: um cliente com 16 anos): bloqueia só aquela linha.
- **Falha estrutural** (coluna ausente, tipo errado, índice repetido): **rejeita o lote inteiro**, porque nenhuma linha dele é confiável.

**Regras de bloqueio**

| Regra | Colunas |
|---|---|
| Idade entre 18 e 115 anos | `DAYS_BIRTH` |
| ID único e não nulo | `SK_ID_CURR` |
| Sem registros duplicados (ignorando o ID e o `TARGET`, que não existe no dia da decisão) | todas as demais |
| Renda presente e maior que zero | `AMT_INCOME_TOTAL` |
| Crédito maior que zero | `AMT_CREDIT` |
| Scores externos entre 0 e 1 | `EXT_SOURCE_1/2/3` |
| Idade do carro ≥ 0; posse de carro em {Y, N} | `OWN_CAR_AGE`, `FLAG_OWN_CAR` |
| Nota da região em {1, 2, 3} | `REGION_RATING_CLIENT_W_CITY` |
| **Coerência:** tem carro ⇔ tem idade do carro | `FLAG_OWN_CAR`, `OWN_CAR_AGE` |
| **Coerência:** código 365.243 só para aposentados e desempregados; demais valores ≤ 0 | `DAYS_EMPLOYED`, `NAME_INCOME_TYPE` |
| `TARGET` em {0, 1}, quando presente | `TARGET` |
| Tipos declarados e colunas obrigatórias (todas as features do modelo) | — |

As regras de crédito, scores externos e nota da região protegem diretamente as **features que o modelo consome**: um contrato que cobrisse só as colunas da EDA deixaria passar, por exemplo, um score externo igual a 7.

**Regras de alerta** (calculadas apenas sobre as linhas aceitas)

| Regra | Limite |
|---|---|
| Renda muito alta | > 900.000 (p99,9 da referência, congelado) |
| Parcela maior que a renda | `AMT_ANNUITY > AMT_INCOME_TOTAL` |
| Idade do carro suspeita | > 60 anos (logo abaixo dos picos de 64 e 65) |
| Categoria desconhecida | `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_CONTRACT_TYPE` fora das categorias da referência (uma regra por coluna) |

A categoria desconhecida é **alerta**, e não bloqueio: pode ser um erro de digitação ou um produto novo legítimo. O modelo não quebra (`handle_unknown="ignore"`), mas a equipe precisa ser avisada. Já uma nota de região fora de {1, 2, 3} **bloqueia**: a escala é fechada, e o valor é certamente inválido.

**Evidência na referência**

| Resultado | Valor |
|---|---|
| Linhas bloqueadas | 5, todas por `car_consistency` (as mesmas 5 inconsistências encontradas na EDA) |
| Alertas | 2.640 carros acima de 60 anos (1.942 + 694 + 2 + 2, conferindo com a EDA); 222 rendas acima do limite; 25 parcelas acima da renda; 0 categorias desconhecidas |

São 222 alertas de renda, e não ~246 (0,1% de 246.008): a renda se concentra em valores redondos, e várias linhas ficam exatamente em 900.000, sem disparar a regra (`<=`).

**Evidência com um lote propositalmente inválido** — cópias de clientes válidos, com **um erro controlado por linha**:

| Caso | Bloqueado por |
|---|---|
| `age_under_18` | `in_range` (idade) |
| `duplicate_id` | `field_uniqueness` |
| `duplicate_record` | `no_duplicate_records` |
| `zero_income` | `greater_than(0)` |
| `missing_income` | `not_nullable` |
| `car_inconsistent` | `car_consistency` |
| `employment_inconsistent` | `employment_consistency` |
| `valid_1`, `valid_2` | aceitos |
| `car_age_alert` | aceito, com alerta `car_age_above_limit` |

Um lote **sem a coluna** `AMT_INCOME_TOTAL` é rejeitado por inteiro (`column_in_dataframe`), com 0 linhas aceitas.

A referência limpa é **definida** como as linhas que passam no contrato: a mesma regra que filtra a produção filtra o treino.

### Transformações

`data/features.py` aplica as decisões da EDA como **regras fixas**, sem estatísticas aprendidas, de modo idêntico à referência e à produção: `HOUSING_INFO_MISSING`, `CAR_AGE_GROUP`, `EMPLOYMENT_GROUP` e `AGE_YEARS`. Se as colunas do imóvel estiverem ausentes, a função falha explicitamente, em vez de marcar todos os clientes como "sem informação".

### Modelo baseline

**Regressão logística** (scikit-learn), escolhida por ser simples, rápida e **interpretável**: a contribuição de cada variável é calculável cliente a cliente, o que facilita explicar decisões (ver [art. 20 da LGPD](#3-base-legal-e-direitos-do-titular)).

- **Features (14):** `EXT_SOURCE_1/2/3`, `AGE_YEARS`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `REGION_RATING_CLIENT_W_CITY`, `HOUSING_INFO_MISSING`, `EMPLOYMENT_GROUP`, `CAR_AGE_GROUP`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_CONTRACT_TYPE`.
- **Fora do modelo:** `CODE_GENDER` (não discriminação, garantido por teste), identificador, colunas brutas substituídas por versões transformadas.
- **Pré-processamento dentro do `Pipeline`:** imputação pela mediana, padronização e one-hot são aprendidos **somente no treino**.
- **Divisão:** treino 196.802, teste 49.201 (estratificado, dentro da referência).

| Métrica (teste) | Valor | Referência |
|---|---|---|
| ROC-AUC | 0,7323 | Aleatório = 0,5; Gini = 0,46 |
| PR-AUC | 0,2054 | Aleatório = 0,0807 (≈ 2,5×) |

A acurácia não é usada: com 8% de inadimplentes, um modelo que aprova todos acertaria 92%.

### Política de decisão

O modelo devolve uma probabilidade; a decisão exige um limiar. Ele é calculado **no treino** para aprovar **90%** dos clientes e depois **congelado**. Comparação no teste:

| Aprovação | Limiar | Inadimplência dos aprovados | Inadimplentes barrados |
|---|---|---|---|
| 80% | 0,1180 | 5,27% | 47,8% |
| 85% | 0,1388 | 5,73% | 39,6% |
| **90%** | **0,1695** | **6,29%** | **29,8%** |
| 95% | 0,2248 | 7,00% | 17,6% |
| 99% | 0,3559 | 7,79% | 4,5% |

**Premissa de negócio:** recusar um bom cliente foi tratado como o erro mais caro. Acima de 90%, cada 5 pontos de aprovação a mais deixam passar mais risco que o passo anterior (+0,56 p.p. de 85→90%; +0,71 p.p. de 90→95%). A escolha é uma premissa informada por esse trade-off, e não um ótimo matemático; o dataset não informa juros nem perdas reais. Na política de 90%, cerca de 3 em cada 4 recusados pagariam, coerente com uma PR-AUC de 0,2. Uma instituição focada em reduzir perdas priorizaria barrar inadimplentes (recall).

Artefatos: `reports/baseline_metrics.json` (métricas, limiar, features, data de treino).

---

## Etapa 2 — Simulação e detecção de drift

### Cenários simulados

`monitoring/simulate.py` gera, a partir da produção, três lotes, todos com semente fixa:

| Cenário | Entradas | Risco real | O que demonstra |
|---|---|---|---|
| **Controle** | iguais | igual | O detector não inventa drift |
| **Data Drift** | mudam | acompanha o perfil | O detector vê a mudança; o modelo erra fora da região que conhecia |
| **Concept Drift** | **iguais** | **muda** | O detector de entradas não vê nada, mas o modelo piora |

**Data Drift — uma crise com inflação e desemprego**

| Mecanismo | Parâmetro | Efeito medido |
|---|---|---|
| Reajustes salariais desiguais | renda × U(1,0; 1,4) por cliente (+20% em média) | Mediana da renda: 146.250 → 176.660 |
| Inflação de preços | +20% em crédito, parcela e preço do bem | Mediana da parcela: 24.939 → 29.927 |
| Desemprego | 15% dos `Working` viram `Unemployed` (tipo de renda **e** código de emprego mudam juntos, para o contrato aceitar) | 4.759 novos desempregados |

Os rótulos dos novos desempregados seguem a inadimplência dos desempregados da referência (29,4%), **preservando a estrutura original**: quem já era inadimplente continua; parte dos bons pagadores passa a inadimplente. A inadimplência desse grupo foi de 9,75% para 29,40%, e a da produção, de 8,07% para 9,59%.

> **Classificação correta:** isso é *covariate shift* para uma região pouco coberta pelo treino (apenas 17 desempregados), combinado com uma má especificação do modelo (`NOT_EMPLOYED` mistura aposentados e desempregados). Não é uma mudança em P(Y|X): o desempregado simulado tem o mesmo risco do desempregado da referência.

**Premissa com incerteza explícita:** a taxa de 29,4% vem de 5 inadimplentes em 17 clientes. O intervalo de confiança de 95% vai aproximadamente de 13% a 53%; é uma premissa de cenário, não uma estimativa precisa.

**Concept Drift — mesmo perfil, mais risco**

Clientes com **comprometimento de renda alto** (parcela/renda no top 20% da referência, limite 0,2467, congelado) passam a ter a inadimplência **dobrada**, sem nenhuma alteração nas features.

| Medida | Valor |
|---|---|
| Clientes no grupo | 12.306 |
| Inadimplência do grupo | 8,10% → 16,20% |
| Inadimplência da produção | 8,07% → 9,69% |
| `features_changed` | `false` (provado pelo script) |

**Artefato verificado:** numa versão anterior, os rótulos eram sorteados ao acaso, o que torna a ordenação dentro do grupo impossível por construção. Com os rótulos preservando a estrutura, a ROC-AUC do Data Drift passou de 0,6978 para 0,7105: cerca de um terço da queda anterior era artefato da simulação.

Artefatos: `reports/simulation_summary.json`.

### Predições na produção

`monitoring/predict.py` aplica a cada lote o mesmo caminho da referência (contrato → features → modelo) e decide com o **limiar congelado** de 0,1695.

| Lote | Aprovação | Inadimplência | Inadimplência dos aprovados | ROC-AUC |
|---|---|---|---|---|
| Referência (teste) | 90,09% | 8,07% | 6,29% | 0,7323 |
| Controle | 90,12% | 8,07% | 6,23% | 0,7356 |
| Data Drift | 89,85% | 9,59% | 7,68% | 0,7105 |
| Concept Drift | 90,12% | 9,69% | 7,92% | 0,6953 |

**A degradação silenciosa, medida:** nos dois cenários de crise, a aprovação ficou praticamente igual (≈ 90%), enquanto a inadimplência entre os aprovados subiu cerca de **23% a 27%**. O modelo continuou decidindo como antes.

Os alertas do contrato também reagiram, sem precisar de rótulos: `income_above_threshold` passou de 56 (controle) para 162 (Data Drift). O alerta `car_age_above_limit`, de uma variável não simulada, ficou em 699 nos três lotes.

### Detecção de Data Drift (PSI e KS)

`monitoring/drift.py` compara a referência (o conjunto de teste do baseline, com scores fora da amostra) com cada lote, usando **apenas informação disponível no dia da decisão**: as features do modelo e o score (`RISK_SCORE`). O `TARGET` não é usado.

**Método**

- **PSI** (métrica de decisão): faixas pelos decis da referência, congeladas, mais uma faixa para valores ausentes. Níveis: < 0,10 estável; 0,10–0,25 moderado; > 0,25 significativo (convenção de mercado em credit scoring).
- **KS** (contexto estatístico): só em variáveis contínuas. As discretas (`REGION_RATING_CLIENT_W_CITY`, `HOUSING_INFO_MISSING`) são tratadas como categóricas.
- **Evidently** (`DataDriftPreset`, método PSI): relatório visual. Os valores podem diferir ligeiramente dos da tabela, porque o Evidently define as faixas de outra forma; a tabela documenta o método exato usado.

**Resultados**

| Variável | Controle | Data Drift | Concept Drift |
|---|---|---|---|
| `NAME_INCOME_TYPE` | 0,0005 | **0,5666 — significativo** | 0,0005 |
| `AMT_INCOME_TOTAL` | 0,0003 | **0,1714 — moderado** | 0,0003 |
| `AMT_CREDIT` | 0,0002 | **0,1467 — moderado** | 0,0002 |
| `AMT_ANNUITY` | 0,0003 | **0,1200 — moderado** | 0,0003 |
| `EMPLOYMENT_GROUP` | 0,0003 | 0,0392 — estável | 0,0003 |
| `RISK_SCORE` | 0,0003 | 0,0004 — estável | 0,0003 |
| Demais variáveis | ≤ 0,0006 | ≤ 0,0006 | ≤ 0,0006 |

Leituras:

- **Controle:** todas as variáveis estáveis, e todos os p-valores do KS acima de 0,05 (o menor é 0,062). O detector não gera falso alarme.
- **Data Drift:** as quatro variáveis simuladas aparecem no topo; nenhuma variável não alterada passa do limite.
- **Concept Drift:** a tabela é **idêntica** à do controle. O monitoramento de entradas não vê nada.
- **O score não se moveu** (PSI 0,0004 no Data Drift): a saída do modelo, sozinha, não revelaria a crise.
- **`NAME_INCOME_TYPE` × `EMPLOYMENT_GROUP`:** o mesmo evento aparece como significativo na coluna original e como estável na feature agrupada, porque os novos desempregados se diluem entre os 44 mil aposentados de `NOT_EMPLOYED`. Agrupar categorias pode esconder mudanças; por isso o monitoramento inclui colunas originais importantes.

**Significância estatística × relevância prática:** com dezenas de milhares de linhas e 15 testes, algum p-valor tende a cair abaixo de 0,05 por acaso. Numa execução anterior (com a referência inteira), o KS acusou "drift" em `EXT_SOURCE_1` no controle (p = 0,038), sem nenhuma alteração; com a referência atual, o mesmo teste deu p = 0,386. O PSI ficou abaixo de 0,001 nas duas versões. Por isso o PSI decide, e o KS é apenas contexto.

Artefatos: `reports/drift_report_control.html`, `drift_report_data_drift.html`, `drift_report_concept_drift.html`, `reports/drift_summary.json`.

### Detecção de Concept Drift (desempenho com rótulos)

Em crédito, os rótulos chegam **meses depois** da decisão. `monitoring/performance.py` mede o desempenho quando eles estão disponíveis. A métrica principal é o **erro de calibração**:

> erro de calibração = inadimplência real − risco médio previsto (positivo: o modelo subestima o risco)

| Lote | Inadimplência real | Risco previsto | **Erro de calibração** | PR-AUC ÷ inadimplência |
|---|---|---|---|---|
| Referência | 8,07% | 8,10% | −0,03 p.p. | 2,54 |
| Controle | 8,07% | 8,06% | +0,01 p.p. | 2,68 |
| Data Drift | 9,59% | 8,18% | **+1,42 p.p.** | 2,33 |
| Concept Drift | 9,69% | 8,06% | **+1,63 p.p.** | 2,23 |

No Concept Drift, o risco previsto é **idêntico** ao do controle (as entradas são as mesmas), mas a realidade ficou 1,6 ponto mais arriscada: a degradação silenciosa em um único número. O limite de alerta adotado é de 1 p.p.

A PR-AUC bruta parece **subir** nos cenários de drift (0,2161 → 0,2232), apenas porque a inadimplência subiu. Normalizada pela prevalência, ela cai, coerente com a ROC-AUC.

**Diagnóstico por segmento** (localiza o problema; não serve para detectá-lo, porque o segmento afetado não é conhecido de antemão):

| Lote | Segmento | Clientes | Real | Previsto | Erro |
|---|---|---|---|---|---|
| Data Drift | Novos desempregados | 4.764 | 29,43% | 9,47% | **+19,96 p.p.** |
| Concept Drift | Comprometimento alto | 12.306 | 16,20% | 8,16% | **+8,04 p.p.** |
| Referência | Comprometimento alto | 9.793 | 9,03% | 8,29% | +0,73 p.p. |

Segmentos com poucos clientes (ex.: os 3 a 5 desempregados preexistentes nos outros lotes) produzem taxas extremas por acaso e não são interpretados.

Artefatos: `reports/performance_report.html`, `reports/performance_summary.json`.

### Comparação dos sinais de monitoramento

| Sinal | Precisa de rótulos? | Data Drift | Concept Drift | Quando avisa |
|---|---|---|---|---|
| PSI das entradas e alertas do contrato | Não | ✅ detecta | ❌ silêncio | No dia |
| Score e taxa de aprovação | Não | ❌ silêncio | ❌ silêncio | No dia |
| Calibração e desempenho | Sim | ✅ detecta | ✅ detecta | Meses depois |

---

## Etapa 3 — Observabilidade

Foco da disciplina: visibilidade operacional dos pipelines, da infraestrutura e dos modelos. O PDF pede três coisas: **centralizar logs e métricas** (incluindo os testes de drift da Etapa 2), **rastrear o ciclo de vida** (MLflow) e **documentar as métricas de saúde**. Prometheus-client é opcional no enunciado; a stack escolhida é logs estruturados + MLflow + dashboard HTML (e um arquivo OpenMetrics para quem quiser raspar depois).

### Três camadas

| Camada | Onde | Pergunta que responde |
|---|---|---|
| Logs estruturados | `logs/pipeline.jsonl` (não versionado) | a tarefa de ingestão, treino, scoring ou drift **passou ou falhou**, em quanto tempo, com que volume? |
| Rastreamento | MLflow (SQLite local `mlflow.db`) | este modelo degradou? desde quando? com quais parâmetros foi treinado? |
| Saúde e alertas | `reports/observability_dashboard.html` | o pipeline está de pé **agora**, e qual lote disparou drift ou degradação silenciosa? |

Nenhuma camada substitui a outra. O dashboard e o MLflow leem o **mesmo** veredito (`reports/health_summary.json`), calculado em `observability/health.py`.

### Logs de execução

Cada módulo do pipeline emite `pipeline_step_started` / `pipeline_step_finished` (e `pipeline_step_failed` se quebrar). O formato no arquivo é JSON Lines, ingerível por Loki, CloudWatch ou Datadog sem parser próprio; o terminal continua legível.

Linhas bloqueadas pelo contrato **não vão para o log** com o `SK_ID_CURR` original. Elas vão para `data/quarantine/`, com o identificador **HMAC-SHA256** da Etapa 4 (`CREDIT_MONITORING_PSEUDONYM_KEY`). Rejeição estrutural do lote inteiro gera só um resumo (`reason: structural`), sem despejar as linhas.

### MLflow

Dois tipos de run no experimento `credit-scoring`, separados pela tag `run_type`:

- **`training`** — parâmetros, ROC-AUC / PR-AUC, política de 90%, modelo no Model Registry (`credit-scoring-baseline`).
- **`monitoring`** — um run por lote (`control`, `data_drift`, `concept_drift`) com PSI máximo, erro de calibração, linhas bloqueadas, flags de alerta e os HTML do Evidently como artefatos.

O backend é **SQLite** (o file store do MLflow 3 está em *maintenance mode* e o Registry exige banco). Falha de rastreamento **não derruba o pipeline**: o erro vira um warning no log e o job segue. Isso está coberto por teste.

### Métricas de saúde documentadas

| Métrica | Origem | O que vigia |
|---|---|---|
| `pipeline_task_status` / `duration_seconds` | logs | sucesso/falha e latência de cada tarefa |
| `rows_processed` | logs e `production_summary` | volume do lote |
| `blocked_rows` / `contract_alerts` | contrato | ingestão recusada vs. linhas suspeitas que passaram |
| `psi_max` e PSI por feature | Etapa 2 (sem rótulos) | mudança de distribuição no dia da decisão |
| `calibration_gap` | Etapa 2 (com rótulos) | modelo subestimando risco quando os rótulos chegam |
| `silent_degradation` | PSI estável **e** \|gap\| ≥ 1 p.p. | o alerta que nenhum monitor sozinho levanta |
| `roc_auc` vs baseline | desempenho − `baseline_metrics` | queda de ranking ≥ 0,03 |

Alertas gerados nos relatórios já existentes:

| Lote | O que o painel mostra |
|---|---|
| Controle | estável (falso alarme ausente) |
| Data Drift | `feature_drift` + `calibration_gap` (PSI de `NAME_INCOME_TYPE` significativo) |
| Concept Drift | **`silent_degradation`** + `calibration_gap` + `performance_drop` (PSI idêntico ao controle, inadimplência dos aprovados sobe) |

O arquivo `reports/metrics.prom` exporta as mesmas séries em OpenMetrics, para um node exporter ou Prometheus raspar se a equipe quiser integrar com a Fase 3. Não é obrigatório subir Grafana para cumprir o entregável: o dashboard HTML abre offline.

### Entregável

- Dashboard: `reports/observability_dashboard.html`
- Snapshot: `reports/health_summary.json`
- UI do MLflow: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

---

## Etapa 4 — Governança e LGPD

> Este é um projeto acadêmico, construído sobre um dataset público (Home Credit Default Risk, Kaggle). Os dados não são redistribuídos: ficam fora do repositório, e quem executa o projeto deve obtê-los diretamente da competição, respeitando os termos de uso do Kaggle. Nenhuma decisão real de crédito é tomada. As conclusões jurídicas abaixo descrevem como uma fintech real deveria agir e dependem de validação por especialistas em proteção de dados.

### 1. Dados tratados e classificação

O dataset não contém nome, CPF ou endereço, mas **contém dados pessoais**: idade, renda, emprego, família e bens descrevem pessoas reais. O identificador `SK_ID_CURR` é um **pseudônimo**: se ligado a uma tabela de cadastro, a pessoa volta a ser identificável. Pela LGPD, dado pseudonimizado continua sendo dado pessoal. O fato de o dataset ser público também não o retira do escopo da lei.

| Categoria | Exemplos | Tratamento |
|---|---|---|
| Identificador (pseudônimo) | `SK_ID_CURR` | Usado só para controle de duplicidade; não é feature do modelo |
| Dados pessoais financeiros e cadastrais | Renda, crédito, parcela, emprego, idade | Usados no modelo, de forma minimizada (faixas e indicadores) |
| Atributos com risco de discriminação | `CODE_GENDER`, idade, estado civil | Gênero fora do modelo, mantido só para auditoria; idade no modelo, sob auditoria |
| Possível inferência de dado sensível | Categoria `Maternity leave` | Pode revelar gestação; registrada como risco, a ser avaliada juridicamente |
| Dados sensíveis (art. 5º, II) | Nenhum identificado diretamente | — |

### 2. Privacy by Design e by Default

A proteção de dados foi aplicada em cada etapa do pipeline, e não ao final:

| Decisão | Efeito |
|---|---|
| O modelo usa 14 das 122 colunas; as 47 colunas do imóvel viraram um único indicador | Tratamento limitado ao necessário (art. 6º, III) |
| Faixas em vez de valores exatos (idade do carro, tempo de emprego) | Menos detalhe individual no modelo |
| Dados e modelos fora do repositório (`/data/` e `/models/` no `.gitignore`; bloqueio de arquivos grandes no pre-commit) | Nenhum registro de cliente é versionado, nem por engano |
| Saídas com registros individuais removidas dos notebooks | O conteúdo dos notebooks não expõe clientes |
| Predições linha a linha ficam em `data/`; só resumos agregados vão para `reports/` | Apenas números agregados são publicados |
| Testes automatizados com dados sintéticos | As regras são verificadas sem usar dados reais |
| `CODE_GENDER` fora das features, com teste que garante isso | A decisão de não discriminação não depende de memória |
| Pseudonimização do identificador com hash com chave secreta (`governance/privacy.py`) | Logs e quarentena não expõem o `SK_ID_CURR` real (ver abaixo) |

**Por que um hash com chave, e não um hash simples:** os IDs são números sequenciais, com cerca de 350 mil valores possíveis. Um SHA-256 simples seria revertido por **ataque de dicionário** (calcular o hash de todos os IDs possíveis e montar a tabela "hash → ID"). O HMAC-SHA256 usa uma chave secreta, lida de uma variável de ambiente e nunca escrita no código nem versionada; sem ela, a tabela não pode ser montada. Se a chave não estiver definida, o programa falha, em vez de usar uma chave fraca em silêncio. O resultado continua sendo **pseudonimização**, e não anonimização: quem tem a chave refaz a ligação, e o dado segue sendo pessoal pela LGPD.

### 3. Base legal e direitos do titular

**No projeto:** uso acadêmico de um dataset público, sem redistribuição dos dados e sem decisões sobre pessoas reais.

**Numa operação real**, a base legal mais adequada para o modelo de score seria a **proteção do crédito** (art. 7º, X), complementada pela execução de procedimentos preliminares ao contrato solicitado pelo cliente. O legítimo interesse seria uma alternativa, mas exigiria um teste de balanceamento documentado. O consentimento seria frágil: é revogável a qualquer momento e dificilmente é livre para quem precisa de crédito.

**Direito de revisão (art. 20):** o titular pode pedir a revisão de decisões tomadas unicamente por tratamento automatizado, incluindo decisões de crédito, e tem direito a informações sobre os critérios usados. A escolha da **regressão logística** favorece esse direito: a contribuição de cada variável é calculável cliente a cliente, o que permite listar os principais motivos de uma recusa. Duas ressalvas: os coeficientes estão em escala padronizada e precisam ser traduzidos em linguagem comum; e **explicar não é revisar** — uma operação real precisaria de um canal de contestação com reavaliação humana.

### 4. Mitigação de vieses

| Medida | Situação |
|---|---|
| Gênero fora do modelo (não discriminação, art. 6º, IX) | Implementado e protegido por teste |
| Gênero mantido fora das features, para auditoria | Implementado |
| Risco de **variáveis proxy** | Remover o gênero não garante um modelo justo: outras colunas podem carregar a mesma informação (ex.: `Maternity leave` revela, quase certamente, uma cliente mulher; estado civil, ocupação e número de dependentes podem fazer o mesmo de forma mais sutil) |
| Idade no modelo | Mantida por ter relação direta com o risco, mas auditada |

**Método de auditoria** (`governance/bias_audit.py`; conjunto de teste, limiar congelado):

1. **Inadimplência real por grupo** — os grupos têm riscos diferentes de fato?
2. **Taxa de aprovação por grupo** — são aprovados em proporções diferentes?
3. **Aprovação entre bons pagadores, por grupo** — pessoas que pagariam são tratadas igualmente? É a pergunta central: uma diferença na aprovação só indica tratamento desigual se pessoas com o mesmo risco real forem tratadas de forma diferente.
4. **Erro de calibração por grupo** — o modelo erra mais o risco de algum grupo?

Grupos com menos de 100 clientes são reportados, mas não interpretados.

**Resultados** (`reports/bias_audit.json`):

| Grupo | Inadimplência real | Aprovação | Aprovação dos bons pagadores | Erro de calibração |
|---|---|---|---|---|
| Mulheres | 6,90% | 91,3% | 92,8% | −0,79 p.p. |
| Homens | 10,31% | 87,7% | 89,9% | +1,41 p.p. |
| 18–29 anos | 11,16% | 76,0% | 78,9% | −1,38 p.p. |
| 30–44 anos | 9,13% | 88,7% | 90,8% | +0,45 p.p. |
| 45–59 anos | 6,50% | 95,0% | 95,9% | 0,00 p.p. |
| 60+ anos | 5,09% | 98,2% | 98,7% | −0,09 p.p. |

- **Gênero:** mesmo fora do modelo, bons pagadores homens são aprovados 2,9 p.p. a menos que mulheres, evidência de proxies. O modelo subestima o risco dos homens (+1,41 p.p., acima do limite de 1 p.p.) e superestima o das mulheres (−0,79 p.p.): sem enxergar o gênero, ele "faz uma média" entre os dois grupos.
- **Idade:** o modelo superestima o risco dos clientes de 18 a 29 anos (−1,38 p.p.), e a aprovação dos bons pagadores dessa faixa é 20 p.p. menor que a dos acima de 60 anos. Parte da diferença reflete risco real maior; parte é penalização além do risco. Como a idade é uma feature, o efeito é direto, e não via proxy.
- **Limite:** com riscos reais diferentes entre grupos, não é possível igualar calibração e aprovação de bons pagadores ao mesmo tempo. A auditoria explicita o trade-off; a escolha exige uma decisão de negócio e avaliação jurídica. Não se afirma que o modelo é justo apenas porque suas métricas gerais são boas.

**Mitigações possíveis** (não implementadas): recalibração por faixa etária; revisão humana das recusas de clientes jovens próximas ao limiar; avaliar a substituição da idade por variáveis menos sensíveis; monitorar essas métricas a cada execução (Etapa 3).

### 5. Plano de retenção



| O que é guardado | Onde | Retenção no projeto |
|---|---|---|
| Dados brutos (Kaggle) | `data/`, fora do Git | Apenas na máquina de quem executa; descartáveis ao fim do projeto |
| Predições linha a linha | `data/`, fora do Git | Regeneráveis; não precisam ser guardadas |
| Modelo treinado | `models/`, fora do Git | Regenerável a partir do código |
| Resultados agregados | `reports/`, versionados | Mantidos: não contêm dados pessoais |
| Logs de execução (JSONL) | `logs/`, fora do Git | Regeneráveis; sem identificador de cliente |
| Quarentena (linhas bloqueadas) | `data/quarantine/`, fora do Git | Contém dados pessoais pseudonimizados (HMAC-SHA256); descartável ao fim do projeto |
| Execuções do MLflow | `mlflow.db` / `mlruns/`, fora do Git | Metadados de treino e monitoramento; sem PII de cliente |

Numa operação real, os prazos de retenção dependeriam das obrigações legais e regulatórias do setor financeiro e de validação jurídica; após a finalidade, os dados deveriam ser eliminados ou anonimizados.

---

## Análise causal da degradação

A análise separa **hipótese** de **evidência**. Como os cenários são simulações controladas, a causa de cada mudança nos dados é conhecida **por construção**; o que se investiga é **por que o modelo falhou** diante delas.

**1. Por que o modelo não reagiu ao desemprego (Data Drift)**

- **Hipótese:** os novos desempregados caem em `EMPLOYMENT_GROUP = NOT_EMPLOYED`, categoria que o modelo aprendeu como de baixo risco (5,4%), porque era formada quase só por aposentados. A categoria `Unemployed` tinha apenas 17 casos no treino, insuficientes para o modelo aprender seu risco.
- **Mecanismo** (`eda/causal_analysis.ipynb`): o coeficiente de `NOT_EMPLOYED` é −0,39, praticamente igual ao de `Pensioner` (−0,39), e o de `Unemployed` é quase nulo (+0,004). Os coeficientes são relativos (one-hot com regularização) e devem ser lidos em comparação entre categorias.
- **Efeito nos mesmos clientes:** quando os 4.759 trabalhadores passaram a desempregados, o risco médio previsto **caiu** de 9,73% para 9,47% e a aprovação **subiu** de 85,7% para 86,5%, enquanto a inadimplência real **triplicou** (9,75% → 29,40%). O modelo reagiu na direção errada.
- **Evidência agregada:** no segmento de novos desempregados, o erro de calibração é de +20 p.p.; a aprovação geral e o score não apresentaram drift (PSI 0,0004).
- **Limite:** a comparação inclui também a inflação de renda e de preços aplicada a esses clientes; isolar só o desemprego exigiria um cenário dedicado.

**2. Renda nominal × renda real (Data Drift)**

O modelo usa a renda nominal. Com a inflação, ele lê "renda maior" como "risco menor", embora os preços também tenham subido 20%. Clientes com reajuste abaixo da inflação perderam poder de compra, e isso é invisível para o modelo.

**3. Por que o modelo não capta o comprometimento de renda (Concept Drift)**

O risco mudou para os clientes com parcela alta em relação à renda. A regressão logística usa renda e parcela **separadamente**, em termos lineares, e não representa bem a **razão** entre elas. Já na referência, antes de qualquer simulação, esse segmento tinha um pequeno erro de calibração (+0,73 p.p.): a crise ampliou uma fraqueza que já existia.

**Limites da conclusão:** as relações acima são demonstradas na simulação. Numa operação real, uma degradação semelhante poderia ter outras causas (mudança na política comercial, nos canais de captação, na qualidade dos dados de origem), e atribuir causa exigiria investigação adicional.

---

## Testes e qualidade de código

- **pytest**, com dados sintéticos (sem dependência dos CSVs): contrato (cada regra de bloqueio e de alerta, falhas estruturais, lote inválido de ponta a ponta), transformações, modelo, política de decisão (incluindo casos sem inadimplentes ou sem recusados), métricas de desempenho, pseudonimização (mesma chave gera o mesmo pseudônimo, chaves diferentes geram pseudônimos diferentes, o ID original não aparece, chave ausente gera erro), a garantia de que o gênero não é feature do modelo, logs JSONL, quarentena sem PII, alerta de degradação silenciosa e isolamento de falha do MLflow.
- **Ruff** (lint e formatação) e **pre-commit** (Ruff, bloqueio de arquivos grandes, finais de linha e espaços). Os relatórios HTML em `reports/` são exceções documentadas: são entregáveis gerados, não editados à mão.

---

## Como executar

**Pré-requisitos:** Python 3.12 e [uv](https://docs.astral.sh/uv/).

1. Baixe o `application_train.csv` da competição Home Credit Default Risk (Kaggle), respeitando os termos de uso, e coloque-o em `data/`. O arquivo não é versionado.

2. Instale as dependências e rode o pipeline, na ordem:

```bash
uv sync

uv run python -m credit_monitoring.data.split
uv run python -m credit_monitoring.data.contract
uv run python -m credit_monitoring.modeling.train
uv run python -m credit_monitoring.monitoring.simulate
uv run python -m credit_monitoring.monitoring.predict
uv run python -m credit_monitoring.monitoring.drift
uv run python -m credit_monitoring.monitoring.performance
uv run python -m credit_monitoring.governance.bias_audit
uv run python -m credit_monitoring.observability
```

A última linha **não precisa reprocessar os CSVs**: ela lê os JSON já gravados em `reports/`, concatena `logs/pipeline.jsonl` se existir, grava o dashboard e registra os runs `monitoring` no MLflow.

Para só inspecionar o rastreamento depois:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra http://localhost:5000 — runs `training` e `monitoring-*`, comparáveis lado a lado (PSI, calibração, flags).

3. Análise causal: execute `eda/causal_analysis.ipynb` depois do pipeline.

4. Testes e qualidade:

```bash
uv run pytest
uv run ruff check .
uv run pre-commit run --all-files
```

### Como testar a Etapa 3

Os testes automatizados **não usam** `application_train.csv`. Cubra logs, quarentena, alertas e MLflow com:

```bash
uv run pytest tests/test_logging.py tests/test_quarantine.py tests/test_health.py tests/test_tracking.py -q
```

O que cada arquivo prova:

| Teste | Como conferir na prática |
|---|---|
| `test_logging` | JSONL com `task`, `status`, `duration_seconds`; falha da tarefa é relançada e registrada |
| `test_quarantine` | o `SK_ID_CURR` original não aparece no CSV; rejeição estrutural não despeja linhas |
| `test_health` | controle sem alerta; data drift com PSI; concept drift = **degradação silenciosa** |
| `test_tracking` | MLflow fora do ar não quebra o pipeline; run `monitoring` grava métricas e flags |

Para testar o entregável visual **com os relatórios já versionados** (Etapa 2 já rodou neste repo):

```bash
# Windows PowerShell
$env:CREDIT_MONITORING_PSEUDONYM_KEY = "troque-por-um-segredo"

uv sync
uv run python -m credit_monitoring.observability
```

1. Abra `reports/observability_dashboard.html` no navegador. No lote `concept_drift` deve aparecer degradação silenciosa; no `data_drift`, PSI alto; no `control`, nenhum alerta de modelo.
2. Confira `reports/health_summary.json` (`active_alerts`) e `reports/metrics.prom` (séries `credit_silent_degradation` e `credit_max_psi`).
3. Suba a UI: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` e compare os runs `monitoring-control` vs `monitoring-concept_drift`.
4. Para ver logs de ponta a ponta, rode de novo o pipeline (split → … → observability) e abra `logs/pipeline.jsonl`: cada tarefa tem `status` e `duration_seconds`.
5. Para ver a quarentena: `uv run python -m credit_monitoring.data.contract` com a chave definida; os 5 clientes inconsistentes da referência saem em `data/quarantine/` sem o ID original.

Se o MLflow não estiver instalado ainda, `uv sync` passa a puxar a dependência nova.

**Chave de pseudonimização:** a Etapa 3 grava a quarentena com o identificador HMAC. Defina `CREDIT_MONITORING_PSEUDONYM_KEY` com um valor secreto, que não deve ser versionado. Sem a chave, o programa falha ao escrever quarentena (não usa chave fraca em silêncio). Os testes passam a chave só no processo de teste.

**Restrição de compatibilidade:** o pandas está fixado abaixo da versão 3, porque o Evidently 0.7.23 importa uma função interna removida no pandas 3.

---

## Estrutura do repositório

```
Tech-Challenge-4/
├── src/credit_monitoring/
│   ├── config.py               # Caminhos, parâmetros e limites (fonte única)
│   ├── data/
│   │   ├── split.py            # Referência × produção
│   │   ├── contract.py         # Contrato Pandera: bloqueio, alertas, lote inválido
│   │   └── features.py         # Transformações fixas da EDA
│   ├── modeling/
│   │   └── train.py            # Baseline, limiar e funções de decisão
│   ├── monitoring/
│   │   ├── simulate.py         # Cenários de Data Drift e Concept Drift
│   │   ├── predict.py          # Score e decisão dos lotes de produção
│   │   ├── drift.py            # PSI/KS e relatórios Evidently
│   │   └── performance.py      # Calibração e desempenho com rótulos
│   ├── observability/          # Etapa 3: logs, MLflow, saúde e dashboard
│   └── governance/
│       ├── bias_audit.py       # Auditoria de viés por gênero e faixa etária
│       └── privacy.py          # Pseudonimização do identificador (HMAC-SHA256)
├── tests/                      # pytest, com dados sintéticos
├── eda/
│   ├── eda.ipynb               # Inspeção inicial (histórico; a divisão oficial é o split.py)
│   ├── eda_reference.ipynb     # EDA na referência, com as decisões documentadas
│   └── causal_analysis.ipynb   # Coeficientes e comparação dos mesmos clientes
├── reports/                    # Resultados agregados (versionados)
├── data/                       # Dados brutos e predições (não versionados)
├── models/                     # Modelo treinado (não versionado)
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
└── pyproject.toml
```

---

## Decisões de design registradas

- **Divisão antes da EDA:** toda decisão de limpeza e todo limite vêm só da referência; a produção apenas recebe o que foi aprendido.
- **Limites congelados:** faixas de idade do carro e de emprego (`pd.cut`, não `qcut`), limiar de decisão, faixas do PSI e limite de comprometimento são calculados uma vez. Limites recalculados na produção se moveriam com os dados e esconderiam o drift.
- **Bloqueio × alerta:** o que é certamente inválido bloqueia; o que é suspeito alerta. O contrato julga se o dado é confiável, não se o cliente é bom pagador.
- **Falha estrutural rejeita o lote:** erros sem linha associada (coluna ausente, tipo errado, índice repetido) nunca são descartados.
- **Referência limpa definida pelo contrato:** a mesma regra filtra o treino e a produção.
- **Nulos estruturais e valores sentinela viram categorias** (`NO_CAR`, `NOT_EMPLOYED`), em vez de valores inventados (0 anos de emprego colidiria com quem acabou de ser contratado).
- **PSI decide, KS contextualiza:** com amostras grandes, o p-valor acusa diferenças irrelevantes.
- **Decisão separada da avaliação:** `decide()` não precisa de rótulos (disponível no dia); `evaluate_policy()` só roda quando eles existem.
- **Métricas globais detectam, segmentos diagnosticam:** o erro de calibração global dispara o alerta; os segmentos explicam onde está o problema.
- **Observabilidade não é o caminho crítico:** falha do MLflow vira warning; o job de scoring continua.
- **Um veredito, três saídas:** dashboard, OpenMetrics e runs `monitoring` leem `health.py`.
- **Convenção de idioma:** código, notebooks e commits em inglês; documentação em português.

---

## Limitações e próximos passos

**Limitações**

- Os cenários de drift são simulados; as premissas (inflação de 20%, 15% de desemprego, inadimplência de 29,4% para desempregados, dobro de risco no comprometimento alto) são escolhas de cenário.
- O desempregado simulado mantém a renda que tinha como trabalhador (ainda inflacionada), o que é pouco realista.
- A taxa de 29,4% vem de 17 clientes (IC 95% ≈ 13% a 53%).
- O Concept Drift é aplicado ao lote bruto, antes do contrato; contagens do resumo da simulação e das predições podem diferir minimamente.
- A política de 90% de aprovação é uma premissa; o dataset não informa juros nem perdas.
- `NOT_EMPLOYED` agrupa aposentados e desempregados, o que contribui para a falha do modelo e reduz a sensibilidade do monitoramento dessa feature.
- Duplicatas são verificadas dentro do lote, não contra o histórico de clientes já vistos.
- A auditoria de viés usa o conjunto de teste da referência; ela não foi repetida nos cenários de drift.
- O Prometheus/Grafana completo (scrape contínuo, Alertmanager) não faz parte deste repositório: o enunciado trata Prometheus-client como opcional. O arquivo `reports/metrics.prom` e o dashboard HTML cobrem o entregável visual.



---
