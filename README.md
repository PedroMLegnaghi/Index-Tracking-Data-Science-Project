# Index Tracking: S&P 100 e IBOVESPA (Bootcamp Fundo Amanhã)

Projeto desenvolvido no Bootcamp de Introdução à Data Science (Associação Fundo Amanhã) com o objetivo de construir carteiras que repliquem índices de mercado usando menos ativos (Index Tracking), mantendo baixo Tracking Error e alta correlação.

**Objetivo do Trabalho**
- Replicar os índices `S&P 100` e `IBOVESPA` com carteiras do menor tamanho possível (tamanho ótimo).
- Minimizar o Tracking Error médio quadrático do retorno da carteira em relação ao índice.
- Validar in-sample e out-of-sample (split temporal e janela rolante), com visualizações e métricas.

**Formulação Matemática (Index Tracking)**
- Objetivo: $\min \frac{1}{T} \sum_{t=1}^T \big(\sum_i w_i r_{t,i} - R_t\big)^2$
- Restrições: $\sum_i w_i=1$, $w_i\ge 0$, $\sum_i z_i\le K$, $w_i\le M z_i$, $z_i\in\{0,1\}$
- Onde: $R_t$ é o retorno do índice, $r_{t,i}$ o retorno do ativo $i$, $w_i$ os pesos e $z_i$ seleção binária (modelo restrito)

**O que foi feito**
- Coleta de dados (7 anos, Yahoo Finance).
- Pré-processamento (alinhamento temporal, retornos, tratamento de faltantes; outliers mantidos por decisão de modelagem).
- EDA com estatísticas e gráficos (correlação, distribuição, volatilidade).
- Modelos de otimização:
	- Unconstrained IT: pesos contínuos (QP) com $\sum w=1$ e $w\ge0$.
	- Constrained IT: controle de cardinalidade (MIQP) com $\sum z\le K$ e $w\le Mz$.
- Validação:
	- Train/Test split (75/25).
    - Teste de sensibilidade evidencia K=20 ações como número ótimo para montar carteira, o qual minimiza erros e ao mesmo tempo mantém baixa quantidade de ações.
	- Janela rolante (train≈504 dias, test≈126 dias, step≈252 dias).
	- Métricas: Tracking Error, Correlação, Information Ratio, MAE, RMSE.
- Visualizações:
	- Performance carteira vs índice (base 100) e painel de Tracking Error.
	- Evolução do TE por janela.
	- Gráfico completo das carteiras de janelas rolantes com trajetória do início ao fim.

**Principais Resultados (Resumo)**
- S&P 100: TE típico < 2%, correlação > 0.95 com K=20.
- IBOVESPA: TE típico ≈ 0.3–1.0%, correlação > 0.95 com K=20.
- Visualizações mostram robustez temporal e generalização out-of-sample.

**Estrutura do Código**
- `src/data_collection.py`: Coleta índices e componentes com cache CSV.
- `src/data_preprocessing.py`: Limpeza, alinhamento e cálculo de retornos.
- `src/exploratory_analysis.py`: EDA (estatísticas e gráficos).
- `src/optimization_model.py`: Modelos de otimização (QP, MIQP, heurística).
- `src/backtesting.py`: Train/Test, janela rolante e métricas.
- `src/visualization.py`: Gráficos de performance, TE e janelas rolantes.
- `notebooks/06_Projeto_Completo.ipynb`: Apresentação completa do projeto.

**Como Visualizar (Notebook)**
-Abra a pasta "notebooks"
-Abra o arquivo "06_Projeto_Completo.ipynb", os resultados já estão carregados e prontos para serem visualizados

**Como Rodar (Notebook)**
- Abra `notebooks/06_Projeto_Completo.ipynb` no VS Code/Jupyter.
- Execute as células em ordem (auto-reload ativo para módulos `src/`).

**Decisões de Modelagem Importantes**
- Outliers mantidos: Index Tracking precisa replicar eventos extremos do índice (COVID-19, crises), evitando enviesar o TE.

**Limitações e Próximos Passos**
- Não considera custos de transação/slippage e restrições de liquidez.
- Rebalanceamento não otimizado (frequência/custos).
- Próximos: incluir custos, penalização de turnover, CVaR/ES, comparação com ML, índices adicionais.

**Créditos**
- Bootcamp de Introdução à Data Science – Associação Fundo Amanhã.
- Finor
- Metodologia baseada em literatura clássica de Index Tracking e otimização.

