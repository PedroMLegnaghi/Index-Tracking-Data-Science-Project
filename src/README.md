# 📦 Módulos do Projeto

Esta pasta contém os módulos Python do projeto de Index Tracking.

---

## 📁 Estrutura

```
src/
├── __init__.py                  # Inicialização do pacote
├── data_collection.py           # Coleta de dados
├── data_preprocessing.py        # Pré-processamento
├── exploratory_analysis.py      # Análise exploratória
├── optimization_model.py        # Modelo de otimização
├── backtesting.py               # Validação
└── visualization.py             # Visualizações
```

---

## 📖 Descrição dos Módulos

### 1. `data_collection.py`

**Responsabilidade**: Coleta de dados históricos do Yahoo Finance

**Classes**:
- `DataCollector`: Classe principal para coleta

**Principais Métodos**:
- `download_index_data()`: Baixa dados de um índice
- `download_stocks_data()`: Baixa dados de múltiplas ações
- `get_sp100_tickers()`: Lista de tickers do S&P 100
- `get_ibov_tickers()`: Lista de tickers do IBOVESPA
- `collect_all_data()`: Pipeline completo de coleta

**Exemplo de Uso**:
```python
from data_collection import DataCollector

collector = DataCollector('2018-01-01', '2025-01-01')
index_data, stocks_data = collector.collect_all_data('SP100')
```

---

### 2. `data_preprocessing.py`

**Responsabilidade**: Limpeza e preparação dos dados

**Classes**:
- `DataPreprocessor`: Classe para pré-processamento

**Principais Métodos**:
- `check_missing_data()`: Analisa dados faltantes
- `remove_high_missing_columns()`: Remove colunas com muitos NaNs
- `interpolate_missing_values()`: Interpola valores faltantes
- `detect_outliers()`: Detecta outliers (disponível, mas não usado no projeto)
- `treat_outliers()`: Trata outliers (disponível, mas não usado no projeto)
- `calculate_returns()`: Calcula retornos (log ou simples)
- `align_data()`: Alinha temporalmente índice e ações
- `preprocess_pipeline()`: Pipeline completo

⚠️ **IMPORTANTE - Tratamento de Outliers:**

**Outliers NÃO são tratados neste projeto por design!**

**Justificativa:**
- **Objetivo**: Replicar o índice inclusive em eventos extremos (crashes, rallies)
- **Realidade**: Outliers são eventos REAIS (COVID-19 -30%, Crise 2008 -20%)
- **Tracking Error**: Se índice cai 20%, carteira DEVE cair ~20% (baixo TE)
- **Out-of-Sample**: Tratar outliers artificialmente piora performance em crises
- **Retornos Log**: Já limitam naturalmente valores extremos
- **Backtesting**: Precisa testar robustez em períodos voláteis

As funções `detect_outliers()` e `treat_outliers()` estão disponíveis para outros
projetos, mas são **desabilitadas por padrão** no pipeline via `treat_outliers=False`.

**Exemplo de Uso**:
```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(max_missing_pct=0.1)

# Index Tracking (padrão - SEM tratar outliers)
index_ret, stocks_ret = preprocessor.preprocess_pipeline(
    index_data, stocks_data, 
    calculate_ret=True,
    treat_outliers=False  # ← Padrão (recomendado para IT)
)

# Outros projetos (COM tratamento de outliers)
index_ret, stocks_ret = preprocessor.preprocess_pipeline(
    index_data, stocks_data, 
    calculate_ret=True,
    treat_outliers=True  # ← Apenas se necessário
)
```

---

### 3. `exploratory_analysis.py`

**Responsabilidade**: Análise exploratória e visualizações

**Classes**:
- `ExploratoryAnalyzer`: Classe para EDA

**Principais Métodos**:
- `descriptive_statistics()`: Estatísticas descritivas
- `plot_time_series()`: Plota séries temporais
- `plot_returns_distribution()`: Distribuição de retornos
- `plot_correlation_matrix()`: Matriz de correlação (heatmap)
- `analyze_correlation_with_index()`: Correlação com índice
- `analyze_volatility()`: Análise de volatilidade
- `identify_crisis_periods()`: Identifica crises
- `full_eda_report()`: Relatório completo de EDA

**Exemplo de Uso**:
```python
from exploratory_analysis import ExploratoryAnalyzer

analyzer = ExploratoryAnalyzer(figsize=(14, 6))
results = analyzer.full_eda_report(
    index_returns, stocks_returns,
    index_name="S&P 100",
    save_dir="../results"
)
```

---

### 4. `optimization_model.py` ⭐

**Responsabilidade**: Modelo de otimização (núcleo do projeto)

**Classes**:
- `IndexTrackingOptimizer`: Classe para otimização

**Principais Métodos**:
- `optimize_unconstrained()`: Modelo sem restrição de nº ativos
- `optimize_constrained()`: Modelo com restrição (máx K ativos)
- `sensitivity_analysis()`: Análise de sensibilidade (vários K)

**Formulação Matemática**:
```
min (1/T) Σ (Σ w_i * r_{t,i} - R_t)²

s.t.:
  Σ w_i = 1
  w_i ≥ 0
  Σ z_i ≤ K
  w_i ≤ z_i
  z_i ∈ {0,1}
```

**Exemplo de Uso**:
```python
from optimization_model import IndexTrackingOptimizer

optimizer = IndexTrackingOptimizer(
    index_returns, stocks_returns, solver='ECOS'
)

# Modelo não restrito
result_unconstrained = optimizer.optimize_unconstrained()

# Modelo restrito (20 ativos)
result_constrained = optimizer.optimize_constrained(max_assets=20)

# Análise de sensibilidade
sensitivity = optimizer.sensitivity_analysis([5, 10, 20, 30, 50])
```

---

### 5. `backtesting.py`

**Responsabilidade**: Validação in-sample e out-of-sample

**Classes**:
- `Backtester`: Classe para backtesting

**Principais Métodos**:
- `train_test_split()`: Divide dados em treino/teste
- `calculate_tracking_error()`: Calcula TE
- `calculate_correlation()`: Calcula correlação
- `calculate_information_ratio()`: Calcula IR
- `calculate_metrics()`: Todas as métricas
- `backtest_single_period()`: Backtest em um período
- `rolling_window_backtest()`: Backtest com janela rolante
- `evaluate_out_of_sample()`: Avaliação agregada OOS

**Métricas Calculadas**:
- Tracking Error (TE)
- Correlação
- Information Ratio (IR)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

**Exemplo de Uso**:
```python
from backtesting import Backtester

backtester = Backtester()

# Split simples
index_train, index_test = backtester.train_test_split(index_returns)

# Backtest com janela rolante
results = backtester.rolling_window_backtest(
    index_returns, stocks_returns,
    optimizer_func=my_optimizer_function,
    train_window=504,  # 2 anos
    test_window=126    # 6 meses
)

# Avaliar
metrics = backtester.evaluate_out_of_sample(results)
```

---

### 6. `visualization.py`

**Responsabilidade**: Visualizações de resultados

**Classes**:
- `ResultsVisualizer`: Classe para visualizações

**Principais Métodos**:
- `plot_portfolio_vs_index()`: Carteira vs Índice
- `plot_tracking_error_evolution()`: Evolução do TE
- `plot_weights_distribution()`: Distribuição de pesos
- `plot_sensitivity_analysis()`: Curva de sensibilidade

**Exemplo de Uso**:
```python
from visualization import ResultsVisualizer

visualizer = ResultsVisualizer(figsize=(14, 6))

visualizer.plot_portfolio_vs_index(
    portfolio_returns, index_returns, dates,
    title="Carteira vs Índice",
    save_path="results/portfolio_vs_index.png"
)

visualizer.plot_sensitivity_analysis(
    sensitivity_df,
    save_path="results/sensitivity.png"
)
```

---

## 🔧 Dependências

Todas os módulos requerem:

```python
pandas
numpy
matplotlib
seaborn
yfinance
cvxpy
```

Instale com:
```bash
pip install -r ../requirements.txt
```

---

## 📊 Fluxo de Uso Típico

```python
# 1. Coleta
collector = DataCollector(start_date, end_date)
index_data, stocks_data = collector.collect_all_data('SP100')

# 2. Pré-processamento
preprocessor = DataPreprocessor()
index_ret, stocks_ret = preprocessor.preprocess_pipeline(
    index_data, stocks_data, calculate_ret=True
)

# 3. EDA
analyzer = ExploratoryAnalyzer()
eda_results = analyzer.full_eda_report(index_ret, stocks_ret)

# 4. Otimização
optimizer = IndexTrackingOptimizer(index_ret, stocks_ret)
result = optimizer.optimize_constrained(max_assets=20)

# 5. Backtesting
backtester = Backtester()
backtest_results = backtester.rolling_window_backtest(
    index_ret, stocks_ret, optimizer_func
)

# 6. Visualização
visualizer = ResultsVisualizer()
visualizer.plot_portfolio_vs_index(...)
```

---

## 🧪 Testes

Execute o script de teste rápido:

```bash
cd ..
python quick_test.py
```

Ou teste módulos individuais:

```bash
cd src
python data_collection.py
python data_preprocessing.py
python exploratory_analysis.py
python optimization_model.py
```

---

## 📝 Notas Importantes

1. **Imports**: Use caminhos relativos ou absolutos corretos
2. **Dados**: Certifique-se de ter conexão com internet para download
3. **Solvers**: CVXPY usará ECOS por padrão (open-source)
4. **Performance**: Para MIP, considere instalar GUROBI ou CPLEX

---

## 🆘 Troubleshooting

**Problema**: Import errors  
**Solução**: Certifique-se de estar no diretório correto ou ajuste `sys.path`

**Problema**: Solver errors  
**Solução**: O código tem fallback automático, mas verifique instalação do CVXPY

**Problema**: Dados não baixam  
**Solução**: Verifique conexão com internet e limites do Yahoo Finance

---

**Documentação completa no README.md principal** 📚
