"""
Módulo de Pré-processamento de Dados
======================================

Este módulo é responsável pela limpeza e preparação dos dados coletados para Index Tracking,
incluindo tratamento de valores faltantes e cálculo de retornos.

⚠️ DECISÃO DE DESIGN - OUTLIERS:
==================================
Outliers NÃO são tratados por padrão neste projeto!

JUSTIFICATIVA:
- Objetivo: Replicar o índice, inclusive em eventos extremos (crashes, rallies)
- Outliers são REAIS (COVID-19 -30%, Crise 2008 -20%, Black Monday -22%)
- Se o índice cai 20%, a carteira DEVE cair ~20% (baixo tracking error)
- Tratar outliers artificialmente reduz TE no treino mas piora out-of-sample
- Retornos logarítmicos já limitam naturalmente valores extremos
- Backtesting precisa testar robustez em períodos voláteis

Para outros projetos (previsão, classificação), as funções de detecção/tratamento
de outliers estão disponíveis mas comentadas no pipeline.

Funcionalidades:
    - Tratamento de valores faltantes (missing data)
    - Detecção de outliers (disponível, mas não usada por padrão)
    - Tratamento de outliers (disponível, mas não usada por padrão)
    - Cálculo de retornos logarítmicos e simples
    - Alinhamento temporal entre índice e ações
    - Remoção de ativos com dados insuficientes

Autor: Projeto Final - Bootcamp Data Science
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Classe para pré-processar dados de mercado financeiro.
    
    Attributes:
        max_missing_pct (float): Percentual máximo permitido de dados faltantes (0-1)
        max_consecutive_missing (int): Número máximo de dias consecutivos faltantes
        outlier_std_threshold (float): Número de desvios padrão para considerar outlier
    """
    
    def __init__(self, 
                 max_missing_pct: float = 0.1,
                 max_consecutive_missing: int = 30,
                 outlier_std_threshold: float = 5.0):
        """
        Inicializa o pré-processador.
        
        Args:
            max_missing_pct: Percentual máximo de dados faltantes (padrão: 10%)
            max_consecutive_missing: Máximo de dias consecutivos faltantes (padrão: 30)
            outlier_std_threshold: Threshold para detecção de outliers (padrão: 5 desvios)
        """
        self.max_missing_pct = max_missing_pct
        self.max_consecutive_missing = max_consecutive_missing
        self.outlier_std_threshold = outlier_std_threshold
        
        print(f"✓ DataPreprocessor inicializado:")
        print(f"  - Max missing: {max_missing_pct*100:.1f}%")
        print(f"  - Max consecutive missing: {max_consecutive_missing} dias")
        print(f"  - Outlier threshold: ±{outlier_std_threshold} desvios padrão")
    
    def check_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa dados faltantes no DataFrame.
        
        Args:
            data: DataFrame com dados a serem analisados
            
        Returns:
            DataFrame com estatísticas de dados faltantes por coluna
        """

        # fazer cópia do dataframe dado em um outro dataframe
        missing_stats = pd.DataFrame({
            'Total_Missing': data.isnull().sum(),
            'Pct_Missing': (data.isnull().sum() / len(data) * 100).round(2),
            # Retorna o indice do primeiro e último valor não nulo
            # importante para saber se o dado está faltando no início ou fim da série temporal que colocamos, ou seja, importante para definirmos 
            # o período de escolha do ativo
            'First_Valid': data.apply(lambda x: x.first_valid_index()),
            'Last_Valid': data.apply(lambda x: x.last_valid_index())
        })
        
        # Filtrar apenas colunas com dados faltantes e ordenar por percentual de dados faltantes
        missing_stats = missing_stats[missing_stats['Total_Missing'] > 0].sort_values(
            'Pct_Missing', ascending=False
        )
        
        if len(missing_stats) > 0:
            print(f"\n⚠ Dados faltantes encontrados em {len(missing_stats)} colunas:")
            print(missing_stats.head(10))
        else:
            print("\n✓ Nenhum dado faltante encontrado")
        
        return missing_stats
    
    def check_consecutive_missing(self, series: pd.Series) -> int:
        """
        Verifica o número máximo de valores consecutivos faltantes em uma série.
        Usada na função de remoção de colunas com muitos dados faltantes. (remove_high_missing_columns)
        
        Args:
            series: Série temporal a ser analisada, ex: data['AAPL']
            
        Returns:
            Número máximo de valores consecutivos faltantes na série/coluna dada
        """
        # Criar série binária (1 = faltante, 0 = presente)
        is_null = series.isnull().astype(int)
        
        # Agrupar valores consecutivos
        groups = (is_null != is_null.shift()).cumsum()
        
        # Contar máximo de 1s consecutivos
        max_consecutive = is_null.groupby(groups).sum().max()
        
        return int(max_consecutive) if not np.isnan(max_consecutive) else 0
    
    def remove_high_missing_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas com muitos dados faltantes.
        
        Args:
            data: DataFrame original
            
        Returns:
            DataFrame filtrado
        """
        print(f"\n🔍 Analisando colunas com dados faltantes...")

        # Guardar o número inicial de colunas para comparar no final com a quantidade de colunas removidas
        initial_cols = len(data.columns)
        
        # Calcular percentual de missing
        missing_pct = data.isnull().sum() / len(data)
        
        # Colunas a remover = percentual de missing > limite posto no objeto de data_preprocessor
        cols_to_remove_due_to_general_missing_data = missing_pct[missing_pct > self.max_missing_pct].index.tolist()
        
        # Verificar missing consecutivos, adicionando à lista de colunas para remover do dataset as
        # colunas que ultrapassem o limite de dias consecutivos com dados faltantes
        cols_to_remove_due_to_consecutive_days_missing_data = []
        for col in data.columns:
            max_consecutive = self.check_consecutive_missing(data[col])
            if max_consecutive > self.max_consecutive_missing:
                cols_to_remove_due_to_consecutive_days_missing_data.append(col)
        
        # Unir ambas as listas
        all_cols_to_remove = list(set(cols_to_remove_due_to_general_missing_data + cols_to_remove_due_to_consecutive_days_missing_data))
        
        # Remover colunas problemáticas
        data_clean = data.drop(columns=all_cols_to_remove)
        
        print(f"✓ Colunas removidas: {len(all_cols_to_remove)}")
        print(f"  - Por % missing: {len(cols_to_remove_due_to_general_missing_data)}")
        print(f"  - Por missing consecutivo: {len(cols_to_remove_due_to_consecutive_days_missing_data)}")
        print(f"  Colunas restantes: {len(data_clean.columns)} de {initial_cols}")
        
        if len(all_cols_to_remove) > 0 and len(all_cols_to_remove) <= 20:
            print(f"  Removidas: {', '.join(all_cols_to_remove)}")
        
        return data_clean
    
    def interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpola valores faltantes restantes usando interpolação linear.
        O método de interpolação é o seguinte:
        - Interpolação linear para valores internos
        - Forward fill e backward fill para bordas
        Utilizado após remoção de colunas com muitos dados faltantes.
        
        Args:
            data: DataFrame com alguns valores faltantes
            
        Returns:
            DataFrame com valores interpolados (sem valores faltantes)
        """
        print(f"\n🔧 Interpolando valores faltantes...")
        
        missing_before = data.isnull().sum().sum()
        
        # Interpolação linear: 
        # a interpolação linear é um método que estima os valores faltantes com base nos valores conhecidos adjacentes.
        # Por exemplo, se temos os valores [1, NaN, 3], a interpolação linear preencherá o NaN com 2, que é a média dos valores 1 e 3.
        # limit_direction='both' garante que a interpolação seja feita em ambas as direções (início e fim da série)
        # Exemplo: Preços da AAPL: [150, NaN, NaN, 153] -> [150, 151, 152, 153]
        data_interpolated = data.interpolate(method='linear', limit_direction='both')
        
        # Forward fill e backward fill para bordas
        # Depois da interpolação linear, ainda podem restar valores NaN nas bordas (início ou fim da série).
        # O forward fill (ffill) preenche valores NaN da cauda com o último valor conhecido
        # O backward fill (bfill) preenche valores NaN da cabeça com o próximo valor conhecido
        # Exemplo: Preços da AAPL: [NaN, NaN, 150, 151, NaN, Nan] 
        # -Depois do ffill-> [NaN, NaN, 150, 151, 151, 151]
        # -Depois do bfill-> [150, 150, 150, 151, 151, 151]
        data_interpolated:pd.DataFrame = data_interpolated.fillna(method='ffill')
        data_interpolated:pd.DataFrame = data_interpolated.fillna(method='bfill')
        
        missing_after = data_interpolated.isnull().sum().sum()
        
        print(f"✓ Interpolação completa:")
        print(f"  - Missing antes: {missing_before}")
        print(f"  - Missing depois: {missing_after}")
        
        return data_interpolated
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Detecta outliers nos dados.
        
        Args:
            data: DataFrame a ser analisado
            method: Método de detecção ('zscore' ou 'iqr')
            
        Returns:
            DataFrame booleano indicando outliers
        """
        print(f"\n🔍 Detectando outliers usando método: {method}")
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > self.outlier_std_threshold
        
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        
        else:
            raise ValueError("method deve ser 'zscore' ou 'iqr'")
        
        total_outliers = outliers.sum().sum()
        pct_outliers = (total_outliers / (data.shape[0] * data.shape[1]) * 100)
        
        print(f"✓ Outliers detectados: {total_outliers} ({pct_outliers:.2f}% dos dados)")
        
        # Mostrar colunas com mais outliers
        outliers_per_col = outliers.sum().sort_values(ascending=False).head(10)
        if len(outliers_per_col) > 0:
            print(f"\n  Top colunas com outliers:")
            for col, count in outliers_per_col.items():
                print(f"    {col}: {count} outliers")
        
        return outliers
    
    def treat_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame, 
                      method: str = 'winsorize') -> pd.DataFrame:
        """
        Trata outliers detectados.
        
        Args:
            data: DataFrame original
            outliers: DataFrame booleano com outliers marcados
            method: Método de tratamento ('winsorize', 'remove', 'median')
            
        Returns:
            DataFrame com outliers tratados
        """
        print(f"\n🔧 Tratando outliers usando método: {method}")
        
        data_treated = data.copy()
        
        if method == 'winsorize':
            # Substituir outliers por valores nos percentis 1% e 99%
            lower = data.quantile(0.01)
            upper = data.quantile(0.99)
            
            for col in data.columns:
                data_treated.loc[outliers[col] & (data[col] < lower[col]), col] = lower[col]
                data_treated.loc[outliers[col] & (data[col] > upper[col]), col] = upper[col]
        
        elif method == 'median':
            # Substituir outliers pela mediana da coluna
            for col in data.columns:
                median_val = data[col].median()
                data_treated.loc[outliers[col], col] = median_val
        
        elif method == 'remove':
            # Remover linhas com outliers (cuidado!)
            mask = ~outliers.any(axis=1)
            data_treated = data[mask]
            print(f"  ⚠ Linhas removidas: {len(data) - len(data_treated)}")
        
        else:
            raise ValueError("method deve ser 'winsorize', 'median' ou 'remove'")
        
        print(f"✓ Tratamento de outliers concluído")
        
        return data_treated
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calcula retornos a partir de preços.
        
        Args:
            prices: DataFrame com preços
            method: Tipo de retorno ('log' ou 'simple')
            
        Returns:
            DataFrame com retornos calculados
        """
        print(f"\n📊 Calculando retornos ({method})...")
        
        if method == 'log':
            # Retornos logarítmicos: ln(P_t / P_{t-1})
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            # Retornos simples: (P_t - P_{t-1}) / P_{t-1}
            returns = prices.pct_change()
        else:
            raise ValueError("method deve ser 'log' ou 'simple'")
        
        # Remover primeira linha (NaN)
        returns = returns.dropna()
        
        print(f"✓ Retornos calculados: {len(returns)} períodos")
        print(f"  Estatísticas médias:")
        print(f"    Retorno médio: {returns.mean().mean()*100:.4f}%")
        print(f"    Volatilidade média: {returns.std().mean()*100:.4f}%")
        
        return returns
    
    def align_data(self, index_data: pd.DataFrame, stocks_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sincroniza temporalmente os dados do índice e das ações, garantindo que ambos tenham exatamente as mesmas datas.
        Crítico para Index Tracking, garantido a consistência dos dados (sem missing values na hora de comparar dados de mesmo dias)
        Args:
            index_data: DataFrame do índice
            stocks_data: DataFrame das ações
            
        Returns:
            Tupla (index_aligned, stocks_aligned)
        """
        print(f"\n🔗 Alinhando dados temporalmente...")
        
        # Obter datas em comum
        common_dates = index_data.index.intersection(stocks_data.index)
        
        # Filtrar ambos DataFrames
        index_aligned = index_data.loc[common_dates]
        stocks_aligned = stocks_data.loc[common_dates]
        
        print(f"✓ Alinhamento concluído:")
        print(f"  Datas do índice: {len(index_data)}")
        print(f"  Datas das ações: {len(stocks_data)}")
        print(f"  Datas em comum: {len(common_dates)}")
        print(f"  Período: {common_dates[0].strftime('%Y-%m-%d')} até {common_dates[-1].strftime('%Y-%m-%d')}")
        
        return index_aligned, stocks_aligned
    
    def preprocess_pipeline(self, index_data: pd.DataFrame, stocks_data: pd.DataFrame,
                          calculate_ret: bool = True, treat_outliers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline completo de pré-processamento para Index Tracking.
        Transforma dados brutos → dados prontos para otimização

        ⚠️ IMPORTANTE: Outliers NÃO são tratados por padrão!
        
        JUSTIFICATIVA:
        Para Index Tracking, eventos extremos (crashes, rallies) são PARTE DO OBJETIVO.
        O modelo deve replicar o índice inclusive em situações extremas. Tratar outliers
        artificialmente reduzirá o tracking error no treino mas piorará a performance
        out-of-sample em períodos de alta volatilidade.
        
        Args:
            index_data: DataFrame do índice (OHLCV)
            stocks_data: DataFrame das ações (preços ajustados)
            calculate_ret: Se True, calcula retornos logarítmicos ao final (padrão: True)
            treat_outliers: Se True, detecta e trata outliers (NÃO recomendado para IT, padrão: False)
            
        Returns:
            Tupla (index_processed, stocks_processed)
            - Se calculate_ret=True: retorna retornos logarítmicos
            - Se calculate_ret=False: retorna preços limpos
        """
        print(f"\n{'='*70}")
        print("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print(f"{'='*70}")
        
        # 1. Alinhar temporalmente
        index_aligned, stocks_aligned = self.align_data(index_data, stocks_data)
        
        # 2. Analisar dados faltantes
        print(f"\n--- ANÁLISE DO ÍNDICE ---")
        self.check_missing_data(index_aligned)
        
        print(f"\n--- ANÁLISE DAS AÇÕES ---")
        self.check_missing_data(stocks_aligned)
        
        # 3. Remover colunas com muitos missing
        stocks_clean = self.remove_high_missing_columns(stocks_aligned)
        
        # 4. Interpolar valores faltantes restantes
        index_clean = self.interpolate_missing_values(index_aligned[['Close']])
        stocks_clean = self.interpolate_missing_values(stocks_clean)
        
        # 5. Detectar e tratar outliers (DESABILITADO por padrão para Index Tracking)
        # 
        # ⚠️ DECISÃO DE DESIGN: Outliers NÃO devem ser tratados para Index Tracking!
        #
        # JUSTIFICATIVA:
        # - Objetivo: Replicar o índice (inclusive em eventos extremos como crashes)
        # - Outliers são REAIS (COVID-19, crises financeiras, etc.)
        # - Se o índice caiu 20%, a carteira DEVE cair ~20% (baixo tracking error)
        # - Tratar outliers artificialmente reduz TE no treino, mas piora out-of-sample
        # - Retornos logarítmicos já limitam naturalmente valores extremos
        #
        # Para ativar (outros projetos), use: treat_outliers=True no pipeline
        if treat_outliers:
            print(f"\n⚠️ TRATANDO OUTLIERS (não recomendado para Index Tracking)")
            outliers = self.detect_outliers(stocks_clean, method='zscore')
            stocks_clean = self.treat_outliers(stocks_clean, outliers, method='winsorize')
        
        # 6. Calcular retornos
        if calculate_ret:
            index_returns = self.calculate_returns(index_clean, method='log')
            stocks_returns = self.calculate_returns(stocks_clean, method='log')
            
            # Uma vez que o "calculate_returns" acaba fazendo com que a primeira linha do dataset se torne NaN (pois
            # a primeira linha não tem como ter um parâmetro de aumento ou descréscimo percentual em relação a ninguém, ou seja
            # ,pois a primeira linha é o referencial), fazemos a exclusão da primeira linha em ambos índice e array das ações
            index_returns = index_returns.iloc[1:]
            stocks_returns = stocks_returns.iloc[1:]
            
            print(f"\n{'='*70}")
            print("PRÉ-PROCESSAMENTO FINALIZADO - RETORNOS CALCULADOS")
            print(f"{'='*70}")
            print(f"  Índice: {index_returns.shape}")
            print(f"  Ações: {stocks_returns.shape}")
            print(f"{'='*70}\n")
            
            return index_returns, stocks_returns
        
        else:
            print(f"\n{'='*70}")
            print("PRÉ-PROCESSAMENTO FINALIZADO - PREÇOS LIMPOS")
            print(f"{'='*70}")
            print(f"  Índice: {index_clean.shape}")
            print(f"  Ações: {stocks_clean.shape}")
            print(f"{'='*70}\n")
            
            return index_clean, stocks_clean


def main():
    """Função principal para teste do módulo."""
    # Carregar dados de exemplo
    print("Carregando dados de exemplo...")
    index_data = pd.read_csv('../data/sp100_index.csv', index_col=0, parse_dates=True)
    stocks_data = pd.read_csv('../data/sp100_stocks.csv', index_col=0, parse_dates=True)
    
    # Criar pré-processador
    preprocessor = DataPreprocessor(
        max_missing_pct=0.1,
        max_consecutive_missing=30,
        outlier_std_threshold=5.0
    )
    
    # Executar pipeline
    index_returns, stocks_returns = preprocessor.preprocess_pipeline(
        index_data, stocks_data, calculate_ret=True
    )
    
    # Salvar dados processados
    index_returns.to_csv('../data/sp100_index_returns.csv')
    stocks_returns.to_csv('../data/sp100_stocks_returns.csv')
    
    print("✓ Dados processados salvos com sucesso!")


if __name__ == "__main__":
    main()
