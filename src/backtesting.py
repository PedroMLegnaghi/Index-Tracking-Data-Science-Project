"""
Módulo de Backtesting e Validação
==================================

Este módulo implementa funcionalidades para validação de estratégias de Index Tracking
tanto dentro da amostra (in-sample) quanto fora da amostra (out-of-sample).

Funcionalidades:
    - Split temporal dos dados (treino/teste)
    - Backtesting com janela rolante
    - Métricas de performance (Tracking Error, Correlação, Information Ratio)
    - Comparação entre diferentes estratégias
    - Validação cruzada temporal

Autor: Projeto Final - Bootcamp Data Science
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    Classe para realizar backtesting de estratégias de Index Tracking.
    """
    
    def __init__(self):
        """Inicializa o backtester."""
        print("✓ Backtester inicializado")
    
    def train_test_split(self, data: pd.DataFrame, train_size: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide dados em treino e teste temporalmente.
        
        Args:
            data: DataFrame com os dados
            train_size: Proporção para treino (0-1)
            
        Returns:
            Tupla (train_data, test_data)
        """
        split_idx = int(len(data) * train_size)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        
        print(f"\n📊 Split temporal realizado:")
        print(f"  Treino: {len(train)} períodos ({train.index[0].strftime('%Y-%m-%d')} até {train.index[-1].strftime('%Y-%m-%d')})")
        print(f"  Teste: {len(test)} períodos ({test.index[0].strftime('%Y-%m-%d')} até {test.index[-1].strftime('%Y-%m-%d')})")
        
        return train, test
    
    def calculate_tracking_error(self, portfolio_returns: np.ndarray, 
                                index_returns: np.ndarray) -> float:
        """Calcula Tracking Error."""
        return np.sqrt(np.mean((portfolio_returns - index_returns)**2))
    
    def calculate_correlation(self, portfolio_returns: np.ndarray,
                            index_returns: np.ndarray) -> float:
        """Calcula correlação."""
        return np.corrcoef(portfolio_returns, index_returns)[0, 1]
    
    def calculate_information_ratio(self, portfolio_returns: np.ndarray,
                                   index_returns: np.ndarray) -> float:
        """Calcula Information Ratio."""
        active_returns = portfolio_returns - index_returns
        return np.mean(active_returns) / np.std(active_returns) if np.std(active_returns) > 0 else 0
    
    def calculate_metrics(self, portfolio_returns: np.ndarray,
                         index_returns: np.ndarray) -> Dict:
        """
        Calcula todas as métricas de performance.
        
        Args:
            portfolio_returns: Retornos da carteira
            index_returns: Retornos do índice
            
        Returns:
            Dicionário com métricas
        """
        metrics = {
            'Tracking_Error': self.calculate_tracking_error(portfolio_returns, index_returns),
            'Tracking_Error_pct': self.calculate_tracking_error(portfolio_returns, index_returns) * 100,
            'Correlation': self.calculate_correlation(portfolio_returns, index_returns),
            'Information_Ratio': self.calculate_information_ratio(portfolio_returns, index_returns),
            'Portfolio_Return_Mean': np.mean(portfolio_returns),
            'Portfolio_Return_Std': np.std(portfolio_returns),
            'Index_Return_Mean': np.mean(index_returns),
            'Index_Return_Std': np.std(index_returns),
            'MAE': np.mean(np.abs(portfolio_returns - index_returns)),
            'RMSE': np.sqrt(np.mean((portfolio_returns - index_returns)**2))
        }
        
        return metrics
    
    def backtest_single_period(self, weights: np.ndarray, 
                              stocks_returns_test: pd.DataFrame,
                              index_returns_test: pd.Series) -> Dict:
        """
        Realiza backtest para um único período de teste.
        
        Args:
            weights: Pesos da carteira (treinados no período de treino)
            stocks_returns_test: Retornos das ações no período de teste
            index_returns_test: Retornos do índice no período de teste
            
        Returns:
            Dicionário com resultados do backtest
        """
        # Calcular retornos da carteira
        portfolio_returns = stocks_returns_test.values @ weights
        
        # Calcular métricas
        metrics = self.calculate_metrics(portfolio_returns, index_returns_test.values.squeeze())
        
        # Adicionar informações adicionais
        metrics['portfolio_returns'] = portfolio_returns
        metrics['index_returns'] = index_returns_test.values.squeeze()
        metrics['dates'] = index_returns_test.index
        
        return metrics
    
    def rolling_window_backtest(self, index_returns: pd.Series,
                                stocks_returns: pd.DataFrame,
                                optimizer_func,
                                train_window: int = 504,  # ~2 anos
                                test_window: int = 126,   # ~6 meses
                                step_size: int = 252) -> List[Dict]:
        """
        Realiza backtest com janela rolante.
        
        Args:
            index_returns: Retornos do índice
            stocks_returns: Retornos das ações
            optimizer_func: Função que treina o modelo e retorna pesos
            train_window: Tamanho da janela de treino (em dias)
            test_window: Tamanho da janela de teste (em dias)
            step_size: Passo para mover a janela
            
        Returns:
            Lista de dicionários com resultados de cada período
        """
        print(f"\n{'='*70}")
        print("BACKTESTING COM JANELA ROLANTE")
        print(f"{'='*70}")
        print(f"  Janela de treino: {train_window} dias (~{train_window/252:.1f} anos)")
        print(f"  Janela de teste: {test_window} dias (~{test_window/252:.1f} anos)")
        print(f"  Passo: {step_size} dias\n")
        
        results = []
        n_windows = 0
        
        # Iterar sobre janelas
        start_idx = 0
        while start_idx + train_window + test_window <= len(index_returns):
            n_windows += 1
            
            # Definir índices
            train_start = start_idx
            train_end = start_idx + train_window
            test_start = train_end
            test_end = min(test_start + test_window, len(index_returns))
            
            # Extrair dados
            index_train = index_returns.iloc[train_start:train_end]
            stocks_train = stocks_returns.iloc[train_start:train_end]
            index_test = index_returns.iloc[test_start:test_end]
            stocks_test = stocks_returns.iloc[test_start:test_end]
            
            print(f"\n--- Janela {n_windows} ---")
            print(f"  Treino: {index_train.index[0].strftime('%Y-%m-%d')} até {index_train.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Teste: {index_test.index[0].strftime('%Y-%m-%d')} até {index_test.index[-1].strftime('%Y-%m-%d')}")
            
            # Treinar modelo
            try:
                weights = optimizer_func(index_train, stocks_train)
                
                # Backtest
                result = self.backtest_single_period(weights, stocks_test, index_test)
                result['window'] = n_windows
                result['train_period'] = (index_train.index[0], index_train.index[-1])
                result['test_period'] = (index_test.index[0], index_test.index[-1])
                result['weights'] = weights  # ✅ ADICIONAR PESOS AO RESULTADO
                
                results.append(result)
                
                print(f"  ✓ Tracking Error: {result['Tracking_Error_pct']:.4f}%")
                print(f"  ✓ Correlação: {result['Correlation']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Erro: {e}")
            
            # Mover janela
            start_idx += step_size
        
        print(f"\n{'='*70}")
        print(f"BACKTEST FINALIZADO: {n_windows} janelas testadas")
        print(f"{'='*70}\n")
        
        return results
    
    def evaluate_out_of_sample(self, results: List[Dict]) -> pd.DataFrame:
        """
        Avalia resultados agregados do backtest out-of-sample.
        
        Args:
            results: Lista de resultados de backtest
            
        Returns:
            DataFrame com métricas agregadas
        """
        print(f"\n{'='*70}")
        print("AVALIAÇÃO OUT-OF-SAMPLE")
        print(f"{'='*70}\n")
        
        # Extrair métricas
        metrics_list = []
        for i, result in enumerate(results, 1):
            metrics_list.append({
                'Window': i,
                'Test_Start': result['test_period'][0].strftime('%Y-%m-%d'),
                'Test_End': result['test_period'][1].strftime('%Y-%m-%d'),
                'Tracking_Error_%': result['Tracking_Error_pct'],
                'Correlation': result['Correlation'],
                'Information_Ratio': result['Information_Ratio'],
                'MAE': result['MAE'],
                'RMSE': result['RMSE']
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Estatísticas agregadas
        print("Estatísticas Agregadas:")
        print(f"  Tracking Error médio: {metrics_df['Tracking_Error_%'].mean():.4f}%")
        print(f"  Tracking Error std: {metrics_df['Tracking_Error_%'].std():.4f}%")
        print(f"  Correlação média: {metrics_df['Correlation'].mean():.4f}")
        print(f"  Information Ratio médio: {metrics_df['Information_Ratio'].mean():.4f}")
        
        print(f"\n{'='*70}")
        print(metrics_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        return metrics_df


