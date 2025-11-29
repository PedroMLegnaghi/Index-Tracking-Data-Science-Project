"""
Módulo de Análise Exploratória de Dados (EDA)
==============================================

Este módulo fornece funcionalidades para análise exploratória de dados financeiros,
incluindo estatísticas descritivas, análise de correlação e visualizações.

Funcionalidades:
    - Estatísticas descritivas completas
    - Análise de correlação entre ativos
    - Visualização de séries temporais
    - Análise de distribuição de retornos
    - Matriz de correlação e heatmaps
    - Análise de volatilidade
    - Identificação de períodos de crise

Autor: Projeto Final - Bootcamp Data Science
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExploratoryAnalyzer:
    """
    Classe para realizar análise exploratória de dados financeiros.
    
    Attributes:
        figsize (tuple): Tamanho padrão das figuras
        dpi (int): Resolução dos gráficos
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 6), dpi: int = 100):
        """
        Inicializa o analisador exploratório.
        
        Args:
            figsize: Tamanho padrão das figuras (largura, altura)
            dpi: Resolução dos gráficos
        """
        self.figsize = figsize
        self.dpi = dpi
        
        print(f"✓ ExploratoryAnalyzer inicializado")
        print(f"  - Figsize: {figsize}")
        print(f"  - DPI: {dpi}")
    
    def descriptive_statistics(self, data: pd.DataFrame, name: str = "Data") -> pd.DataFrame:
        """
        Calcula estatísticas descritivas completas.
        
        Args:
            data: DataFrame ou Series com os dados
            name: Nome do dataset para impressão
            
        Returns:
            DataFrame com estatísticas descritivas
        """
        print(f"\n{'='*70}")
        print(f"ESTATÍSTICAS DESCRITIVAS: {name}")
        print(f"{'='*70}")
        
        # Converter Series para DataFrame (1 coluna)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=name)
        
        # Calcular estatísticas
        stats = pd.DataFrame({
            'count': data.count(),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            '25%': data.quantile(0.25),
            '50%': data.quantile(0.50),
            '75%': data.quantile(0.75),
            'max': data.max(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        })
        
        print(stats)
        print(f"{'='*70}\n")
        
        return stats
    
    def plot_time_series(self, data: pd.DataFrame, title: str = "Séries Temporais",
                        normalize: bool = False, save_path: Optional[str] = None):
        """
        Plota séries temporais.
        
        Args:
            data: DataFrame ou Series com séries temporais
            title: Título do gráfico
            normalize: Se True, normaliza para base 100
            save_path: Caminho para salvar a figura (opcional)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Converter Series para DataFrame (1 coluna)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name if data.name else "Value")
        
        if normalize:
            # Normalizar para base 100
            data_plot = data / data.iloc[0] * 100
            ylabel = "Valor Normalizado (Base 100)"
        else:
            data_plot = data
            ylabel = "Valor"
        
        # Plotar
        for col in data_plot.columns:
            ax.plot(data_plot.index, data_plot[col], label=col, linewidth=1.5, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, returns: pd.DataFrame, 
                                 cols: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        Plota a distribuição dos retornos com histograma e curva normal.
        
        Args:
            returns: DataFrame ou Series com retornos
            cols: Colunas específicas para plotar (se None, plota todas)
            save_path: Caminho para salvar a figura (opcional)
        """
        # Converter Series para DataFrame (1 coluna)
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=returns.name if returns.name else "Returns")
        
        if cols is None:
            cols = returns.columns[:min(6, len(returns.columns))]  # Máximo 6 colunas
        
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3  # 3 colunas por linha
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows*4), dpi=self.dpi)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(cols):
            ax = axes[i]
            
            # Histograma
            ax.hist(returns[col].dropna(), bins=50, density=True, alpha=0.6, 
                   color='skyblue', edgecolor='black')
            
            # Curva normal teórica
            mu = returns[col].mean()
            sigma = returns[col].std()
            x = np.linspace(returns[col].min(), returns[col].max(), 100)
            ax.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2),
                   'r-', linewidth=2, label='Normal')
            
            ax.set_title(f"Distribuição: {col}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Retorno", fontsize=10)
            ax.set_ylabel("Densidade", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remover subplots vazios
        for i in range(n_cols, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Matriz de Correlação",
                               method: str = 'pearson', save_path: Optional[str] = None):
        """
        Plota matriz de correlação como heatmap.
        
        Args:
            data: DataFrame com os dados
            title: Título do gráfico
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            save_path: Caminho para salvar a figura (opcional)
        """
        # Calcular correlação
        corr_matrix = data.corr(method=method)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
        
        return corr_matrix
    
    def analyze_correlation_with_index(self, index_returns: pd.Series, 
                                      stocks_returns: pd.DataFrame,
                                      top_n: int = 10) -> pd.DataFrame:
        """
        Analisa correlação de cada ação com o índice.
        
        Args:
            index_returns: Série com retornos do índice
            stocks_returns: DataFrame com retornos das ações
            top_n: Número de ações mais/menos correlacionadas a mostrar
            
        Returns:
            DataFrame com correlações ordenadas
        """
        print(f"\n{'='*70}")
        print("ANÁLISE DE CORRELAÇÃO COM O ÍNDICE")
        print(f"{'='*70}")
        
        # Calcular correlação
        correlations = stocks_returns.corrwith(index_returns.squeeze()).sort_values(ascending=False)
        
        print(f"\n✓ Top {top_n} ações MAIS correlacionadas:")
        print(correlations.head(top_n))
        
        print(f"\n✓ Top {top_n} ações MENOS correlacionadas:")
        print(correlations.tail(top_n))
        
        print(f"\n✓ Estatísticas de correlação:")
        print(f"  Correlação média: {correlations.mean():.4f}")
        print(f"  Correlação mediana: {correlations.median():.4f}")
        print(f"  Desvio padrão: {correlations.std():.4f}")
        print(f"  Mínimo: {correlations.min():.4f}")
        print(f"  Máximo: {correlations.max():.4f}")
        print(f"{'='*70}\n")
        
        return correlations
    
    def plot_correlation_distribution(self, correlations: pd.Series,
                                     save_path: Optional[str] = None):
        """
        Plota distribuição das correlações com o índice.
        
        Args:
            correlations: Série com correlações
            save_path: Caminho para salvar a figura (opcional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Histograma
        axes[0].hist(correlations, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(correlations.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Média: {correlations.mean():.3f}')
        axes[0].axvline(correlations.median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mediana: {correlations.median():.3f}')
        axes[0].set_xlabel("Correlação com o Índice", fontsize=12)
        axes[0].set_ylabel("Frequência", fontsize=12)
        axes[0].set_title("Distribuição das Correlações", fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot
        axes[1].boxplot(correlations, vert=True)
        axes[1].set_ylabel("Correlação com o Índice", fontsize=12)
        axes[1].set_title("Boxplot das Correlações", fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def analyze_volatility(self, returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calcula e analisa volatilidade rolante.
        
        Args:
            returns: DataFrame com retornos
            window: Janela para cálculo da volatilidade rolante
            
        Returns:
            DataFrame com volatilidades
        """
        print(f"\n📊 Calculando volatilidade rolante (janela: {window} dias)...")
        
        # Volatilidade rolante (anualizada)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
        print(f"✓ Volatilidade média: {volatility.mean().mean()*100:.2f}%")
        print(f"  Volatilidade mínima: {volatility.min().min()*100:.2f}%")
        print(f"  Volatilidade máxima: {volatility.max().max()*100:.2f}%")
        
        return volatility
    
    def plot_volatility(self, volatility: pd.DataFrame, 
                       cols: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
        """
        Plota volatilidade rolante ao longo do tempo.
        
        Args:
            volatility: DataFrame com volatilidades
            cols: Colunas específicas para plotar (se None, plota todas)
            save_path: Caminho para salvar a figura (opcional)
        """
        if cols is None:
            cols = volatility.columns[:min(10, len(volatility.columns))]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for col in cols:
            ax.plot(volatility.index, volatility[col] * 100, label=col, alpha=0.7)
        
        ax.set_title("Volatilidade Rolante (Anualizada)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel("Volatilidade (%)", fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def identify_crisis_periods(self, index_returns: pd.Series, 
                               threshold: float = -0.05) -> pd.DataFrame:
        """
        Identifica períodos de crise (grandes quedas no índice).
        
        Args:
            index_returns: Série com retornos do índice
            threshold: Threshold para considerar crise (ex: -5%)
            
        Returns:
            DataFrame com períodos de crise
        """
        print(f"\n🔍 Identificando períodos de crise (threshold: {threshold*100:.1f}%)...")
        
        # Encontrar dias com quedas severas
        crisis_days = index_returns[index_returns < threshold]
        
        if len(crisis_days) > 0:
            print(f"✓ {len(crisis_days)} dias de crise identificados:")
            print(crisis_days.sort_values().head(10))
        else:
            print(f"✓ Nenhum dia de crise identificado com threshold {threshold*100:.1f}%")
        
        return crisis_days
    
    def full_eda_report(self, index_returns: pd.Series, stocks_returns: pd.DataFrame,
                       index_name: str = "Índice", save_dir: Optional[str] = None):
        """
        Gera relatório completo de EDA.
        
        Args:
            index_returns: Retornos do índice
            stocks_returns: Retornos das ações
            index_name: Nome do índice
            save_dir: Diretório para salvar gráficos (opcional)
        """
        print(f"\n{'='*70}")
        print(f"RELATÓRIO COMPLETO DE ANÁLISE EXPLORATÓRIA: {index_name}")
        print(f"{'='*70}\n")
        
        # 1. Estatísticas descritivas
        print("\n--- 1. ESTATÍSTICAS DO ÍNDICE ---")
        self.descriptive_statistics(index_returns, name=f"{index_name} - Retornos")
        
        print("\n--- 2. ESTATÍSTICAS DAS AÇÕES ---")
        stats_stocks = self.descriptive_statistics(stocks_returns, name="Ações - Retornos")
        
        # 2. Séries temporais (cumulativo)
        print("\n--- 3. VISUALIZAÇÃO DE RETORNOS CUMULATIVOS ---")
        cumulative = (1 + stocks_returns).cumprod()
        
        cumulative_index = (1 + index_returns).cumprod()
        
        # Plotar índice
        save_path_1 = f"{save_dir}/01_retornos_cumulativos_indice.png" if save_dir else None
        self.plot_time_series(cumulative_index, 
                            title=f"Retornos Cumulativos: {index_name}",
                            normalize=True, save_path=save_path_1)
        
        # Plotar algumas ações
        save_path_2 = f"{save_dir}/02_retornos_cumulativos_acoes.png" if save_dir else None
        self.plot_time_series(cumulative.iloc[:, :10], 
                            title="Retornos Cumulativos das 10 primeiras ações",
                            normalize=True, save_path=save_path_2)
        
        # 3. Distribuição de retornos
        print("\n--- 4. DISTRIBUIÇÃO DE RETORNOS ---")
        save_path_3 = f"{save_dir}/03_distribuicao_retornos.png" if save_dir else None
        self.plot_returns_distribution(stocks_returns, save_path=save_path_3)
        
        # 4. Correlação com índice
        print("\n--- 5. CORRELAÇÃO COM O ÍNDICE ---")
        correlations = self.analyze_correlation_with_index(index_returns, stocks_returns)
        
        save_path_4 = f"{save_dir}/04_distribuicao_correlacoes.png" if save_dir else None
        self.plot_correlation_distribution(correlations, save_path=save_path_4)
        
        # 5. Matriz de correlação (subset)
        print("\n--- 6. MATRIZ DE CORRELAÇÃO ---")
        save_path_5 = f"{save_dir}/05_matriz_correlacao.png" if save_dir else None
        top_corr_stocks = correlations.head(20).index
        self.plot_correlation_matrix(stocks_returns[top_corr_stocks], 
                                     title="Matriz de Correlação: Top 20 Ações",
                                     save_path=save_path_5)
        
        # 6. Volatilidade
        print("\n--- 7. ANÁLISE DE VOLATILIDADE ---")
        volatility = self.analyze_volatility(stocks_returns, window=30)
        
        save_path_6 = f"{save_dir}/06_volatilidade_rolante.png" if save_dir else None
        self.plot_volatility(volatility, cols=top_corr_stocks[:5], save_path=save_path_6)
        
        # 7. Períodos de crise
        print("\n--- 8. PERÍODOS DE CRISE ---")
        crisis = self.identify_crisis_periods(index_returns, threshold=-0.03)
        
        print(f"\n{'='*70}")
        print("RELATÓRIO EDA FINALIZADO")
        print(f"{'='*70}\n")
        
        return {
            'statistics': stats_stocks,
            'correlations': correlations,
            'volatility': volatility,
            'crisis_periods': crisis
        }

