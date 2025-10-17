"""
Módulo de Visualização de Resultados
====================================

Este módulo fornece funções para visualizar resultados das análises e otimizações.

Autor: Projeto Final - Bootcamp Data Science
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class ResultsVisualizer:
    """Classe para visualização de resultados."""
    
    def __init__(self, figsize=(14, 6), dpi=100):
        """Inicializa o visualizador."""
        self.figsize = figsize
        self.dpi = dpi
        print(f"✓ ResultsVisualizer inicializado")
    
    def plot_portfolio_vs_index(self, portfolio_returns, index_returns, dates,
                                title="Carteira vs Índice", save_path=None):
        """Plota comparação entre carteira e índice."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calcular retornos cumulativos normalizados para base 100
        # Isso garante que ambas as linhas comecem em 100
        cum_portfolio = 100 * (1 + pd.Series(portfolio_returns, index=dates)).cumprod()
        cum_index = 100 * (1 + pd.Series(index_returns, index=dates)).cumprod()
        
        ax.plot(dates, cum_index, label='Índice', linewidth=2.5, color='steelblue', linestyle='-')
        ax.plot(dates, cum_portfolio, label='Carteira', linewidth=2, color='coral', linestyle='--', alpha=0.9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Data", fontsize=12)
        ax.set_ylabel("Retorno Cumulativo (Base 100)", fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Adicionar anotação com informações
        ax.text(0.02, 0.98, f'Início: 100.00', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_portfolio_vs_index_with_te(self, portfolio_returns, index_returns, dates,
                                        title="Carteira vs Índice", save_path=None):
        """Plota comparação entre carteira e índice COM painel de Tracking Error."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3), 
                                       dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]})
        
        # Painel superior: Retornos cumulativos
        cum_portfolio = 100 * (1 + pd.Series(portfolio_returns, index=dates)).cumprod()
        cum_index = 100 * (1 + pd.Series(index_returns, index=dates)).cumprod()
        
        ax1.plot(dates, cum_index, label='Índice', linewidth=2.5, color='steelblue', linestyle='-')
        ax1.plot(dates, cum_portfolio, label='Carteira', linewidth=2, color='coral', linestyle='--', alpha=0.9)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel("Retorno Cumulativo (Base 100)", fontsize=12)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Painel inferior: Diferença (Tracking Error ao longo do tempo)
        diff = cum_portfolio - cum_index
        ax2.fill_between(dates, 0, diff, color='red', alpha=0.3, label='Tracking Error')
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')
        ax2.plot(dates, diff, color='darkred', linewidth=1.5)
        
        ax2.set_xlabel("Data", fontsize=12)
        ax2.set_ylabel("Diferença", fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_tracking_error_evolution(self, results, save_path=None):
        """Plota evolução do tracking error."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        windows = [r['window'] for r in results]
        te = [r['Tracking_Error_pct'] for r in results]
        
        ax.plot(windows, te, marker='o', linewidth=2, markersize=8)
        ax.axhline(np.mean(te), color='r', linestyle='--', label=f'Média: {np.mean(te):.4f}%')
        
        ax.set_title("Evolução do Tracking Error (Out-of-Sample)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Janela de Teste", fontsize=12)
        ax.set_ylabel("Tracking Error (%)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_weights_distribution(self, weights_df, top_n=15, save_path=None):
        """Plota distribuição de pesos da carteira."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        top_weights = weights_df.head(top_n)
        
        ax.barh(range(len(top_weights)), top_weights['Weight']*100, color='steelblue')
        ax.set_yticks(range(len(top_weights)))
        ax.set_yticklabels(top_weights['Stock'])
        ax.invert_yaxis()
        
        ax.set_title(f"Top {top_n} Ativos - Distribuição de Pesos", fontsize=14, fontweight='bold')
        ax.set_xlabel("Peso (%)", fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_sensitivity_analysis(self, sensitivity_df, save_path=None):
        """Plota análise de sensibilidade."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.plot(sensitivity_df['K'], sensitivity_df['Tracking_Error_pct'], 
               marker='o', linewidth=2, markersize=10, color='darkred')
        
        ax.set_title("Análise de Sensibilidade: Tracking Error vs Número de Ativos",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Número de Ativos (K)", fontsize=12)
        ax.set_ylabel("Tracking Error (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_all_rolling_windows(self, rolling_results, index_returns_full, 
                                 stocks_returns_full, index_name="Índice", 
                                 max_windows=5, save_path=None):
        """
        Plota o índice real junto com a trajetória COMPLETA das carteiras 
        de diferentes janelas rolantes (do início ao fim de TODOS os dados).
        
        Args:
            rolling_results: Lista de resultados do rolling_window_backtest
            index_returns_full: Series completa dos retornos do índice
            stocks_returns_full: DataFrame completo dos retornos das ações
            index_name: Nome do índice (ex: "S&P 100", "IBOVESPA")
            max_windows: Número máximo de janelas a plotar (default: 5)
            save_path: Caminho para salvar o gráfico
        """
        fig, ax = plt.subplots(figsize=(16, 8), dpi=self.dpi)
        
        # Calcular retorno cumulativo do índice completo (base 100)
        dates_full = index_returns_full.index
        cum_index_full = 100 * (1 + index_returns_full).cumprod()
        
        # Plotar índice real (linha grossa azul escuro)
        ax.plot(dates_full, cum_index_full, label=f'{index_name} (Real)', 
               linewidth=3.5, color='#2c3e50', alpha=1.0, zorder=10)
        
        # Cores para cada janela (paleta vibrante e distinta)
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#3498db']
        
        # Selecionar apenas as primeiras max_windows janelas
        selected_results = rolling_results[:max_windows]
        
        # Plotar cada carteira (trajetória COMPLETA desde o início)
        for i, result in enumerate(selected_results):
            window_num = result['window']
            weights = result['weights']  # Pesos otimizados nesta janela
            test_start_date = result['dates'][0]
            test_end_date = result['dates'][-1]
            te = result['Tracking_Error_pct']
            
            # Calcular retornos da carteira em TODOS os períodos (não só no teste!)
            portfolio_returns_full = stocks_returns_full.values @ weights
            
            # Calcular retorno cumulativo completo (base 100)
            cum_portfolio_full = 100 * (1 + pd.Series(portfolio_returns_full, index=dates_full)).cumprod()
            
            # Formatar datas para legenda
            start_date = test_start_date.strftime('%Y-%m')
            end_date = test_end_date.strftime('%Y-%m')
            
            # Plotar trajetória completa da carteira
            color = colors[i % len(colors)]
            ax.plot(dates_full, cum_portfolio_full, 
                   label=f'Carteira {window_num}: testada em {start_date}-{end_date} (TE={te:.2f}%)',
                   linewidth=2.0, color=color, alpha=0.75, linestyle='--')
            
            # Adicionar marcador no período de teste
            test_mask = (dates_full >= test_start_date) & (dates_full <= test_end_date)
            ax.plot(dates_full[test_mask], cum_portfolio_full[test_mask],
                   linewidth=3.0, color=color, alpha=0.95, zorder=5)
        
        # Configurações do gráfico
        ax.set_title(f'Evolução Temporal: {index_name} vs {max_windows} Carteiras Diferentes\n(Linha grossa = período de teste out-of-sample)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel("Data", fontsize=13)
        ax.set_ylabel("Retorno Cumulativo (Base 100)", fontsize=13)
        
        # Legenda fora do gráfico (à direita) para evitar sobreposição
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, 
                 framealpha=0.95, edgecolor='gray', fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adicionar anotação com informações
        avg_te = np.mean([r['Tracking_Error_pct'] for r in selected_results])
        
        textstr = f'Carteiras exibidas: {len(selected_results)}\nTE médio (OOS): {avg_te:.3f}%\n\n💡 Cada carteira foi treinada\nem período diferente'
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        plt.show()

