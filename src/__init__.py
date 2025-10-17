"""
Projeto Final - Index Tracking
===============================

Sistema modular para replicação de índices de mercado (S&P 100, IBOVESPA)
utilizando otimização matemática.

Módulos disponíveis:
    - data_collection: Coleta de dados via Yahoo Finance
    - data_preprocessing: Limpeza e preparação de dados
    - exploratory_analysis: Análise exploratória de dados
    - optimization_model: Modelo de otimização (Index Tracking)
    - backtesting: Validação in-sample e out-of-sample
    - visualization: Visualizações de resultados

Autor: Bootcamp Data Science BAH + FINOR
Data: Outubro 2025
"""

__version__ = "1.0.0"
__author__ = "Bootcamp Data Science - 4ª Edição"

from .data_collection import DataCollector
from .data_preprocessing import DataPreprocessor
from .exploratory_analysis import ExploratoryAnalyzer
from .optimization_model import IndexTrackingOptimizer
from .backtesting import Backtester
from .visualization import ResultsVisualizer

__all__ = [
    'DataCollector',
    'DataPreprocessor',
    'ExploratoryAnalyzer',
    'IndexTrackingOptimizer',
    'Backtester',
    'ResultsVisualizer'
]
