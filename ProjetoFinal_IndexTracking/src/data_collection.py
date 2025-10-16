"""
Módulo de Coleta de Dados
==========================

Este módulo é responsável pela coleta de dados históricos de índices e ações
através da API do Yahoo Finance.

Funcionalidades:
    - Download de dados históricos de índices (S&P100, IBOVESPA)
    - Download de dados de ações individuais que compõem os índices
    - Obtenção da lista de ativos que compõem cada índice
    - Tratamento de erros e validação dos dados coletados

"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

# remover warnings desnecessários do yfinance
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    """
    Classe para coletar dados históricos de mercado usando Yahoo Finance.
    
    Attributes:
        start_date (str): Data inicial para coleta (formato: 'YYYY-MM-DD')
        end_date (str): Data final para coleta (formato: 'YYYY-MM-DD')
    """
    
    def __init__(self, start_date: str, end_date: str):
        """
        Inicializa o coletor de dados.
        
        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'
            end_date: Data final no formato 'YYYY-MM-DD'
        """
        self.start_date = start_date
        self.end_date = end_date
        self.validate_dates()
    
    def validate_dates(self):
        """Valida se as datas estão no formato correto (ANO/MÊS/DIA) e são consistentes."""
        try:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            if start >= end:
                raise ValueError("Data inicial deve ser anterior à data final")
            
            if end > datetime.now():
                raise ValueError("Data final não pode ser no futuro")
                
            print(f"✓ Período de coleta validado: {self.start_date} até {self.end_date}")
            print(f"  Total de anos: {(end - start).days / 365.25:.1f}")
            
        except ValueError as e:
            raise ValueError(f"Erro no formato das datas: {e}")
    
    def download_index_data(self, index_ticker: str) -> pd.DataFrame:
        """
        Baixa dados históricos de um índice específico.
        
        Args:
            index_ticker: Ticker do índice (ex: '^BVSP', '^OEX')
            
        Returns:
            DataFrame com dados históricos do índice (OHLCV) com os seguintes campos:
                - Open, High, Low, Close, Adj Close, Volume e Data (em formato de timestamp, funciona como o "index" das linhas)
        """
        print(f"\n📊 Baixando dados do índice: {index_ticker}")
        
        try:
            data = yf.download(
                index_ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False
            )
            
            # Verificar se dados foram retornados
            if data.empty:
                raise ValueError(f"Nenhum dado retornado para {index_ticker}")
            
            print(f"✓ Download completo: {len(data)} dias de dados")
            
            # Como o índice tem formato de timestamp, tranformo para string para imprimir de forma legível
            print(f"  Período: {data.index[0].strftime('%Y-%m-%d')} até {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Dados faltantes: {data.isnull().sum().sum()}")
            
            return data
            
        except Exception as e:
            print(f"✗ Erro ao baixar dados de {index_ticker}: {e}")
            return pd.DataFrame()
    
    def download_stocks_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Baixa dados históricos de múltiplas ações.
        
        Args:
            tickers: Lista de tickers das ações
            
        Returns:
            DataFrame com preços de fechamento ajustados de todas as ações
        """
        print(f"\n📈 Baixando dados de {len(tickers)} ações...")
        
        # Dividir em batches de 20 ações (não sobrecarrega a API)
        batch_size = 20
        all_stocks_data = []
        
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  Lote {batch_num}/{total_batches}: baixando {len(batch)} ações...")
            
            try:
                # Download do batch
                data = yf.download(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    timeout=30  # Timeout maior para batches
                )
                
                if data.empty:
                    print(f"  ⚠ Lote {batch_num}: nenhum dado retornado")
                    continue
                
                # Extrair 'Adj Close'
                if isinstance(data.columns, pd.MultiIndex):
                    # Múltiplos tickers
                    if 'Adj Close' in data.columns.get_level_values(0):
                        batch_data = data['Adj Close']
                    else:
                        batch_data = data['Close']
                else:
                    # Apenas 1 ticker
                    if 'Adj Close' in data.columns:
                        batch_data = pd.DataFrame({batch[0]: data['Adj Close']})
                    else:
                        batch_data = pd.DataFrame({batch[0]: data['Close']})
                
                all_stocks_data.append(batch_data)
                print(f"  ✓ Lote {batch_num}: {len(batch_data.columns)} ações coletadas")
                
            except Exception as e:
                print(f"  ✗ Lote {batch_num}: Erro - {e}")
                continue
        
        # Consolidar todos os batches
        if not all_stocks_data:
            print(f"✗ Nenhum dado coletado!")
            return pd.DataFrame()
        
        stocks_data = pd.concat(all_stocks_data, axis=1)
        
        # Identificar ações que falharam
        tickers_baixados = set(stocks_data.columns)
        tickers_solicitados = set(tickers)
        tickers_falhados = tickers_solicitados - tickers_baixados
        
        print(f"\n✓ Download completo!")
        print(f"  Ações com sucesso: {len(stocks_data.columns)}/{len(tickers)}")
        print(f"  Período: {stocks_data.index[0].strftime('%Y-%m-%d')} até {stocks_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Dados faltantes: {stocks_data.isnull().sum().sum()}")
        
        if tickers_falhados:
            print(f"  ⚠ Ações não encontradas ({len(tickers_falhados)}): {', '.join(list(tickers_falhados)[:5])}{'...' if len(tickers_falhados) > 5 else ''}")
        
        return stocks_data
    
    def get_sp100_tickers(self) -> List[str]:
        """
        Obtém a lista de tickers que compõem o S&P 100.
        
        Returns:
            Lista de tickers do S&P 100
        """
        # Lista manualmente curada dos principais componentes do S&P 100
        # Fonte: https://en.wikipedia.org/wiki/S%26P_100
        sp100_tickers = [
            'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
            'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
            'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
            'CVX', 'DE', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX',
            'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM',
            'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD',
            'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE',
            'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM',
            'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TSLA', 'TXN',
            'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
        ]
        
        print(f"✓ Lista S&P 100: {len(sp100_tickers)} tickers")
        return sp100_tickers
    
    def get_ibov_tickers(self) -> List[str]:
        """
        Obtém a lista de tickers que compõem o IBOVESPA.
        
        Returns:
            Lista de tickers do IBOVESPA (com sufixo .SA)
        """
        # Lista atualizada dos principais componentes do IBOVESPA (Outubro 2025)
        # Tickers validados e atualizados - removidos os deslistados
        ibov_base = [
            'ABEV3', 'ALPA4', 'AMER3', 'ASAI3', 'AZUL4', 'B3SA3', 'BBAS3', 'BBDC3', 'BBDC4',
            'BBSE3', 'BPAC11', 'BRAP4', 'BRFS3', 'BRKM5', 'CMIG4', 'CMIN3', 'COGN3',
            'CPFE3', 'CPLE6', 'CSAN3', 'CSNA3', 'CVCB3', 'CYRE3', 'DXCO3', 'ECOR3',
            'EGIE3', 'ELET3', 'ELET6', 'EMBR3', 'ENEV3', 'ENGI11', 'EQTL3', 'EZTC3',
            'FLRY3', 'GGBR4', 'GOAU4', 'HAPV3', 'HYPE3', 'IGTI11', 'ITSA4', 'ITUB4',
            'KLBN11', 'LREN3', 'LWSA3', 'MGLU3', 'MRFG3', 'MRVE3', 'MULT3', 'PCAR3',
            'PETR3', 'PETR4', 'PETZ3', 'PRIO3', 'RADL3', 'RAIL3', 'RAIZ4', 'RDOR3',
            'RENT3', 'SANB11', 'SBSP3', 'SLCE3', 'SMTO3', 'SUZB3', 'TAEE11', 'TIMS3',
            'TOTS3', 'UGPA3', 'USIM5', 'VALE3', 'VBBR3', 'VIVT3', 'WEGE3', 'YDUQ3'
        ]
        
        # Adicionar sufixo .SA para Yahoo Finance
        ibov_tickers = [ticker + '.SA' for ticker in ibov_base]
        
        print(f"✓ Lista IBOVESPA: {len(ibov_tickers)} tickers")
        return ibov_tickers
    
    def collect_all_data(self, index_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Coleta todos os dados necessários para um índice específico.
        
        Args:
            index_name: Nome do índice ('SP100' ou 'IBOV')
            
        Returns:
            Tupla (dados_indice, dados_acoes)
        """
        print(f"\n{'='*70}")
        print(f"INICIANDO COLETA DE DADOS: {index_name}")
        print(f"{'='*70}")
        
        if index_name.upper() == 'SP100':
            index_ticker = '^OEX'
            tickers = self.get_sp100_tickers()
        elif index_name.upper() == 'IBOV':
            index_ticker = '^BVSP'
            tickers = self.get_ibov_tickers()
        else:
            raise ValueError("index_name deve ser 'SP100' ou 'IBOV'")
        
        # Coletar dados do índice
        index_data = self.download_index_data(index_ticker)
        
        # Coletar dados das ações
        stocks_data = self.download_stocks_data(tickers)
        
        print(f"\n{'='*70}")
        print(f"COLETA FINALIZADA: {index_name}")
        print(f"{'='*70}\n")
        
        return index_data, stocks_data


