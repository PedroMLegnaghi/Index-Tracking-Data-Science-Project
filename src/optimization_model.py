"""
Módulo de Otimização para Index Tracking
=========================================

Este módulo implementa modelos de otimização para replicar índices de mercado
com um número reduzido de ativos (Index Tracking).

Funcionalidades:
    - Modelo não restrito (Unconstrained IT)
    - Modelo restrito com número máximo de ativos (Constrained IT)
    - Otimização usando CVXPY
    - Diferentes funções objetivo (tracking error, variância)
    - Análise de sensibilidade para número de ativos

Formulação Matemática:
    min (1/T) Σ (Σ w_i * r_{t,i} - R_t)²
    
    s.t.:
        Σ w_i = 1  (soma dos pesos = 1)
        w_i ≥ 0    (sem venda a descoberto)
        w_i ≤ z_i  (ligação com variável binária)
        Σ z_i ≤ K  (máximo K ativos)
        z_i ∈ {0,1}

Autor: Projeto Final - Bootcamp Data Science
Data: Outubro 2025
"""
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

import gurobipy as gp
from gurobipy import GRB


class IndexTrackingOptimizer:
    """
    Classe para otimização de carteiras de Index Tracking.
    
    Attributes:
        index_returns: Retornos do índice a ser replicado
        stocks_returns: Retornos das ações disponíveis
    
    OBS: O solver utilizado é o Gurobi.
    """

    def __init__(self, index_returns: pd.Series, stocks_returns: pd.DataFrame):
        """
    Inicializa o otimizador.
    
    Args:
        index_returns: Série com retornos do índice
        stocks_returns: DataFrame com retornos das ações
    """
        self.index_returns = index_returns.squeeze().values
        self.stocks_returns = stocks_returns.values
        self.stock_names = stocks_returns.columns.tolist()
        self.n_stocks = len(self.stock_names)
        self.n_periods = len(self.index_returns)
        self.solver = 'GUROBI'

        print(f"✓ IndexTrackingOptimizer inicializado:")
        print(f"  - Número de ações: {self.n_stocks}")
        print(f"  - Períodos: {self.n_periods}")
        print(f"  - Solver: {self.solver}")

    def _build_qp_data(self):
        R = self.stocks_returns  # (T, n)
        r = self.index_returns   # (T,)
        T = float(self.n_periods)
        Q = (R.T @ R) / T        # (n, n)
        b = (R.T @ r) / T        # (n,)
        lin_coefs = -2.0 * b     # linear part in objective
        return Q, lin_coefs, R, r

    def optimize_unconstrained(self, verbose: bool = False) -> Dict:
        """
        Resolve o problema de Index Tracking SEM restrição no número de ativos.
        
        Minimiza: (1/T) * ||R @ w - r_index||²
        
        Subject to:
            sum(w) = 1
            w >= 0
        
        Args:
            verbose: Se True, imprime informações do solver
            
        Returns:
            Dicionário com resultados da otimização
        """
        
        if verbose:
            print(f"\n{'='*70}")
            print("OTIMIZANDO: Modelo Não Restrito (Gurobi)")
            print(f"{'='*70}")

        Q, lin_coefs, R, r = self._build_qp_data()
        n = self.n_stocks

        try:
            model = gp.Model("unconstrained_it")
            model.setParam('OutputFlag', 1 if verbose else 0)

            w_vars = model.addVars(n, lb=0.0, ub=1.0, name="w")
            model.addConstr(gp.quicksum(w_vars[i] for i in range(n)) == 1.0, name="sum_weights")

            # Construir objeto quadrático (w^T Q w) + lin_coefs^T w
            quad = gp.QuadExpr()
            # linear terms
            for i in range(n):
                if lin_coefs[i] != 0.0:
                    quad.add(lin_coefs[i] * w_vars[i])
            # quadratic terms
            for i in range(n):
                for j in range(i, n):
                    coef = Q[i, j]
                    if coef != 0.0:
                        if i == j:
                            quad.add(coef * w_vars[i] * w_vars[i])
                        else:
                            quad.add(2.0 * coef * w_vars[i] * w_vars[j])

            model.setObjective(quad, GRB.MINIMIZE)
            model.optimize()

            if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                raise ValueError(f"Gurobi status: {model.Status}")

            weights = np.array([w_vars[i].X for i in range(n)])
            weights_df = pd.DataFrame({'Stock': self.stock_names, 'Weight': weights}).sort_values('Weight', ascending=False)

            portfolio_ret = R @ weights
            tracking_error_value = np.sqrt(np.mean((portfolio_ret - r) ** 2))

            # objective numeric value (w^T Q w + lin_coefs^T w)
            objective_value = float(weights @ (Q @ weights) + lin_coefs @ weights)

            n_active_assets = int(np.sum(weights > 1e-6))

            if verbose:
                print(f"\n✓ Otimização bem-sucedida!")
                print(f"  Status (Gurobi): {model.Status}")
                print(f"  Objective (QP): {objective_value:.8f}")
                print(f"  Tracking Error: {tracking_error_value:.6f} ({tracking_error_value*100:.4f}%)")
                print(f"  Ativos ativos (peso > 1e-6): {n_active_assets}")
                print(f"\n  Ativos com peso > 0:")
            sel = weights_df[weights_df['Weight'] > 1e-12]
            if sel.empty:
                print(weights_df.head(10).to_string(index=False))
            else:
                print(sel.to_string(index=False))
            print(f"{'='*70}\n")

            return {
                'weights': weights,
                'weights_df': weights_df,
                'n_active_assets': n_active_assets,
                'tracking_error': tracking_error_value,
                'objective_value': objective_value,
                'status': model.Status,
                'portfolio_returns': portfolio_ret
            }

        except Exception as e:
            print(f"✗ Erro na otimização (Gurobi): {e}")
            return None

    def optimize_constrained(self, max_assets: int, time_limit: Optional[float] = 10, verbose: bool = False) -> Dict:
        """
        Resolve o problema de Index Tracking COM restrição no número de ativos.
        
        Minimiza: (1/T) * ||R @ w - r_index||²
        
        Subject to:
            sum(w) = 1
            w >= 0
            w <= z * M  (Big-M constraint)
            sum(z) <= K
            z binary
        
        Args:
            max_assets: Número máximo de ativos permitidos
            verbose: Se True, imprime informações do solver
            
        Returns:
            Dicionário com resultados da otimização
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"OTIMIZANDO: Modelo Restrito (Gurobi, K={max_assets})")
            print(f"{'='*70}")

        Q, lin_coefs, R, r = self._build_qp_data()
        n = self.n_stocks
        M = 1.0  # Big-M (ajustável conforme necessário)

        try:
            model = gp.Model("constrained_it")
            model.setParam('OutputFlag', 1 if verbose else 0)
            if time_limit is not None:
                model.setParam('TimeLimit', float(time_limit))

            w_vars = model.addVars(n, lb=0.0, ub=1.0, name="w")
            z_vars = model.addVars(n, vtype=GRB.BINARY, name="z")

            model.addConstr(gp.quicksum(w_vars[i] for i in range(n)) == 1.0, name="sum_w")
            model.addConstr(gp.quicksum(z_vars[i] for i in range(n)) <= max_assets, name="cardinality")

            for i in range(n):
                model.addConstr(w_vars[i] <= M * z_vars[i], name=f"bigM_{i}")

            # Quadratic objective
            quad = gp.QuadExpr()
            for i in range(n):
                if lin_coefs[i] != 0.0:
                    quad.add(lin_coefs[i] * w_vars[i])
            for i in range(n):
                for j in range(i, n):
                    coef = Q[i, j]
                    if coef != 0.0:
                        if i == j:
                            quad.add(coef * w_vars[i] * w_vars[i])
                        else:
                            quad.add(2.0 * coef * w_vars[i] * w_vars[j])

            model.setObjective(quad, GRB.MINIMIZE)
            model.optimize()

            if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                # fallback para heurística relaxada
                if verbose:
                    print("ℹ️  Gurobi não encontrou solução ótima; usando heurística relaxada Top-K")
                return self._optimize_constrained_relaxed(max_assets, verbose)

            weights = np.array([w_vars[i].X for i in range(n)])
            selected = np.array([z_vars[i].X for i in range(n)])
            weights_df = pd.DataFrame({
                'Stock': self.stock_names,
                'Weight': weights,
                'Selected': selected
            })
            sel_df = weights_df[weights_df['Selected'] > 0.5].sort_values('Weight', ascending=False)

            portfolio_ret = R @ weights
            tracking_error_value = np.sqrt(np.mean((portfolio_ret - r) ** 2))
            objective_value = float(weights @ (Q @ weights) + lin_coefs @ weights)
            n_active_assets = int(np.sum(selected > 0.5))

            if verbose:
                print(f"\n✓ Otimização bem-sucedida!")
                print(f"  Status (Gurobi): {model.Status}")
                print(f"  Objective (QP): {objective_value:.8f}")
                print(f"  Tracking Error: {tracking_error_value:.6f} ({tracking_error_value*100:.4f}%)")
                print(f"  Ativos selecionados: {n_active_assets} / {max_assets}")
                print(f"\n  Carteira otimizada:")
                print(sel_df.to_string(index=False))
                print(f"{'='*70}\n")

            return {
                'weights': weights,
                'weights_df': sel_df,
                'selected': selected,
                'n_active_assets': n_active_assets,
                'tracking_error': tracking_error_value,
                'objective_value': objective_value,
                'status': model.Status,
                'portfolio_returns': portfolio_ret
            }

        except Exception as e:
            print(f"✗ Erro na otimização (Gurobi MIP): {e}")
            return None

    def _optimize_constrained_relaxed(self, max_assets: int, verbose: bool = False) -> Dict:
        """
        Resolve versão relaxada do problema (sem variáveis binárias).
        Seleciona os K ativos com maiores pesos da solução não restrita.
        
        Esta heurística é muito eficiente para Index Tracking e geralmente
        encontra soluções muito próximas (ou idênticas) ao ótimo global.
        
        Args:
            max_assets: Número máximo de ativos
            verbose: Verbosidade
            
        Returns:
            Dicionário com resultados
        """
        if verbose:
            print(f"\n  → Heurística Top-{max_assets}: selecionando ativos...")

        result_unconstrained = self.optimize_unconstrained(verbose=False)
        if result_unconstrained is None:
            return None

        weights_full = result_unconstrained['weights']
        top_k_indices = np.argsort(weights_full)[-max_assets:]
        R_subset = self.stocks_returns[:, top_k_indices]
        r = self.index_returns
        T = float(self.n_periods)
        Qsub = (R_subset.T @ R_subset) / T
        bsub = (R_subset.T @ r) / T
        linsub = -2.0 * bsub
        k = len(top_k_indices)

        try:
            model = gp.Model("topk_reopt")
            model.setParam('OutputFlag', 1 if verbose else 0)
            w_sub = model.addVars(k, lb=0.0, ub=1.0, name="w_sub")
            model.addConstr(gp.quicksum(w_sub[i] for i in range(k)) == 1.0, name="sum_w_sub")

            quad = gp.QuadExpr()
            for i in range(k):
                if linsub[i] != 0.0:
                    quad.add(linsub[i] * w_sub[i])
            for i in range(k):
                for j in range(i, k):
                    coef = Qsub[i, j]
                    if coef != 0.0:
                        if i == j:
                            quad.add(coef * w_sub[i] * w_sub[i])
                        else:
                            quad.add(2.0 * coef * w_sub[i] * w_sub[j])

            model.setObjective(quad, GRB.MINIMIZE)
            model.optimize()

            weights = np.zeros(self.n_stocks)
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                vals = np.array([w_sub[i].X for i in range(k)])
                weights[top_k_indices] = vals
            else:
                # fallback: usar pesos originais truncados
                weights[top_k_indices] = weights_full[top_k_indices]

            weights_df = pd.DataFrame({
                'Stock': [self.stock_names[i] for i in top_k_indices],
                'Weight': weights[top_k_indices]
            }).sort_values('Weight', ascending=False)

            portfolio_ret = self.stocks_returns @ weights
            tracking_error_value = np.sqrt(np.mean((portfolio_ret - r) ** 2))
            objective_value = float(weights @ ((self.stocks_returns.T @ self.stocks_returns) / T @ weights) - 2.0 * (self.stocks_returns.T @ r / T) @ weights)

            print(f"\n✓ Heurística resolvida com sucesso!")
            print(f"  Método: Top-{max_assets} + Re-otimização (Gurobi)")
            print(f"  Tracking Error: {tracking_error_value:.6f} ({tracking_error_value*100:.4f}%)")
            print(f"  Ativos selecionados: {max_assets}")

            return {
                'weights': weights,
                'weights_df': weights_df,
                'selected': None,
                'n_active_assets': max_assets,
                'tracking_error': tracking_error_value,
                'objective_value': objective_value,
                'status': model.Status,
                'portfolio_returns': portfolio_ret
            }

        except Exception as e:
            print(f"✗ Erro na re-otimização Top-K (Gurobi): {e}")
            return None

    
    def sensitivity_analysis(self, k_range: List[int], time_limit: Optional[float] = 5) -> pd.DataFrame:
        """
        Analisa sensibilidade do tracking error ao número de ativos.
        
        Args:
            k_range: Lista com diferentes valores de K para testar
            
        Returns:
            DataFrame com resultados para cada K
        """
        print(f"\n{'='*70}")
        print("ANÁLISE DE SENSIBILIDADE (Gurobi)")
        print(f"{'='*70}")
        results = []
        for k in k_range:
            print(f"\n--- Testando K = {k} ---")
            result = self.optimize_constrained(max_assets=k, time_limit=time_limit, verbose=False)
            if result:
                print(result['weights_df'])
                results.append({
                    'K': k,
                    'Tracking_Error': result['tracking_error'],
                    'Tracking_Error_pct': result['tracking_error'] * 100,
                    'Objective_Value': result['objective_value'],
                    'N_Active': result['n_active_assets']
                })
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        return results_df
