import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        call_count = [0]
        best_f = [np.inf]
        best_x = [np.zeros(dim)]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise BudgetExceeded()
            call_count[0] += 1
            x = np.asarray(x, dtype=float)
            fx = func(x)
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        # CMA-ES implementation with fixed covariance update
        def cma_es(start_x, max_evals, seed=42):
            rng_local = np.random.default_rng(seed)
            
            x = np.array(start_x, dtype=float)
            sigma = 0.5
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Cumulative step size
            chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            evals_used = 0
            best_x_local = x.copy()
            best_f_local = wrapped_func(x)
            evals_used += 1
            
            while evals_used < max_evals:
                if time.time() > deadline:
                    break
                
                # Sample population
                population = []
                for i in range(lambda_):
                    z = rng_local.standard_normal(dim)
                    y = sigma * C_sqrt @ z
                    x_trial = x + y
                    x_trial = np.clip(x_trial, lb, ub)
                    population.append((x_trial, z, y))
                
                # Evaluate
                fitness = []
                for x_trial, z, y in population:
                    f = wrapped_func(x_trial)
                    evals_used += 1
                    fitness.append(f)
                    if f < best_f_local:
                        best_f_local = f
                        best_x_local = x_trial.copy()
                    if evals_used >= max_evals:
                        break
                
                if evals_used >= max_evals:
                    break
                
                # Sort by fitness
                sorted_pop = sorted(zip(fitness, [p[0] for p in population[:len(fitness)]]), key=lambda t: t[0])
                
                if len(sorted_pop) < mu:
                    break
                
                # Weighted mean of best mu
                mean_weighted = np.zeros(dim)
                for i in range(mu):
                    mean_weighted += w[i] * sorted_pop[i][1]
                
                # Update evolution paths
                y_w = mean_weighted - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * y_w
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix (FIXED: use old mean x for rank updates)
                C = (1 + c_1 + c_mu * mueff) * C + c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    diff = sorted_pop[i][1] - x
                    C += c_mu * w[i] * np.outer(diff, diff)
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                
                # Update mean
                x = mean_weighted
                x = np.clip(x, lb, ub)
            
            return best_x_local, best_f_local

        # Phase 1: Initial sampling to find a good starting point
        initial_budget = min(20, budget)
        for i in range(initial_budget):
            if time.time() > deadline or call_count[0] >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            fx = wrapped_func(x)
        
        # Phase 2: CMA-ES with most of the budget
        remaining_for_cma = budget - call_count[0] - 30
        if remaining_for_cma > 10:
            try:
                cx, cf = cma_es(best_x[0], remaining_for_cma, seed=42)
                if cf < best_f[0]:
                    best_f[0] = cf
                    best_x[0] = cx.copy()
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        # Phase 3: Local refinement with Nelder-Mead if budget remains
        remaining = budget - call_count[0]
        if best_x[0] is not None and remaining > 5:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: wrapped_func(x),
                    best_x[0],
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        return best_x[0], best_f[0]
