import numpy as np
import time

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

        # CMA-ES implementation
        def cma_es(func_wrapper, dim, lb, ub, max_evals, seed=42):
            rng_local = np.random.default_rng(seed)
            
            # Initial point
            x = np.zeros(dim)
            
            # Initial step size
            sigma = 1.0
            
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
            
            # Covariance matrix
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Cumulative step size
            chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            evals_used = 0
            best_x_local = x.copy()
            best_f_local = func_wrapper(x)
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
                    # Clip to bounds
                    x_trial = np.clip(x_trial, lb, ub)
                    population.append(x_trial)
                
                # Evaluate
                fitness = []
                for x_trial in population:
                    f = func_wrapper(x_trial)
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
                sorted_indices = np.argsort(fitness)
                sorted_pop = [population[i] for i in sorted_indices[:mu]]
                sorted_fitness = [fitness[i] for i in sorted_indices[:mu]]
                
                if len(sorted_pop) < mu:
                    break
                
                # Weighted mean
                mean_weighted = np.zeros(dim)
                for i in range(mu):
                    mean_weighted += w[i] * sorted_pop[i]
                
                # Update evolution paths
                y_w = mean_weighted - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * y_w
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix
                delta_d = (1 + 2 * (mueff > mu + 1) * c_mu) * c_c
                outer = np.outer(p_c, p_c)
                C = (1 + c_1 + c_mu * mueff - c_1) * C + c_1 * (outer - C)
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # If Cholesky fails, reset
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                
                # Update mean
                x = mean_weighted
                x = np.clip(x, lb, ub)
            
            return best_x_local, best_f_local

        # Phase 1: Run CMA-ES with most of the budget
        cma_budget = int(budget * 0.75)
        
        try:
            cx, cf = cma_es(wrapped_func, dim, lb, ub, cma_budget, seed=42)
            if cx is not None:
                if cf < best_f[0]:
                    best_f[0] = cf
                    best_x[0] = cx.copy()
        except (TimeoutError, BudgetExceeded):
            pass
        except Exception:
            pass

        # Phase 2: Local refinement with Nelder-Mead if budget remains
        if best_x[0] is not None and call_count[0] < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: wrapped_func(x),
                    best_x[0],
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        # Phase 3: Final local search with small perturbations
        if call_count[0] < budget and best_x[0] is not None:
            try:
                for _ in range(20):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    # Small random perturbations around best
                    scale = max(0.01, (best_f[0] - (-100)) / 200 * 0.1)
                    x_trial = best_x[0] + rng.normal(0, scale, dim)
                    x_trial = np.clip(x_trial, lb, ub)
                    f = wrapped_func(x_trial)
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        return best_x[0], best_f[0]
