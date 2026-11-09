import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Split budget: 70% for Global Search (DE), 30% for Local Polish
        de_budget = int(total_budget * 0.7)
        polish_budget = total_budget - de_budget

        # Calculate maxiter for DE
        pop_factor = 2
        pop_size = pop_factor * n
        maxiter_de = max(1, int((de_budget - pop_size) / pop_size))

        # Run Differential Evolution
        bounds = [(-5, 5)] * n
        seed = 42
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds=bounds, 
                maxiter=maxiter_de,
                popsize=pop_factor,
                seed=seed,
                polish=False,
                tol=1e-8
            )
        except (TimeoutError, ValueError) as e:
            pass
        except Exception as e:
            pass

        # Now polish the best result found
        x0 = best_x
        if np.isinf(best_f):
             x0 = np.random.uniform(-5, 5, n)
             best_f = wrapped_func(x0)

        # Phase 1: Powell for broad local convergence
        try:
            res_local = minimize(
                wrapped_func, 
                x0, 
                method='Powell',
                bounds=bounds,
                options={'maxiter': 50}
            )
            if res_local.fun < best_f:
                best_f = res_local.fun
                best_x = np.clip(res_local.x, -5, 5)
        except (TimeoutError, ValueError) as e:
            pass
        except Exception as e:
            pass

        # Phase 2: High-precision polish with Nelder-Mead
        if time.time() < deadline - 10.0 and calls_made[0] < total_budget - 10:
            try:
                # Nelder-Mead for fine-tuning
                res_nm = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.clip(res_nm.x, -5, 5)
            except (TimeoutError, ValueError) as e:
                pass
            except Exception as e:
                pass

        return best_x, best_f
