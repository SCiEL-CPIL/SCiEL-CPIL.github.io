import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290.0
            max_calls = self.budget
            calls_made = 0
            dim = self.dim

            # Wrapper for counting and clipping
            def eval_func(x):
                nonlocal calls_made
                if calls_made >= max_calls:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls_made += 1
                return val

            # (1+1)-ES with 1/5 success rule
            rng = np.random.default_rng(42)
            x = np.zeros(dim)
            fx = eval_func(x)
            
            sigma = 1.0
            # Success history
            success_hist = []
            hist_len = 10
            
            es_budget = int(self.budget * 0.8)
            nm_budget = self.budget - es_budget

            for i in range(es_budget - 1):
                if time.time() > deadline:
                    break
                if calls_made >= max_calls:
                    break

                y = x + sigma * rng.standard_normal(dim)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    success_hist.append(1)
                else:
                    success_hist.append(0)
                
                if len(success_hist) > hist_len:
                    success_hist.pop(0)
                
                # Adapt sigma
                if len(success_hist) == hist_len:
                    success_rate = np.mean(success_hist)
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
            
            # Nelder-Mead Polish
            if calls_made < max_calls and time.time() < deadline:
                nm_calls = 0
                nm_limit = max_calls - calls_made
                
                def nm_func(z):
                    nonlocal nm_calls
                    if nm_calls >= nm_limit:
                        return np.inf
                    nm_calls += 1
                    return func(np.clip(z, -5, 5))
                
                try:
                    # maxiter is a heuristic. 
                    # Nelder-Mead in 5D uses ~6 evals/iter.
                    # We allow nm_limit evals.
                    max_iter = max(1, nm_limit // 6)
                    res = minimize(nm_func, x, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < fx:
                        x, fx = res.x, res.fun
                except Exception:
                    pass

            return x, fx
