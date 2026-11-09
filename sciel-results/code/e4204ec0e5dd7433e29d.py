import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal call_count, best_fx, best_x
            if call_count >= budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            try:
                x = np.asarray(x, dtype=float).flatten()
                if x.shape[0] != dim:
                    return 1e10
                x = np.clip(x, -5, 5)
                val = float(func(x))
                if np.isfinite(val):
                    call_count += 1
                    if val < best_fx:
                        best_fx = val
                        best_x = x.copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        x0 = np.zeros(dim)
        wrapper(x0) # Initial evaluation

        # Try Nelder-Mead first
        try:
            res = minimize(
                wrapper, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 500, 
                    'maxfev': budget, 
                    'adaptive': True
                }
            )
        except Exception:
            pass

        # If we have budget left, try Powell? Or just return.
        # Nelder-Mead with maxfev=budget will use up most of the budget.
        
        return best_x, best_fx
