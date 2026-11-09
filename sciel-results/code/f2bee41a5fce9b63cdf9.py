"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Random restarts with Nelder-Mead local optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def count_func(x):
            nonlocal calls
            calls += 1
            return func(x)
        
        # Phase 1: Random search to find a good starting point
        # Use 50 evaluations for random search
        for i in range(50):
            x = rng.uniform(-5, 5, self.dim)
            fx = count_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local optimization from the best point
        # Use Nelder-Mead with the remaining budget
        remaining = self.budget - calls
        
        if remaining > 0:
            res = minimize(
                count_func,
                best_x,
                method='Nelder-Mead',
                options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        return best_x, best_f

# EVOLVE-BLOCK-END
