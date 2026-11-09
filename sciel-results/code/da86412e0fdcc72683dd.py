"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with global exploration phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')

        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal calls
            if calls >= self.budget:
                return best_f
            calls += 1
            # Clip to box as per problem statement, though Nelder-Mead might go out
            # The problem says func clips, so we can just pass x.
            # However, Nelder-Mead might generate points far outside.
            # It's safer to let func handle clipping as stated.
            return func(x)

        rng = np.random.default_rng(0)

        # Phase 1: Global Exploration via Multi-Start
        # Use 20 starts with limited iterations each
        n_starts = 20
        maxiter_local = 10
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
            if calls >= self.budget - 50:  # Reserve budget for final polish
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Initial evaluation
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Run local Nelder-Mead
            try:
                res = minimize(counted_func, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter_local, 
                                        'xatol': 1e-4, 
                                        'fatol': 1e-4,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 2: Local Exploitation / Polish
        if best_x is not None and time.time() < deadline and calls < self.budget:
            # Run a longer, more precise local search from the best point
            try:
                res = minimize(counted_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 
                                        'xatol': 1e-8, 
                                        'fatol': 1e-8,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # If we still have budget and time, do one more polish
        if best_x is not None and time.time() < deadline and calls < self.budget:
             try:
                res = minimize(counted_func, best_x, method='Powell', 
                               options={'maxiter': 50, 
                                        'ftol': 1e-10,
                                        'xtol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
             except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
