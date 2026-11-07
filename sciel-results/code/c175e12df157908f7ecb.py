"""Rewritten optimizer using Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return x, f

        # Phase 1: Random Sampling to find diverse basins
        n_init = max(5, budget // 5)
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            res = eval_point(x)
            if res is None:
                break
            x, fx = res
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization from top seeds
        remaining = budget - calls_made
        if remaining < 10:
            return best_x, best_f

        n_seeds = min(3, len(candidates))
        
        for i in range(n_seeds):
            if calls_made >= budget:
                break
            
            start_x = candidates[i][1].copy()
            start_f = candidates[i][0]
            
            # Local Nelder-Mead
            def local_func(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return start_f
                res = eval_point(x)
                if res is None:
                    return start_f
                return res[1]

            def callback(xk):
                return calls_made >= budget - 5

            try:
                res = minimize(
                    local_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6, 'callback': callback}
                )
                new_x = np.clip(res.x, -5, 5)
                new_f = res.fun
                
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
