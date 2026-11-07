"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start local search optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # 1. Global Exploration using Stratified Sampling (Latin Hypercube)
        # We allocate 20% of the budget to global search.
        n_global = min(100, budget // 5)
        
        # Generate LHS points
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            perm = np.random.permutation(n_global)
            lower = -5.0 + 10.0 * perm / n_global
            upper = -5.0 + 10.0 * (perm + 1) / n_global
            lhs_points[:, d] = lower + (upper - lower) * np.random.rand(n_global)
            
        calls_made = 0
        max_calls = budget
        best_f = np.inf
        best_x = None
        candidate_starts = []
        
        # Strict function wrapper to prevent budget overrun
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return np.inf
            calls_made += 1
            return func(x)

        # Evaluate global points
        for i in range(n_global):
            x = lhs_points[i]
            fx = safe_func(x)
            candidate_starts.append((fx, x))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # 2. Local Refinement
        # Select top K candidates
        # Reduced K to 3 to allow more iterations per local search
        K = 3
        candidate_starts.sort(key=lambda t: t[0])
        top_starts = [x for _, x in candidate_starts[:K]]
        
        # Allocate remaining budget to local search
        remaining_budget = budget - calls_made
        budget_per_start = remaining_budget // K
        
        for start_x in top_starts:
            if budget_per_start <= 0:
                break
                
            # Use Nelder-Mead with strict call limiting via safe_func
            # maxfev is a soft limit, safe_func is the hard limit
            res = minimize(safe_func, start_x, method='Nelder-Mead', 
                           options={'maxfev': budget_per_start, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
