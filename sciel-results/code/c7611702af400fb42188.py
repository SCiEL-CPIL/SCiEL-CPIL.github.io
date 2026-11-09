"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell optimizer with adaptive budgeting."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def objective(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling
        # Use 10% of budget for global exploration
        n_random = max(10, budget // 10)
        xs = rng.uniform(-5, 5, (n_random, dim))
        fs = np.array([objective(x) for x in xs])
        
        # Sort by function value
        sorted_indices = np.argsort(fs)
        
        # Phase 2: Multi-start Powell
        # Use top 5 starting points
        n_starts = min(5, len(sorted_indices))
        starts = [xs[idx].copy() for idx in sorted_indices[:n_starts]]

        remaining_budget = budget - calls_used
        
        bounds = [(-5, 5)] * dim
        
        for i, start in enumerate(starts):
            if calls_used >= budget:
                break
            
            # Calculate remaining budget for this specific run
            # We want to leave some buffer for the next starts if this one fails to improve
            # But since we are sorting, the first one is the best. 
            # Let's give the first start more budget.
            # Simple approach: equal split of remaining budget among remaining starts.
            local_budget = (budget - calls_used) // (n_starts - i)
            
            if local_budget < 10:
                local_budget = budget - calls_used
            
            try:
                res = minimize(
                    objective,
                    start,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxfev': local_budget,
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
