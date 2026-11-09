"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Global L-BFGS-B local search from diverse random starting points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Count function evaluations
        eval_count = [0]
        
        def wrapped_func(x):
            eval_count[0] += 1
            if time.time() > deadline:
                return float('inf')
            return func(x)
        
        # Generate diverse starting points
        # Use a mix of random points and points near the center
        n_starts = min(50, budget // 10)
        starts = []
        
        # Add center
        starts.append(np.zeros(dim))
        
        # Add random points
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Add some points near corners
        corners = []
        for i in range(dim):
            for val in [-3, 3]:
                pt = np.zeros(dim)
                pt[i] = val
                corners.append(pt)
        for c in corners[:10]:
            if len(starts) < n_starts:
                starts.append(c)
        
        # Shuffle starts
        rng.shuffle(starts)
        
        best_x = None
        best_f = float('inf')
        bounds = [(-5, 5)] * dim
        
        # Allocate budget: use remaining budget for L-BFGS-B maxiter
        # Each L-BFGS-B call uses ~10-30 evaluations typically
        # We'll let it run until budget is exhausted or deadline
        
        for x0 in starts:
            if eval_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'maxfun': budget - eval_count[0]})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
        
        # If no valid result, use the first evaluation
        if best_x is None:
            x0 = starts[0]
            best_x = x0
            best_f = wrapped_func(x0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
