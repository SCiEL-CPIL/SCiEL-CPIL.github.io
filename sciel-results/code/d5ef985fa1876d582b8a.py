"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Derivative-free optimizer for the bent cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        try:
            # Phase 1: Coarse sampling to find a good basin
            # Use ~10% of budget for initial exploration
            phase1_budget = max(20, budget // 10)
            
            # Start with a reasonable initial point
            x0 = rng.uniform(-5, 5, dim)
            f0 = evaluate(x0)
            
            # Generate a set of points to find a good starting location
            for i in range(phase1_budget - 1):
                if time.time() > deadline or calls >= budget:
                    break
                x_new = rng.uniform(-5, 5, dim)
                f_new = evaluate(x_new)
            
            # Phase 2: Local refinement with Powell's Method
            # Powell's method is robust for narrow valleys (like Bent Cigar)
            if calls < budget and best_x is not None:
                remaining_budget = budget - calls
                maxfev = max(10, remaining_budget - 5)
                
                options = {
                    'maxfev': maxfev,
                    'xatol': 1e-8,
                    'fatol': 1e-10
                }
                
                res = scipy.optimize.minimize(
                    evaluate, 
                    best_x, 
                    method='Powell', 
                    options=options
                )
                
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
