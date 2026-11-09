"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start L-BFGS-B Optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = [0]
        
        def func_wrapper(x):
            calls_made[0] += 1
            return func(x)
        
        def check_time():
            return time.time() >= deadline - 5.0  # 5s safety margin
        
        # Phase 1: Global exploration with random starts
        # Use a reasonable number of random starts to find good initial points
        n_starts = 15
        points = rng.uniform(-5, 5, (n_starts, self.dim))
        
        for i in range(n_starts):
            if check_time():
                break
            x = points[i]
            fx = func_wrapper(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local exploitation with L-BFGS-B
        # Use the best point from Phase 1 and potentially some other good points
        remaining_budget = self.budget - calls_made[0]
        
        if remaining_budget > 20 and best_x is not None:
            # Try L-BFGS-B from the best point
            try:
                res = minimize(
                    func_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining_budget,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: If we have budget left, try a few more local optimizations
        # from other promising points
        if not check_time():
            # Try L-BFGS-B from the second and third best points from Phase 1
            sorted_indices = np.argsort([func_wrapper(p) for p in points])[:3]
            for idx in sorted_indices:
                if check_time():
                    break
                if calls_made[0] >= self.budget - 10:
                    break
                x_start = points[idx].copy()
                try:
                    res = minimize(
                        func_wrapper,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 50,
                            'maxfun': max(10, self.budget - calls_made[0] - 5),
                            'ftol': 1e-10,
                            'gtol': 1e-8,
                            'maxls': 30
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Ensure we don't exceed budget
        if calls_made[0] > self.budget:
            # This shouldn't happen due to maxfun, but just in case
            pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
