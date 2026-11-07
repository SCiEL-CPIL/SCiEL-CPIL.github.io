"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit
        
        # Count function calls
        call_count = 0
        max_calls = self.budget
        
        # Best solution found
        best_x = None
        best_f = np.inf
        
        # Wrap func to count calls and enforce budget
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                return 1e10  # Return large value to stop optimization
            call_count += 1
            # Clip to box
            x = np.clip(x, -5, 5)
            return func(x)
        
        # Use L-BFGS-B with multiple restarts
        rng = np.random.default_rng(42)
        
        # Generate diverse initial points
        n_restarts = 10
        for i in range(n_restarts):
            if time.time() > deadline - 5:
                break
            if call_count >= max_calls:
                break
            
            # Initial point: mix of center and random
            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': max_calls - call_count}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                continue
        
        # If no valid result, use a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
