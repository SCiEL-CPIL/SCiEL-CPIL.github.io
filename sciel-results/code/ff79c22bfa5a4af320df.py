"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        start_time = time.time()
        deadline = start_time + 285.0

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        bounds = [(-5.0, 5.0)] * dim

        # Generate diverse start points
        # The true minimum is at 1 + shift, where shift in [-4,4]^5.
        # So true min is in [-3, 5]^5.
        # We sample broadly in [-5, 5]^5 to be safe, with some bias to center.
        
        starts = []
        # Known Rosenbrock min
        starts.append(np.ones(dim))
        # Negative
        starts.append(-np.ones(dim))
        # Origin
        starts.append(np.zeros(dim))
        
        # Random points
        # Use 15 random points to ensure coverage
        for _ in range(15):
            # Sample in [-4, 4] to focus on likely region, but clip to [-5,5]
            x = rng.uniform(-4.0, 4.0, dim)
            starts.append(x)

        # Iterate through starts, running L-BFGS-B from each
        for x0 in starts:
            if call_count >= budget - 5:
                break
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
            except Exception:
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
