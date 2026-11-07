"""Optimizer for BBOB f9 (Rosenbrock rotated) using Multi-Start BFGS with finite differences."""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-Start BFGS optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            # Check budget
            if calls_made >= budget:
                return 1e10
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            
            # Clip to box as per task description, though func does it too
            x = np.clip(x, -5.0, 5.0)
            
            f = float(func(x))
            calls_made += 1
            
            if np.isnan(f) or np.isinf(f):
                f = 1e10
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        # Multi-Start Strategy
        # We divide the budget among several restarts.
        # BFGS is a local method, so diverse starts are crucial.
        n_restarts = 5
        fev_per_restart = budget // n_restarts
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # 1. Center
        # 2-5. Random points in a slightly larger box to ensure we cover the shift
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(n_restarts - 1):
            # Sample from [-4.5, 4.5] to stay within [-5,5] but cover the likely minimum location [-4,4]
            x0 = rng.uniform(-4.5, 4.5, size=dim)
            starts.append(x0)

        for i in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            x0 = starts[i]
            
            try:
                # Use BFGS with finite difference gradients
                # maxfev limits the number of function evaluations
                res = minimize(safe_func, x0, method='BFGS', options={'maxfev': fev_per_restart, 'maxiter': 100})
            except Exception:
                # If an exception occurs, we stop and use the best found so far
                break

        return best_x, best_f
