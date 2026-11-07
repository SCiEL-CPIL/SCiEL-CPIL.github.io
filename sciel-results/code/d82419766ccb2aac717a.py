"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # Track calls and best
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Perform multiple restarts of Nelder-Mead
        # Budget is 500. Let's allocate ~40-50 evals per restart.
        # 500 / 45 ~ 11 restarts.
        
        num_restarts = 10
        evals_per_restart = 45
        
        for r in range(num_restarts):
            if time.time() > deadline or calls >= budget:
                break
            
            # Random start point
            x0 = rng.uniform(-5.0, 5.0, n)
            
            # Run Nelder-Mead
            # We use a wrapper that checks time and budget
            def objective(x):
                if time.time() > deadline or calls >= budget:
                    return np.inf
                return safe_call(x)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'maxfev': evals_per_restart, 
                                        'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                pass

            # Update best if res.success or just compare
            if res.fun < best_f:
                best_x = res.x
                best_f = res.fun

        # If budget remains, do a final local polish on the best point
        if time.time() < deadline and calls < budget and best_x is not None:
             def objective_polish(x):
                if time.time() > deadline or calls >= budget:
                    return np.inf
                return safe_call(x)
             
             try:
                 res = minimize(objective_polish, best_x, method='Nelder-Mead', 
                                options={'maxiter': 100, 'maxfev': budget - calls, 
                                         'xatol': 1e-6, 'fatol': 1e-6})
                 if res.fun < best_f:
                     best_x = res.x
                     best_f = res.fun
             except Exception:
                 pass

        return best_x, best_f
# EVOLVE-BLOCK-END
