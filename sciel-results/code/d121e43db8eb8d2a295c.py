"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        
        # Phase 1: Exploration with (1+1)-ES
        while calls < self.budget - 50:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        
        # Phase 2: Final polish pass with tight tolerances
        # Use Nelder-Mead for high-precision refinement
        if calls < self.budget:
            remaining = self.budget - calls
            if remaining > 10:
                try:
                    from scipy.optimize import minimize
                    # Use a small number of iterations to stay within budget
                    # Nelder-Mead uses ~dim+1 initial points + iterations
                    # We'll limit function calls by wrapping
                    call_count = [0]
                    def wrapped_func(z):
                        call_count[0] += 1
                        if call_count[0] > remaining:
                            return fx  # Return current best to stop
                        return func(z)
                    
                    # Start from current best
                    # Use tight tolerances
                    res = minimize(wrapped_func, x, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                    if res.fun < fx:
                        x, fx = res.x, res.fun
                except Exception:
                    pass
        return x, fx
# EVOLVE-BLOCK-END
