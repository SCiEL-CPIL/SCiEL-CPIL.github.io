"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive step size, Nelder-Mead polish, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Phase 1: (1+1)-ES with 1/5 success rule
        for i in range(self.budget - 1):
            if time.time() >= deadline or calls >= self.budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.95
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        # Phase 2: Nelder-Mead polish on the best point found
        if time.time() < deadline - 10 and calls < self.budget:
            remaining_budget = self.budget - calls
            # Use a subset of remaining budget for Nelder-Mead
            max_nm_calls = min(remaining_budget, 100)
            
            nm_x = best_x.copy()
            nm_f = best_f
            
            # Nelder-Mead with callback to track calls and stop early
            nm_calls = [0]
            
            def nm_func(x_in):
                nm_calls[0] += 1
                if nm_calls[0] > max_nm_calls or time.time() >= deadline:
                    return nm_f  # Return current best to stop progress
                x_clipped = np.clip(x_in, -5, 5)
                return func(x_clipped)
            
            try:
                result = minimize(nm_func, nm_x, method='Nelder-Mead',
                                  options={'maxiter': max_nm_calls, 'xatol': 1e-8, 'fatol': 1e-10,
                                           'adaptive': True})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
