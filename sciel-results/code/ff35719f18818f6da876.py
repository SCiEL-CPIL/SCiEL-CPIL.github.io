"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Center-biased initialization: optimum is in [-4,4]^5, center is better
        x = rng.uniform(-2, 2, self.dim)
        fx = func(x)
        sigma = 1.5
        min_sigma = 0.001
        calls = 1

        # Phase 1: Exploration with (1+1)-ES using 1/5 success rule
        # Reserve ~15% of budget for polish phase
        es_budget = int(self.budget * 0.85)
        while calls < es_budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.25, 5.0)
            else:
                sigma = max(sigma * 0.85, min_sigma)

        # Phase 2: Nelder-Mead polish
        if calls < self.budget:
            remaining = self.budget - calls
            if remaining > 15:
                try:
                    from scipy.optimize import minimize
                    call_count = [0]
                    best_fx = [fx]
                    
                    def wrapped_func(z):
                        nonlocal fx
                        call_count[0] += 1
                        if call_count[0] > remaining:
                            return best_fx[0]
                        val = func(np.clip(z, -5, 5))
                        if val < best_fx[0]:
                            best_fx[0] = val
                        return val

                    # First polish: moderate tolerance to find basin
                    res = minimize(wrapped_func, x, method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-8,
                                            'fatol': 1e-10, 'adaptive': True})
                    if res.fun < fx:
                        x, fx = np.clip(res.x, -5, 5), res.fun
                        calls += call_count[0]
                        remaining = self.budget - calls

                    # Second polish: very tight tolerance if budget remains
                    if remaining > 10:
                        call_count[0] = 0
                        best_fx[0] = fx
                        res2 = minimize(wrapped_func, x, method='Nelder-Mead',
                                        options={'maxiter': 500, 'xatol': 1e-12,
                                                 'fatol': 1e-14, 'adaptive': True})
                        if res2.fun < fx:
                            x, fx = np.clip(res2.x, -5, 5), res2.fun
                        calls += call_count[0]
                except Exception:
                    pass

        return x, fx
# EVOLVE-BLOCK-END
