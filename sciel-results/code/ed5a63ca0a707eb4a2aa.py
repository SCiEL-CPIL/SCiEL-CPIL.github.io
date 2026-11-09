"""Rewritten optimizer for BBOB f11 (Discus) using Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        self.calls = 0

        def wrapper(x):
            if self.calls >= self.budget:
                return np.inf
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            return float(func(x_clipped))

        # Phase 1: Random sampling to find a good initial point
        n_init = 15
        for _ in range(n_init):
            if self.calls >= self.budget:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx = wrapper(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if self.calls >= self.budget:
            return best_x, best_f

        # Phase 2: Nelder-Mead optimization
        remaining = self.budget - self.calls
        
        # Run NM from the best start
        # We use maxfun to limit evaluations
        try:
            res = minimize(
                wrapper, 
                best_x, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'maxfun': remaining, 
                    'xatol': 1e-8, 
                    'fatol': 1e-8
                }
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass # If NM fails, we still have the best from Phase 1

        # Phase 3: Restarts if budget remains
        # Try a few local restarts from the best point to escape potential stagnation
        n_restarts = 5
        for _ in range(n_restarts):
            if self.calls >= self.budget:
                break
            
            remaining = self.budget - self.calls
            if remaining < 10:
                break
            
            # Perturb the best point
            scale = 0.1
            x_pert = best_x + rng.normal(0, scale, self.dim)
            x_pert = np.clip(x_pert, -5, 5)
            
            try:
                res2 = minimize(
                    wrapper, 
                    x_pert, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, 
                        'maxfun': remaining, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8
                    }
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
