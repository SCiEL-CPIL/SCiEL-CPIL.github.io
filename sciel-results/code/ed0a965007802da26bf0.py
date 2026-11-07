"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with coarse grid for early progress."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def nm_func(x):
            nonlocal call_count
            if call_count >= budget:
                return best_fx
            val = func(x)
            call_count += 1
            record(x, val)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid search for early progress
        n_per_dim = 3
        grid_vals = np.linspace(-4, 4, n_per_dim)
        grid_points = np.array(np.meshgrid(*([grid_vals] * dim)))
        grid_points = grid_points.reshape(dim, -1).T

        # Shuffle grid points for diversity
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]

        # Evaluate first portion of grid (limited by budget and time)
        max_grid_calls = min(len(grid_points), 100)
        for i in range(max_grid_calls):
            if call_count >= budget or time.time() > deadline - 10:
                break
            x = grid_points[i]
            fx = func(x)
            call_count += 1
            record(x, fx)

        # Phase 2: Nelder-Mead local optimization from best point
        # and a few other promising points
        start_points = [best_x.copy()]
        
        # Add a few random points near the best
        for _ in range(3):
            if call_count >= budget or time.time() > deadline - 10:
                break
            perturbed = best_x + rng.normal(0, 1.5, size=dim)
            perturbed = np.clip(perturbed, -5, 5)
            start_points.append(perturbed)

        for start in start_points:
            if call_count >= budget or time.time() > deadline - 10:
                break
            
            # Nelder-Mead settings
            options = {
                'maxiter': 200,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                result = minimize(nm_func, start, method='Nelder-Mead', options=options)
            except Exception:
                continue

            # Record the result
            if call_count < budget:
                record(result.x, result.fun)

        return best_x, best_fx
# EVOLVE-BLOCK-END
