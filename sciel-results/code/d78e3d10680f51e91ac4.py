"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: Sobol-based initial exploration + (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # Use Sobol sequence for initial space-filling exploration
        # First 32 points give good coverage of [-5, 5]^5
        n_initial = min(32, budget)
        
        # Generate Sobol sequence points in [0,1]^dim, then map to [-5,5]^dim
        # Use scipy's Sobol for better space-filling properties
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=False)
            points_unit = sobol.random(n_initial)
            # Map from [0,1] to [-5,5]
            points = 10.0 * points_unit - 5.0
        except ImportError:
            # Fallback: Latin Hypercube
            points = np.zeros((n_initial, dim))
            for d in range(dim):
                perm = np.random.default_rng(42 + d).permutation(n_initial)
                points[:, d] = (perm + 0.5) / n_initial * 10.0 - 5.0
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # If no initial point was evaluated (shouldn't happen), use center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        calls_made = n_initial
        remaining = budget - calls_made
        
        # (1+1)-ES refinement from the best initial point
        if remaining > 0:
            x = best_x
            fx = best_f
            sigma = 1.0  # Start with moderate step size
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng().standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        sigma = 1e-6
        
        return best_x, best_f
# EVOLVE-BLOCK-END
