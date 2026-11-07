"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Rastrigin-aware multi-start + CMA-ES-like exploitation + periodic snap refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Multi-start (1+1)-ES
        # This is robust for rotated Rastrigin because it adapts step size to hop between basins.
        
        # Initial best guess
        x = np.zeros(dim)
        f = eval_func(x)
        
        # Try a few random starts to find a better basin
        for _ in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = eval_func(x_try)
            if f_try < f:
                x = x_try
                f = f_try

        sigma = 1.0
        no_improve = 0
        max_no_improve = 15
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = eval_func(y)
            
            if fy < f:
                x = y
                f = fy
                no_improve = 0
                sigma *= 0.95 # Refine
            else:
                no_improve += 1
                sigma *= 1.05 # Explore
                
            # Clamp sigma
            if sigma < 1e-4:
                sigma = 1e-4
            if sigma > 5.0:
                sigma = 5.0

            # If stuck, restart from a random point or large jump
            if no_improve >= max_no_improve:
                if rng.random() < 0.5:
                    x = x + 2.0 * rng.standard_normal(dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                f = eval_func(x)
                sigma = 1.0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
