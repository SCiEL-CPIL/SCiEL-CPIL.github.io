"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(0)
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        # Phase 1: Diversified initialization
        # Allocate a small fraction of budget to sample diverse starting points
        n_starts = 10
        candidates = []
        
        # Include center and various scales to cover the [-5, 5] box
        candidates.append(np.zeros(dim))
        for i in range(n_starts - 1):
            if i < 2:
                # Tight around center
                x0 = rng.uniform(-1, 1, dim)
            elif i < 5:
                # Medium spread
                x0 = rng.uniform(-3, 3, dim)
            else:
                # Wide spread
                x0 = rng.uniform(-5, 5, dim)
            candidates.append(x0)

        best_x = None
        best_f = np.inf
        
        for x0 in candidates:
            if calls_made >= total_budget or time.time() > deadline:
                break
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Fallback if no candidates were evaluated
        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            best_x = x.copy()
            best_f = fx
        else:
            x = best_x.copy()
            fx = best_f

        # Phase 2: (1+1)-ES from the best starting point
        sigma = 2.0
        success_counter = 0
        total_steps = 0
        window = 10

        while calls_made < total_budget:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                success_counter += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            total_steps += 1

            # Adapt sigma based on success rate in the last 'window' steps
            if total_steps >= window:
                success_rate = success_counter / total_steps
                if success_rate > 0.3:
                    sigma *= 1.5
                elif success_rate < 0.1:
                    sigma *= 0.5
                else:
                    sigma *= 1.0
                
                success_counter = 0
                total_steps = 0
                
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
