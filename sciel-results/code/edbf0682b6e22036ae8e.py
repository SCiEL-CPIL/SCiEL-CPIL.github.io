"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global grid exploration followed by targeted local search (CMA-ES style)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse global grid exploration
        # Use a 3x3x3x3x3 grid = 243 points to cover [-5,5]^5
        # This gives early AOCC gains by quickly finding a good basin
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([
                                -5 + 10 * i / 2,
                                -5 + 10 * j / 2,
                                -5 + 10 * k / 2,
                                -5 + 10 * l / 2,
                                -5 + 10 * m / 2,
                            ])
                            grid_points.append(x)
        
        # Add center and some random points for diversity
        grid_points.append(np.zeros(dim))
        rng = np.random.default_rng(42)
        for _ in range(20):
            grid_points.append(rng.uniform(-5, 5, dim))

        # Evaluate grid points
        best_x = None
        best_f = np.inf
        calls = 0

        for x in grid_points:
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: should not happen
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local refinement using (1+1)-ES from best grid point
        # Use a larger initial step for exploration near the basin
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Start with moderate step size
        remaining = budget - calls

        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            
            # If sigma becomes too small, try a larger jump
            if sigma < 0.01:
                # Try a random restart near current best
                x_new = np.clip(x + rng.uniform(-2, 2, dim), -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < fx:
                    x, fx = x_new, fx_new
                sigma = 1.0

        return x, fx
# EVOLVE-BLOCK-END
