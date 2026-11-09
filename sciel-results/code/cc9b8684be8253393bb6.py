"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + (1+1)-ES refinement for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # Phase 1: Coarse grid search (3^5 = 243 points)
        # This guarantees we sample the entire domain with decent resolution
        # Grid: 3 points per dimension over [-5, 5]
        grid_points_1d = np.array([-5.0, 0.0, 5.0])
        
        # Generate all 3^5 grid points
        grid_points = np.array(np.meshgrid(*([grid_points_1d] * n)), dtype=np.float64)
        grid_points = grid_points.reshape(n, -1).T  # shape (243, 5)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate all grid points
        for i in range(grid_points.shape[0]):
            if time.time() > deadline:
                break
            x = grid_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Fine grid around the best point found
        # Use a 3^5 grid in a smaller box around best_x
        remaining = budget - calls_made
        if remaining > 100 and best_x is not None:
            # Fine grid: 3 points per dim in [-2, 2] around best
            fine_grid_1d = np.array([-2.0, 0.0, 2.0])
            fine_points = np.array(np.meshgrid(*([fine_grid_1d] * n)), dtype=np.float64)
            fine_points = fine_points.reshape(n, -1).T + best_x
            fine_points = np.clip(fine_points, -5, 5)
            
            for i in range(fine_points.shape[0]):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = fine_points[i]
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Phase 3: (1+1)-ES refinement
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            # Adaptive step size
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(42 + calls_made).standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if sigma < 1e-6:
                    break
        
        return x, fx
# EVOLVE-BLOCK-END
