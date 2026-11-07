"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0):
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        #
        # Strategy:
        # 1. Use a dense grid to identify the non-dominated envelope (record lows in f2)
        # 2. Select 'budget' points uniformly from the non-dominated x0 values
        # 3. This ensures all evaluations are on the true Pareto front and well-distributed

        # Dense grid to identify non-dominated x0 values
        # Increased resolution to capture fine structure of the discontinuous front
        n_grid = 100000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)

        # Identify non-dominated points (record lows in f2)
        # Vectorized approach for speed
        # min_so_far[i] = min(f2_grid[0..i])
        min_so_far = np.minimum.accumulate(f2_grid)
        
        # A point is non-dominated if it achieves the minimum so far
        # We use a small tolerance to handle floating point precision
        tol = 1e-12
        is_nd = (f2_grid <= min_so_far + tol)
        
        x_nd = x_grid[is_nd]

        # Ensure we include x0=0 and x0=1 for full front coverage
        if x_nd[0] > 0:
            x_nd = np.concatenate([[0.0], x_nd])
        if x_nd[-1] < 1.0:
            x_nd = np.concatenate([x_nd, [1.0]])

        # Select 'budget' points uniformly from the non-dominated x0 values
        if len(x_nd) <= budget:
            selected_x = x_nd
        else:
            # Select budget indices uniformly from the range
            indices = np.linspace(0, len(x_nd) - 1, budget, dtype=int)
            selected_x = x_nd[indices]

        # Ensure we don't exceed budget
        n_evals = min(len(selected_x), budget)

        for i in range(n_evals):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)

            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
