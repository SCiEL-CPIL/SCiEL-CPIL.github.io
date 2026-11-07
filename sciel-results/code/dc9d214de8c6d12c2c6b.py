"""Hierarchical grid search exploiting Katsuura fractal structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse 3^5 grid
        # Evaluate all 3^5 = 243 points
        grid_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        
        # Start with the coarsest grid
        for grid_size in grid_sizes:
            if calls + grid_size ** dim > budget:
                break
            if time.time() > deadline:
                break

            # Generate grid points
            points = np.array(np.meshgrid(*[np.linspace(-5, 5, grid_size) for _ in range(dim)])).T.reshape(-1, dim)
            
            # Evaluate all points
            for p in points:
                if time.time() > deadline:
                    break
                evaluate(p)

            # After coarse grid, refine around the best point
            # Do a local refinement with progressively finer grids
            if grid_size <= 5:
                # After the 5^5 grid, do local refinement
                # Find the best region and refine
                pass

        # Phase 2: Local refinement around best point
        # Use a series of local grids with decreasing step sizes
        step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]
        
        for step in step_sizes:
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Local grid around best_x
            # Use a 5x5x5x5x5 local grid
            local_grid_size = 5
            num_points = local_grid_size ** dim
            if calls + num_points > budget:
                # Reduce grid size
                local_grid_size = 3
                num_points = local_grid_size ** dim
                if calls + num_points > budget:
                    break

            # Generate local grid
            offsets = np.linspace(-step * (local_grid_size - 1) / 2, step * (local_grid_size - 1) / 2, local_grid_size)
            points = np.array(np.meshgrid(*[offsets for _ in range(dim)])).T.reshape(-1, dim)
            
            for p in points:
                if time.time() > deadline:
                    break
                x = np.clip(best_x + p, -5, 5)
                evaluate(x)

        # Phase 3: Fine-tune with coordinate descent
        if calls < budget and time.time() < deadline:
            # Coordinate descent for final polish
            for _ in range(20):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    # Try small perturbations in each coordinate
                    for delta in [-0.01, 0.01, -0.005, 0.005, -0.001, 0.001]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x = best_x.copy()
                        x[d] += delta
                        x = np.clip(x, -5, 5)
                        f = evaluate(x)
                        if f < best_f:
                            improved = True
                            break
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
