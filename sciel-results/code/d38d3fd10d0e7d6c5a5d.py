"""Two-phase optimizer: coarse grid search + BFGS local minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: grid search for basin finding, then BFGS refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Coarse grid search to find basin of attraction
        # Use a 3x3x3x3x3 grid = 243 points, well within budget
        grid_points_per_dim = 3
        total_grid = grid_points_per_dim ** self.dim
        # Use remaining budget wisely: allocate ~60% to grid, 40% to BFGS
        grid_budget = min(total_grid, int(self.budget * 0.5))

        # Generate grid points in [-5, 5]^dim
        if grid_budget >= total_grid:
            # Full grid
            axes = np.linspace(-5, 5, grid_points_per_dim)
            grid = np.array(np.meshgrid(*([axes] * self.dim), indexing='ij')).reshape(self.dim, -1).T
        else:
            # Subsample the grid
            axes = np.linspace(-5, 5, grid_points_per_dim)
            full_grid = np.array(np.meshgrid(*([axes] * self.dim), indexing='ij')).reshape(self.dim, -1).T
            rng = np.random.default_rng(42)
            indices = rng.choice(len(full_grid), size=grid_budget, replace=False)
            grid = full_grid[indices]

        best_x = None
        best_f = np.inf

        for point in grid:
            if time.time() > deadline or calls_made >= self.budget:
                break
            fx = safe_call(point)
            if fx < best_f:
                best_f = fx
                best_x = point.copy()

        # Phase 2: BFGS local minimization from best grid point
        if best_x is not None and calls_made < self.budget:
            # Use multiple starting points around the best for robustness
            # Try BFGS from the best point and a few nearby perturbations
            rng = np.random.default_rng(123)
            candidates = [best_x.copy()]

            # Add some perturbations around the best point
            for sigma in [2.0, 1.0, 0.5]:
                for _ in range(3):
                    perturbed = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                    candidates.append(perturbed)

            # Also try a few random points if budget allows
            if self.budget - calls_made > 50:
                for _ in range(5):
                    candidates.append(rng.uniform(-5, 5, self.dim))

            best_bfgs_x = best_x.copy()
            best_bfgs_f = best_f

            for start in candidates:
                if time.time() > deadline or calls_made >= self.budget:
                    break

                # Define objective for BFGS that counts calls
                def objective(x):
                    nonlocal calls_made
                    if calls_made >= self.budget:
                        return np.inf
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    calls_made += 1
                    return fx

                try:
                    result = minimize(
                        objective,
                        start,
                        method='BFGS',
                        options={'maxiter': 100, 'gtol': 1e-8}
                    )
                    if result.fun < best_bfgs_f:
                        best_bfgs_f = result.fun
                        best_bfgs_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

            # Final polish: a few more BFGS iterations from the best found
            if calls_made < self.budget and time.time() < deadline:
                def final_objective(x):
                    nonlocal calls_made
                    if calls_made >= self.budget:
                        return np.inf
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    calls_made += 1
                    return fx

                try:
                    result = minimize(
                        final_objective,
                        best_bfgs_x,
                        method='BFGS',
                        options={'maxiter': 50, 'gtol': 1e-10}
                    )
                    if result.fun < best_bfgs_f:
                        best_bfgs_f = result.fun
                        best_bfgs_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

            return best_bfgs_x, best_bfgs_f
        else:
            return best_x, best_f
# EVOLVE-BLOCK-END
