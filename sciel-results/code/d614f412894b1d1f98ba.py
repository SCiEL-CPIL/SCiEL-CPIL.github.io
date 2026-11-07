"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol-based global search with adaptive local refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Use scipy's Sobol sequence for low-discrepancy sampling
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Systematic Sobol sampling to cover the domain
        # Generate Sobol points and map to [-5, 5]^dim
        # We'll do this in batches to check the deadline
        total_sobol_points = 200  # initial global exploration
        sobol_points = sobol.random(total_sobol_points)
        # Map [0,1]^d to [-5,5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        for i in range(total_sobol_points):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            update_best(x, fx)

        # Phase 2: Adaptive local refinement around the best point
        # Use a multi-scale approach: try different step sizes
        if best_x is not None and time.time() < deadline:
            # Try a series of local searches with decreasing step sizes
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
            for step in step_sizes:
                if time.time() > deadline:
                    break
                # Generate a few perturbed points around the best
                n_perturbations = 10
                for _ in range(n_perturbations):
                    if time.time() > deadline:
                        break
                    perturbation = np.random.default_rng().standard_normal(dim) * step
                    x_new = np.clip(best_x + perturbation, -5, 5)
                    fx_new = func(x_new)
                    if fx_new < best_f:
                        best_x = x_new.copy()
                        best_f = fx_new

        # Phase 3: If we still have budget, do more refined searches
        # Generate additional Sobol points in a smaller region around the best
        if best_x is not None and time.time() < deadline:
            # Create a smaller search region around the best point
            region_size = 1.0
            n_additional = 100
            sobol2 = qmc.Sobol(d=dim, scramble=True, seed=123)
            sobol_points2 = sobol2.random(n_additional)
            # Map to region around best_x
            sobol_points2 = (sobol_points2 * 2.0 - 1.0) * region_size + best_x
            sobol_points2 = np.clip(sobol_points2, -5, 5)

            for i in range(n_additional):
                if time.time() > deadline:
                    break
                x = sobol_points2[i]
                fx = func(x)
                update_best(x, fx)

        # Phase 4: Final fine-tuning with very small steps
        if best_x is not None and time.time() < deadline:
            step_sizes_fine = [0.01, 0.005, 0.001]
            for step in step_sizes_fine:
                if time.time() > deadline:
                    break
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    perturbation = np.random.default_rng().standard_normal(dim) * step
                    x_new = np.clip(best_x + perturbation, -5, 5)
                    fx_new = func(x_new)
                    if fx_new < best_f:
                        best_x = x_new.copy()
                        best_f = fx_new

        return best_x, best_f

# EVOLVE-BLOCK-END
