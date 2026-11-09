"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid search with local refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid search
        # Start with a 3^5 = 243 point grid over [-5, 5]^5
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() < deadline

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            if not check_time():
                return False
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return True

        # Coarse grid: 3 levels per dimension
        coarse_levels = 3
        coarse_points = coarse_levels ** dim
        if coarse_points <= budget:
            # Generate all combinations
            coords = np.linspace(-5, 5, coarse_levels)
            grid = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)
            for x in grid:
                if not eval_point(x):
                    break

        # Phase 2: Medium grid around best region
        if check_time() and calls_made < budget:
            if best_x is not None:
                # Focus on region around best point with margin
                margin = 2.0
                center = np.clip(best_x, -margin, margin)
                med_levels = 5
                med_coords = np.linspace(center - margin, center + margin, med_levels)
                med_coords = np.clip(med_coords, -5, 5)
                med_grid = np.array(np.meshgrid(*([med_coords] * dim))).T.reshape(-1, dim)
                for x in med_grid:
                    if not check_time() or calls_made >= budget:
                        break
                    eval_point(x)

        # Phase 3: Fine local search around best point
        if check_time() and calls_made < budget and best_x is not None:
            # Use a local gradient-free optimizer with shrinking steps
            x = best_x.copy()
            fx = best_f
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if not check_time() or calls_made >= budget:
                    break
                improved = True
                while improved:
                    improved = False
                    if not check_time() or calls_made >= budget:
                        break
                    for d in range(dim):
                        for direction in [1, -1]:
                            if not check_time() or calls_made >= budget:
                                break
                            y = x.copy()
                            y[d] += direction * step
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls_made += 1
                            if fy < fx:
                                x, fx = y, fy
                                best_x, best_f = x.copy(), fx
                                improved = True

        # Phase 4: If we still have budget, do a finer global exploration
        if check_time() and calls_made < budget and best_x is not None:
            # Try a few random perturbations around best
            rng = np.random.default_rng(42)
            remaining = budget - calls_made
            n_random = min(remaining, 50)
            for _ in range(n_random):
                if not check_time() or calls_made >= budget:
                    break
                x_trial = best_x + rng.normal(0, 0.5, dim)
                x_trial = np.clip(x_trial, -5, 5)
                eval_point(x_trial)

        return best_x, best_f
# EVOLVE-BLOCK-END
