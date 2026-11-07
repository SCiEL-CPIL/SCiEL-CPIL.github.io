"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pattern search with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Directions: coordinate axes and their negatives only (10 directions for dim=5)
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            directions.append(-d)

        step = 1.0
        min_step = 1e-6
        max_step = 10.0
        no_progress = 0
        max_no_progress = 5

        while calls < self.budget:
            if time.time() > deadline:
                break

            progress = False
            for d in directions:
                if calls >= self.budget - 1:  # Leave room for potential restart
                    break
                if time.time() > deadline:
                    break

                # Try to move in direction d
                trial_x = np.clip(x + step * d, -5, 5)
                if np.allclose(trial_x, x):
                    continue
                trial_f = func(trial_x)
                calls += 1

                if trial_f < fx:
                    x = trial_x
                    fx = trial_f
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    progress = True
                    step = min(step * 1.2, max_step)
                    no_progress = 0
                    break  # restart direction loop with new x
                else:
                    step = max(step * 0.5, min_step)

            if not progress:
                no_progress += 1
                if no_progress >= max_no_progress or step <= min_step * 1.01:
                    # Local minimum found; try a random restart
                    if time.time() > deadline or calls >= self.budget - 1:
                        break
                    # Random perturbation
                    perturbation = rng.normal(0, 1.0, dim)
                    x_new = np.clip(x + perturbation, -5, 5)
                    fx_new = func(x_new)
                    calls += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    step = max(step, 0.5)
                    no_progress = 0
                else:
                    # Reduce step further
                    step = max(step * 0.5, min_step)

        return best_x, best_f
# EVOLVE-BLOCK-END
