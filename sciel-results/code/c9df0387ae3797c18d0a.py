"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive step sizes for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension step sizes
        steps = np.full(dim, 2.0)
        # Track recent progress per dimension
        recent_improvement = np.zeros(dim)

        calls_used = 1

        # Main loop: coordinate descent with adaptive steps
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Try each dimension in random order
            order = rng.permutation(dim)
            for d in order:
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try positive and negative directions
                for sign in [1.0, -1.0]:
                    if calls_used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * steps[d], -5.0, 5.0)
                    if np.array_equal(trial, x):
                        continue

                    fy = func(trial)
                    calls_used += 1

                    if fy < fx:
                        # Improvement: move and increase step
                        x = trial
                        fx = fy
                        steps[d] = min(steps[d] * 1.3, 10.0)
                        recent_improvement[d] = fx - fy  # positive value

                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        break  # Found improvement, stop trying other direction
                    else:
                        # No improvement: decrease step
                        steps[d] = max(steps[d] * 0.7, 0.01)
                        recent_improvement[d] = 0.0

            # Periodic global exploration: if no recent improvement, try a larger random jump
            if np.all(recent_improvement == 0.0) and calls_used < budget:
                # Check if we've been stuck for a while
                # Try a random perturbation in a random subset of dimensions
                if time.time() > deadline:
                    break
                subset_size = max(1, dim // 2)
                subset = rng.choice(dim, size=subset_size, replace=False)
                trial = x.copy()
                for d in subset:
                    trial[d] = np.clip(trial[d] + rng.uniform(-2.0, 2.0), -5.0, 5.0)
                if not np.array_equal(trial, x):
                    fy = func(trial)
                    calls_used += 1
                    if fy < fx:
                        x = trial
                        fx = fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        # Reset steps for improved dimensions
                        for d in subset:
                            steps[d] = max(steps[d], 1.0)
                recent_improvement = np.zeros(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
