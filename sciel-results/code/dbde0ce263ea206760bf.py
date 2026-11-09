"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0

        def _time_ok():
            return time.time() < deadline

        # Initial broad sampling to find a good starting point
        n_initial = 30
        best_x = None
        best_f = np.inf

        for _ in range(n_initial):
            if not _time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            best_x = x.copy()
            best_f = fx

        # Multi-start (1+1)-ES from the best initial point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        sigma_min = 1e-8
        sigma_max = 10.0

        remaining = budget - calls_used

        # Adaptive (1+1)-ES with restarts
        # Use a simple multiplicative update for sigma
        # Success: sigma *= 1.2
        # Failure: sigma *= 0.85
        steps_since_best_improvement = 0
        restart_threshold = 30

        while calls_used < budget:
            if not _time_ok():
                break

            # Generate candidate
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            try:
                fy = func(y)
            except Exception:
                break

            calls_used += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.2

                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
                    steps_since_best_improvement = 0
                else:
                    steps_since_best_improvement += 1
            else:
                # Failure
                sigma *= 0.85
                steps_since_best_improvement += 1

            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            if sigma < sigma_min:
                sigma = sigma_min

            # Restart mechanism
            # If we haven't improved the global best in 'restart_threshold' steps,
            # reset the local search to the best point with a larger step size
            if steps_since_best_improvement >= restart_threshold:
                x = best_x.copy()
                fx = best_f
                sigma = 2.0  # Reset to a reasonable exploration size
                steps_since_best_improvement = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
