"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initialize at origin (sphere minimum is often near origin after shift)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        sigma = 1.0
        success_rate = 0.0
        n_success = 0
        n_total = 0

        call_count = 1

        for _ in range(self.budget - 1):
            # Time guard
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1

            n_total += 1
            if fy < fx:
                x, fx = y, fy
                n_success += 1
                success_rate = n_success / n_total
                # Increase step size on success
                sigma *= 1.2
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Decrease step size on failure
                sigma *= 0.9

            # Keep sigma in reasonable bounds
            sigma = max(1e-6, min(sigma, 10.0))

            # Reset success tracking periodically
            if n_total >= 50:
                n_success = 0
                n_total = 0
                success_rate = 0.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
