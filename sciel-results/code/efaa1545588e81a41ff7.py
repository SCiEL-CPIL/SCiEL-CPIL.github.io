"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with reset-based 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step size
        sigma = 2.5

        # Reset-based success counting for 1/5 rule
        window_size = 5
        successes = 0
        total_in_window = 0

        total_calls = 1  # already made one call

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Generate a candidate
            y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Update success counter and accept if better
            if fy < fx:
                successes += 1
                x = y
                fx = fy

            total_in_window += 1

            # Adapt sigma based on 1/5 success rule (reset after each window)
            if total_in_window >= window_size:
                success_rate = successes / total_in_window
                if success_rate > 1.0 / 5.0:
                    sigma *= 1.06
                elif success_rate < 1.0 / 5.0:
                    sigma *= 0.94

                # Reset counters
                successes = 0
                total_in_window = 0

            # Clamp sigma
            sigma = max(1e-7, min(12.0, sigma))

        return best_x, best_fx
# EVOLVE-BLOCK-END
