"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with zero-centered init, stable step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 2.0

        # Track consecutive failures for occasional large jumps
        fail_count = 0
        max_fail = 20

        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Occasionally make a larger exploratory step if stuck
            if fail_count >= max_fail:
                step_scale = 3.0
                fail_count = 0
            else:
                step_scale = 1.0

            y = np.clip(x + sigma * step_scale * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                fail_count = 0
            else:
                sigma *= 0.9
                fail_count += 1

                # If sigma gets too small, reset to encourage exploration
                if sigma < 0.01:
                    sigma = 0.5
                    fail_count = 0

        return x, fx
# EVOLVE-BLOCK-END
