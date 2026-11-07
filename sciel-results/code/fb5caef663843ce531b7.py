"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Initialize at a random point in the box
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step size
        sigma = 2.0

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            z = rng.standard_normal(n)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)

            # Evaluate
            fx_trial = func(x_trial)
            calls += 1

            # Update best
            if fx_trial < best_f:
                best_f = fx_trial
                best_x = x_trial.copy()

            # Update step size
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                sigma *= 1.5
            else:
                sigma *= 0.87

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
