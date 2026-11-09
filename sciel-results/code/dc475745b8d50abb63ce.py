"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim

        # Initialize at center of box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        step_success = 0
        step_attempts = 0
        no_improve_count = 0

        # Run (1+1)-ES until budget or time
        calls_made = 1
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(n)
            y = np.clip(x + sigma * direction, -5, 5)

            fy = func(y)
            calls_made += 1

            # Track best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            if fy < fx:
                # Move
                x = y
                fx = fy
                step_success += 1
                no_improve_count = 0
            else:
                no_improve_count += 1

            step_attempts += 1

            # Adapt step size every 10 steps (1/5 rule)
            if step_attempts >= 10:
                success_rate = step_success / step_attempts
                if success_rate > 0.2:
                    sigma *= 1.25
                else:
                    sigma *= 0.8
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)
                step_success = 0
                step_attempts = 0

            # If stuck for a while, try a larger exploratory step
            if no_improve_count >= 50:
                # Big jump to a random point
                x_new = rng.uniform(-5, 5, n)
                fx_new = func(x_new)
                calls_made += 1
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    best_f = min(best_f, fx_new)
                    best_x = x.copy()
                no_improve_count = 0
                sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
