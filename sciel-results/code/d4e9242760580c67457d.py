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

        # Initialize at origin - typically closer to minimizer in [-4,4]^5
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step size - larger to explore quickly
        sigma = 3.0

        # Track recent successes for adaptive sigma
        success_count = 0
        window_size = 10
        recent_successes = []
        no_improvement_count = 0
        polish_mode = False

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Detect convergence and switch to polish mode
            if not polish_mode:
                no_improvement_count += 1
                if no_improvement_count >= 50 and i > 100:
                    polish_mode = True
                    sigma = min(sigma, 0.5)

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
                no_improvement_count = 0

            # Update step size based on success
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                success_count += 1
                recent_successes.append(1)
            else:
                recent_successes.append(0)

            # Keep window size
            if len(recent_successes) > window_size:
                recent_successes.pop(0)

            # Adaptive sigma: target ~25% success rate for (1+1)-ES
            if len(recent_successes) == window_size:
                rate = sum(recent_successes) / window_size
                if rate > 0.3:
                    sigma *= 1.2
                elif rate < 0.2:
                    sigma *= 0.85

            # In polish mode, use smaller steps
            if polish_mode:
                sigma = min(sigma, 0.3)

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-5, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
