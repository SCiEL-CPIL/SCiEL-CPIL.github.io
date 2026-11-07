"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive sigma, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial exploration: try a few random points to find a good starting point
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # Try a few random initializations
        for _ in range(5):
            if calls_used >= budget or time.time() >= deadline:
                break
            candidate = rng.uniform(-5, 5, dim)
            cf = func(candidate)
            calls_used += 1
            if cf < best_f:
                best_x = candidate
                best_f = cf

        # (1+1)-ES parameters
        sigma = 0.5
        success_count = 0
        success_window = 0
        window_size = 10
        no_improve_calls = 0
        max_no_improve = 80
        sigma_history = []

        while calls_used < budget and time.time() < deadline:
            # Generate trial point
            step = sigma * rng.standard_normal(dim)
            trial_x = best_x + step
            trial_x = np.clip(trial_x, -5, 5)
            
            trial_f = func(trial_x)
            calls_used += 1

            if trial_f < best_f:
                # Accept
                best_x = trial_x
                best_f = trial_f
                success_count += 1
                no_improve_calls = 0
            else:
                # Reject
                no_improve_calls += 1

            success_window += 1

            # Adapt sigma based on 1/5 success rule over window
            if success_window >= window_size:
                rate = success_count / success_window
                # Smooth adaptation: scale sigma proportionally to deviation from 1/5
                # If rate > 0.2, increase sigma; if rate < 0.2, decrease
                if rate > 0.2:
                    sigma *= 1.15
                elif rate < 0.2:
                    sigma /= 1.15
                # Clamp sigma
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 5.0)
                success_count = 0
                success_window = 0

            # Restart if stuck: use a smaller perturbation
            if no_improve_calls >= max_no_improve:
                no_improve_calls = 0
                # Reset sigma to a moderate value
                sigma = max(sigma * 0.5, 0.1)
                # Small random perturbation to escape local optimum
                best_x = best_x + 0.3 * sigma * rng.standard_normal(dim)
                best_x = np.clip(best_x, -5, 5)
                # Evaluate the perturbed point
                new_f = func(best_x)
                calls_used += 1
                if new_f < best_f:
                    best_f = new_f
                # If no improvement after restart, try a larger jump
                if calls_used >= budget or time.time() >= deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
