"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive window, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step size
        sigma = 3.0

        # Track success rate for 1/5 rule with adaptive window
        window_size = 10
        successes = 0
        total_in_window = 0

        total_calls = 1  # already made one call

        # Track stagnation for potential restart
        stagnation_count = 0
        best_fx_at_last_improve = best_fx
        max_stagnation = 50

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Check for stagnation - if no improvement for max_stagnation steps, restart
            if stagnation_count >= max_stagnation:
                # Restart from best with larger sigma
                x = best_x.copy()
                fx = best_fx
                sigma = max(sigma * 1.5, 1.0)
                stagnation_count = 0
                successes = 0
                total_in_window = 0

            # Generate a candidate
            y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                best_fx_at_last_improve = best_fx
                stagnation_count = 0

            # Update success counter
            if fy < fx:
                successes += 1
                # Accept the candidate
                x = y
                fx = fy
            else:
                stagnation_count += 1

            total_in_window += 1

            # Adapt sigma based on 1/5 success rule
            if total_in_window >= window_size:
                success_rate = successes / total_in_window
                if success_rate > 1.0 / 5.0:
                    sigma *= 1.10  # increase more aggressively
                elif success_rate < 1.0 / 5.0:
                    sigma *= 0.90  # decrease more aggressively

                # Reset counters
                successes = 0
                total_in_window = 0

            # Clamp sigma
            sigma = max(1e-4, min(10.0, sigma))

        # Final polish: use a local search around best_x
        if total_calls < budget and time.time() < deadline - 10:
            # Nelder-Mead style local refinement
            polish_x = best_x.copy()
            polish_fx = best_fx
            polish_sigma = max(sigma * 0.1, 0.01)
            polish_budget = min(50, budget - total_calls)
            
            for _ in range(polish_budget):
                if time.time() >= deadline - 5:
                    break
                for d in range(dim):
                    for direction in [1, -1]:
                        trial = polish_x.copy()
                        trial[d] += direction * polish_sigma
                        trial = np.clip(trial, -5, 5)
                        trial_fx = func(trial)
                        total_calls += 1
                        
                        if trial_fx < polish_fx:
                            polish_x = trial
                            polish_fx = trial_fx
                            if polish_fx < best_fx:
                                best_fx = polish_fx
                                best_x = polish_x.copy()
                        if total_calls >= budget or time.time() >= deadline - 5:
                            break
                    if total_calls >= budget or time.time() >= deadline - 5:
                        break
                # Shrink polish step
                polish_sigma *= 0.8

        return best_x, best_fx
# EVOLVE-BLOCK-END
