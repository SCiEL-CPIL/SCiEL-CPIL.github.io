"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+4)-ES with 1/5 success rule for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 3.0
        pop_size = 4
        success_count = 0
        total_trials = 0
        calls_made = 1

        # Adaptive step size parameters
        sigma_min = 0.01
        sigma_max = 10.0
        target_success_rate = 0.2  # 1/5 rule

        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for j in range(pop_size):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Select best offspring
            offspring.sort(key=lambda t: t[0])
            best_fy, best_y = offspring[0]

            total_trials += pop_size

            if best_fy < fx:
                # Success: move to best offspring
                x = best_y
                fx = best_fy
                success_count += 1
                # Increase sigma
                sigma = min(sigma * 1.3, sigma_max)
            else:
                # Failure: stay at current x
                # Decrease sigma
                sigma = max(sigma * 0.7, sigma_min)

            # Adaptive sigma adjustment based on success rate
            if total_trials >= 20:
                current_rate = success_count / total_trials
                if current_rate > target_success_rate + 0.1:
                    sigma = min(sigma * 1.1, sigma_max)
                elif current_rate < target_success_rate - 0.1:
                    sigma = max(sigma * 0.9, sigma_min)

            # Occasionally do a larger jump to escape local minima (Rastrigin has many)
            if total_trials % 50 == 0 and sigma < 1.0:
                # Try a random restart around best
                x = np.clip(best_x + sigma * 2.0 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls_made += 1
                sigma = max(sigma * 1.5, sigma_min)

        return best_x, best_fx
# EVOLVE-BLOCK-END
