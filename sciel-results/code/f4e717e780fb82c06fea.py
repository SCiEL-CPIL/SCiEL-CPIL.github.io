"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive sigma, restarts on stagnation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Use a population-based (1+lambda) strategy with restarts
        # For f6 (Attractive sector), the function has a single global minimum
        # but can have flat regions. A larger population helps escape plateaus.

        lambda_pop = 10  # offspring count per generation
        max_generations = budget // lambda_pop  # ~50 generations

        # Adaptive sigma: start with a reasonable step size
        sigma = 3.0
        sigma_min = 1e-4
        sigma_max = 10.0

        # Stagnation tracking: if no improvement in N evaluations, restart
        stagnation_window = 50  # evaluations without improvement triggers restart
        calls_since_improvement = 0
        total_calls = 0

        # Initial point: try center first, then random
        x = np.zeros(dim)
        fx = func(x)
        total_calls += 1
        best_x = x.copy()
        best_f = fx

        # If center isn't great, try a few random points to get a better start
        for _ in range(5):
            if total_calls >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            total_calls += 1
            if fx_try < best_f:
                best_x = x_try.copy()
                best_f = fx_try

        x = best_x.copy()
        fx = best_f
        calls_since_improvement = 0

        # Main optimization loop
        while total_calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda offspring
            offspring = []
            for _ in range(lambda_pop):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Adaptive step: scale with sigma
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Find best offspring
            best_off = min(offspring, key=lambda t: t[1])
            best_off_x, best_off_f = best_off

            if best_off_f < fx:
                # Accept: move to best offspring
                x = best_off_x
                fx = best_off_f
                calls_since_improvement = 0

                # Increase sigma slightly on success
                sigma = min(sigma * 1.2, sigma_max)

                # Update global best
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
            else:
                # No improvement: decrease sigma
                sigma = max(sigma * 0.8, sigma_min)
                calls_since_improvement += lambda_pop

                # If significantly worse, try a small random perturbation from best
                if calls_since_improvement >= stagnation_window:
                    # Restart: jump to a random point near the best
                    jump_size = max(sigma, 1.0)
                    x = np.clip(best_x + jump_size * rng.standard_normal(dim), -5, 5)
                    fx = func(x)
                    total_calls += 1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                    calls_since_improvement = 0
                    sigma = 2.0  # reset sigma

            # If sigma is too small, we're stuck; do a bigger jump
            if sigma < 0.05 and calls_since_improvement > 20:
                jump_size = 3.0
                x = np.clip(best_x + jump_size * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                calls_since_improvement = 0
                sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
