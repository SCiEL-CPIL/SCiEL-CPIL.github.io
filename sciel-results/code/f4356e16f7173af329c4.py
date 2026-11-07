"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size and restarts for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            nonlocal calls_made
            return calls_made >= budget or (time.time() - t0) > deadline

        # Try a few different initial strategies
        # Strategy 1: Start near center (0,0,0,...)
        # Strategy 2: Random start
        # Strategy 3: Grid-like exploration first

        # Phase 1: Quick exploration with multiple random starts
        # Use a (1+lambda)-ES with lambda=10, adaptive sigma

        lambda_ = 10
        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # Start at origin since sphere minimum is often near center after shift
        x = np.zeros(dim)
        fx = func(x)
        calls_made += 1

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # Adaptive step size control
        # Use success rate based adaptation
        successes = 0
        window = 50  # adaptation window

        for i in range(budget - 1):
            if check_time():
                break

            # Generate lambda offspring
            offspring = []
            for j in range(lambda_):
                if check_time():
                    break
                step = rng.standard_normal(dim)
                y = np.clip(x + sigma * step, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            if not offspring:
                break

            offspring.sort(key=lambda t: t[0])

            # Select best offspring
            fy_best, y_best = offspring[0]

            # Count successes (offspring better than parent)
            n_success = sum(1 for fy, _ in offspring if fy < fx)
            successes += n_success

            if fy_best < fx:
                x = y_best
                fx = fy_best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase sigma slightly on success
                sigma = min(sigma * 1.05, sigma_max)
            else:
                # Decrease sigma on failure
                sigma = max(sigma * 0.95, sigma_min)

            # Periodic adaptation: adjust sigma based on success rate over window
            if (i + 1) % window == 0:
                # Target success rate ~ 1/lambda (for (1+lambda)-ES)
                target = 1.0 / lambda_
                observed = successes / (window * lambda_)
                if observed > target * 1.5:
                    sigma = min(sigma * 1.2, sigma_max)
                elif observed < target * 0.5:
                    sigma = max(sigma * 0.8, sigma_min)
                successes = 0

            # Check if we're stuck: if sigma is very small and no progress
            if sigma < 1e-6 and i > 100:
                # Try a restart with a larger step
                sigma = 1.0
                # Maybe jump to a new random point
                if rng.random() < 0.5:
                    x_new = rng.uniform(-5, 5, dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
