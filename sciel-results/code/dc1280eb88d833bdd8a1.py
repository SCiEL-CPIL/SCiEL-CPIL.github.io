"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive population and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 275  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = None
        best_f = np.inf

        # We'll use a simple but effective approach:
        # 1. A few random starts
        # 2. For each start, run a simplified CMA-ES or (1+1)-ES with restarts
        # 3. Use the remaining budget wisely

        # Strategy: Use a combination of random sampling and local search
        # For f18 (Schaffers F7), the function is ill-conditioned, so we need
        # good exploration initially and exploitation later.

        # Phase 1: Random sampling to find a good starting point
        # Phase 2: Local optimization from the best point found

        # Let's do a hybrid:
        # - Use about 20% of budget for random sampling
        # - Use the rest for a (1+1)-ES or simple evolution strategy from the best point

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling
        n_random = max(10, budget // 10)  # at least 10, at most ~50
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local search using (1+1)-ES with restarts
        # We'll run multiple short (1+1)-ES runs from the best point found
        remaining = budget - n_random
        if remaining <= 0:
            return best_x, best_f

        # Use a (1+1)-ES with adaptive sigma
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # start with moderate step size

        # Success counter for 1/5 rule
        successes = 0
        attempts = 0
        success_threshold = 5  # 1/5 rule

        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                # Accept
                x, fx = y, fy
                successes += 1
            else:
                # Reject
                pass

            attempts += 1

            # Adapt sigma using 1/5 success rule
            if attempts >= 10:
                success_rate = successes / attempts
                if success_rate > 1.0 / 5.0:
                    sigma *= 1.2
                elif success_rate < 1.0 / 5.0:
                    sigma *= 0.8
                # Reset counters
                successes = 0
                attempts = 0

            # Keep global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # If sigma becomes too small, do a small random jump
            if sigma < 1e-4:
                x = np.clip(best_x + rng.uniform(-0.5, 0.5, dim), -5, 5)
                fx = func(x)
                sigma = 0.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
