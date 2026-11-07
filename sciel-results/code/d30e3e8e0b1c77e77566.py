"""Model-and-optimize optimizer for BBOB f5 (linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin out of 300s

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx

        # Step size for (1+1)-ES
        sigma = 2.0  # initial step size
        success_count = 0
        counter = 0

        # 1/5 success rule parameters
        delta = 1.0  # increase factor
        sigma_min = 1e-10
        sigma_max = 10.0

        while calls < budget and time_ok():
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -4, 4)
            fy = func(y)
            calls += 1

            # Check if improvement
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                success_count = 0

            counter += 1

            # Apply 1/5 success rule every 10 steps
            if counter >= 10:
                if success_count >= 2:  # 2/10 = 1/5
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_count <= 1:  # 1/10 < 1/5
                    sigma = max(sigma / 1.2, sigma_min)
                else:
                    # Keep sigma
                    pass
                success_count = 0
                counter = 0

            # If sigma is very small, try a random restart
            if sigma < 1e-6 and calls < budget - 50:
                # Random restart with smaller perturbation
                x = best_x + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -4, 4)
                fx = func(x)
                calls += 1
                sigma = 0.5

        return best_x, best_fx

# EVOLVE-BLOCK-END
