"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        rng = np.random.default_rng(0)
        
        # Initial point: uniform random in the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Step size
        sigma = 2.0
        # 1/5 rule parameters
        alpha_up = 1.1
        alpha_down = 1.0 / 1.1
        window = 10
        successes = 0

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            # Clip to box
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                successes += 1
            else:
                successes -= 1

            # 1/5 success rule: adjust sigma based on success rate over window
            if successes >= window // 5:
                sigma *= alpha_up
            elif successes <= -window // 5:
                sigma *= alpha_down
            successes = 0

        return best_x, best_f

# EVOLVE-BLOCK-END
