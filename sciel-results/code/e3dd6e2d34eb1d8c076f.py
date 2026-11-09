"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize from broad uniform scout
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size parameters
        sigma = 2.0
        temp = 5.0
        cool_rate = 0.97
        min_temp = 1e-6

        for i in range(budget - 1):
            # Check deadline with safety margin
            if time.time() > deadline - 5.0:
                break

            # Occasionally take a larger jump to escape plateaus
            if rng.random() < 0.1:
                step_sigma = sigma * 2.0
            else:
                step_sigma = sigma

            # Generate candidate
            y = np.clip(x + step_sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)

            if fy < fx:
                # Always accept improvements
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = y.copy(), fy
                # Increase step size slightly on improvement
                sigma = min(sigma * 1.1, 10.0)
            else:
                # Accept worse solutions with probability exp(-delta/temp)
                delta = fy - fx
                if delta < 1e-10:
                    # Plateau: accept to keep moving
                    x, fx = y, fy
                elif delta / temp < 50.0:  # avoid overflow in exp
                    if rng.random() < np.exp(-delta / temp):
                        x, fx = y, fy
                # Decrease step size on rejection
                sigma = max(sigma * 0.9, 0.01)

            # Cool down
            temp = max(temp * cool_rate, min_temp)

        # Final check: ensure we have a valid result
        if fx > best_fx:
            best_x, best_fx = x.copy(), fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
