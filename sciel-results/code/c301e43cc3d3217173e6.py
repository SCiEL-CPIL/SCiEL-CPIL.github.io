"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        # We'll do multiple restarts to explore different regions
        # f21 has 101 peaks, so exploration is important
        max_restarts = 50  # max number of restarts
        calls_per_restart = max(10, self.budget // max_restarts)

        for restart in range(max_restarts):
            if calls >= self.budget:
                break
            if time_left() < 2.0:
                break

            # Initialize at a random point, with some bias towards center
            # for later restarts
            if restart == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Mix of random and near-best
                if rng.random() < 0.5 and best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Adaptive step size
            for step in range(calls_per_restart):
                if calls >= self.budget:
                    break
                if time_left() < 1.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    sigma *= 0.8
                    if sigma < 0.01:
                        sigma = 0.01

        return best_x, best_f
# EVOLVE-BLOCK-END
