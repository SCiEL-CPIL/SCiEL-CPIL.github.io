"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and strict wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Counting wrapper
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def remaining_time():
            return deadline - time.time()

        # Initial diverse starting points: center, a few random, and a few
        # near-boundary points to cover the box.
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        for _ in range(2):
            starts.append(rng.uniform(-5.0, 5.0, dim) * rng.choice([-1.0, 1.0], size=dim))

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        for x0 in starts:
            if call_count >= budget or remaining_time() < 5.0:
                break
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)

        # (1+1)-ES refinement on the best point with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        min_sigma = 1e-12

        while call_count < budget and remaining_time() > 5.0:
            # Adaptive sigma: increase on success, decrease on failure
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = counted_func(y)

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.2
            else:
                sigma *= 0.87
                if sigma < min_sigma:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
