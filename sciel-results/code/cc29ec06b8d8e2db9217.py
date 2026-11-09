"""Black-box continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start exploration
        # Use a few restarts to explore different regions
        n_starts = 5
        calls_per_start = budget // n_starts
        extra = budget - calls_per_start * n_starts

        for s in range(n_starts):
            # Check deadline
            if time.time() > deadline - 10.0:
                break

            # Start point: mix of random and center-biased
            if s == 0:
                x = rng.uniform(-4, 4, dim)
            elif s == 1:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES for this start
            sigma = 3.0  # Initial step size
            calls_used = 1

            max_calls_for_this_start = calls_per_start + (1 if s < extra else 0)

            for i in range(max_calls_for_this_start - 1):
                if time.time() > deadline - 5.0:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # If sigma becomes too small, break early
                if sigma < 1e-6:
                    break

        # Phase 2: Fine-tuning around the best point found
        if best_x is not None and time.time() < deadline - 2.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            # Use remaining budget for fine-tuning
            remaining = budget - sum(1 for _ in range(0))  # We'll just do a fixed number
            # Actually, let's do a fixed number of fine-tuning steps
            for i in range(100):
                if time.time() > deadline - 1.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

                if sigma < 1e-7:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
