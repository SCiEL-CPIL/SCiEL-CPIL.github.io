"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with periodic restarts and a time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def remaining():
            return self.budget - calls_made

        def time_ok():
            return time.time() < deadline

        # Phase 1: coarse exploration with multiple restarts
        # Use a large initial step to cover the search space quickly
        n_restarts = 5
        calls_per_restart = self.budget // (n_restarts + 1)

        for r in range(n_restarts):
            if not time_ok() or remaining() <= 0:
                break

            # Initialize near center for first restart, random for others
            if r == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 3.0
            # Run (1+1)-ES for this restart
            max_steps = min(calls_per_restart, remaining())
            for step in range(max_steps):
                if not time_ok() or remaining() <= 0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.8

                # Early termination if sigma is too small
                if sigma < 1e-6:
                    break

        # Phase 2: fine-tuning around the best point found
        if best_x is not None and remaining() > 0 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            max_steps = remaining()
            for step in range(max_steps):
                if not time_ok() or remaining() <= 0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                if sigma < 1e-7:
                    break

        # Phase 3: very fine refinement if budget remains
        if best_x is not None and remaining() > 0 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            max_steps = remaining()
            for step in range(max_steps):
                if not time_ok() or remaining() <= 0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
