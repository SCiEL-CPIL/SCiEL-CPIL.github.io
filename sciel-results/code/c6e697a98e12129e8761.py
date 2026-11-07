"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 320s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        def now():
            return time.time()

        # Global best
        x_best = rng.uniform(-5, 5, dim)
        fx_best = func(x_best)
        calls_made = 1

        # Multi-start (1+1)-ES: restart when stuck, keep best-so-far.
        # Each phase is a short (1+1)-ES; restarts escape local minima.
        max_phase_calls = 120  # calls per phase before forcing a restart
        min_sigma = 1e-4
        max_sigma = 4.0

        while calls_made < budget:
            if now() >= deadline:
                break

            # Start a phase: with some probability restart from scratch,
            # otherwise continue from the current best (refine).
            # Use a fresh start to escape local minima.
            if rng.random() < 0.5:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
            else:
                x = x_best.copy()
                fx = fx_best

            sigma = 1.5
            phase_calls = 0
            no_improve = 0
            max_no_improve = 30  # restart phase if stuck

            while phase_calls < max_phase_calls and calls_made < budget:
                if now() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                phase_calls += 1

                if fy < fx:
                    # Accept step
                    x, fx = y, fy
                    if fx < fx_best:
                        x_best, fx_best = x.copy(), fx
                    sigma = min(sigma * 1.25, max_sigma)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.8, min_sigma)
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break

            # After a phase, if we improved the global best, we may
            # continue refining; otherwise restart from a fresh point.
            if calls_made >= budget:
                break
            if now() >= deadline:
                break

        return x_best, fx_best
# EVOLVE-BLOCK-END
