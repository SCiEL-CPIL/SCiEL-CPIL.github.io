"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and deadline guard.

    Phase 1: cheap multi-start exploration to locate a promising basin.
    Phase 2: a single committed (1+1)-ES run with 1/5 success-rule
    step-size adaptation from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _es_run(self, func, x, fx, sigma, n_calls, rng, deadline):
        """Run (1+1)-ES with 1/5 success rule for n_calls evaluations.

        Returns (x, fx, sigma, total_calls_used).
        """
        dim = self.dim
        success = 0
        used = 0
        for _ in range(n_calls):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            used += 1
            if fy < fx:
                x, fx = y, fy
                success += 1
            else:
                sigma *= 0.87
        # 1/5 success-rule update
        if success > n_calls * 0.2:
            sigma *= 1.5
        elif success < n_calls * 0.2:
            sigma *= 0.75
        return x, fx, sigma, used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # ---- Phase 1: multi-start exploration ----
        n_starts = 6
        phase1 = min(60, max(1, budget // 8))
        per_start = max(2, phase1 // n_starts)

        best_x = None
        best_f = np.inf
        calls = 0

        for _ in range(n_starts):
            if calls >= phase1 or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            # short local descent from this start
            x, fx, sigma, used = self._es_run(
                func, x, fx, 2.0, per_start, rng, deadline
            )
            calls += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # ---- Phase 2: committed (1+1)-ES from best point ----
        remaining = max(0, budget - calls)
        if remaining > 0:
            x, fx, sigma, used = self._es_run(
                func, best_x, best_f, 2.0, remaining, rng, deadline
            )
            calls += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
