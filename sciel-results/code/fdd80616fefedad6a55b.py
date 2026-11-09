"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: multi-start adaptive (1+1)-ES with 1/5-rule sigma
    adaptation and a final high-precision coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(42)

        def now():
            return time.time()

        # ---- Phase 1: Multi-start adaptive (1+1)-ES with 1/5 success rule ----
        # Diverse starts: origin, structured points, and random draws.
        starts = [
            np.array([0.0] * dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf

        # Allocate budget per start: first few get a short exploration,
        # then a longer adaptive refinement on the best-so-far.
        # Strategy: give each start a small adaptive ES run, keep the best,
        # then do a long adaptive ES run from the best point.

        def adaptive_es(x0, fx0, budget_for_run, sigma0):
            """Run an adaptive (1+1)-ES with 1/5 success rule for up to
            budget_for_run calls. Returns (x, fx)."""
            x = np.clip(x0, -5, 5).copy()
            fx = fx0
            sigma = sigma0
            # 1/5 success rule bookkeeping
            successes = 0
            window = 0
            window_size = 12
            used = 0
            while used < budget_for_run:
                if now() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = safe_func(y)
                except StopIteration:
                    break
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                else:
                    successes = 0
                window += 1
                if window >= window_size:
                    rate = successes / window
                    if rate > 0.2:
                        sigma *= 1.4
                    elif rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                    window = 0
                if sigma < 1e-9:
                    break
            return x, fx

        # Short exploratory adaptive runs from each start (small budget each).
        # Reserve the bulk of the budget for the final long run.
        explore_budget_each = 12
        for s in starts:
            if calls[0] >= budget - 20 or now() > deadline:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            # Short adaptive refinement from this start
            x, fx = adaptive_es(x, fx, explore_budget_each, sigma0=1.5)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: Long adaptive (1+1)-ES from the best point ----
        remaining = budget - calls[0]
        # Leave a small reserve for the final polish.
        polish_reserve = max(10, int(0.02 * budget))
        long_budget = max(0, remaining - polish_reserve)
        if best_x is not None and long_budget > 0:
            x, fx = adaptive_es(best_x, best_f, long_budget, sigma0=0.8)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 3: High-precision coordinate polish ----
        # Fine coordinate descent with geometrically shrinking steps.
        if best_x is not None:
            step = 0.1
            for _ in range(30):
                if calls[0] >= budget or now() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if now() > deadline:
                        break
                    for sign in (1, -1):
                        if calls[0] >= budget:
                            break
                        x2 = best_x.copy()
                        x2[d] = np.clip(best_x[d] + sign * step, -5, 5)
                        try:
                            f2 = safe_func(x2)
                        except StopIteration:
                            break
                        if f2 < best_f:
                            best_f = f2
                            best_x = x2
                            improved = True
                    if now() > deadline:
                        break
                if improved:
                    step *= 0.7
                else:
                    step *= 0.5
                if step < 1e-7:
                    break

        # Ensure best_x/best_f are set (fallback if no successful calls).
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
