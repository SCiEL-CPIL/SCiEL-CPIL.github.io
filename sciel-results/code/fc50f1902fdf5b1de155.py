"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and anti-stagnation kicks."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _es_run(self, x, fx, sigma, rng, func, max_calls, deadline):
        """Run a (1+1)-ES from starting point x for up to max_calls evaluations."""
        d = self.dim
        calls = 0
        no_improve = 0
        for _ in range(max_calls):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.3
            else:
                sigma *= 0.9
                no_improve += 1
                # Anti-stagnation: if stuck, kick with larger step
                if no_improve >= 15:
                    kick = rng.uniform(1.0, 3.0, d)
                    y2 = np.clip(x + kick * rng.standard_normal(d), -5, 5)
                    fy2 = func(y2)
                    calls += 1
                    if fy2 < fx:
                        x, fx = y2, fy2
                        no_improve = 0
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    no_improve = 0
            # Keep sigma in reasonable range
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 10.0)
        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        d = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start initialization with 10 random points
        n_starts = 10
        candidates = []
        for i in range(n_starts):
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            candidates.append((fx, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run (1+1)-ES from each starting point, allocating remaining budget
        remaining_budget = self.budget - n_starts
        calls_per_start = max(1, remaining_budget // n_starts)

        best_x = candidates[0][1]
        best_f = candidates[0][0]

        for i, (fx, x) in enumerate(candidates):
            if time.time() > deadline:
                break
            if remaining_budget <= 0:
                break

            # Allocate slightly more to better starts
            weight = 1.0 / (i + 1)
            allocated = max(5, int(calls_per_start * weight))
            allocated = min(allocated, remaining_budget)

            x_new, fx_new, used = self._es_run(
                x, fx, 1.5, rng, func, allocated, deadline
            )
            remaining_budget -= used

            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
