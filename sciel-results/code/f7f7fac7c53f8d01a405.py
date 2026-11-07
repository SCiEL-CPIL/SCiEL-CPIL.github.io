"""Multi-start adaptive (1+1)-ES with larger initial sigma for BBOB Weierstrass (f16)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin below the 330s kill
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        # ---- Phase 1: minimal multi-start to find a good basin early ----
        # Center + a few random starts to conserve budget for the ES loop
        seeds = [np.zeros(dim)]
        for _ in range(5):
            seeds.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        for s in seeds:
            if time.time() > deadline and best_x is not None:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # ---- Phase 2: adaptive (1+1)-ES with larger initial sigma ----
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # start large to explore the oscillatory landscape
        min_sigma, max_sigma = 1e-7, 8.0
        target = 1.0 / 5.0  # 1/5 success rule
        c_sigma = 0.3  # moderate adaptation speed

        remaining = self.budget - call_count
        for step in range(remaining):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_func(y)
            improved = fy < fx
            if improved:
                x, fx = y, fy

            # Per-step sigma adaptation
            sigma *= np.exp(c_sigma * ((1.0 if improved else 0.0) - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
