"""Multi-start adaptive (1+1)-ES with deadline guard for BBOB Weierstrass (f16)."""
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
        deadline = time.time() + 285.0  # safety margin below the 330s kill

        # ---- Phase 1: minimal multi-start to find a good basin early ----
        # Use fewer starts to preserve budget for local optimization.
        # The origin is a strong default for BBOB f16.
        n_starts = 4
        seeds = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            seeds.append(rng.uniform(-3.0, 3.0, dim))

        # De-duplicate
        seen = []
        uniq = []
        for s in seeds:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.append(key)
                uniq.append(s)

        best_x = None
        best_f = np.inf
        for s in uniq:
            if best_x is not None and time.time() > deadline:
                break
            fx = func(np.clip(s, -5.0, 5.0))
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        if best_x is None:
            x0 = np.zeros(dim)
            best_f = func(x0)
            best_x = x0.copy()

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # smaller initial step for faster local convergence
        min_sigma, max_sigma = 1e-8, 5.0
        # 1/5 success rule
        target = 1.0 / 5.0
        c_sigma = 0.25  # slightly faster adaptation

        n_calls = len(uniq)
        for _ in range(self.budget - n_calls):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
            # sigma adaptation
            sigma *= np.exp(c_sigma * ((1.0 if fy < fx else 0.0) - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
