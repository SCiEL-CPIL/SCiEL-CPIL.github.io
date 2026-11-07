"""Fresh design: multi-start + adaptive (1+1)-ES with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin below the 330s kill
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        # ---- Phase 1: cheap multi-start to find a good basin early ----
        n_starts = 8
        seeds = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            seeds.append(rng.uniform(-4.5, 4.5, dim))
        # De-duplicate (avoid duplicate calls)
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
            if time.time() > deadline and best_x is not None:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        min_sigma, max_sigma = 1e-6, 6.0
        target = 1.0 / 5.0
        c_sigma = 0.2

        remaining = self.budget - call_count
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_func(y)
            improved = fy < fx
            if improved:
                x, fx = y, fy
            # sigma adaptation via 1/5 success rule (per step)
            sigma *= np.exp(c_sigma * ((1.0 if improved else 0.0) - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        # ---- Phase 3: local polish with Nelder-Mead if budget remains ----
        if call_count < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    lambda z: safe_func(z),
                    x0=x,
                    method='Nelder-Mead',
                    options={'maxiter': max(1, self.budget - call_count),
                             'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < fx:
                    x = np.clip(res.x, -5.0, 5.0)
                    fx = res.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
