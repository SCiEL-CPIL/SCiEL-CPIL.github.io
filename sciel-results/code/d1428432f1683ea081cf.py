"""Optimizer for BBOB f5 (Linear slope) via projected gradient descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Projected gradient descent exploiting the linear structure of f5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _project(self, x):
        return np.clip(x, -5.0, 5.0)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        # ---- Phase 1: establish an initial point and gradient estimate ----
        x0 = np.zeros(dim)
        f0 = func(x0)
        x_best, f_best = x0.copy(), f0
        calls = 1

        # Estimate gradient at x0 via central differences
        h = 0.05
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            fp = func(self._project(xp))
            fm = func(self._project(xm))
            grad[i] = (fp - fm) / (2.0 * h)
            calls += 2
            if calls > budget - 5 or not time_ok():
                return x_best, f_best

        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            # Degenerate: fall back to random search
            sigma = 2.0
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            for _ in range(budget - calls):
                y = self._project(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < f_best:
                        x_best, f_best = x.copy(), fx
                else:
                    sigma *= 0.85
                if not time_ok():
                    break
            return x_best, f_best

        g_dir = grad / gnorm  # unit descent direction

        # ---- Phase 2: line search along the descent direction to the boundary ----
        # Since f is linear (up to the offset/rotation), the minimum along the
        # ray x0 - t*g_dir is at the largest t where x stays in the box.
        # Compute the maximum step to the boundary.
        max_t = np.inf
        for i in range(dim):
            if g_dir[i] > 1e-15:
                # x0[i] - t*g_dir[i] >= -5  =>  t <= (x0[i]+5)/g_dir[i]
                t_i = (x0[i] + 5.0) / g_dir[i]
            elif g_dir[i] < -1e-15:
                # x0[i] - t*g_dir[i] <= 5  =>  t <= (5 - x0[i]) / (-g_dir[i])
                t_i = (5.0 - x0[i]) / (-g_dir[i])
            else:
                continue
            if t_i < max_t:
                max_t = t_i

        if np.isinf(max_t):
            max_t = 10.0
        max_t = max(max_t, 0.0)

        # Evaluate at the boundary point and a few interior points
        candidates = []
        for frac in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
            t = max_t * frac
            xc = self._project(x0 - t * g_dir)
            candidates.append(xc)

        for xc in candidates:
            if calls >= budget or not time_ok():
                break
            fc = func(xc)
            calls += 1
            if fc < f_best:
                x_best, f_best = xc.copy(), fc

        # ---- Phase 3: local refinement around the best point ----
        # Use a small step-size adaptive search to fine-tune
        x = x_best.copy()
        fx = f_best
        sigma = 0.1
        for _ in range(budget - calls):
            if not time_ok() or calls >= budget:
                break
            y = self._project(x + sigma * rng.standard_normal(dim))
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < f_best:
                    x_best, f_best = x.copy(), fx
            else:
                sigma *= 0.8
            # Shrink sigma over time for convergence
            sigma *= 0.999
            if sigma < 1e-4:
                sigma = 1e-4

        # ---- Phase 4: final polish with very small steps ----
        if calls < budget and time_ok():
            sigma = 1e-3
            for _ in range(min(budget - calls, 100)):
                if not time_ok() or calls >= budget:
                    break
                y = self._project(x_best + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                if fy < f_best:
                    x_best, f_best = y.copy(), fy

        return x_best, f_best
# EVOLVE-BLOCK-END
