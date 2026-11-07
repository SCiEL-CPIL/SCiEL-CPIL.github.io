"""Fresh design: global sampling + Nelder-Mead refinement + (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: global sampling, Nelder-Mead local refinement, (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s wall-clock kill
        dim = self.dim
        budget = self.budget

        calls = [0]

        def f(x):
            x = np.asarray(x, dtype=float)
            calls[0] += 1
            return func(x)

        def time_ok():
            return calls[0] < budget and (time.time() - start_time) < deadline

        # ---- Phase 1: Global sampling to find diverse basins ----
        rng = np.random.default_rng(12345)
        candidates = []
        # center
        candidates.append(np.zeros(dim))
        # vertices
        for v in np.array([-5.0, 5.0]) ** np.arange(dim):
            candidates.append(v)
        # mid-edges
        for i in range(dim):
            for s in (-5.0, 5.0):
                pt = np.zeros(dim)
                pt[i] = s
                candidates.append(pt)
        # random points
        candidates.append(rng.uniform(-5, 5, dim))
        candidates.append(rng.uniform(-5, 5, dim))
        candidates.append(rng.uniform(-4, 4, dim))
        candidates.append(rng.uniform(-4, 4, dim))
        candidates.append(rng.uniform(-3, 3, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            if not time_ok():
                break
            c = np.clip(c, -5, 5)
            fc = f(c)
            if fc < best_f:
                best_f, best_x = fc, c

        # ---- Phase 2: Nelder-Mead local refinement from several starts ----
        # Build a set of local starts: best point + a few random + center
        local_starts = [best_x.copy()]
        local_starts.append(np.zeros(dim))
        for _ in range(3):
            local_starts.append(rng.uniform(-4, 4, dim))

        for ls in local_starts:
            if not time_ok():
                break
            ls = np.clip(ls, -5, 5)
            try:
                res = minimize(
                    f, ls, method="Nelder-Mead",
                    options={"maxiter": 300, "xatol": 1e-8, "fatol": 1e-10,
                             "adaptive": True},
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # ---- Phase 3: (1+1)-ES polish from the best point ----
        if time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(50):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
            if fx < best_f:
                best_f, best_x = fx, x

        return best_x, best_f
# EVOLVE-BLOCK-END
