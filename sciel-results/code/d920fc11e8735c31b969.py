import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin well under the 330s wall-clock kill
        dim = self.dim
        budget = self.budget

        calls = [0]
        best_x = [np.zeros(dim)]
        best_f = [np.inf]

        def f_wrap(x):
            # x may be a list (scipy) or ndarray
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = func(x)
            calls[0] += 1
            if fx < best_f[0]:
                best_f[0] = float(fx)
                best_x[0] = x.copy()
            return float(fx)

        def time_ok(margin=2.0):
            return (time.time() + margin) < deadline and calls[0] < budget

        rng = np.random.default_rng(42)

        # ---- Phase 1: cheap global pass to locate a good basin quickly ----
        # Center + structured corners (scaled to the likely minimizer region) + randoms.
        candidates = [np.zeros(dim)]
        # 4 scaled corners of [-4,4]^5 (minimizer lies in [-4,4]^5)
        for _ in range(4):
            candidates.append(rng.choice([-4.0, 4.0], size=dim))
        # random points in [-4.5, 4.5]^5
        for _ in range(16):
            candidates.append(rng.uniform(-4.5, 4.5, size=dim))

        for x in candidates:
            if not time_ok():
                break
            f_wrap(x)

        # rank the sampled points (we have their values recorded implicitly via best,
        # but we need per-point values; recompute-free approach: track them)
        # We'll re-collect the top starting points by re-evaluating is wasteful,
        # so instead keep a small list of (value, x) as we go.
        # (best_x/best_f already hold the global best; for a 2nd start we use a
        #  slightly perturbed best, which is cheap and robust.)

        # ---- Phase 2: CMA-ES from the best point (primary convergence) ----
        x0 = best_x[0].copy()
        # small sigma: near-deterministic start, fast early progress
        sigma0 = 0.5
        # CMA-ES needs ~ a few hundred evals to converge in 5-D; cap by budget.
        maxfev = min(budget - calls[0] - 1, 260)
        if maxfev > 1 and time_ok():
            try:
                res = minimize(
                    f_wrap,
                    x0,
                    method="CovarianceMatrixAdaptation",
                    options={
                        "maxfev": maxfev,
                        "sigma0": sigma0,
                        "xatol": 1e-9,
                        "fatol": 1e-12,
                        "tolx": 1e-12,
                        "verbose": 0,
                    },
                )
            except Exception:
                pass

        # ---- Phase 3: CMA-ES restart from a perturbed best (polish / second basin) ----
        if time_ok():
            x1 = best_x[0] + rng.normal(0.0, 0.3, size=dim)
            x1 = np.clip(x1, -5.0, 5.0)
            maxfev2 = min(budget - calls[0] - 1, 180)
            if maxfev2 > 1:
                try:
                    res2 = minimize(
                        f_wrap,
                        x1,
                        method="CovarianceMatrixAdaptation",
                        options={
                            "maxfev": maxfev2,
                            "sigma0": 0.4,
                            "xatol": 1e-9,
                            "fatol": 1e-12,
                            "tolx": 1e-12,
                            "verbose": 0,
                        },
                    )
                except Exception:
                    pass

        return best_x[0], best_f[0]
