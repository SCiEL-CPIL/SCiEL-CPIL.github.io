"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5, 5]^5.

Strategy:
  * Multi-start (1+1)-ES with adaptive step size.
  * Front-load a coarse global search (Latin hypercube + random) to quickly
    find a good region.
  * Then use (1+1)-ES with restarts to polish.
  * L-BFGS-B for final polish.
  * Explicit wall-clock deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize as _sp_min


class Optimizer:
    """Continuous black-box minimizer for f15 (Rastrigin rotated) on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget

        def _left():
            return time.time() < deadline

        # ---- state ----
        x_best = np.zeros(dim)
        f_best = np.inf
        calls = 0

        def _wrap(x):
            nonlocal calls
            calls += 1
            return func(x)

        def _update(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # ---- 1) Latin hypercube sampling for early progress ----
        # Sample ~80 points spread across the box
        rng = np.random.RandomState(42)
        n_lhs = 80
        if _left():
            # Latin hypercube: for each dimension, divide into n_lhs equal intervals,
            # pick one random point in each, then permute
            lhs_points = np.zeros((n_lhs, dim))
            for d in range(dim):
                intervals = np.linspace(-5.0, 5.0, n_lhs + 1)
                for i in range(n_lhs):
                    lhs_points[i, d] = rng.uniform(intervals[i], intervals[i + 1])
            # Permute each dimension independently
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                lhs_points[:, d] = lhs_points[perm, d]

            for i in range(n_lhs):
                if not _left():
                    break
                x = lhs_points[i].copy()
                _update(x, _wrap(x))

        # ---- 2) Additional random samples to fill gaps ----
        n_random = 40
        for _ in range(n_random):
            if not _left():
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            _update(x, _wrap(x))

        # ---- 3) Multi-start (1+1)-ES ----
        # Use the best point found so far as the main starting point,
        # plus a few other good points as secondary starts.

        def run_11es(x0, sigma0, max_evals, sigma_min=1e-4, sigma_max=10.0):
            """Run (1+1)-ES from x0 with initial step sigma0."""
            nonlocal x_best, f_best
            x = x0.copy()
            f = _wrap(x)
            _update(x, f)
            sigma = sigma0
            n_evals = 1
            no_improve = 0

            while n_evals < max_evals and _left():
                # Generate candidate
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                x_new = x + sigma * direction
                # Clip to bounds
                x_new = np.clip(x_new, -5.0, 5.0)

                f_new = _wrap(x_new)
                n_evals += 1

                if f_new < f:
                    # Accept
                    x = x_new
                    f = f_new
                    _update(x, f)
                    no_improve = 0
                    # Increase sigma slightly
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    no_improve += 1
                    # Decrease sigma
                    sigma = max(sigma * 0.8, sigma_min)

                    if no_improve > 30:
                        break

            return x, f

        # Primary (1+1)-ES from best point
        if _left():
            x0 = x_best.copy()
            sigma0 = 1.5
            max_evals_primary = 150
            run_11es(x0, sigma0, max_evals_primary)

        # Secondary starts from other good points
        # Try a few restarts with different initial sigmas
        if _left():
            for sigma_try in [0.5, 2.0, 0.2]:
                if not _left():
                    break
                x0 = x_best.copy()
                max_evals_sec = 80
                run_11es(x0, sigma_try, max_evals_sec)

        # ---- 4) L-BFGS-B polish ----
        if _left():
            try:
                res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                              bounds=[(-5.0, 5.0)] * dim,
                              options={"maxiter": 50, "maxls": 20})
                if res.x is not None:
                    _update(res.x, float(res.fun))
            except Exception:
                pass

        # ---- 5) Final (1+1)-ES with small step for fine-tuning ----
        if _left():
            x0 = x_best.copy()
            sigma0 = 0.1
            run_11es(x0, sigma0, 50, sigma_min=1e-5, sigma_max=0.5)

        # ---- 6) Another L-BFGS-B polish ----
        if _left():
            try:
                res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                              bounds=[(-5.0, 5.0)] * dim,
                              options={"maxiter": 30, "maxls": 15})
                if res.x is not None:
                    _update(res.x, float(res.fun))
            except Exception:
                pass

        return x_best, f_best
