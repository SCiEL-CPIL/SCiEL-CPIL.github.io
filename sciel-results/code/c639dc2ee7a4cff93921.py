"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5, 5]^5.

Strategy:
  * Front-load a coarse global grid (2^5 = 32 points) to get a decent initial
    best-so-far value quickly.
  * Run multiple rounds of Nelder-Mead simplex from diverse starting points
    (including the current best and random points near the best).
  * Use a (1+1)-ES with adaptive step size for robust local refinement.
  * Final L-BFGS-B polish on the best point found.
  * Explicit wall-clock deadline guard with a safety margin.
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
        deadline = t0 + 280.0  # 330 s hard kill -> 50 s safety margin
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
            if calls >= budget:
                return f_best  # avoid exceeding budget
            calls += 1
            return func(x)

        def _update(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # ---- 1) coarse global grid: front-load early progress ----
        # 2^5 = 32 points, step 5.0. For Rastrigin with shift in [-4,4],
        # this gives a reasonable initial best.
        grid_levels = 2
        gvals = np.linspace(-5.0, 5.0, grid_levels)
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        for i4 in range(grid_levels):
                            if not _left():
                                break
                            x = np.array([gvals[i0], gvals[i1], gvals[i2],
                                          gvals[i3], gvals[i4]], dtype=float)
                            _update(x, _wrap(x))
                        if not _left():
                            break
                    if not _left():
                        break
                if not _left():
                    break
            if not _left():
                break
        if not _left():
            return x_best, f_best

        # ---- 2) Multi-start Nelder-Mead from diverse starting points ----
        # Use the best point and several random points near it.
        rng = np.random.RandomState(42)
        nm_starts = [x_best.copy()]
        for _ in range(10):
            perturbation = rng.randn(dim) * 1.5
            x_start = np.clip(x_best + perturbation, -5.0, 5.0)
            nm_starts.append(x_start)

        for x_start in nm_starts:
            if not _left() or calls >= budget:
                break
            try:
                res = _sp_min(_wrap, x_start, method="Nelder-Mead",
                              options={"maxiter": 100, "xatol": 1e-6, "fatol": 1e-6})
                if res.x is not None and np.isfinite(res.fun):
                    _update(res.x, float(res.fun))
            except Exception:
                pass

        if not _left() or calls >= budget:
            return x_best, f_best

        # ---- 3) (1+1)-ES with adaptive step size for robust local refinement ----
        sigma = 0.5
        x_current = x_best.copy()
        f_current = f_best
        success_count = 0
        total_count = 0

        while _left() and calls < budget:
            # Generate candidate
            direction = rng.randn(dim)
            x_candidate = np.clip(x_current + sigma * direction, -5.0, 5.0)
            f_candidate = _wrap(x_candidate)
            total_count += 1

            if f_candidate < f_current:
                x_current = x_candidate.copy()
                f_current = f_candidate
                _update(x_candidate, f_candidate)
                success_count += 1
                sigma = min(sigma * 1.25, 4.0)  # increase step on success
            else:
                sigma = max(sigma * 0.75, 0.01)  # decrease step on failure

            # Reset counters periodically
            if total_count % 20 == 0:
                # If we've had many failures, restart from best
                if success_count / total_count < 0.1:
                    x_current = x_best.copy()
                    f_current = f_best
                    sigma = 0.5
                    success_count = 0
                    total_count = 0

        if not _left() or calls >= budget:
            return x_best, f_best

        # ---- 4) Final polish with L-BFGS-B on the best point ----
        try:
            res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                          bounds=[(-5.0, 5.0)] * dim,
                          options={"maxiter": 50, "maxls": 20})
            if res.x is not None and np.isfinite(res.fun):
                _update(res.x, float(res.fun))
        except Exception:
            pass

        return x_best, f_best
