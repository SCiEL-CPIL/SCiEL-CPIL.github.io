"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import sobol
from scipy.optimize import minimize


class Optimizer:
    """Two-stage global-to-local search: Sobol sampling + multi-start Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()

        def check_deadline():
            return time.time() > deadline or calls_used >= total_budget

        # ---------- Stage 1: Sobol quasi-random global sampling ----------
        # Draw a Sobol sequence of dim dimensions; map [0,1)^dim -> [-5,5]^dim.
        # 400 points for diverse coverage.
        n_sobol = min(400, max(50, total_budget - 100))
        n_sobol = max(n_sobol, 50)
        sobol_sampler = sobol(dim, scramble=True, seed=12345)
        points = sobol_sampler.next(n_sobol)  # shape (n_sobol, dim), in [0,1)
        points = points * 10.0 - 5.0  # map to [-5,5]

        candidates = []  # list of (f, x)
        for i in range(n_sobol):
            if check_deadline():
                break
            x = points[i]
            fx = func(x)
            track_call(x, fx)
            candidates.append((float(fx), x.copy()))

        if not candidates:
            # Fallback: single evaluation at origin
            x0 = np.zeros(dim)
            fx0 = func(x0)
            track_call(x0, fx0)
            candidates.append((float(fx0), x0.copy()))

        # Keep the top-5 diverse candidates (by value).
        candidates.sort(key=lambda t: t[0])
        top_k = 5
        top = candidates[:top_k]

        # ---------- Stage 2: multi-start Nelder-Mead local refinement ----------
        # Compute pairwise distances among top candidates to scale initial simplex.
        xs = np.array([t[1] for t in top])
        if len(xs) >= 2:
            # Mean pairwise distance
            from scipy.spatial.distance import pdist
            dists = pdist(xs)
            mean_dist = float(np.mean(dists)) if len(dists) > 0 else 1.0
        else:
            mean_dist = 1.0
        # Scale initial simplex size: a fraction of the mean distance, bounded.
        simplex_scale = max(0.3, min(3.0, 0.5 * mean_dist))

        remaining_budget = total_budget - calls_used
        # Reserve a small tail for a final polish if needed.
        per_start = max(8, remaining_budget // max(len(top), 1))

        for idx, (f0, x0) in enumerate(top):
            if check_deadline():
                break
            if calls_used >= total_budget:
                break

            # Local budget for this start (capped by remaining).
            local_budget = min(per_start, total_budget - calls_used)
            if local_budget < 6:
                break

            x_start = x0.copy()

            # Objective wrapper that tracks calls and enforces the deadline.
            def objective(z, _lb=local_budget):
                # z is a 1D array; clip to box (func also clips, but be safe).
                zc = np.clip(z, -5.0, 5.0)
                val = func(zc)
                track_call(zc, val)
                return val

            # Nelder-Mead with adaptive initial simplex size.
            # We build a custom initial simplex scaled by simplex_scale.
            try:
                res = minimize(
                    objective,
                    x_start,
                    method="Nelder-Mead",
                    options={
                        "maxiter": local_budget,
                        "maxfev": local_budget,
                        "initial_simplex": None,  # default; we rely on scale via x_start spread
                        "xatol": 1e-8,
                        "fatol": 1e-8,
                        "adaptive": True,
                    },
                )
            except Exception:
                # If Nelder-Mead fails, fall back to a tiny (1+1) local step.
                try:
                    rng = np.random.default_rng(999 + idx)
                    x = x_start.copy()
                    fx = objective(x)
                    sigma = simplex_scale * 0.5
                    for _ in range(local_budget):
                        if check_deadline():
                            break
                        step = sigma * rng.standard_normal(dim)
                        y = np.clip(x + step, -5, 5)
                        fy = objective(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma *= 1.03
                        else:
                            sigma *= 0.96
                        sigma = float(np.clip(sigma, 1e-9, 5.0))
                except Exception:
                    pass

        # ---------- Stage 3: tiny final polish from the global best ----------
        remaining_budget = total_budget - calls_used
        if best_x is not None and remaining_budget > 3:
            rng = np.random.default_rng(7)
            x = best_x.copy()
            fx = func(x)
            track_call(x, fx)
            sigma = 0.02
            for _ in range(remaining_budget):
                if check_deadline():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                track_call(y, fy)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.02
                else:
                    sigma *= 0.97
                sigma = float(np.clip(sigma, 1e-9, 0.2))

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
