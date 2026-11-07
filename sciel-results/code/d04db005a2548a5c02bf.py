"""Direct-approach optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Coarse global grid (3^5 = 243 points) to locate the best basin
           early, which strongly improves the AOCC area-under-curve score.
  Phase 2: L-BFGS-B refinement on the top grid candidates for fast
           local convergence to the minimum.
  Phase 3: If budget remains, a second, finer local search around the
           current best using Nelder-Mead or additional L-BFGS-B restarts
           with jitter to escape shallow local minima.
  A wall-clock deadline guard (325 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-then-L-BFGS-B optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 325.0  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = np.asarray(x, dtype=float).copy()

        # ---------------- Phase 1: coarse global grid ----------------
        # 3^5 = 243 points over [-5, 5]^5 (spacing ~2.5). This quickly locates
        # the best of the 21 basins, giving strong early AOCC progress.
        grid_n = 3
        grid_vals = np.linspace(-5.0, 5.0, grid_n)
        grid_points = list(np.meshgrid(*([grid_vals] * n), indexing="ij"))
        grid_points = np.array([p for p in zip(*grid_points)])
        grid_points = np.asarray(grid_points).reshape(-1, n)

        grid_results = []
        for gp in grid_points:
            if time.time() >= deadline:
                break
            gp = np.asarray(gp, dtype=float)
            try:
                fg = fwrap(gp)
                grid_results.append((fg, gp.copy()))
                update_best(gp, fg)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0.copy()
            except Exception:
                return (None, np.inf)

        # Sort grid results to get top candidates for local optimization
        grid_results.sort(key=lambda x: x[0])

        # ---------------- Phase 2: L-BFGS-B refinement ----------------
        candidates = []
        for i in range(min(8, len(grid_results))):
            candidates.append(grid_results[i][1])

        bounds = [(-5.0, 5.0) for _ in range(n)]

        for start_x in candidates:
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 5:
                break

            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'maxfun': 100})
                if res.fun < best["f"]:
                    update_best(res.x, res.fun)
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # ---------------- Phase 3: Fine local search / jitter restarts ----------------
        # If we still have budget, do a few small jittered L-BFGS-B runs
        # around the current best to escape potential shallow local minima.
        if best["x"] is not None and calls["count"] < budget - 10 and time.time() < deadline:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if time.time() >= deadline:
                    break
                if calls["count"] >= budget - 5:
                    break
                jitter = rng.normal(0, 0.1, size=n)
                x_start = best["x"] + jitter
                x_start = np.clip(x_start, -5.0, 5.0)
                try:
                    res = minimize(fwrap, x_start, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 50, 'maxfun': 50})
                    if res.fun < best["f"]:
                        update_best(res.x, res.fun)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
