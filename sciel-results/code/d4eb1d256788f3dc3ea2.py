"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Nelder-Mead polish and a (1+1)-ES fallback."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        # Generate a set of diverse starting points
        # Include center, corners, and random points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(20):
            starts.append(rng.uniform(-5, 5, dim))
        # Some structured points
        for i in range(dim):
            for val in [-3, -1, 0, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate approximately
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Sort starts by some heuristic? Just try them in order, but maybe randomize
        rng.shuffle(starts)

        for s in starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            s = np.clip(s, -5, 5)

            # Try L-BFGS-B
            try:
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 50, 'gtol': 1e-6, 'ftol': 1e-12}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            # If we have budget left, try Nelder-Mead polish from the best point
            if best_x is not None and calls[0] < budget:
                try:
                    # Use a smaller maxiter for NM to save budget
                    nm_maxiter = min(50, max(10, (budget - calls[0]) // 2))
                    res_nm = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                    )
                    update_best(res_nm.x, res_nm.fun)
                except Exception:
                    pass

        # If we still have budget, do a (1+1)-ES from the best point
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                if sigma < 1e-6:
                    break

        if best_x is None:
            # Fallback: return a random point
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
