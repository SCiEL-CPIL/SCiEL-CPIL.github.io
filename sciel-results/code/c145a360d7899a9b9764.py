"""Rewritten optimizer for f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start global search + adaptive (1+1)-ES + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        calls = [0]

        def count_call(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # ---- Phase 1: coarse global exploration on a grid + random points ----
        # f21 has 101 peaks; a coarse grid over [-5,5]^5 catches many basins.
        levels = 4
        grid = np.array(np.meshgrid(*([np.linspace(-5, 5, levels)] * dim)),
                        dtype=float).reshape(-1, dim)
        # Add random jitter so we don't sit exactly on grid nodes
        grid = grid + rng.uniform(-0.5, 0.5, grid.shape)
        grid = np.clip(grid, -5, 5)
        # Add a handful of pure-random points
        grid = np.vstack([grid, rng.uniform(-5, 5, (30, dim))])

        best_x = None
        best_f = np.inf
        # Evaluate grid points while budget/time allows
        for pt in grid:
            if calls[0] >= budget or time.time() > deadline:
                break
            f = count_call(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # ---- Phase 2: adaptive (1+1)-ES from the best point (and a few others) ----
        # Sort top candidates for restarts
        cand = []
        if best_x is not None:
            cand.append((best_f, best_x))

        def es_refine(x0, f0, remaining):
            x = x0.copy()
            fx = f0
            sigma = 1.0
            no_improve = 0
            while calls[0] < budget and time.time() < deadline:
                if remaining <= 0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = count_call(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.35
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    # Escape local minima: if stuck, reset sigma and take a
                    # larger exploratory step (helps jump between f21 peaks)
                    if no_improve >= 40:
                        sigma = 2.0
                        no_improve = 0
                        jump = x + 2.0 * rng.standard_normal(dim)
                        jump = np.clip(jump, -5, 5)
                        fj = count_call(jump)
                        remaining -= 1
                        if fj < fx:
                            x, fx = jump, fj
                            sigma = 1.0
            return x, fx

        # Refine the global best with a generous slice of the budget
        es_budget = max(0, budget - calls[0] - 200)  # reserve for CMA-ES
        if es_budget > 0:
            best_x, best_f = es_refine(best_x, best_f, es_budget)

        # ---- Phase 3: CMA-ES fine convergence in the final basin ----
        if calls[0] < budget - 5 and time.time() < deadline:
            # Cap CMA-ES maxiter so we never blow the call budget
            max_iter = max(1, (budget - calls[0]) // (dim + 1))
            try:
                res = minimize(
                    count_call,
                    best_x,
                    method="Nelder-Mead",
                    options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3},
                )
                if res.fun < best_f:
                    best_x, best_f = res.x, res.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
