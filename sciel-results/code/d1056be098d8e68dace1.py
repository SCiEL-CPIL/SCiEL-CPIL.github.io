"""Optimizer for the BBOB-style f3 (Rastrigin) task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Coarse global grid search to locate the basin, then local (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.monotonic() + 285.0   # safety margin vs 300s
        dim = self.dim
        budget = self.budget
        calls = 0

        def call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: coarse global grid (10^5 = 100k evaluations) ----
        # Exploits the periodic structure of Rastrigin: a coarse lattice
        # quickly identifies the basin containing the shifted minimum.
        grid = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
        pts = np.array(np.meshgrid(*([grid] * dim)), dtype=float).reshape(-1, dim)

        best_x = None
        best_f = np.inf
        for p in pts:
            f = call(p)
            if f < best_f:
                best_f = f
                best_x = p
            if calls >= budget:
                break
            if time.monotonic() > deadline:
                break

        # ---- Phase 2: local (1+1)-ES refinement from the best basin ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.25
        rng = np.random.default_rng(0)
        while calls < budget:
            if time.monotonic() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = call(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.90
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END
