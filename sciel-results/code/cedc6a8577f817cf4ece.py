"""Rewritten optimizer: multi-start Powell with LHS initialization, deadline-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell's method with Latin Hypercube initialization for Lunacek bi-Rastrigin f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: Latin Hypercube Sampling for diverse initial points
        n_lhs = 25
        # Stratified sampling in each dimension
        lhs_points = np.zeros((n_lhs, dim))
        for j in range(dim):
            # Generate stratified samples: one per bin
            bins = np.linspace(-5, 5, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, j] = rng.uniform(bins[i], bins[i + 1])
        # Shuffle columns independently to break correlation
        for j in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, j] = lhs_points[perm, j]

        # Evaluate all LHS points
        lhs_values = []
        for i in range(n_lhs):
            if calls[0] >= budget or time.time() > self._deadline:
                break
            xi = lhs_points[i]
            fi = f(xi)
            lhs_values.append((fi, xi.copy()))
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Sort by function value
        lhs_values.sort(key=lambda x: x[0])

        # Phase 2: Powell's method from top candidates
        n_powell = min(8, len(lhs_values))
        # Allocate remaining budget among Powell runs
        remaining = budget - calls[0]
        per_powell = max(20, remaining // (n_powell + 2))  # +2 for final polish

        for idx in range(n_powell):
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fi, xi = lhs_values[idx]
            try:
                res = minimize(
                    f,
                    xi,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfev': per_powell, 'xtol': 1e-8, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final polish with L-BFGS-B on the best point
        if calls[0] < budget and best_x is not None and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': 30}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
