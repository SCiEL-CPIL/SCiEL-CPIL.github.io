"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global-local optimizer: random restarts + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def wrapped_func(x):
            nonlocal best_f, best_x, calls_used
            calls_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def lbfgs_refine(x0, max_evals=50):
            """L-BFGS-B local refinement."""
            nonlocal best_x, best_f, calls_used
            try:
                result = minimize(wrapped_func, x0, method='L-BFGS-B',
                                 bounds=[(-5, 5)] * dim,
                                 options={'maxiter': max_evals, 'maxfun': max_evals})
            except Exception:
                pass

        # Phase 1: Global exploration with random restarts
        # Use more points for better coverage of 21 peaks
        n_restarts = min(20, budget // 10)
        start_points = []
        for _ in range(n_restarts):
            if calls_used >= budget - 20:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)
            start_points.append((fx, x))

        # Sort by function value and pick top candidates
        start_points.sort(key=lambda t: t[0])
        n_local = min(6, len(start_points))

        # Phase 2: Local refinement with L-BFGS-B on best candidates
        for i in range(n_local):
            if time.time() > deadline or calls_used >= budget - 10:
                break
            fx, x = start_points[i]
            remaining = budget - calls_used
            if remaining <= 15:
                break
            max_evals = max(20, remaining // (n_local - i + 1))
            lbfgs_refine(x, max_evals=max_evals)

        # Phase 3: If budget remains, do more refinement from best point
        if best_x is not None and time.time() < deadline and calls_used < budget - 10:
            remaining = budget - calls_used
            if remaining > 20:
                lbfgs_refine(best_x, max_evals=min(100, remaining))

        return best_x, best_f
# EVOLVE-BLOCK-END
