# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B with polish for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf
        deadline = time.time() + 280  # safety margin before 330s kill

        def F(x):
            calls[0] += 1
            if calls[0] > budget:
                return 1e10
            val = func(np.asarray(x, dtype=float))
            return val

        bounds = [(-5.0, 5.0)] * d
        rng = np.random.default_rng(42)

        # Phase 1: Multi-restart to find a good basin
        # We use 4 restarts. The first is at origin, others are random.
        # We use moderate tolerances to save budget for polish.
        n_restarts = 4
        
        # Reserve budget for polish
        polish_budget = 50
        available_for_search = budget - polish_budget

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls[0] >= available_for_search:
                break

            # Generate diverse starting points
            if r == 0:
                x0 = np.zeros(d)
            elif r == 1:
                x0 = rng.uniform(-3.0, 3.0, d)
            elif r == 2:
                x0 = rng.uniform(-4.0, 4.0, d)
            else:
                x0 = rng.uniform(-5.0, 5.0, d)

            # Clip to bounds
            x0 = np.clip(x0, -5.0, 5.0)

            # Evaluate starting point
            f0 = F(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if calls[0] >= available_for_search:
                break

            # L-BFGS-B with moderate tolerance
            try:
                res = minimize(
                    F, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(1, available_for_search - calls[0]),
                        'gtol': 1e-8,  # Moderate tolerance for search phase
                        'ftol': 1e-10,
                    }
                )
            except Exception:
                break

            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5.0, 5.0)

        # Phase 2: Polish
        # If we have a candidate and budget left, do a high-precision run
        if best_x is not None and calls[0] < budget - 5 and time.time() < deadline:
            try:
                res_polish = minimize(
                    F, best_x, method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': max(1, budget - calls[0]),
                        'gtol': 1e-12,  # Tight tolerance for polish
                        'ftol': 1e-14,
                    }
                )
                if res_polish.fun < best_f:
                    best_f = res_polish.fun
                    best_x = np.clip(res_polish.x, -5.0, 5.0)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(d)
            if calls[0] < budget:
                best_f = F(best_x)
            else:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
