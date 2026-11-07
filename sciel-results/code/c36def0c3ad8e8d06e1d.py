"""Optimizer using L-BFGS-B with multi-start for fast convergence."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Quick coarse sampling to find a good starting region
        # We spend very few calls here to get a decent initial guess quickly.
        n_coarse = 10
        rng = np.random.default_rng(42)

        for i in range(n_coarse):
            if not time_ok() or call_count[0] >= budget - 20:
                break
            
            # Structured points first, then random
            if i < dim:
                x0 = np.zeros(dim)
                x0[i] = 1.0
            elif i == dim:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-3, 3, dim)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Intensive local refinement using L-BFGS-B
        # L-BFGS-B is much faster than Nelder-Mead for smooth functions in low dimensions.
        # We perform multiple starts from the best point and nearby random points.
        
        # First, refine the current best
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Then, try a few more starts around the best point to escape local minima
        # if the first one got stuck, or just to find a better basin.
        n_extra_starts = 5
        scales = [0.5, 1.0, 2.0, 0.2, 1.5]
        
        for i in range(n_extra_starts):
            if not time_ok() or call_count[0] >= budget - 10:
                break
            
            # Generate a new start point near the current best
            # We use a fixed scale to ensure we explore the neighborhood
            scale = scales[i]
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5, 5)
            
            # Check if this point is already good enough or if we should try to optimize from it
            # We just run L-BFGS-B from here.
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final high-precision refinement
        if time_ok() and best_x is not None and call_count[0] < budget - 5:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
