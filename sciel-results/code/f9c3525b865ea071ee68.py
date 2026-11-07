"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with global exploration via random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                val = func(np.asarray(x, dtype=np.float64))
                val = float(val)
                if np.isfinite(val):
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        # Phase 1: Random exploration to find promising regions
        # Use a fraction of budget for initial sampling
        n_random = min(20, budget // 10)
        rng = np.random.default_rng(42)

        candidates = []
        for i in range(n_random):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization from best candidates
        # Use L-BFGS-B for each promising starting point
        n_local = min(10, len(candidates), max(1, (budget - calls_made) // 50))

        for i in range(n_local):
            if time.time() > deadline or calls_made >= budget - 10:
                break

            fx, x0 = candidates[i]

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': max(10, budget - calls_made - 50),
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )

                if result.success or result.fun < best_f:
                    if np.isfinite(result.fun):
                        if result.fun < best_f:
                            best_f = float(result.fun)
                            best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Check time
            if time.time() > deadline - 50:
                break

        # Phase 3: If we have remaining budget, do more random restarts
        # and short local optimizations
        while calls_made < budget - 5 and time.time() < deadline - 10:
            # Random restart
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Quick local refinement
            if calls_made < budget - 20 and time.time() < deadline - 20:
                try:
                    result = minimize(
                        safe_func,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 50,
                            'maxfun': max(5, budget - calls_made - 10),
                            'ftol': 1e-10,
                            'gtol': 1e-6,
                        }
                    )
                    if result.success or (np.isfinite(result.fun) and result.fun < best_f):
                        if np.isfinite(result.fun):
                            best_f = float(result.fun)
                            best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 4: Final polish — tighten the best solution found
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 500,
                        'maxfun': max(5, budget - calls_made - 2),
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                    }
                )
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
