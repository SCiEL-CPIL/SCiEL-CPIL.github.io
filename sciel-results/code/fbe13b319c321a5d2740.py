"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with global exploration and multi-scale refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=np.float64)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                val = func(x_arr)
                val = float(val)
                if np.isfinite(val):
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        def check_deadline():
            return time.time() < deadline and calls_made < budget

        rng = np.random.default_rng(42)

        # Phase 1: Random exploration to find promising regions
        # Use a fraction of budget for initial sampling
        n_random = min(25, max(10, budget // 15))
        candidates = []

        for i in range(n_random):
            if not check_deadline():
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
        n_local = min(8, len(candidates), max(1, (budget - calls_made) // 60))

        for i in range(n_local):
            if not check_deadline() or calls_made >= budget - 30:
                break

            fx, x0 = candidates[i]

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 300,
                        'maxfun': max(15, budget - calls_made - 40),
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )

                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Check time
            if time.time() > deadline - 60:
                break

        # Phase 3: Multi-scale refinement around the best point
        if best_x is not None and check_deadline():
            # Try refinement with decreasing perturbation scales
            scales = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            for scale in scales:
                if not check_deadline() or calls_made >= budget - 20:
                    break

                # Generate a few perturbed points around best_x
                n_perturb = min(3, max(1, (budget - calls_made - 10) // 20))
                for _ in range(n_perturb):
                    if not check_deadline():
                        break

                    perturbation = rng.normal(0, scale, dim)
                    x_pert = np.clip(best_x + perturbation, -5, 5)
                    fx_pert = safe_func(x_pert)

                    if fx_pert < best_f:
                        best_f = fx_pert
                        best_x = x_pert.copy()

                    # Quick local refinement from perturbed point
                    if check_deadline() and calls_made < budget - 15:
                        try:
                            result = minimize(
                                safe_func,
                                x_pert,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={
                                    'maxiter': 100,
                                    'maxfun': max(8, budget - calls_made - 10),
                                    'ftol': 1e-13,
                                    'gtol': 1e-9,
                                }
                            )
                            if np.isfinite(result.fun) and result.fun < best_f:
                                best_f = float(result.fun)
                                best_x = np.clip(result.x, -5, 5).copy()
                        except Exception:
                            pass

        # Phase 4: If we have remaining budget, do more random restarts
        # and short local optimizations
        while check_deadline() and calls_made < budget - 10:
            # Random restart
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Quick local refinement
            if check_deadline() and calls_made < budget - 15:
                try:
                    result = minimize(
                        safe_func,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 80,
                            'maxfun': max(10, budget - calls_made - 8),
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                        }
                    )
                    if np.isfinite(result.fun) and result.fun < best_f:
                        best_f = float(result.fun)
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 5: Final polish — tighten the best solution found
        if best_x is not None and check_deadline() and calls_made < budget - 5:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 500,
                        'maxfun': max(10, budget - calls_made - 3),
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                    }
                )
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

            # Phase 5b: Ultra-high-precision polish with tighter tolerances
            if best_x is not None and check_deadline() and calls_made < budget - 2:
                try:
                    result2 = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 1000,
                            'maxfun': max(5, budget - calls_made),
                            'ftol': 1e-18,
                            'gtol': 1e-16,
                        }
                    )
                    if np.isfinite(result2.fun) and result2.fun < best_f:
                        best_f = float(result2.fun)
                        best_x = np.clip(result2.x, -5, 5).copy()
                except Exception:
                    pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f


# EVOLVE-BLOCK-END
