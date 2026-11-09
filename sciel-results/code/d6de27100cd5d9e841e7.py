"""Optimized solver for BBOB f9 (rotated Rosenbrock) using CMA-ES global search + L-BFGS-B + BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """DE global search + high-precision L-BFGS-B/BFGS polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            call_count[0] += 1
            x_arr = np.asarray(x, dtype=float)
            if x_arr.shape != (dim,):
                x_arr = np.resize(x_arr, (dim,))
            x_clipped = np.clip(x_arr, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 3.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution global search
        # DE is robust for global search and handles the rotated valley structure
        # reasonably well with a sufficient population size.
        
        de_bounds = [(-5.0, 5.0)] * dim
        
        # We want to use about 300 function evaluations for global search.
        # popsize=15, maxiter=20 -> ~300 evaluations.
        try:
            res_de = differential_evolution(
                counted_func,
                de_bounds,
                seed=42,
                maxiter=20,
                popsize=15,
                tol=1e-8,
                init='latinhypercube',
                polish=False,  # We do our own polish
                update_interval=1,
                callback=None
            )
            if res_de.fun < best_f[0]:
                best_f[0] = res_de.fun
                best_x[0] = np.clip(res_de.x, -5.0, 5.0)
        except Exception:
            pass

        # If CMA-ES didn't find anything better, fall back to standard anchors
        if best_x[0] is None:
            # Try standard Rosenbrock minimum (1,1,...,1) and origin
            for anchor in [np.ones(dim), np.zeros(dim)]:
                if has_budget():
                    f = counted_func(anchor)
                    if f < best_f[0]:
                        best_f[0] = f
                        best_x[0] = anchor.copy()

        if best_x[0] is None:
            return np.zeros(dim), 0.0

        current_best_x = best_x[0].copy()
        current_best_f = best_f[0]

        # Phase 2: High-precision L-BFGS-B polish
        # L-BFGS-B handles box constraints and converges rapidly near the minimum.
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-15,
                        'gtol': 1e-14,
                        'maxls': 50
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: BFGS polish for quadratic convergence near minimum
        # BFGS uses dense Hessian approximation, which converges faster near the optimum.
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='BFGS',
                    options={
                        'maxiter': 50,
                        'gtol': 1e-12,
                        'ftol': 1e-15,
                        'maxls': 50
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final ultra-precise L-BFGS-B pass
        if has_budget() and remaining_time() > 2.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-16,
                        'gtol': 1e-15,
                        'maxls': 30
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if it's better
        if best_x[0] is not None and best_f[0] < current_best_f:
            current_best_f = best_f[0]
            current_best_x = best_x[0]

        if current_best_x is None:
            return np.zeros(dim), 0.0

        return current_best_x, current_best_f
