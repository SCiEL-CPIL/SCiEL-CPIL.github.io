"""Optimized solver for BBOB f9 (rotated Rosenbrock) using DE global search + L-BFGS-B + BFGS polish."""
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
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
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

        # Phase 1: Deterministic Differential Evolution for global search
        # DE provides population-based diversity to handle unknown rotation/shift
        # without relying on random restarts. Use a moderate population to stay
        # within budget while covering the search space.
        de_popsize = 12  # 12 * 5 = 60 initial evaluations
        de_maxiter = 25  # 25 * 12 = 300 additional evaluations (total ~360, saving ~140 for polish)
        
        try:
            de_result = differential_evolution(
                counted_func,
                bounds=bounds,
                popsize=de_popsize,
                maxiter=de_maxiter,
                seed=42,
                tol=1e-12,
                init='sobol',
                polish=False,  # We do our own high-precision polish
                update_interval=1,
                callback=None
            )
            if de_result.success and de_result.fun < best_f[0]:
                best_f[0] = de_result.fun
                best_x[0] = np.clip(de_result.x, -5, 5)
        except Exception:
            pass

        # If DE didn't find anything better, fall back to standard anchors
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
                        'maxiter': 300,
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
                        'maxiter': 100,
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

        # Phase 4: Perturbed restart to escape potential local minima
        # Small perturbations around the best point can help if we're in a shallow local min
        if has_budget() and remaining_time() > 5.0:
            rng2 = np.random.default_rng(123)
            for _ in range(3):
                if not has_budget() or remaining_time() < 3.0:
                    break
                perturbation = rng2.normal(0, 0.1, size=dim)
                start_x = np.clip(current_best_x + perturbation, -5.0, 5.0)
                try:
                    res = minimize(
                        counted_func,
                        start_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'ftol': 1e-14,
                            'gtol': 1e-13,
                            'maxls': 30
                        }
                    )
                    if res.fun < current_best_f:
                        current_best_f = res.fun
                        current_best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 5: Final ultra-precise L-BFGS-B pass
        if has_budget() and remaining_time() > 2.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-16,
                        'gtol': 1e-15,
                        'maxls': 50
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
