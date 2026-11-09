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

        # Phase 1: Lightweight Deterministic Differential Evolution for global search
        # Reduced population and iterations to leave budget for multiple local polish runs.
        # 10 * 5 (init) + 15 * 10 (iter) = 200 calls approx.
        de_popsize = 10
        de_maxiter = 15
        
        de_candidates = []
        try:
            de_result = differential_evolution(
                counted_func,
                bounds=bounds,
                popsize=de_popsize,
                maxiter=de_maxiter,
                seed=42,
                tol=1e-8, # Looser tol for DE, we will polish later
                init='sobol',
                polish=False,
                update_interval=1,
                callback=None
            )
            if de_result.success and de_result.fun < best_f[0]:
                best_f[0] = de_result.fun
                best_x[0] = np.clip(de_result.x, -5, 5)
                de_candidates.append(de_result.x)
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
                    de_candidates.append(anchor)

        if best_x[0] is None:
            return np.zeros(dim), 0.0

        current_best_x = best_x[0].copy()
        current_best_f = best_f[0]
        
        # Add the best found so far to candidates
        if current_best_x not in de_candidates:
            de_candidates.append(current_best_x)

        # Phase 2: Multi-start L-BFGS-B Polish
        # L-BFGS-B is efficient for Rosenbrock. We try to polish the best candidate
        # and potentially a few others if budget allows, to escape local minima
        # or sub-optimal valleys.
        
        # First, polish the best candidate from DE
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

        # If we have significant budget left, try polishing other candidates
        # We limit the number of restarts to avoid wasting budget on bad starts.
        if has_budget() and remaining_time() > 10.0:
            # Try up to 3 other candidates from the DE list (if any) or random points
            # Since DE is deterministic, we can just use the ones we stored if we had them,
            # but we only stored the best. Let's generate a few random points near the best
            # or just random points in the box to see if we can find a better basin.
            
            # Strategy: Random restarts near the current best (local search) 
            # and a few global random points.
            n_restarts = 3
            for i in range(n_restarts):
                if not has_budget() or remaining_time() < 5.0:
                    break
                
                # 50% chance: perturb current best (local refinement)
                # 50% chance: random point in box (global exploration)
                if i % 2 == 0:
                    # Perturb current best
                    sigma = 0.5
                    x_start = current_best_x + rng.normal(0, sigma, dim)
                    x_start = np.clip(x_start, -5, 5)
                else:
                    # Random point
                    x_start = rng.uniform(-5, 5, dim)
                
                # Evaluate start point to ensure it's valid and count it
                f_start = counted_func(x_start)
                
                # Polish this start point
                if has_budget():
                    try:
                        res = minimize(
                            counted_func,
                            x_start,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 100, # Lower maxiter for restarts
                                'ftol': 1e-12,
                                'gtol': 1e-10,
                                'maxls': 30
                            }
                        )
                        if res.fun < current_best_f:
                            current_best_f = res.fun
                            current_best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Phase 3: Final ultra-precise L-BFGS-B pass on the best found
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
