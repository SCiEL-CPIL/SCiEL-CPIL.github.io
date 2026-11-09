"""Optimized solver for BBOB f9 (rotated Rosenbrock) using multi-start L-BFGS-B with enhanced polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid L-BFGS-B and BFGS optimizer for rotated Rosenbrock with enhanced polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin
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
            return call_count[0] < budget and remaining_time() > 2.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Phase 1: Generate diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, 1.5))
        starts.append(np.full(dim, -1.0))
        starts.append(np.full(dim, 2.0))
        
        # Random points in the likely region [-4, 4]
        for _ in range(12):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Points near the valley structure
        for _ in range(4):
            p = rng.uniform(-2, 2, dim)
            p = 0.5 * p + 0.5 * np.ones(dim)
            starts.append(p)

        # Deduplicate and clip
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.linalg.norm(s - u) < 0.15 for u in unique_starts):
                unique_starts.append(s)

        # Phase 2: Screen with reduced budget to save calls for polish
        max_screening = min(12, len(unique_starts), max(10, budget - 150))
        screening_results = []
        for s in unique_starts[:max_screening]:
            if not has_budget():
                break
            f = counted_func(s)
            screening_results.append((f, s.copy()))
        
        screening_results.sort(key=lambda t: t[0])
        
        # Select top 4 for deep optimization (fewer starts, more polish per start)
        top_starts = [s for _, s in screening_results[:4]]
        
        current_best_f = best_f[0]
        current_best_x = best_x[0].copy() if best_x[0] is not None else np.zeros(dim)

        # Phase 3: L-BFGS-B from top starts with adaptive budget allocation
        for i, start in enumerate(top_starts):
            if not has_budget():
                break
            if remaining_time() < 15.0:
                break
            
            try:
                # Allocate more iterations to the best start
                if i == 0:
                    maxiter = 250
                    ftol = 1e-15
                    gtol = 1e-13
                elif i == 1:
                    maxiter = 150
                    ftol = 1e-14
                    gtol = 1e-12
                else:
                    maxiter = 80
                    ftol = 1e-12
                    gtol = 1e-10

                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter, 
                        'ftol': ftol, 
                        'gtol': gtol, 
                        'maxls': 60
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                continue

        # Phase 4: BFGS Polish - multiple passes for better convergence
        polish_passes = 3
        for pass_num in range(polish_passes):
            if not has_budget() or remaining_time() < 8.0:
                break
            try:
                if pass_num == 0:
                    maxiter_bfgs = 80
                    gtol_bfgs = 1e-12
                    ftol_bfgs = 1e-14
                elif pass_num == 1:
                    maxiter_bfgs = 50
                    gtol_bfgs = 1e-14
                    ftol_bfgs = 1e-16
                else:
                    maxiter_bfgs = 30
                    gtol_bfgs = 1e-16
                    ftol_bfgs = 1e-18

                res = minimize(
                    counted_func,
                    current_best_x,
                    method='BFGS',
                    options={
                        'maxiter': maxiter_bfgs, 
                        'gtol': gtol_bfgs, 
                        'ftol': ftol_bfgs,
                        'maxls': 60
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 5: Final L-BFGS-B Polish with extreme tightness
        if has_budget() and remaining_time() > 4.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'ftol': 1e-17, 
                        'gtol': 1e-15, 
                        'maxls': 80
                    }
                )
                if res.fun < current_best_f:
                    current_best_f = res.fun
                    current_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 6: Additional BFGS polish if budget remains
        if has_budget() and remaining_time() > 3.0:
            try:
                res = minimize(
                    counted_func,
                    current_best_x,
                    method='BFGS',
                    options={
                        'maxiter': 40, 
                        'gtol': 1e-15, 
                        'ftol': 1e-17,
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
