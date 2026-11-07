"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            try:
                val = func(x)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Phase 1: Global exploration with a coarse grid + random jitter
        # The Discus function has a narrow valley. We need good starting points.
        # 10 starting points is a reasonable balance.
        n_starts = 10
        rng = np.random.default_rng(42)
        
        # Create a grid of starting points
        # Using a stratified sampling approach to cover the space
        # 10 points in 5D: we can use a Sobol sequence or just random.
        # Random is simple and effective with multi-start.
        # Let's use a mix: center, corners, and random.
        start_points = []
        
        # Center
        start_points.append(np.zeros(d))
        
        # Random points
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-5, 5, d))
            
        # Evaluate all start points
        best_x = start_points[0].copy()
        best_f = wrapped_func(best_x)
        
        start_fits = [best_f]
        for x in start_points[1:]:
            if not time_ok() or calls >= budget:
                break
            fx = wrapped_func(x)
            start_fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort start points by fitness
        sorted_indices = np.argsort(start_fits)
        sorted_starts = [start_points[i] for i in sorted_indices]
        
        # Phase 2: Local refinement with L-BFGS-B
        # Allocate remaining budget among the best starts
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f
        
        # We have 10 starts. Let's allocate budget to the top 5.
        # L-BFGS-B typically converges in < 100 iterations for 5D smooth functions.
        # Each iteration can use multiple function calls (line search).
        # Setting maxiter to a high value and letting it stop on time/budget.
        
        # To ensure we use the budget effectively, we can run L-BFGS-B on the top starts.
        # If a start is far away, it might waste budget.
        # We'll run it on the top 5 starts, with a max_iter constraint based on remaining budget.
        
        n_refine = 5
        per_start_max_iter = max(1, (remaining_budget - 10) // n_refine)
        
        for i in range(n_refine):
            if not time_ok() or calls >= budget:
                break
            
            x0 = sorted_starts[i]
            f0 = start_fits[i]
            
            # Skip if this start is significantly worse than the current best
            # to avoid wasting budget on poor local minima
            if f0 > best_f + 50: 
                continue

            # Run L-BFGS-B
            # options: maxiter is the max number of iterations, not function calls.
            # But we have a global call counter.
            # We can use a callback to check time/budget, but scipy's minimize
            # doesn't easily expose early stopping via callback for budget.
            # However, if wrapped_func returns 1e10, it might stop or behave weirdly.
            # Better to just let it run, but limit maxiter.
            
            # A safer approach: run with a moderate maxiter.
            # If it doesn't converge, it returns the best found.
            # 100 iterations is usually plenty for 5D.
            
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * d,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            if not time_ok() or calls >= budget:
                break

        # Phase 3: Polish with very tight L-BFGS-B from the best found
        if not time_ok() or calls >= budget:
            return best_x, best_f

        # If we have budget left, do one more very precise run from the best point
        if calls < budget - 10:
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * d,
                               options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
