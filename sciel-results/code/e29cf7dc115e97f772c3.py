"""Rewritten program using LHS + SLSQP + L-BFGS-B hybrid strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    """Hybrid LHS, SLSQP, and L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Deadline guard: keep 10s margin (300s limit)
        deadline = start_time + 290

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        # 1. Generate LHS points in [-4, 4]^dim
        # Using 30 points for global exploration
        n_lhs = 30
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_lhs)
        # Scale to [-4, 4] to ensure we are well within bounds for local solvers
        lhs_points = lhs_points * 8.0 - 4.0

        # Evaluate LHS points
        lhs_results = []
        for x in lhs_points:
            if call_count[0] >= budget - 150: # Reserve budget for local
                break
            f_val = wrapped_func(x)
            lhs_results.append((f_val, x))
            
        if not lhs_results:
             # Fallback
             best_x = np.zeros(dim)
             best_f = wrapped_func(best_x)
        else:
            lhs_results.sort(key=lambda item: item[0])
            best_f, best_x = lhs_results[0]
            
            # Take top 4 distinct points
            top_k = 4
            starts = [p[1] for p in lhs_results[:top_k]]
            
            bounds = [(-5.0, 5.0)] * dim
            
            for x0 in starts:
                if time.time() > deadline:
                    break
                if call_count[0] >= budget - 20:
                    break
                    
                # 2. SLSQP Refinement
                # SLSQP is robust for smooth functions with bounds
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

                # 3. L-BFGS-B Polish
                # L-BFGS-B is faster for final convergence
                if time.time() < deadline and call_count[0] < budget - 10:
                    try:
                        res2 = minimize(
                            wrapped_func,
                            best_x, # Start from the current best
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                        )
                        if res2.fun < best_f:
                            best_f = res2.fun
                            best_x = res2.x
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
