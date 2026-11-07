"""Optimizer using LHS global exploration and hybrid SLSQP/L-BFGS-B local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time


class Optimizer:
    """Hybrid LHS + SLSQP + L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count[0] < budget

        # Phase 1: Global Exploration with Latin Hypercube Sampling
        # LHS provides better coverage than pure random or structured grids
        n_lhs = 25
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_lhs)
        # Scale to [-4, 4] to stay well within bounds and cover the likely region
        lhs_points = lhs_points * 8.0 - 4.0

        lhs_results = []
        for x in lhs_points:
            if not (time_ok() and budget_ok()):
                break
            f_val = wrapped_func(x)
            lhs_results.append((f_val, x))

        if not lhs_results:
            # Fallback if we failed immediately
            best_x = np.zeros(dim)
            if budget_ok():
                best_f = wrapped_func(best_x)
            else:
                best_f = 0.0
            return best_x, best_f

        # Sort results to find the best starting points
        lhs_results.sort(key=lambda item: item[0])
        best_f, best_x = lhs_results[0]

        # Phase 2: Local Refinement on Top Candidates
        # Take top 5 distinct points from LHS
        top_k = min(5, len(lhs_results))
        starts = [p[1] for p in lhs_results[:top_k]]
        
        bounds = [(-5.0, 5.0)] * dim

        for x0 in starts:
            if not (time_ok() and budget_ok()):
                break
            
            # Reserve some budget for final polishing
            if call_count[0] > budget - 50:
                break

            # Step 2a: SLSQP for robustness
            # SLSQP handles bounds well and can escape shallow local minima better than L-BFGS-B in some cases
            try:
                res_slsqp = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
                )
                if res_slsqp.fun < best_f:
                    best_f = res_slsqp.fun
                    best_x = res_slsqp.x.copy()
            except Exception:
                pass

            # Step 2b: L-BFGS-B for high-precision polishing
            # Start from the best found so far (either original or SLSQP result)
            if time_ok() and budget_ok():
                try:
                    res_lbfgs = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                    )
                    if res_lbfgs.fun < best_f:
                        best_f = res_lbfgs.fun
                        best_x = res_lbfgs.x.copy()
                except Exception:
                    pass

        # Phase 3: Final Polish
        # If we have budget left, try to squeeze out more precision from the current best
        if best_x is not None and time_ok() and budget_ok():
            try:
                res_final = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
