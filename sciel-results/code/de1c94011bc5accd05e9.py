"""Hybrid optimizer for Bent Cigar (f12): Sobol exploration + Adaptive Coordinate Descent + Powell polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Three-phase optimizer for Bent Cigar:
    Phase 1: Deterministic Sobol sampling for systematic coarse exploration.
    Phase 2: Adaptive Coordinate Descent (ACD) to exploit anisotropy.
    Phase 3: Powell's method for final local polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def make_call(x):
            nonlocal calls_used
            if not check_time() or calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            return fx

        best_x = np.zeros(dim)
        best_fx = float('inf')

        # Phase 1: Sobol sampling for systematic coarse search
        # Use ~20% of budget for exploration
        n_sobol = max(10, min(50, budget // 5))
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * 10.0 - 5.0  # Map to [-5, 5]^d

        for i in range(n_sobol):
            if not check_time() or calls_used >= budget:
                break
            x = sobol_points[i]
            fx = make_call(x)
            if fx is None:
                break
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not check_time() or calls_used >= budget:
            return best_x, best_fx

        # Phase 2: Adaptive Coordinate Descent
        # This exploits the anisotropy of Bent Cigar by adapting step sizes per dimension
        x = best_x.copy()
        fx = best_fx
        
        # Initialize step sizes
        step_sizes = np.ones(dim) * 1.0
        min_step = 1e-6
        max_step = 5.0
        no_improve_count = 0
        max_no_improve = 20

        while check_time() and calls_used < budget:
            improved = False
            for d in range(dim):
                if not check_time() or calls_used >= budget:
                    break
                
                # Try positive and negative steps
                for sign in [1.0, -1.0]:
                    y = x.copy()
                    y[d] = np.clip(y[d] + sign * step_sizes[d], -5, 5)
                    fy = make_call(y)
                    if fy is None:
                        break
                    
                    if fy < fx - 1e-12:
                        # Success: move and expand step slightly
                        x = y
                        fx = fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        step_sizes[d] = min(step_sizes[d] * 1.2, max_step)
                        improved = True
                        no_improve_count = 0
                        break  # Don't try negative if positive worked (or vice versa)
                if not check_time() or calls_used >= budget:
                    break
            
            if not check_time() or calls_used >= budget:
                break
            
            if not improved:
                # Shrink all steps
                step_sizes *= 0.5
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                # Check if steps are too small
                if np.all(step_sizes < min_step):
                    break
            else:
                no_improve_count = 0

        if not check_time() or calls_used >= budget:
            return best_x, best_fx

        # Phase 3: Powell's method for final polish
        remaining_budget = budget - calls_used
        if remaining_budget > 10:
            try:
                res = minimize(func, best_x, method='Powell',
                               options={'maxfev': remaining_budget, 'maxiter': 1000})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
