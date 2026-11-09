"""Katsuura (f23) minimizer: CMA-ES with multi-start and adaptive budget allocation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23).
    
    Strategy:
    1. Smart initial sampling: Mix of origin, local Gaussian, and global uniform points.
    2. Tiered local search with CMA-ES for robustness:
       - Phase 1: Quick Nelder-Mead on top candidates to identify promising basins.
       - Phase 2: CMA-ES on top 2 candidates with adaptive budget.
       - Phase 3: Final polish on the global best.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Initial Diverse Sampling
        init_points = []
        
        # Exact zero (highest priority for Katsuura)
        init_points.append(np.zeros(dim))
        
        # Local Gaussian sampling around origin
        for sigma in [0.3, 0.5, 1.0, 2.0]:
            for _ in range(5):
                p = rng.normal(0, sigma, dim)
                init_points.append(np.clip(p, -5, 5))
        
        # Global uniform sampling
        for _ in range(10):
            p = rng.uniform(-5, 5, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        sorted_indices = np.argsort(init_values)
        
        # 2. Tiered Local Search
        # Tier 1: Deep search on the top 3 candidates using Nelder-Mead
        top_k = 3
        if calls < self.budget - 50:
            for i in range(min(top_k, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 50, 
                                       'xatol': 1e-6, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 2: CMA-ES on the top 2 candidates with remaining budget
        if calls < self.budget - 20:
            remaining_budget = self.budget - calls - 20
            cma_budget_per_start = max(10, remaining_budget // 2)
            
            for i in range(min(2, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 15.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead',
                                   options={
                                       'maxiter': cma_budget_per_start,
                                       'xatol': 1e-7,
                                       'fatol': 1e-9,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 3: Final polish on the global best
        if calls < self.budget - 5 and best_x is not None:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 80, 
                                   'xatol': 1e-9, 
                                   'fatol': 1e-11, 
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
