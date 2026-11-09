"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        
        # Strategy:
        # 1. Global Search: Use Sobol sequence (quasi-random) for better space-filling
        #    than Latin Hypercube in 5D. This helps locate the rotated Rosenbrock valley.
        # 2. Local Polish 1: Multi-start L-BFGS-B from the top 10 best global points.
        #    L-BFGS-B is efficient for smooth functions.
        # 3. Local Polish 2: Nelder-Mead refinement on the absolute best point found.
        #    Nelder-Mead is derivative-free and can provide a final high-precision polish
        #    or correct minor inaccuracies from numerical gradients in L-BFGS-B.

        # Phase 1: Sobol Global Search
        n_samples = 200
        if budget < 300:
            n_samples = int(budget * 0.6)
        
        # Sobol generator
        sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        samples = sampler.random(n=n_samples)
        # Scale from [0, 1] to [-5, 5]
        samples = samples * 10 - 5
        
        candidates = []
        for x in samples:
            if call_count >= budget - 100: # Reserve budget for local search
                break
            f_val = counted_func(x)
            candidates.append((f_val, x.copy()))
        
        if not candidates:
            # Fallback if no calls were made
            return best_x, best_f

        # Sort by function value
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-start L-BFGS-B Polish
        # Select top K candidates. Use distinct points to maximize coverage.
        K = 10
        top_candidates = []
        seen = set()
        for f, x in candidates:
            x_key = tuple(np.round(x, 6))
            if x_key not in seen:
                seen.add(x_key)
                top_candidates.append(x)
                if len(top_candidates) >= K:
                    break
        
        for x0 in top_candidates:
            if call_count >= budget - 20: # Reserve for final polish
                break
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                # Update best if improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead Final Polish
        # Run NM on the best point found so far to ensure high precision.
        if call_count < budget - 10 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
