"""Adaptive coordinate-wise optimizer for Bent Cigar with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Adaptive coordinate-wise step size optimizer with Nelder-Mead refinement.
    
    Strategy:
    1. Phase 1: Sobol-based initialization to find a good starting point
    2. Phase 2: Coordinate-wise adaptive descent (exploits Bent Cigar anisotropy)
    3. Phase 3: Nelder-Mead local refinement on the best candidate
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Sobol-based initialization (use ~20% of budget)
        # Use scipy's Sobol or fallback to Halton-like sequence
        try:
            from scipy.stats import qmc
            n_sobol = min(32, budget // 5)
            sobol = qmc.Sobol(d=dim, seed=42)
            points = sobol.random(n_sobol) * 10.0 - 5.0  # Map to [-5, 5]
        except ImportError:
            n_sobol = min(32, budget // 5)
            rng_init = np.random.default_rng(42)
            points = rng_init.uniform(-5, 5, (n_sobol, dim))
        
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        for pt in points:
            if time.time() >= deadline:
                break
            if calls_used >= budget:
                break
            pt_clipped = np.clip(pt, -5, 5)
            f_val = func(pt_clipped)
            calls_used += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = pt_clipped.copy()
        
        # Phase 2: Coordinate-wise adaptive descent
        # Each dimension has its own step size
        x = best_x.copy()
        fx = best_fx
        sigmas = np.ones(dim) * 1.0  # Initial step sizes per dimension
        
        # Adaptive factors
        increase_factor = 1.3
        decrease_factor = 0.7
        min_sigma = 1e-10
        max_sigma = 5.0
        
        # Track per-dimension success streaks
        dim_success = np.zeros(dim)  # Consecutive successes per dimension
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # For each dimension, try a step in both directions
            for i in range(dim):
                if time.time() >= deadline:
                    break
                if calls_used >= budget:
                    break
                
                # Try positive step
                x_trial = x.copy()
                x_trial[i] += sigmas[i]
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls_used += 1
                
                if f_trial < fx:
                    # Success: accept and increase step size
                    x = x_trial
                    fx = f_trial
                    sigmas[i] = min(sigmas[i] * increase_factor, max_sigma)
                    dim_success[i] += 1
                else:
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[i] -= sigmas[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_used += 1
                    
                    if f_trial < fx:
                        # Success: accept and increase step size
                        x = x_trial
                        fx = f_trial
                        sigmas[i] = min(sigmas[i] * increase_factor, max_sigma)
                        dim_success[i] += 1
                    else:
                        # Failure: decrease step size
                        sigmas[i] = max(sigmas[i] * decrease_factor, min_sigma)
                        dim_success[i] = 0
                
                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            
            # Early termination if all sigmas are very small
            if np.all(sigmas < 1e-8):
                break
        
        # Phase 3: Nelder-Mead refinement (use remaining budget)
        remaining_budget = budget - calls_used
        if remaining_budget > 10 and time.time() < deadline:
            try:
                def wrapper(x_vec):
                    nonlocal calls_used
                    if calls_used >= budget or time.time() >= deadline:
                        return 1e10
                    x_vec = np.clip(x_vec, -5, 5)
                    val = func(x_vec)
                    calls_used += 1
                    return val
                
                # Nelder-Mead options
                nm_options = {
                    'maxiter': min(remaining_budget, 200),
                    'maxfev': min(remaining_budget, 200),
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
                
                result = minimize(wrapper, best_x, method='Nelder-Mead', options=nm_options)
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass  # Keep best found so far
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
