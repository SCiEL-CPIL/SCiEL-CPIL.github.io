"""Sobol Global Search + Nelder-Mead Local Exploitation for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic hybrid: Sobol sampling followed by Nelder-Mead local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (300s limit)
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Search using Sobol Quasi-Random Sampling
        # We use the first 40% of the budget for global exploration.
        # Sobol sequences provide better uniform coverage than pure random, 
        # which is crucial for finding the basin of attraction in ill-conditioned landscapes.
        global_calls = int(budget * 0.4)
        
        # Generate Sobol points in [0, 1]^d
        sampler = qmc.Sobol(d=dim, scramble=True)
        # We need at least 2 points for Sobol, but we want global_calls points
        # Sobol requires power of 2 for best properties, but we can just take the first N.
        # However, qmc.Sobol.random(n) generates n points.
        # To be safe and efficient, let's generate a power of 2 >= global_calls if possible, 
        # or just generate global_calls points. The quality is still good for moderate N.
        # Let's stick to generating exactly global_calls points for strict budget control.
        # Note: Sobol generator state is not persistent across calls in a simple way here, 
        # but we create a new one.
        
        # We need to map [0,1] to [-5,5]
        # Actually, the problem states minimizer is in [-4,4]^5, so [-5,5] is safe.
        
        points = sampler.random(global_calls)
        # Map to [-5, 5]
        points = points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate all global points
        for p in points:
            if time.time() > deadline:
                break
            fx = func(p)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = p.copy()
                
        # If no calls were made (should not happen), initialize at center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Local Exploitation using Nelder-Mead
        # Use the remaining budget for local search starting from the global best.
        remaining_budget = budget - calls_made
        
        # Define a wrapper to count function calls and enforce the remaining budget
        # and deadline.
        local_calls = [0]
        
        def wrapped_func(x):
            # Check budget and time
            if local_calls[0] >= remaining_budget:
                return best_f  # Return current best to stop optimization effectively
            if time.time() > deadline:
                return best_f
            
            fx = func(x)
            local_calls[0] += 1
            return fx

        # Nelder-Mead options
        # maxiter is not the strict limit here, maxfev is.
        # We set maxfev to remaining_budget to strictly respect the call limit.
        # However, NM might call the function more than maxfev if not careful? 
        # SciPy's minimize with Nelder-Mead respects maxfev.
        
        options = {
            'maxfev': remaining_budget,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        try:
            result = minimize(wrapped_func, best_x, method='Nelder-Mead', options=options)
            # Update best if local search found something better
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except Exception:
            # If an exception occurs, we keep the best found so far
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
