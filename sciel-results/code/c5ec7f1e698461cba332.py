"""
Optimizer for Bent Cigar (f12) using Sobol global search followed by Powell's method.
"""
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
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline:
                raise Exception("Time limit exceeded")
            
            # Ensure input is valid
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        # 2^6 = 64 points is a good balance for dim=5
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline:
                break
            wrapped_obj(x)
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Local Refinement with Powell's Method ---
        # Powell's method is effective for anisotropic functions like Bent Cigar
        # because it builds conjugate directions adaptively.
        
        # We need to limit the number of function evaluations in Powell.
        # Remaining budget: budget - calls_used
        remaining_budget = budget - calls_used
        
        if remaining_budget > 0:
            # scipy.optimize.minimize with method='Powell'
            # maxiter is not the best way to limit function calls directly in all versions,
            # but we can use a callback or just rely on the budget check in wrapped_obj.
            # However, Powell might run away if not constrained.
            # We'll use a custom callback to stop if we hit the deadline or budget.
            
            stop_flag = [False]
            
            def callback(xk):
                if time.time() >= deadline or calls_used >= budget:
                    stop_flag[0] = True
                # Also stop if we've used too many calls relative to remaining
                return stop_flag[0]
            
            try:
                res = minimize(
                    wrapped_obj, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 1000,  # Generous maxiter, controlled by callback/budget
                        'xtol': 1e-12,
                        'ftol': 1e-12
                    },
                    callback=callback
                )
            except Exception:
                pass # If it fails, we already have best_x
            
        # Return the best result found
        return best_x, best_fx
