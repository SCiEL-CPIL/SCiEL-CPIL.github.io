"""
Optimizer for Bent Cigar (f12) using Sobol global search followed by
adaptive coordinate descent with per-dimension step size adaptation.
"""
import numpy as np
import time
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
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
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
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            wrapped_obj(x)
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Adaptive Coordinate Descent ---
        # For Bent Cigar, we need to align with the narrow valley.
        # Coordinate descent with adaptive step sizes per dimension
        # can be effective if we allow step sizes to grow/shrink
        # based on improvement.
        
        remaining_budget = budget - calls_used
        if remaining_budget > 0:
            x = x0.copy()
            fx = best_fx
            
            # Initialize step sizes: start with a moderate step
            # Bent Cigar has curvature 1 in one direction and 100 in others
            # So we want small steps in steep directions, large in flat
            # But we don't know which is which. Start with 1.0 and adapt.
            step_sizes = np.ones(dim)
            
            # Number of iterations for coordinate descent
            # Each iteration does dim function evaluations (one per coordinate)
            # We can do many iterations
            max_cd_iters = remaining_budget // (dim + 1)  # +1 for safety
            
            for it in range(max_cd_iters):
                if time.time() >= deadline or calls_used >= budget:
                    break
                
                improved = False
                for i in range(dim):
                    if time.time() >= deadline or calls_used >= budget:
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] += step_sizes[i]
                    fx_trial = wrapped_obj(x_trial)
                    
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        # Increase step size (we're moving in a good direction)
                        step_sizes[i] *= 1.5
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[i] -= step_sizes[i]
                        fx_trial = wrapped_obj(x_trial)
                        
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            # Increase step size
                            step_sizes[i] *= 1.5
                        else:
                            # No improvement, shrink step size
                            step_sizes[i] *= 0.5
                    
                    # Keep step sizes within reasonable bounds
                    step_sizes[i] = np.clip(step_sizes[i], 1e-6, 10.0)
                
                # If no improvement in this full pass, try to restart
                # from a slightly perturbed version of best_x
                if not improved:
                    # Perturb best_x slightly and continue
                    if calls_used + dim < budget:
                        x = best_x.copy() + 0.1 * np.random.randn(dim)
                        x = np.clip(x, -5, 5)
                        fx = wrapped_obj(x)
                        step_sizes = np.ones(dim) * 0.5
                    else:
                        break
        
        # Return the best result found
        return best_x, best_fx
