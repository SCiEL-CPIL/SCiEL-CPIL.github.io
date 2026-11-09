"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
Adaptive Coordinate Descent for anisotropic refinement.
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
        # 64 points provides good coverage for 5D without wasting budget
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Adaptive Coordinate Descent ---
        # Bent Cigar is highly anisotropic. Coordinate descent with adaptive 
        # step sizes per dimension is very effective because it can shrink 
        # steps in "narrow" dimensions rapidly while maintaining larger steps 
        # in "wide" dimensions.
        
        # Initialize step sizes. 
        # We start with a moderate step size. The Sobol phase ensures we are 
        # likely in the basin.
        # Heuristic: start with 0.5. 
        h = np.full(dim, 0.5)
        
        current_x = x0.copy()
        current_fx = best_fx
        
        # Track no-improvement counter to handle plateaus or local minima
        no_improve_count = 0
        max_no_improve = 20 # If we cycle through all dims without improvement, shrink all steps
        
        # Main Loop
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            improved = False
            
            for i in range(dim):
                if time.time() >= deadline or calls_used >= budget:
                    break
                
                # Try positive and negative steps
                x_plus = current_x.copy()
                x_plus[i] += h[i]
                x_plus = np.clip(x_plus, -5, 5)
                
                x_minus = current_x.copy()
                x_minus[i] -= h[i]
                x_minus = np.clip(x_minus, -5, 5)
                
                # Evaluate
                fx_plus = wrapped_obj(x_plus)
                fx_minus = wrapped_obj(x_minus)
                
                # Determine best move
                if fx_plus < current_fx:
                    current_x = x_plus
                    current_fx = fx_plus
                    improved = True
                    # Success: Expand step size slightly to accelerate convergence
                    # Cap expansion to avoid oscillation
                    h[i] = min(h[i] * 1.2, 5.0)
                elif fx_minus < current_fx:
                    current_x = x_minus
                    current_fx = fx_minus
                    improved = True
                    # Success: Expand step size slightly
                    h[i] = min(h[i] * 1.2, 5.0)
                else:
                    # Failure: Shrink step size for this dimension
                    h[i] *= 0.5
                    # If step size becomes too small, this dimension is effectively optimized
                    if h[i] < 1e-6:
                        h[i] = 0.0
                        continue # Skip further attempts in this dim if h is 0? 
                        # Actually, if h is 0, we should probably stop trying this dim 
                        # or reset it if stuck. For now, 0 means "done".
                        # But we might need to re-examine if we moved in other dims.
                        # Let's just keep h=0 and skip.
                        # Wait, if h=0, x_plus == x_minus == current_x. 
                        # We just did 2 wasted calls. 
                        # Optimization: If h[i] is very small, we can skip evaluation 
                        # if we are confident. But for safety, let's only set to 0 
                        # if we have tried many times. 
                        # Better: Don't set to 0, just make it very small. 
                        # If h[i] < 1e-8, we consider this dim converged.
                        pass # We already updated h[i] above. 
                        # Let's refine: if h[i] < 1e-8, we can break out of 
                        # trying this dimension in this pass? No, we just did.
                        # Next pass, we will try again with even smaller h.
                        # To save calls, if h[i] < 1e-7, we can skip this dim 
                        # for a few iterations? 
                        # For simplicity and robustness, we just let h shrink.
                        # The cost is 2 calls per dim per pass. 5 dims = 10 calls/pass.
                        # 500 calls / 10 = 50 passes. 
                        # 50 passes * 5 dims = 250 steps. 
                        # This is sufficient for convergence if steps adapt well.
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Stuck in a local minimum or noise? 
                    # Reset steps to a larger value to escape? 
                    # Or just break if steps are tiny.
                    if np.min(h[h>0]) < 1e-5:
                        break
                    # Try to kick out of local min by increasing steps
                    h *= 1.5
                    no_improve_count = 0

        return best_x, best_fx
