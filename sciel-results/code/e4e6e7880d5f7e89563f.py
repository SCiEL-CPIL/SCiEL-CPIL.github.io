import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Hybrid Optimizer: Global (1+1)-ES followed by Local Nelder-Mead refinement.
    
    Phase 1: (1+1)-ES with adaptive step size to locate the global basin.
    Phase 2: Nelder-Mead to refine the solution to high precision.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        
        # Split budget: 70% for Global Search (ES), 30% for Local Refinement (NM)
        # We ensure at least 10 calls for NM if possible, but prioritize ES if budget is small
        es_budget = int(budget * 0.7)
        nm_budget = budget - es_budget
        
        # Ensure we have some budget for both phases
        if es_budget < 10:
            es_budget = max(1, budget // 2)
            nm_budget = budget - es_budget
            
        if nm_budget < 10 and budget > 20:
            nm_budget = 10
            es_budget = budget - nm_budget

        rng = np.random.default_rng(42)
        
        # --- PHASE 1: (1+1)-ES Global Search ---
        # Start from a random point in the center to avoid boundary bias initially
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        calls_used = 1
        
        # Simple (1+1)-ES loop
        # We use a success-based adaptation similar to Inspiration 1
        # But we track global best separately in case the current trajectory fails
        # though (1+1) keeps the best so far by definition of the update rule
        
        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            # Generate offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, lower, upper)
            
            fy = func(y)
            calls_used += 1
            
            # Update
            if fy < fx:
                x, fx = y, fy
                # Increase step size on success
                sigma *= 1.2
                if sigma > 5.0: sigma = 5.0 # Cap sigma to stay within reasonable bounds
            else:
                # Decrease step size on failure
                sigma *= 0.9
                if sigma < 1e-4:
                    sigma = 0.1
                    
            # Update global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Check for stagnation in ES to potentially trigger early exit to NM?
            # For now, we stick to the fixed split to ensure we always get refinement.
            # If ES is stuck, NM will just fail to improve much, which is fine.

        # --- PHASE 2: Nelder-Mead Local Refinement ---
        # Use the best point found in Phase 1 as the starting point for NM
        x_start = best_x.copy()
        
        # Create initial simplex for NM
        # Small initial step to ensure we are in the local basin
        simplex_step = 0.5 
        simplex = [x_start.copy()]
        for i in range(dim):
            p = x_start.copy()
            p[i] += simplex_step
            p = np.clip(p, lower, upper)
            simplex.append(p)
            
        # Wrapper for NM to count calls and enforce deadline
        nm_calls = 0
        nm_best_x = x_start.copy()
        nm_best_f = best_fx
        
        def wrapped_func(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            
            # Check budget and time
            if calls_used >= budget or time.time() > deadline:
                return 1e6
                
            x = np.clip(x, lower, upper)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = 1e6
                
            calls_used += 1
            nm_calls += 1
            
            if np.isnan(f_val) or np.isinf(f_val):
                return 1e6
                
            # Update local best for NM (also updates global best implicitly)
            if f_val < nm_best_f:
                nm_best_f = f_val
                nm_best_x = x.copy()
                # Also update the overall best_x/fx for the final return
                # But we will return nm_best_x at the end
                
            return f_val

        def callback(xk):
            # Stop if we hit budget or time
            if calls_used >= budget or time.time() > deadline:
                return True
            return False

        try:
            # Run Nelder-Mead
            # maxiter is set high, but callback will stop it
            result = scipy.optimize.minimize(
                wrapped_func, 
                x_start, 
                method='Nelder-Mead',
                options={
                    'maxiter': nm_budget * 2, # Allow enough iterations
                    'xatol': 1e-8, 
                    'fatol': 1e-8, 
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except Exception:
            pass

        # Final result is the best found in either phase
        # nm_best_x/f is guaranteed to be <= best_x/f from ES
        # because NM started at best_x and only moves if it improves
        
        if nm_best_x is not None:
            return nm_best_x, nm_best_f
        else:
            return best_x, best_fx
