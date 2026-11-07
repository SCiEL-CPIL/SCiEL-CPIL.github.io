"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start coarse sampling followed by local L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Wrap func to track calls and best value
        call_count = 0
        best_val = float('inf')
        best_x = None
        
        def wrapped(x):
            nonlocal call_count, best_val, best_x
            if time.time() > deadline:
                return best_val
            call_count += 1
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse Random Sampling
        # Use 10% of budget or at least 50 calls, capped at 100 to save time
        n_coarse = min(100, max(50, self.budget // 10))
        
        # Initial best is infinity, so first call will set it
        candidates = []
        
        for _ in range(n_coarse):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, self.dim)
            wrapped(x)
            candidates.append((best_val, x.copy())) # We don't know individual val easily without modifying wrapped, 
            # Actually wrapped updates best_val. Let's just collect x and evaluate later? 
            # No, we must record every call.
            # Let's modify wrapped to return the actual value for this specific x?
            # The contract says "Every call is recorded". The wrapper calls func.
            # To get the value of the current x, we can just call func directly? 
            # No, we must use the same interface.
            # Let's re-implement wrapped to return the value.
            
        # Re-do Phase 1 properly
        call_count = 0
        best_val = float('inf')
        best_x = None
        sampled_points = []
        
        def eval_func(x):
            nonlocal call_count, best_val, best_x
            if time.time() > deadline:
                return best_val
            call_count += 1
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        for _ in range(n_coarse):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, self.dim)
            val = eval_func(x)
            sampled_points.append((val, x))

        # Phase 2: Local Refinement
        # Select top diverse points from coarse sampling
        # Sort by value
        sampled_points.sort(key=lambda t: t[0])
        
        # Take top 5 unique points (simple check: just take first 5, they are likely distinct enough in 5D)
        # To ensure diversity, we could cluster, but with only 100 points and 5D, top 5 are usually good enough for Rosenbrock valley entry.
        starts = [p[1] for p in sampled_points[:5]]
        
        # If we haven't used much budget, maybe add a few more random starts? 
        # Let's stick to top 5 from coarse.
        
        # Allocate remaining budget to local optimization
        # L-BFGS-B is efficient. We want to run multiple optimizations.
        # We can set maxiter high and let it run until budget/time is out.
        
        # We will run minimize sequentially. 
        # Each minimize call will use eval_func.
        
        # To prevent one bad start from eating all time, we can set a maxiter per start.
        # But Rosenbrock can be tricky. 
        # Let's just run them. The deadline guard in eval_func will stop everything.
        
        for start_x in starts:
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
                
            # Try to refine
            try:
                res = minimize(
                    eval_func, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': self.budget - call_count, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass
        
        # Return the best found
        return best_x, best_val

# EVOLVE-BLOCK-END
