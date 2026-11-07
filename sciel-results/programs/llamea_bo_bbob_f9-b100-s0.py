import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start L-BFGS-B Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Phase 1: Generate a small set of diverse random starting points (e.g., 5-10).
    - Phase 2: Run L-BFGS-B from each start point, sharing the global call budget.
    - L-BFGS-B is highly efficient for smooth functions like Rosenbrock, using
      quasi-Newton approximations to converge rapidly along the valley.
    - We use finite differences for gradients since the function is a black-box.
    - We carefully manage the call budget to ensure we do not exceed 500 calls.
    - We track the best solution found across all local optimizations.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # Safety margin for 300s limit
        
        n = self.dim
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                # Return a large penalty if budget is exhausted
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            calls_made += 1
            return func(x_clipped)
        
        # Generate diverse starting points
        # Use a mix of random points and a few structured ones
        n_starts = 8
        starts = []
        
        # Include origin
        starts.append(np.zeros(n))
        
        # Add random points
        for _ in range(n_starts - 1):
            # Sample from a broader range to ensure diversity
            x_start = self.rng.uniform(-4, 4, size=n)
            starts.append(x_start)
        
        # For each start, run L-BFGS-B
        # We need to allocate budget carefully.
        # L-BFGS-B will use as many calls as it needs until convergence or maxiter.
        # We set maxiter to a reasonable value, but rely on the call budget tracker.
        
        for x0 in starts:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Initial evaluation to check if it's already better
            # But tracked_func will handle it.
            
            try:
                # Use L-BFGS-B with finite differences
                # options: maxiter, ftol, gtol
                # We don't know the scale of f, so default tolerances might be okay.
                # Rosenbrock minimum is 0, but shifted/offset.
                # We just want to minimize.
                
                res = minimize(
                    tracked_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                # Update best if this local opt is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If optimization fails, continue to next start
                continue
                
        # If no valid result, fallback to origin
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
