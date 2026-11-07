import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        best_x = None
        best_f = float('inf')
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal self
            if self.calls >= self.budget:
                # Return a large value to discourage further exploration in this run
                return 1e10 
            self.calls += 1
            # Check time limit (safety margin)
            if time.time() - self.start_time > 290: # 290s limit to be safe
                return 1e10
            return func(x)
            
        rng = np.random.default_rng(0)
        
        # Strategy: Multiple restarts with L-BFGS-B
        # Budget 300. Let's aim for ~15 restarts with ~20 calls each.
        # We will run restarts until budget is exhausted or time is up.
        
        n_restarts = 20 # Max number of restarts to try
        per_run_budget = 15 # Approx calls per run allowed by L-BFGS-B
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > 290:
                break
                
            # Generate random start
            x0 = rng.uniform(-5, 5, self.dim)
            
            # If we have a previous best, we could also restart from there with perturbation?
            # For now, pure random restarts is a solid baseline.
            
            # Run L-BFGS-B
            # maxiter is not the same as maxfun. maxfun is more direct.
            # L-BFGS-B usually uses 1-2 fun evals per iteration.
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * self.dim, 
                               options={'maxfun': 20, 'maxiter': 20})
                
                if res.success or res.fun < best_f:
                    # Note: res.fun might be the value at the last iteration, 
                    # but we need to ensure it's a valid point within bounds.
                    # L-BFGS-B respects bounds.
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                        
            except Exception:
                continue
                
        # If no valid result was found (e.g., all failed), return a random point
        if best_x is None:
             best_x = rng.uniform(-5, 5, self.dim)
             best_f = func(best_x) # This would exceed budget if not careful, but we assume at least one call happened or we return a default.
             # Actually, if self.calls is 0, we must make at least one call to be valid? 
             # "exception before the first call makes the whole result invalid".
             # So we must ensure at least one call.
             if self.calls == 0:
                 x_test = rng.uniform(-5, 5, self.dim)
                 best_f = func(x_test)
                 best_x = x_test
                 self.calls = 1

        return best_x, best_f
