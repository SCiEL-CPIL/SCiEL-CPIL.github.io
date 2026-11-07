import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Initial random point
        x0 = rng.uniform(-5, 5, self.dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # We will use a multi-start Nelder-Mead strategy.
        # Each restart uses a simplex of size dim+1.
        # We define a list of (max_iter, prob_local) for each restart.
        # prob_local: probability to start near best_x instead of random.
        
        # Schedule:
        # 1. 5 quick explorations (10 steps)
        # 2. 5 medium explorations (20 steps)
        # 3. 1 long exploitation (remaining budget)
        
        restarts = [
            (10, 0.0), (10, 0.0), (10, 0.0), (10, 0.0), (10, 0.0),
            (20, 0.5), (20, 0.5), (20, 0.5), (20, 0.5), (20, 0.5),
            (100, 1.0) # This one will be adjusted based on remaining budget
        ]

        for i, (max_iter, prob_local) in enumerate(restarts):
            if calls >= self.budget:
                break
            if time.time() >= deadline:
                break

            # Determine start point
            if rng.random() < prob_local:
                # Perturb best_x
                # Sigma for perturbation: start large, shrink?
                # For now, fixed sigma based on dimension
                sigma = 2.0
                x_start = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
            else:
                x_start = rng.uniform(-5, 5, self.dim)

            # Calculate allowed iterations to not exceed budget
            # NM uses ~1 call per iteration after init.
            # Init uses dim+1 calls.
            # Total cost approx dim+1 + max_iter.
            # We have self.budget - calls remaining.
            allowed_iter = self.budget - calls - (self.dim + 1)
            
            # If allowed_iter is too small, skip or do minimal
            if allowed_iter < 5:
                continue
                
            # Cap max_iter to allowed_iter and the planned max_iter
            current_max_iter = min(max_iter, allowed_iter)
            
            # Run Nelder-Mead
            # options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': False, 'maxiter': current_max_iter}
            # We need to count calls accurately. SciPy's minimize doesn't return call count directly in a standard way easily accessible without wrapping, but we can estimate or use a wrapper.
            
            # Let's wrap func to count calls
            call_count = [0]
            def wrapped_func(x):
                call_count[0] += 1
                return func(np.clip(x, -5, 5))
            
            # Simplex options
            # Default simplex is based on x_start.
            # We can specify initial simplex if we want, but default is fine.
            
            try:
                res = minimize(wrapped_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': current_max_iter, 'xatol': 1e-5, 'fatol': 1e-5})
                calls += call_count[0]
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If it fails, we might have used some calls, but it's hard to know exactly if it threw mid-run.
                # However, the wrapper increments call_count.
                # If an exception happens, the calls made are still counted in the evaluator's global count.
                # We just stop.
                break

        return best_x, best_f
