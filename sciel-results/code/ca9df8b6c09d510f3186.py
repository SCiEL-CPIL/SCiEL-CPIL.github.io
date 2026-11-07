import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        rng = np.random.default_rng(42)
        
        # State
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def safe_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls_used += 1
            return func(x)

        # 1. Initialization: Diverse Sampling
        # We use a mix of scales to ensure we cover the box and the center.
        n_init = 60
        points = []
        # 20 uniform in [-5, 5]
        points.extend(rng.uniform(-5, 5, (20, self.dim)))
        # 20 uniform in [-3, 3]
        points.extend(rng.uniform(-3, 3, (20, self.dim)))
        # 20 uniform in [-1, 1]
        points.extend(rng.uniform(-1, 1, (20, self.dim)))
        
        X_init = np.array(points)
        for x in X_init:
            try:
                f = safe_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            except Exception:
                break
        
        if best_x is None:
            # Fallback if something went wrong
            return np.zeros(self.dim), np.inf

        # 2. Main Loop: Annealed Local Search
        # We will run L-BFGS from perturbed versions of the best point.
        # Temperature starts high and decays.
        
        max_iter_outer = 20 # Max number of restarts
        # Budget per L-BFGS run
        # We have 500 - 60 = 440 calls left.
        # 20 restarts * 20 calls/run = 400 calls.
        # Leaves 40 calls for a final polish.
        
        budget_per_run = 20
        
        # Temperature schedule
        # Start with a large step (2.0) and decay to small (0.01)
        temp = 2.0
        decay = 0.8
        
        for i in range(max_iter_outer):
            if time.time() > deadline - 5:
                break
            if calls_used + budget_per_run > self.budget:
                budget_per_run = self.budget - calls_used
                if budget_per_run <= 0:
                    break
            
            # Perturb best_x
            noise = rng.normal(0, temp, self.dim)
            x_start = best_x + noise
            x_start = np.clip(x_start, -5, 5)
            
            # Run L-BFGS
            try:
                # Use a wrapper that stops if budget/time is hit
                local_calls = [0]
                def local_wrapper(x):
                    if local_calls[0] >= budget_per_run:
                        raise ValueError("Local Budget")
                    if time.time() > deadline - 5:
                        raise ValueError("Time")
                    local_calls[0] += 1
                    return func(x) # Note: using raw func here to avoid double counting in safe_func? 
                                   # No, we need to count calls. 
                                   # Let's use safe_func but pass it the budget limit?
                                   # safe_func increments global calls_used.
                                   # So we just call safe_func.
                    # But safe_func raises if global budget is hit.
                    # So local_wrapper should just call safe_func.
                    return safe_func(x)

                # Actually, safe_func already handles the global count.
                # We just need to limit the number of calls in this specific run.
                # So we use a counter in the closure.
                
                res = minimize(local_wrapper, x_start, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': budget_per_run, 'maxls': 10, 'gtol': 1e-5})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
            # Update temperature
            temp *= decay
            
            # Check if we have budget for a final polish
            if calls_used < self.budget - 10:
                continue
            else:
                break
        
        # 3. Final Polish
        # If we have budget left, do one more high-precision run from the best point
        if calls_used < self.budget - 5 and time.time() < deadline - 10:
            remaining = self.budget - calls_used
            try:
                local_calls = [0]
                def final_wrapper(x):
                    if local_calls[0] >= remaining:
                        raise ValueError("Final Budget")
                    if time.time() > deadline - 2:
                        raise ValueError("Time")
                    local_calls[0] += 1
                    return safe_func(x)

                res = minimize(final_wrapper, best_x, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': remaining, 'maxls': 15, 'gtol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
