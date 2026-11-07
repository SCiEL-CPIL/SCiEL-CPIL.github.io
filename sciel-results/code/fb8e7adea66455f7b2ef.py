import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290  # 10s margin
            
            # Initial random sampling to find a good start
            rng = np.random.default_rng(42)
            best_x = None
            best_f = float('inf')
            evals = 0
            max_initial_evals = 10
            
            for _ in range(max_initial_evals):
                if time.time() > deadline: break
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                evals += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x
            
            # Use remaining budget for local optimization
            remaining_budget = self.budget - evals
            
            # Define a wrapper to count evals and check time
            # Nelder-Mead doesn't have a strict eval count limit in older scipy, 
            # but maxiter controls simplex iterations.
            # We need to be careful with the number of function calls.
            # Nelder-Mead uses d+1 simplex points.
            # Each iteration does ~d function calls? No, 1 or 2.
            # Let's set maxiter to remaining_budget // (self.dim + 1) to be safe?
            # Or just let it run and check time?
            
            # A safer approach: Manual Nelder-Mead or use Powell?
            # Powell is also derivative-free.
            
            # Let's try Nelder-Mead with a tight maxiter.
            # In SciPy, maxiter is the maximum number of iterations allowed.
            # Each iteration of NM is roughly 1 function call (sometimes 2).
            # So maxiter = remaining_budget is a good upper bound.
            
            options = {
                'maxiter': remaining_budget,
                'xatol': 1e-6,
                'fatol': 1e-6
            }
            
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options=options)
                # Check if res is better
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

            return best_x, best_f
