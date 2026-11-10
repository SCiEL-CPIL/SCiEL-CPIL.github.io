import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0
        
        n = self.dim
        budget = self.budget
        
        # Initial point
        x0 = np.random.uniform(-5, 5, n)
        
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1
        
        def wrapped_func(x):
            nonlocal best_x, best_f, calls
            if time.time() > deadline or calls >= budget:
                return best_f + 1e10 # Ensure it's worse than best, but not inf
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
            
        # Estimate max iterations
        # Scipy CMA lambda ~ 10 for n=5
        # 500 calls / 10 = 50 generations
        # Give it a bit more to be safe, the wrapper will stop it.
        
        try:
            res = minimize(wrapped_func, x0, method='CMA', 
                           options={'maxiter': 100, 'disp': False, 'tol': 1e-12})
        except Exception:
            pass
            
        return best_x, best_f
