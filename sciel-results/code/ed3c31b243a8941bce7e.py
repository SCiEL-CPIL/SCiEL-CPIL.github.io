import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin
        
        # Phase 1: Global Exploration
        # Use a subset of the budget for initial sampling
        n_explore = 50
        if self.budget < 100:
            n_explore = self.budget // 2
        
        points = np.random.uniform(-5, 5, (n_explore, self.dim))
        best_x = None
        best_f = float('inf')
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = points[i]
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local Refinement
        # Use remaining budget for Nelder-Mead
        remaining_budget = self.budget - n_explore
        if remaining_budget > 0 and best_x is not None:
            # Define objective
            def obj(x):
                return func(x)
            
            # Nelder-Mead options
            # maxfev is the max number of function evaluations
            # We don't want to exceed the deadline
            options = {
                'maxiter': 1000,
                'maxfev': remaining_budget,
                'xatol': 1e-8,
                'fatol': 1e-8
            }
            
            res = minimize(obj, best_x, method='Nelder-Mead', options=options)
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        return best_x, best_f
