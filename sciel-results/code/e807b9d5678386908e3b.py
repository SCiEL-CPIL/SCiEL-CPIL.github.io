import numpy as np
import time
from scipy.optimize import powell

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.call_count = 0
        self.start_time = None
        self.max_time = 290.0  # Safety margin below 300s

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Track calls
        def wrapper(x):
            if time.time() - self.start_time > self.max_time:
                raise TimeoutError("Time limit exceeded")
            self.call_count += 1
            return func(x)

        # We will distribute the budget among multiple starts.
        # Powell's method can be efficient. 
        # Let's try 5 starts.
        num_starts = 5
        remaining_budget = self.budget
        
        # Initial points: one at center, others random
        # Also add a few points closer to the center to help with local convergence
        starts = []
        starts.append(np.zeros(self.dim))
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        for i, x0 in enumerate(starts):
            if self.call_count >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
            
            # Give the remaining budget to this start.
            # If it converges early, the rest is used by subsequent starts.
            maxfev = remaining_budget
            
            try:
                res = powell(wrapper, x0, bounds=[(-5, 5)] * self.dim, 
                             maxfev=maxfev, tol=1e-10, options={'maxiter': 200})
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
                    
            except Exception:
                # If an exception occurs, we stop.
                break
            
            # Update remaining budget
            remaining_budget = self.budget - self.call_count
            
            # If we have no budget left, break
            if remaining_budget <= 0:
                break

        return self.best_x, self.best_f
