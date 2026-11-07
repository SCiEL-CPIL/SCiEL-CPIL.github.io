import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0) for _ in range(dim)]
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            if time.time() - self.start_time > 290:
                raise TimeoutError("Time limit exceeded")
            
            # Ensure x is within bounds (though DE and NM usually handle this)
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Global Search using Differential Evolution
        # Use ~80% of the budget
        # popsize=5 is suitable for dim=5 to allow more generations
        try:
            differential_evolution(
                wrapped_func, 
                self.bounds, 
                maxiter=100, # Will be limited by budget/time
                popsize=5,
                tol=1e-8,
                seed=0,
                polish=False # We will do our own polish
            )
        except (TimeoutError, ValueError):
            pass # Stop and use best found

        # Phase 2: Local Polish using Nelder-Mead
        # Use remaining budget
        remaining = self.budget - self.calls
        if remaining > 0 and self.best_x is not None:
            try:
                minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except (TimeoutError, ValueError):
                pass

        # If no calls were made (unlikely), return a default
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)

        return self.best_x, self.best_f
