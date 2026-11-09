import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    The task description indicates the minimizer lies in [-4, 4]^5, which 
    contradicts a strictly linear function (whose minimum on a box would be 
    at a vertex). This suggests the function is likely convex and smooth 
    (e.g., Sphere or Ellipsoidal). 
    
    The previous strategy wasted budget probing vertices. This strategy 
    uses Powell's Method, which is highly efficient for low-dimensional 
    (5D) smooth functions. It performs coordinate-wise line searches, 
    which are effective for separable or nearly separable functions common 
    in BBOB benchmarks. It starts from the origin, which is a reasonable 
    prior for the center of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                val = float(func(x_arr))
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_arr.copy()
                return val
            except Exception:
                return float('inf')

        # Start from the origin, which is the center of the box [-5, 5]^5
        # and a likely region for the minimum given the shift is in [-4, 4].
        x0 = np.zeros(self.dim)
        
        # Evaluate the starting point to ensure we have a valid best_x
        if self.calls_made < self.budget and time.time() < deadline:
            _ = wrapped_func(x0)

        # Use Powell's Method for the remaining budget.
        # Powell is robust for low dimensions and doesn't require gradient estimation.
        try:
            res = minimize(
                wrapped_func, 
                self.best_x if self.best_x is not None else x0, 
                method='Powell', 
                options={
                    'maxiter': 1000, # Let it run until budget/time
                    'xtol': 1e-8,
                    'ftol': 1e-8,
                    'maxfev': self.budget # Hard limit on function evaluations
                }
            )
        except Exception:
            pass

        # If Powell didn't use all budget (converged), we could try a restart, 
        # but for 5D smooth functions, Powell usually converges well.
        # We return the best found.
        return self.best_x, self.best_f
