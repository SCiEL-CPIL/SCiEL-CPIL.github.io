import numpy as np
from scipy.optimize import dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        call_count = [0]
        max_calls = self.budget
        
        def wrapped_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            call_count[0] += 1
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        bounds = [(-5, 5)] * self.dim
        
        # dual_annealing uses maxfun to limit evaluations
        # It also has maxiter, but maxfun is the hard limit on function calls
        try:
            res = dual_annealing(
                wrapped_func, 
                bounds, 
                maxfun=max_calls,
                seed=42 # Fixed seed for reproducibility if needed, or random
            )
            return res.x, res.fun
        except Exception:
            # Fallback to a simple random search or Nelder-Mead if it fails
            # But we should try to return something valid
            best_x = np.zeros(self.dim)
            best_f = float('inf')
            # Quick random search to find a better starting point if dual_annealing failed immediately
            for _ in range(min(100, max_calls - call_count[0])):
                x = np.random.uniform(-5, 5, self.dim)
                f = wrapped_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f
