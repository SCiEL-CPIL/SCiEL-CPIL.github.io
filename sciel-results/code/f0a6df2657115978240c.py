import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = time.time() + 285
        
        def eval_func(x):
            if self.calls >= budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f
        
        # Multi-start L-BFGS-B for smooth quasi-convex functions
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        n_starts = 12
        starts = []
        
        # Origin
        starts.append(np.zeros(dim))
        
        # Grid-like points
        for i in range(1, 4):
            for j in range(1, 4):
                start = np.zeros(dim)
                start[0] = -4 + 8 * i / 4
                start[1] = -4 + 8 * j / 4
                starts.append(start)
        
        # Random points
        while len(starts) < n_starts:
            start = rng.uniform(-4, 4, size=dim)
            starts.append(start)
        
        bounds = [(-5, 5)] * dim
        
        for start in starts:
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            try:
                result = minimize(
                    eval_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxls': 20}
                )
                
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = result.x.copy()
                    
            except Exception:
                continue
        
        return self.best_x, self.best_f
