import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        calls_made = 0
        max_calls = self.budget
        
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return 1e10
            calls_made += 1
            return func(x)

        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Configuration: (n_random, maxiter)
        # Restart 1: 20 random, 15 iters NM
        # Restart 2: 20 random, 15 iters NM
        # Restart 3: 20 random, 10 iters NM
        
        restarts = [
            (20, 15),
            (20, 15),
            (20, 10)
        ]
        
        for n_random, max_iter in restarts:
            if calls_made >= max_calls:
                break
                
            # Phase 1: Random Search
            points = rng.uniform(-4, 4, (n_random, self.dim))
            
            for p in points:
                if calls_made >= max_calls:
                    break
                f_val = counted_func(p)
                if f_val < best_f:
                    best_f = f_val
                    best_x = p.copy()
            
            if best_x is None or calls_made >= max_calls:
                continue
            
            # Phase 2: Local Optimization
            start_x = best_x.copy()
            
            try:
                res = minimize(counted_func, start_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
