import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        calls = 0
        def counted_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Initial random point
        x0 = np.random.uniform(-5, 5, D)
        best_x = x0.copy()
        best_f = counted_func(x0)
        
        # Main loop: Multi-Start Powell
        while calls < budget:
            remaining = budget - calls
            # Limit evaluations per local run to ensure multiple restarts
            chunk_size = min(40, remaining)
            
            # Choose start point
            # Exploit if we have found a good solution and have some budget used
            if calls > 50 and np.random.rand() < 0.7:
                # Start near best found with small jitter
                jitter = np.random.uniform(-0.5, 0.5, D)
                start_x = best_x + jitter
                start_x = np.clip(start_x, -5, 5)
            else:
                # Start at a random point
                start_x = np.random.uniform(-5, 5, D)
                
            # Run Powell local search
            try:
                res = minimize(counted_func, start_x, method='Powell', 
                               options={'maxfev': chunk_size, 'xtol': 1e-6, 'ftol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
                
            if calls >= budget:
                break

        return best_x, best_f
