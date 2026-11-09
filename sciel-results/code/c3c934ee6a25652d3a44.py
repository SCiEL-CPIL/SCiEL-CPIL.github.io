import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal calls_made
            calls_made += 1
            if calls_made > total_budget:
                raise ValueError("Budget exceeded")
            return func(x)

        # Deterministic diverse starting points
        # Center and corners
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim) * 4.0,
            -np.ones(self.dim) * 4.0,
            np.array([4.0, -4.0, 4.0, -4.0, 4.0])[:self.dim],
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0])[:self.dim]
        ]
        
        # We have 5 starts.
        # Let's allocate 30 calls per start (150 total)
        phase1_budget_per_start = 30
        
        for x0 in starts:
            local_calls = 0
            def local_func(x):
                nonlocal local_calls
                local_calls += 1
                if local_calls > phase1_budget_per_start:
                    raise ValueError("Local budget exceeded")
                return wrapped_func(x)
            
            try:
                max_iter = phase1_budget_per_start // self.dim + 1
                res = minimize(local_func, x0, method='Powell', options={'maxiter': max_iter, 'disp': False})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except ValueError:
                pass
            except Exception:
                pass

        # Phase 2: Refine the best point found with remaining budget
        remaining_budget = total_budget - calls_made
        if remaining_budget > 0 and best_x is not None:
            local_calls = 0
            def local_func2(x):
                nonlocal local_calls
                local_calls += 1
                if local_calls > remaining_budget:
                    raise ValueError("Local budget exceeded")
                return wrapped_func(x)
            
            try:
                max_iter = remaining_budget // self.dim + 1
                res = minimize(local_func2, best_x, method='Powell', options={'maxiter': max_iter, 'disp': False})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except ValueError:
                pass
            except Exception:
                pass
        
        return best_x, best_f
