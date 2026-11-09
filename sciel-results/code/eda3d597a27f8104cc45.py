import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Vertex Search to find a good starting point
        x = np.zeros(self.dim)
        calls_made = 0
        
        for i in range(self.dim):
            x_pos = x.copy()
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            calls_made += 1
            
            x_neg = x.copy()
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            calls_made += 1
            
            if f_pos < f_neg:
                x[i] = 5.0
            else:
                x[i] = -5.0

        # Step 2: Refine using Nelder-Mead
        # Remaining budget
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 10:
            # Define a wrapper to count calls and respect budget
            self.current_calls = 0
            self.total_calls = calls_made
            self.budget_limit = self.budget
            
            def func_wrapper(x_eval):
                if self.total_calls >= self.budget_limit:
                    return 1e10 # Return large value to stop
                self.total_calls += 1
                return func(x_eval)

            # Nelder-Mead options
            # maxiter is the number of function calls per dimension? No, it's iterations.
            # Each iteration uses dim+1 calls roughly?
            # Let's be conservative.
            max_iter = max(10, (remaining_budget // (self.dim + 1)) - 5)
            
            res = minimize(func_wrapper, x, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            
            # Update x to the best point found
            x = res.x

        return x, func(x) if self.total_calls < self.budget else 0.0
