import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        calls = [0]
        
        def wrapper(x):
            calls[0] += 1
            if calls[0] > self.budget:
                # Stop immediately if we exceed budget
                # This is a safety net, minimize should respect maxfev
                raise Exception("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        rng = np.random.default_rng(123)
        
        # Define start points
        # 1. Origin
        # 2-5. Random points
        # Maybe add points near boundaries?
        starts = [np.zeros(self.dim)]
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        best_x = None
        best_f = float('inf')
        
        # We will run Nelder-Mead from each start
        # We divide the budget roughly, but let maxfev handle it
        # To ensure we don't waste all budget on one bad start, 
        # we can cap maxfev per start if we have many starts.
        # With 5 starts, ~100 each is reasonable.
        
        for start in starts:
            if calls[0] >= self.budget:
                break
            
            remaining = self.budget - calls[0]
            if remaining <= 0:
                break
            
            # Use a portion of the remaining budget for this start
            # If it's the last start, use all remaining.
            # Otherwise, use min(remaining, 100) or similar?
            # Let's just use remaining, but Nelder-Mead might converge early.
            # If it doesn't converge, it will use up the budget.
            # This is risky if the first start is bad and consumes 500 calls.
            # Better to cap per-start budget to ensure diversity.
            maxfev_for_start = min(remaining, 100)
            
            try:
                res = minimize(wrapper, start, method='Nelder-Mead', 
                               options={'maxiter': 200, 'maxfev': maxfev_for_start, 'xatol': 1e-8, 'fatol': 1e-8})
                
                # Update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception as e:
                # If minimize fails, try to get the current best from the exception?
                # Or just ignore and move to next start.
                pass
        
        # Fallback if no valid result
        if best_x is None:
            # Evaluate origin
            x0 = np.zeros(self.dim)
            f0 = wrapper(x0)
            best_x = x0
            best_f = f0

        return best_x, best_f
