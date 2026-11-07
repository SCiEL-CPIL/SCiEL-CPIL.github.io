import numpy as np
import time
from scipy.optimize import minimize

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
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            self.call_count += 1
            return func(x)

        # We will use Nelder-Mead with multiple starts.
        # Nelder-Mead is robust for 5D and can be efficient.
        # We'll do a few random starts and one at center.
        
        # Initial points: one at center, others random
        num_starts = 5
        starts = []
        starts.append(np.zeros(self.dim))
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Distribute budget roughly equally, but allow early stopping if converged
        # or if time is running out.
        
        # We will iterate through starts. For each start, we run Nelder-Mead.
        # We need to be careful with maxfev to not exceed total budget.
        
        # Strategy:
        # 1. Run NM with a limited maxfev (e.g., 100) for each start.
        # 2. If we have budget left, do a final polish on the best point found so far.
        
        # Let's try 100 function evaluations per start initially.
        # 5 starts * 100 = 500. This uses the whole budget.
        # But NM might stop early.
        
        # To be safe, let's calculate maxfev for each run.
        # We want to ensure total calls <= budget.
        
        # Let's use a simple loop.
        # For each start:
        #   maxfev_for_this_run = min(remaining_budget, 100)
        #   Run NM
        #   Update best
        
        # If we have remaining budget after all starts, do a final NM run on best_x.
        
        for i, x0 in enumerate(starts):
            if self.call_count >= self.budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
            
            # Determine maxfev for this run
            # Leave some buffer for the final polish? 
            # Or just use what's left?
            # If we use what's left, and it's the last start, it might use all of it.
            # If it's not the last, we should cap it so we have budget for others.
            
            # Let's cap at 100 for the first 4 starts, and use remaining for the last.
            # Actually, simpler: cap at min(remaining_budget, 100)
            # If remaining_budget > 100, we cap at 100.
            # If remaining_budget <= 100, we use all remaining.
            
            maxfev = min(self.budget - self.call_count, 100)
            
            if maxfev <= 0:
                break
            
            try:
                res = minimize(wrapper, x0, method='Nelder-Mead', 
                               options={'maxiter': maxfev, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
                    
            except Exception as e:
                # If an exception occurs (e.g. budget exceeded or timeout), stop.
                # The wrapper might have already recorded the call that triggered it?
                # If wrapper raises, the call is not returned, but was it counted?
                # In my wrapper: 
                # if self.call_count >= self.budget: raise
                # self.call_count += 1
                # return func(x)
                # So if budget is reached, it raises BEFORE incrementing? 
                # No, I checked `if self.call_count >= self.budget`.
                # If call_count is 500, and budget is 500, it raises.
                # So the 501st call is not made.
                # This is correct.
                
                # However, if NM tries to make a call that exceeds budget, 
                # wrapper raises. NM catches it? No, NM propagates it.
                # We catch it here and break.
                break
            
            # Update remaining budget implicitly via self.call_count
            
            # Check if we should stop early if we've used all budget
            if self.call_count >= self.budget:
                break
            
            # Check time
            if time.time() - self.start_time > self.max_time:
                break
        
        # If we have budget left, do a final polish on the best point
        if self.best_x is not None and self.call_count < self.budget and time.time() - self.start_time < self.max_time:
            remaining = self.budget - self.call_count
            if remaining > 0:
                try:
                    # Use a larger maxfev for the polish
                    res = minimize(wrapper, self.best_x, method='Nelder-Mead', 
                                   options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x
                except Exception:
                    pass

        return self.best_x, self.best_f
