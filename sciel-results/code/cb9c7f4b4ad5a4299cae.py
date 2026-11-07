import numpy as np
import time
import sys
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(0)
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # Safety margin for 300s limit
        
        # Initialize with a random point
        x0 = self.rng.uniform(-5, 5, self.dim)
        self.best_x = x0
        self.best_f = func(x0)
        
        calls_made = 1
        remaining_budget = self.budget - calls_made
        
        # We will use Nelder-Mead with restarts
        # Each run of NM will use a chunk of the budget
        # We track total calls to ensure we don't exceed budget
        
        # Helper to run NM and update best
        def run_nm(x_start, maxfev):
            nonlocal calls_made, remaining_budget
            
            # Define a wrapper to count calls
            # scipy's minimize handles the function calls internally, 
            # but we need to track them against our global budget.
            # It's safer to use a wrapper that raises an exception or stops if budget is exceeded,
            # but scipy doesn't support that natively well.
            # Instead, we estimate maxfev based on remaining budget.
            
            # Simple Nelder-Mead options
            options = {
                'maxiter': maxfev, 
                'maxfev': maxfev,
                'xatol': 1e-8,
                'fatol': 1e-8
            }
            
            # We need to be careful: scipy minimize might not stop exactly at maxfev if it converges faster.
            # But it won't exceed maxfev.
            
            try:
                res = minimize(func, x_start, method='Nelder-Mead', options=options)
                # Update best
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
                # We don't know exactly how many calls were made by scipy, 
                # but we allocated maxfev. 
                # To be safe, we assume it used up to maxfev.
                # Actually, we can track calls by wrapping func.
            except Exception:
                pass

        # Better approach: Wrap func to count calls and enforce budget.
        call_count = [0]
        def counting_func(x):
            if call_count[0] >= self.budget:
                raise ValueError("Budget exceeded")
            call_count[0] += 1
            return func(x)
            
        # Reset counter for the loop, but we already used 1 call.
        # Let's restructure to use a single counter.
        
        total_calls = 1
        x_current = x0
        f_current = self.best_f
        
        # We will run multiple phases of Nelder-Mead
        # Phase 1: Global exploration with several restarts
        # Phase 2: Local refinement
        
        # Strategy:
        # 1. Pick a random start.
        # 2. Run NM with maxfev = min(remaining_budget, 100)
        # 3. If improved, continue from there with a smaller maxfev or restart.
        
        while total_calls < self.budget and time.time() < deadline:
            # Determine maxfev for this NM run
            # Use a portion of remaining budget, e.g., 50 or 100
            maxfev = min(self.budget - total_calls, 100)
            if maxfev < 10:
                break
                
            # Choose starting point
            # 50% chance to restart from random, 50% from current best with noise
            if self.rng.random() < 0.5 or total_calls < 10:
                x_start = self.rng.uniform(-5, 5, self.dim)
            else:
                # Perturb current best
                noise_scale = 1.0
                x_start = np.clip(self.best_x + noise_scale * self.rng.standard_normal(self.dim), -5, 5)
                
            # Run NM
            # We need to ensure counting_func is used.
            # But counting_func raises exception if budget exceeded.
            # We should catch that.
            
            try:
                res = minimize(counting_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': maxfev, 'maxfev': maxfev, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except ValueError:
                # Budget exceeded during NM
                break
            except Exception:
                pass
                
            total_calls = call_count[0]
            
        return self.best_x, self.best_f
