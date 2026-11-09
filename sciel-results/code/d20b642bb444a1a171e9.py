"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Pre-allocate some memory for tracking best
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # We will perform multiple restarts.
        # Each restart uses a local Nelder-Mead optimization.
        # To respect the global budget, we estimate cost per NM run and limit restarts.
        # A typical NM run in 5D takes ~50-100 evaluations to converge reasonably.
        # With 500 budget, we can afford ~5-10 restarts.
        
        # Strategy: 
        # 1. Initial random sample to get a sense of the landscape (optional but cheap)
        # 2. Start NM from that point.
        # 3. If time/budget allows, pick a new random point and restart.
        
        # We track function calls manually to ensure we don't exceed budget.
        # However, scipy.optimize.minimize doesn't easily expose a counter that stops exactly at budget.
        # Instead, we can wrap the function to count and raise an exception if exceeded, 
        # or simply rely on the fact that NM converges fast. 
        # Given the 5D space and sharp ridge, NM might get stuck. 
        # Multi-start is key.
        
        # Let's use a wrapper that counts calls.
        class CallCounter:
            def __init__(self, func, budget_limit):
                self.func = func
                self.budget_limit = budget_limit
                self.count = 0
            
            def __call__(self, x):
                self.count += 1
                if self.count > self.budget_limit:
                    # Return a large value to signal failure/stopping, or raise
                    # Raising might be cleaner for scipy to stop
                    raise ValueError("Budget exceeded")
                return self.func(x)

        # Since we need to handle the budget globally across restarts, 
        # and scipy doesn't support stopping mid-run based on external counter easily 
        # without exceptions, we will do this:
        # 1. Create a counter for the entire run.
        # 2. Run NM with a maxiter limit that corresponds to a fraction of remaining budget.
        # 3. If it finishes, restart.
        
        total_budget = self.budget
        remaining_budget = total_budget
        
        # Number of restarts we attempt
        # Let's try up to 8 restarts, each with a capped maxiter.
        # In 5D, NM simplex has 6 points. Each iteration is ~6 calls.
        # 50 iterations -> 300 calls. That's too many for one run if we want restarts.
        # Let's cap each run to ~50-80 calls.
        
        max_calls_per_run = 60 
        
        for restart_idx in range(10): # Max 10 restarts
            if remaining_budget <= 0:
                break
            if time.time() > deadline:
                break
                
            # Choose initial point
            # First restart: random uniform
            # Subsequent: random uniform
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Wrap function to stop after max_calls_per_run OR global budget
            local_counter = 0
            def local_func(x):
                nonlocal local_counter, remaining_budget
                local_counter += 1
                if local_counter > max_calls_per_run or remaining_budget <= 0:
                    raise ValueError("Local budget exceeded")
                remaining_budget -= 1
                return func(x)
            
            try:
                res = minimize(local_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_calls_per_run, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # Budget exceeded or other error, break out
                pass

            # If we used all budget, break
            if remaining_budget <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
