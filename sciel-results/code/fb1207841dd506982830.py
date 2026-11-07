import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        start_time = time.time()
        # Safety margin: 330s limit, use 325s to be safe
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # (1+1)-ES with restarts
        def es_1plus1_with_restarts():
            nonlocal calls_made, best_f, best_x, deadline, total_budget

            # Initial point
            x = np.zeros(dim)
            fx = wrapper(x)
            
            sigma = 1.0
            min_sigma = 1e-8
            max_sigma = 5.0
            
            steps_since_improvement = 0
            stagnation_limit = 30 # Restart if no improvement for this many steps
            
            # We will run until budget or time is out
            while calls_made < total_budget and time.time() < deadline - 2:
                # Generate trial point
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                
                fx_trial = wrapper(x_trial)
                
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Success: increase step size
                    sigma *= 1.1
                    steps_since_improvement = 0
                else:
                    # Failure: decrease step size
                    sigma *= 0.9
                    steps_since_improvement += 1
                
                # Clip sigma
                sigma = np.clip(sigma, min_sigma, max_sigma)
                
                # Check for stagnation
                if steps_since_improvement >= stagnation_limit:
                    # Restart from a random point
                    # This helps escape local minima
                    x = rng.uniform(-4.0, 4.0, dim)
                    fx = wrapper(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Reset step size for exploration
                    sigma = 1.0
                    steps_since_improvement = 0

        es_1plus1_with_restarts()
        
        return best_x, best_f
