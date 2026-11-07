import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        # Parameters for the (1+1)-ES
        # We will use a stall-based restart strategy
        stall_limit = 30  # If no improvement in 30 steps, consider restarting
        restart_sigma = 2.0
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
                
            # Initialize a new trajectory
            # If we have a previous best, we could start from there, 
            # but for multi-modal, random is often better to escape.
            # Let's alternate: first try random, subsequent restarts random too.
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            sigma = 1.0
            stall_count = 0
            local_best_fx = fx
            
            # Run (1+1)-ES for this trajectory
            while calls_made < total_budget:
                if time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    stall_count = 0
                    # Adaptive sigma: increase on success
                    sigma *= 1.1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = x.copy()
                else:
                    # Decrease on failure
                    sigma *= 0.9
                    stall_count += 1
                    
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)
                
                # Check for stagnation
                if stall_count >= stall_limit:
                    # Break out of this trajectory and restart
                    break

        if best_x is None:
            # Fallback: should not happen if budget > 0
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            
        return best_x, best_fx
