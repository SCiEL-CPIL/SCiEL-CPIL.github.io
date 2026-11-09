"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        
        # Use 5 restarts to increase the chance of finding a better basin
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(0)
        
        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
                
            # Check if we have enough budget left for a meaningful run
            remaining_budget = total_budget - (r * budget_per_restart)
            if remaining_budget <= 0:
                break
            
            # Use slightly less than the full allocated budget to account for overhead
            # and ensure we don't exceed total budget
            current_budget = min(budget_per_restart, remaining_budget)
            
            # Initialize
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            
            # If this start is better than the global best, update
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # (1+1)-ES loop
            sigma = 2.0
            success_count = 0
            window_size = 10
            window = []
            
            for i in range(current_budget - 1):
                if time.time() > deadline:
                    break
                    
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                
                # Update
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    # Update global best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    success_count -= 1
                    
                # 1/5 success rule with window
                window.append(1 if fy < fx else 0)
                if len(window) > window_size:
                    window.pop(0)
                
                # Adjust sigma based on success rate in window
                if len(window) == window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    
                # Prevent sigma from exploding or vanishing
                if sigma > 10.0: sigma = 10.0
                if sigma < 1e-8: sigma = 1e-8
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
