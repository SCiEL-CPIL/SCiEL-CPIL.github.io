import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # We will use a multi-start strategy.
        # Split budget into several segments for different restarts.
        # Let's do 5 restarts.
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        # Keep track of total calls to ensure we don't exceed budget
        total_calls = 0
        
        for r in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Local search using (1+1)-ES for this restart
            x = x0
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0  # Start with a larger sigma to explore
            successes = 0
            attempts = 0
            window_size = 10
            
            # Run for budget_per_restart steps
            for _ in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                
                attempts += 1
                
                if attempts % window_size == 0:
                    success_rate = successes / attempts
                    if success_rate > 0.2:
                        sigma *= 1.5
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0
                    attempts = 0
        
        # Use remaining budget to polish the best solution
        remaining = budget - total_calls
        if remaining > 0 and best_x is not None:
            if time.time() < deadline:
                # Run a final (1+1)-ES from the best point
                x = best_x.copy()
                fx = best_f
                sigma = 0.5  # Smaller sigma for polishing
                
                successes = 0
                attempts = 0
                window_size = 5
                
                for _ in range(remaining - 1):
                    if time.time() > deadline:
                        break
                    if total_calls >= budget:
                        break
                        
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        successes += 1
                        if fy < best_f:
                            best_f = fy
                            best_x = x.copy()
                    
                    attempts += 1
                    
                    if attempts % window_size == 0:
                        success_rate = successes / attempts
                        if success_rate > 0.2:
                            sigma *= 1.2
                        elif success_rate < 0.2:
                            sigma *= 0.8
                        sigma = np.clip(sigma, 0.001, 1.0)
                        successes = 0
                        attempts = 0

        return best_x, best_f
