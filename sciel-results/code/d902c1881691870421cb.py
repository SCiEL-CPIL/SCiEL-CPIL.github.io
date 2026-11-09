import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """(1+1)-ES with Sobol initialization, adaptive step size, and progress-based restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Use Sobol for better initial spread
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(50) * 10 - 5 # Map to [-5, 5]
        sobol_idx = 0
        
        remaining = budget
        restart_count = 0
        max_restarts = 30 # Cap restarts
        
        while remaining > 0 and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            # Choose start point
            if sobol_idx < len(sobol_points):
                x = sobol_points[sobol_idx].copy()
                sobol_idx += 1
            else:
                x = rng.uniform(-4.5, 4.5, dim)
                
            fx = func(x)
            remaining -= 1
            if remaining < 0:
                break
                
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            # Local Search (1+1)-ES
            sigma = 1.5 # Start with a slightly larger step
            no_improve_count = 0
            stall_threshold = 30 # If no improvement in 30 steps, restart
            local_steps = 0
            max_local_steps = 100 # Cap local steps to avoid wasting budget on one run
            
            while local_steps < max_local_steps and remaining > 0:
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                local_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    # Increase step size on success
                    sigma *= 1.1
                    no_improve_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    # Decrease step size on failure
                    sigma *= 0.9
                    no_improve_count += 1
                    
                    # If stuck, break out to restart
                    if no_improve_count > stall_threshold:
                        break
            
            restart_count += 1
            
            # If we've used up budget, break
            if remaining <= 0:
                break

        # Final Polish: A very short, high-precision run from the best found
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
