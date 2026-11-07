"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES for Büche-Rastrigin with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # (1+1)-ES parameters
        sigma = 0.5  # Initial step size
        sigma_min = 1e-6
        sigma_max = 5.0
        alpha = 0.2  # Step size adaptation rate
        restart_threshold = 100  # Number of steps without improvement to restart
        
        best_x_global = None
        best_f_global = np.inf
        
        calls = 0
        
        # Initial random point to establish a baseline
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls += 1
        best_x_global = x0.copy()
        best_f_global = f0
        
        # Multi-start loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Start a new local search from a random point or near the best
            # Strategy: 50% chance to start near best, 50% chance to start random
            # This balances exploitation and exploration
            if rng.random() < 0.5 and best_x_global is not None:
                x = best_x_global + rng.normal(0, sigma, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            
            f = func(x)
            calls += 1
            
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
            
            # Local (1+1)-ES loop from this starting point
            steps_without_improvement = 0
            current_best_f = f
            current_best_x = x.copy()
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                dx = rng.normal(0, sigma, dim)
                x_new = x + dx
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls += 1
                
                if f_new < current_best_f:
                    # Success: move to new point, increase step size slightly
                    x = x_new
                    current_best_f = f_new
                    current_best_x = x_new.copy()
                    sigma = min(sigma * (1 + alpha), sigma_max)
                    steps_without_improvement = 0
                    
                    # Update global best
                    if f_new < best_f_global:
                        best_f_global = f_new
                        best_x_global = x_new.copy()
                else:
                    # Failure: stay, decrease step size
                    sigma = max(sigma * (1 - alpha), sigma_min)
                    steps_without_improvement += 1
                    
                    # If stuck too long, break out to start a new search
                    if steps_without_improvement >= restart_threshold:
                        break
            
            # If we broke out due to stagnation, we might want to adjust global sigma
            # but since we reset x for the next outer loop iteration, we keep sigma adaptive
            # or reset it to a moderate value to explore from the new random point.
            # Let's reset sigma to a medium value for the next start to ensure exploration.
            sigma = 0.5
            
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
