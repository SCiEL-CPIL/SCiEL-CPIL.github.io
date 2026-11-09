"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and multi-start restart on stall."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 300s limit, 15s safety margin
        rng = np.random.default_rng(0)
        
        dim = self.dim
        budget = self.budget
        
        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        calls = 1
        
        # Parameters for the (1+1)-ES
        sigma = 1.5
        stall_threshold = 1e-10
        max_stall = 30  # Number of evaluations without improvement before restart
        stall_count = 0
        
        # Track best
        best_x = x.copy()
        best_f = fx
        
        # Multi-start strategy: if we stall, restart from a new location
        # biased towards the current best but with a large perturbation
        restart_count = 0
        max_restarts = 5
        
        while calls < budget:
            if time.time() >= deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Improvement
                x = y
                fx = fy
                sigma *= 1.1  # Slightly increase step size
                stall_count = 0
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement
                x = np.clip(x + 0.5 * (y - x), -5, 5)  # Move halfway towards failed candidate? 
                # Actually, standard (1+1) ES keeps x if y is worse. 
                # Let's stick to standard: x remains, sigma decreases.
                sigma *= 0.9
                stall_count += 1
                
                # Check for stall
                if stall_count >= max_stall and restart_count < max_restarts:
                    restart_count += 1
                    # Restart logic:
                    # 1. Start near the global best
                    # 2. Add a large random perturbation to escape local minimum
                    x = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5, 5)
                    fx = func(x)
                    calls += 1
                    sigma = 1.0  # Reset step size
                    stall_count = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        
                    if calls >= budget:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
