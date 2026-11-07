"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        
        # Try starting at origin first (often closer to Lunacek minimum)
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Also evaluate a random point to have a backup
        x_rand = rng.uniform(-5, 5, self.dim)
        fx_rand = func(x_rand)
        
        if fx_rand < fx:
            x, fx = x_rand, fx_rand
        
        sigma = 1.5
        no_improve = 0
        max_no_improve = 50
        best_x = x.copy()
        best_fx = fx
        calls = 2
        
        for _ in range(self.budget - calls):
            if time.time() - start_time > 280:
                break
            if calls >= self.budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.15, 4.0)
                no_improve = 0
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                sigma = max(sigma * 0.92, 0.1)
                no_improve += 1
                
                # Restart if stuck
                if no_improve >= max_no_improve:
                    no_improve = 0
                    # Restart from best with larger sigma
                    x = best_x + sigma * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    sigma = max(sigma * 1.5, 2.0)
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
