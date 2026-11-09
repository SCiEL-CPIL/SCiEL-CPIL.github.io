"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        # Use different seeds per call to diversify across instances
        call_seed = hash(id(func)) % (2**31)
        rng = np.random.default_rng(call_seed)
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        # Multi-start: try a few random starting points
        n_starts = 3
        per_start_budget = self.budget // n_starts
        
        for s in range(n_starts):
            if time.time() > deadline:
                break
            
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()
            
            sigma = 3.0
            window = []
            window_size = 10
            
            for _ in range(per_start_budget - 1):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                    window.append(1)
                else:
                    window.append(0)
                
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    window = window[-window_size:]
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
