"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        
        # Phase 1: Global multi-start initialization
        n_starts = 20
        candidates = []
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            candidates.append((fx, x))
        
        # Sort by fitness and keep top 5
        candidates.sort(key=lambda t: t[0])
        top5 = candidates[:5]
        
        # Phase 2: Run (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly
        calls_used = n_starts
        remaining_budget = self.budget - calls_used
        per_start_budget = remaining_budget // 5
        
        best_x = top5[0][1]
        best_fx = top5[0][0]
        
        for i in range(5):
            x = top5[i][1].copy()
            fx = top5[i][0]
            sigma = 1.0
            
            for _ in range(per_start_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            
            if time.time() > deadline:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
