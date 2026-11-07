"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: random restarts + adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Random sampling to find promising regions
        # Use ~20% of budget for exploration
        n_explore = max(10, budget // 5)
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
        
        # Sort candidates and pick top ones for local optimization
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization from best starting points
        # Use remaining budget for exploitation
        remaining_budget = budget - n_explore
        n_local = min(3, len(candidates))  # Optimize from up to 3 best starts
        
        calls_made = n_explore
        for start_idx in range(n_local):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            x = candidates[start_idx][1].copy()
            fx = candidates[start_idx][0]
            sigma = 1.0
            no_improve = 0
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve > 20:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
