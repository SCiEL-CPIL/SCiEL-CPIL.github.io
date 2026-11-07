"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time
        start_time = time.time()
        deadline = start_time + 290.0

        # Phase 1: Multi-start Initialization
        # The minimizer is in [-4, 4]^5. We sample this region.
        # We use 20% of the budget for initialization (max 100 calls, min 10)
        init_budget = max(10, min(100, budget // 5))
        
        candidates = []
        fvals = []
        
        # Always try the origin and center
        candidates.append(np.zeros(dim))
        candidates.append(rng.uniform(-4, 4, dim))
        
        # Fill the rest with random points in [-4, 4]
        for _ in range(init_budget - 2):
            candidates.append(rng.uniform(-4, 4, dim))
            
        best_x = None
        best_f = np.inf
        calls_made = 0

        for x_cand in candidates:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            x_cand = np.clip(x_cand, -5, 5)
            f_cand = func(x_cand)
            calls_made += 1
            
            if f_cand < best_f:
                best_f = f_cand
                best_x = x_cand.copy()

        # Phase 2: (1+1)-ES Refinement
        if best_x is None:
            # Fallback if init failed or budget exhausted
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            
        x = best_x
        fx = best_f
        sigma = 0.5 # Start with a smaller step since we are in a good region
        
        # Tuned adaptation rates for Weierstrass
        # Faster convergence than 1.5/0.87, more stable
        sigma_inc = 1.2
        sigma_dec = 0.85
        
        for _ in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec
                
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10.0:
                sigma = 10.0

        return x, fx
# EVOLVE-BLOCK-END
