"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+4)-ES with 1/5 success rule and multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Deadline guard: 330s limit, use 320s for safety
        start_time = time.time()
        deadline = start_time + 320.0
        
        # Phase 1: Multi-start initialization
        # Use 5 random starts to find a good initial basin
        n_starts = 5
        best_x = None
        best_f = float('inf')
        
        for _ in range(n_starts):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if time ran out during init (unlikely)
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            
        calls_made = n_starts
        x = best_x
        fx = best_f
        sigma = 1.5 # Initial step size
        
        # Phase 2: (1+4)-ES
        lambda_ = 4
        # Adaptation parameters
        sigma_inc = 1.1
        sigma_dec = 0.9
        # 1/5 rule: if more than 1/5 of offspring are better, increase sigma
        threshold = 1/5.0
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            successes = 0
            best_offspring_x = None
            best_offspring_f = float('inf')
            
            for _ in range(lambda_):
                if time.time() > deadline:
                    break
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    successes += 1
                if fy < best_offspring_f:
                    best_offspring_f = fy
                    best_offspring_x = y.copy()
            
            # Update best
            if best_offspring_f < fx:
                x = best_offspring_x
                fx = best_offspring_f
                
            # Adapt sigma
            p_succ = successes / lambda_
            if p_succ > threshold:
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-4, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
