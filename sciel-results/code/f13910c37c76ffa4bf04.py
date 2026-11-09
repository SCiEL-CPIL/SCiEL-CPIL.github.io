"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with stagnation detection for Rastrigin-type functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_calls = 0
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        
        # Stagnation threshold: if no improvement for this many steps, consider restarting
        stagnation_limit = 30
        
        # Phase 1: Start from origin (Rastrigin's unshifted minimum is at origin)
        # Then use (1+1)-ES with adaptive sigma
        x = np.zeros(dim)
        fx = func(x)
        total_calls += 1
        best_x = x.copy()
        best_f = fx
        
        sigma = 2.0
        no_improve = 0
        
        # Main optimization loop with stagnation-based restarts
        while total_calls < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma = min(sigma * 1.3, 5.0)  # Increase sigma on success, cap at 5
                no_improve = 0
            else:
                sigma = max(sigma * 0.85, 0.05)  # Decrease sigma on failure, floor at 0.05
                no_improve += 1
            
            # Track global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Stagnation detection: if no improvement for stagnation_limit steps
            # and sigma is small, restart from a random point
            if no_improve >= stagnation_limit and sigma < 0.5:
                # Restart from a random point, but keep the best found so far
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                total_calls += 1
                sigma = 2.0
                no_improve = 0
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
