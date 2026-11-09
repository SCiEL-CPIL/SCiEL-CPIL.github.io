"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time guard: must finish within 300s, use 290s as safety margin
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        # Strategy: Multiple restarts with (1+1)-ES
        # Each restart gets a portion of the budget
        # We'll do adaptive restarts: if no improvement for a while, restart
        
        # Initial: try a few random points to get a sense of scale
        # Then use (1+1)-ES with sigma adaptation
        
        # Allocate budget into chunks for restarts
        # If we detect stagnation, restart from a new random point or from best so far
        
        max_steps_per_restart = 100
        min_steps_per_restart = 20
        
        while calls_made < budget and time.time() < deadline:
            # Choose starting point
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                # Mix: sometimes restart from scratch, sometimes perturb best
                if rng.random() < 0.3:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5, 5)
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            no_improve = 0
            max_no_improve = 30
            
            while calls_made < budget and time.time() < deadline:
                # Check time more frequently
                if (calls_made % 50 == 0) and time.time() > deadline - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    # Increase sigma slightly on success
                    sigma = min(sigma * 1.3, 10.0)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.8, 0.01)
                
                # Restart if stuck
                if no_improve >= max_no_improve:
                    break
                
                # Also restart if we've done enough steps
                if calls_made >= budget:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
