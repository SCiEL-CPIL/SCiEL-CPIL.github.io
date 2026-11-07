"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for BBOB-style Schwefel function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Multi-start strategy: try several random starting points
        # For 5D Schwefel, the global minimum is at x_i = 4.20968...
        # But since there's a hidden shift, we need to search broadly.
        
        # Number of restarts: use most of the budget for the best run
        # Reserve some calls for initial exploration
        n_starts = 5
        budget_per_start = budget // n_starts
        last_start_budget = budget - (n_starts - 1) * budget_per_start
        
        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            
            # Choose starting point
            if start_idx == 0:
                # First start: center of the box (origin)
                x = np.zeros(dim)
            else:
                # Subsequent starts: random points in [-4, 4]^5
                x = rng.uniform(-4.0, 4.0, size=dim)
            
            # Evaluate starting point
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES with adaptive step size
            sigma = 2.0  # Initial step size
            current_x = x.copy()
            current_f = fx
            calls_in_this_start = 1
            
            # Adaptive step size: increase if successful, decrease if not
            success_count = 0
            total_attempts = 0
            target_success_rate = 0.2  # Typical for (1+1)-ES
            
            while calls_in_this_start < budget_per_start and calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate one trial point
                direction = rng.normal(0, 1, size=dim)
                trial_x = current_x + sigma * direction
                
                # Evaluate
                trial_f = func(trial_x)
                calls_made += 1
                calls_in_this_start += 1
                total_attempts += 1
                
                if trial_f < current_f:
                    current_x = trial_x.copy()
                    current_f = trial_f
                    success_count += 1
                    
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial_x.copy()
                else:
                    pass  # No update to current_x
                
                # Adapt step size every 10 attempts
                if total_attempts % 10 == 0:
                    success_rate = success_count / total_attempts
                    if success_rate > target_success_rate:
                        sigma *= 1.2
                    elif success_rate < target_success_rate / 2:
                        sigma *= 0.8
                    success_count = 0
                    total_attempts = 0
                
                # If sigma becomes too small or too large, restart this start
                if sigma < 1e-4 or sigma > 10.0:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
