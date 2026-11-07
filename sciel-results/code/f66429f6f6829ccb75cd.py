"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with random restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Initial Exploration
        # Sample a few points to find a good starting region.
        # This helps with early progress in AOCC.
        n_explore = min(10, budget // 10)
        if n_explore < 1:
            n_explore = 1
            
        for _ in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with Restarts
        # We run (1+1)-ES from the best point found in Phase 1.
        # If it stagnates, we restart from a new random point.
        
        current_x = best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim)
        current_f = best_f if best_f != np.inf else func(current_x)
        if best_f == np.inf:
            calls_made += 1

        sigma = 1.0
        no_improve = 0
        min_sigma = 1e-4
        max_no_improve = 30 # Restart if no improvement for 30 steps

        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(current_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < current_f:
                # Accept
                current_x = y
                current_f = fy
                no_improve = 0
                sigma *= 1.5 # Increase step size on success
                
                # Update global best
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
            else:
                # Reject
                sigma *= 0.8 # Decrease step size on failure
                no_improve += 1
                
            # Check for stagnation/convergence
            if sigma < min_sigma or no_improve > max_no_improve:
                # Restart strategy:
                # 1. If we have used less than 50% of budget, pick a completely new random point.
                # 2. Otherwise, perturb the current global best slightly to escape local minima.
                
                if calls_made < budget * 0.5:
                    current_x = rng.uniform(-5, 5, dim)
                    current_f = func(current_x)
                    calls_made += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                else:
                    # Perturb global best
                    current_x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                    current_f = func(current_x)
                    calls_made += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                
                # Reset parameters for new search
                sigma = 1.0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
