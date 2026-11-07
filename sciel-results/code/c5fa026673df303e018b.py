"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with deadline guard for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Multi-start strategy: divide budget among restarts
        # Büche-Rastrigin has many local minima, so restarts help
        # Use 10 restarts. We must account for the init call in each restart.
        # Total calls = sum(init_calls + opt_calls).
        # To be safe, we track global budget.
        n_restarts = 10
        # Allocate budget roughly evenly, but let the loop handle exact limits.
        # Each restart consumes at least 1 call for init.
        # Let's reserve 1 call per restart for init, and distribute the rest.
        # Remaining budget for optimization: budget - n_restarts (approx)
        # But it's safer to just check global budget inside the loop.
        
        budget_per_restart = max(1, budget // n_restarts)

        for r in range(n_restarts):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Check budget
            if calls_made >= budget:
                break

            # Initialize: for first restart, try center; for others, random
            if r == 0:
                x = np.zeros(dim)
            else:
                # Random restart in [-4, 4] to avoid boundary issues
                x = rng.uniform(-4, 4, dim)

            # Check budget before init call
            if calls_made >= budget:
                break
                
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.0
            min_sigma = 0.001
            max_sigma = 5.0

            for i in range(budget_per_restart):
                # Check deadline
                if time.time() > deadline:
                    return best_x, best_f
                
                # Check budget
                if calls_made >= budget:
                    return best_x, best_f

                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                # Check budget before call
                if calls_made >= budget:
                    return best_x, best_f

                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma = min(sigma * 1.2, max_sigma)
                else:
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.85, min_sigma)

                # If sigma is very small, we've likely converged to a local min
                if sigma <= min_sigma * 1.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
