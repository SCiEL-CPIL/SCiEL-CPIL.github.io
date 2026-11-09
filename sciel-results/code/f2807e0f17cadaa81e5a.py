"""Optimized program for BBOB f24 (Lunacek bi-Rastrigin) using Multi-Restart (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Restart (1+1)-ES with adaptive step size and seeding."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Initial random sampling to find a good starting point
        # Use ~10% of budget for initial exploration
        n_init = max(10, budget // 10)
        
        for _ in range(n_init):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Multi-Restart (1+1)-ES
        # Define restart budgets. Sum should be <= budget - n_init
        remaining_budget = budget - calls_made
        
        # Schedule of restart budgets
        # We want several restarts to escape local minima.
        # Geometric decay or fixed steps.
        # Let's use a schedule that allows for ~5-6 restarts.
        # Example: 150, 100, 80, 60, 50, 30 (sum=470)
        # Adjust to fit remaining_budget.
        restart_budgets = [150, 100, 80, 60, 50, 30]
        
        # Scale restart budgets if remaining is less than sum
        total_restart_budget = sum(restart_budgets)
        if remaining_budget < total_restart_budget:
            # Proportional scaling
            scale_factor = remaining_budget / total_restart_budget
            restart_budgets = [int(b * scale_factor) for b in restart_budgets]
            # Ensure at least 10 for each
            restart_budgets = [max(10, b) for b in restart_budgets]
            # If still too much, truncate
            current_sum = sum(restart_budgets)
            if current_sum > remaining_budget:
                # Reduce the last one
                restart_budgets[-1] -= (current_sum - remaining_budget)
                if restart_budgets[-1] < 5:
                    restart_budgets = restart_budgets[:-1]

        for r, restart_budget in enumerate(restart_budgets):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            if restart_budget <= 0:
                break

            # Initialize x for this restart
            if r == 0:
                # First restart: start from the best found in Phase 1
                x = best_x.copy()
                fx = best_f
                sigma = 1.5
            else:
                # Subsequent restarts: seed near global best with perturbation
                # Perturbation scale decreases slightly for later restarts?
                # Or keep it constant to ensure escape capability.
                # Let's use a scale of 1.0 * sqrt(dim) to allow significant jumps.
                perturb_scale = 1.0 * np.sqrt(dim)
                x = best_x + perturb_scale * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
            
            # Run (1+1)-ES for this restart
            for i in range(restart_budget - 1):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                # Update step size and current best
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # Increase step size on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85  # Decrease step size on failure
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)
            
            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
