"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """Multi-Start (1+1)-ES with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Initial Random Sampling ---
        # Sample a few points to get a decent starting point
        best_x = clip(rng.uniform(-5, 5, d))
        best_f = float(func(best_x))
        calls += 1
        
        # 4 more random samples
        for _ in range(4):
            if calls >= budget or time.time() > deadline:
                break
            x = clip(rng.uniform(-5, 5, d))
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # --- Phase 2: Multi-Start (1+1)-ES ---
        # We will run several (1+1)-ES instances.
        # Each instance gets a portion of the budget.
        # If an instance stagnates, we restart.
        
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 5.0
        success_rate = 0.5 # Initial guess
        target_success_rate = 0.2 # Aim for 20% success rate (common in ES)
        
        # Adaptation parameters
        adapt_step = 0.2
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate candidate
            # Use a normal distribution scaled by sigma
            # To handle the box constraint, we clip.
            y = clip(best_x + sigma * rng.standard_normal(d))
            fy = float(func(y))
            calls += 1
            
            if fy < best_f:
                best_f, best_x = fy, y.copy()
                success_rate = 0.8 * success_rate + 0.2 * 1.0 # Update success rate
            else:
                success_rate = 0.8 * success_rate + 0.2 * 0.0
            
            # Adapt sigma based on success rate
            # If success rate is too high, increase sigma (search too local)
            # If success rate is too low, decrease sigma (search too global)
            if success_rate > target_success_rate:
                sigma *= (1 + adapt_step)
            elif success_rate < target_success_rate:
                sigma *= (1 - adapt_step)
            
            # Clip sigma
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Stagnation detection and restart
            # If sigma is very small and no progress, restart from a random point
            # or reset sigma to a larger value to escape local minima.
            # Simple heuristic: if sigma < 0.01, reset sigma to 1.0 and maybe perturb best_x
            if sigma < 0.05:
                # Reset sigma
                sigma = 1.0
                success_rate = 0.5
                # Optional: Perturb best_x slightly or restart from random
                # Let's try a small perturbation to escape
                best_x = clip(best_x + 0.5 * rng.standard_normal(d))
                best_f = float(func(best_x))
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
