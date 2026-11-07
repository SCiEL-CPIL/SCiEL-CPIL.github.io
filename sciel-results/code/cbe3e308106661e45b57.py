"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES with strict budget control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 290
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # We use a multi-start strategy.
        # In 5D, CMA-ES with a population size of ~10-20 is efficient.
        # Let's allocate budget to multiple independent CMA-ES runs.
        # Each run uses a portion of the budget.
        
        # Strategy: 
        # 1. Start with a few global random starts.
        # 2. Use CMA-ES logic manually to strictly control calls.
        #    However, implementing full CMA-ES from scratch is complex and error-prone.
        #    Alternative: Use a robust variant of (1+lambda)-ES or a simplified CMA-ES.
        #    
        #    Given the constraint "Python standard library, NumPy and SciPy are available",
        #    and the need for strict call counting, using scipy.optimize.minimize with 'CMA'
        #    is risky if it doesn't respect the callback limit exactly or if it's too slow.
        #    
        #    Let's implement a simplified, robust (1+1)-ES with adaptive step size 
        #    but with **multiple restarts** and **diverse initializations**.
        #    The previous score was 0.5195 with a grid+walk. 
        #    Inspiration 1 (1.1971) likely used a better local search or more restarts.
        #    
        #    Let's try a "Hill Climbing with Adaptive Step and Restarts" approach.
        #    Key improvement over current: 
        #    1. No coarse grid (wastes budget on bad areas).
        #    2. Start with a random point, run local search.
        #    3. If stuck or budget allows, restart from a new random point or 
        #       a point perturbed from the current best.
        #    4. Use a more sophisticated step size adaptation (e.g., based on success rate).

        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Initialize current run
            # First restart: random. Subsequent: perturb best found so far.
            if r == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Perturb the global best to jump out of local minima
                # Use a larger step for restarts
                x = np.clip(best_x + rng.uniform(-2, 2, self.dim), -5, 5)
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local search parameters
            sigma = 1.0
            # Adaptive step size: 
            # If we improve, keep or increase sigma slightly.
            # If we don't improve, decrease sigma.
            # To avoid getting stuck in very narrow basins of Rastrigin, 
            # we should not shrink sigma too fast.
            
            local_calls = 0
            max_local_calls = budget_per_restart
            
            while calls_made < self.budget and local_calls < max_local_calls and time.time() < deadline:
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                local_calls += 1
                
                if fy < fx:
                    # Success
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase step size slightly if we are making good progress
                    # But be careful not to overshoot
                    sigma *= 1.05 
                else:
                    # Failure
                    # Decrease step size
                    sigma *= 0.85
                    
                # If sigma becomes too small, we are likely stuck in a local minimum
                # or at the optimum. Restart helps here.
                if sigma < 0.05:
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
