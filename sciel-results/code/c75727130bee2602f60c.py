import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # Strategy: Multi-start (1+1)-ES with aggressive restarts
        # f22 has 21 peaks. (1+1)-ES is fast and can converge quickly to local minima.
        # By restarting frequently when progress stalls, we explore more peaks.
        # This is "materially different" from CMA-ES which maintains population/covariance 
        # and tends to get stuck in local basins due to its robustness to noise (which we don't have).
        
        rng = np.random.default_rng(42)
        
        global_best_x = None
        global_best_f = float('inf')
        
        # We will run multiple independent (1+1)-ES instances.
        # Each instance gets a small budget. If it doesn't improve significantly, we restart.
        
        # Parameters for (1+1)-ES
        # sigma: step size
        # success_ratio: target success rate (0.5 is standard, but we might tune)
        
        # How to allocate budget?
        # Instead of fixed 5 restarts, let's do adaptive restarts.
        # Start with a few large steps, then refine.
        
        # Let's try a "Levy Flight" inspired or simple random restart hill climbing.
        # Actually, (1+1)-ES is very effective if the step size adapts well.
        
        # Key insight from inspirations: "Multi-start Random Hill Climbing with Adaptive Step Sizes"
        
        # Let's implement a robust (1+1)-ES with the following loop:
        # 1. Pick a start point (random or best found so far with perturbation).
        # 2. Run (1+1)-ES for a fixed number of steps (e.g., 50).
        # 3. If the best f in this run is significantly better than global best, continue refining.
        # 4. If not, restart from a new random point.
        
        # To maximize AOCC, early progress is crucial.
        # So we should find *some* good point quickly, then refine.
        
        # Let's do 10 restarts of 50 calls each.
        n_restarts = 10
        calls_per_restart = total_budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize start point
            # For the first restart, random. For subsequent, maybe near global best?
            # No, for global exploration, random is better to hit different peaks.
            # But if we are already very good, we should exploit.
            
            # Let's use a mix: 50% chance to start near global best (exploit), 50% random (explore).
            # But at the beginning, global best is poor, so random is good.
            
            if global_best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                if rng.random() < 0.5:
                    # Exploit: start near best
                    x = global_best_x + rng.normal(0, 1.0, dim)
                else:
                    # Explore: start random
                    x = rng.uniform(-5, 5, dim)
            
            x = np.clip(x, -5, 5)
            fx = func(x)
            
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            
            # (1+1)-ES parameters
            sigma = 1.0
            successes = 0
            total_steps = 0
            best_f_run = fx
            best_x_run = x.copy()
            
            # Run for calls_per_restart steps
            for i in range(calls_per_restart):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                # Update success counter
                total_steps += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.5  # Increase step size if success (aggressive)
                    
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                    if fx < best_f_run:
                        best_f_run = fx
                        best_x_run = x.copy()
                else:
                    # Failure
                    sigma *= 0.7  # Decrease step size if failure
                    # Keep x as is
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Early exit if sigma is very small (converged locally)
                if sigma < 1e-4:
                    break
            
            # Optional: If we converged very well, we might want to spend more time here?
            # But the fixed budget allocation is simpler and ensures global exploration.
            
        return global_best_x, global_best_f
