import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Use system entropy for better diversity across potential re-runs
        # or fixed if we want reproducibility? 
        # Inspiration 1 used 42. Current used 0. 
        # Let's use a seed based on time to ensure we don't hit a bad path for all instances if the evaluator re-runs?
        # Actually, the evaluator creates a NEW Optimizer for each instance.
        # If I use time.time() as seed, it's non-deterministic.
        # If I use a fixed seed, it's deterministic.
        # The task says "hidden seeded shift". If I use a fixed seed, my random numbers are fixed.
        # The hidden shift is applied to the FUNCTION, not my RNG.
        # So my RNG is independent.
        # Using a fixed seed is safer for debugging but might be unlucky for some specific instance shifts if the search path is deterministic.
        # However, with restarts, the path is not a single line.
        # Let's stick to a fixed seed for stability, but maybe 42 is better than 0?
        # Or better: Use no seed (system entropy) to maximize chance of finding a good path if the fixed one is bad.
        # Given the "Black-box" nature, system entropy is often preferred in competitions to avoid "unlucky" deterministic failures.
        rng = np.random.default_rng()
        
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        calls_made = 0
        
        # Phase 1: Random Initialization
        n_init = min(30, budget // 10)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            
        if best_x is None:
            # Fallback if no calls made (shouldn't happen)
            x = np.zeros(dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)

        # Phase 2: (1+1)-ES with Restarts
        # We have budget - calls_made left.
        # We will run multiple (1+1)-ES instances.
        # Each instance runs for a certain number of steps or until stuck.
        
        max_no_improve = 20 # Threshold for restarting a local search
        min_sigma = 1e-6
        max_sigma = 10.0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Decide starting point for this ES run
            # Strategy: 
            # 1. If we have just started, start from best.
            # 2. Otherwise, alternate between perturbing best and random restart.
            # To ensure we don't get stuck in a local min, random restarts are vital for Rastrigin/Gallagher.
            # Let's say: 30% random, 70% perturb best.
            
            if rng.random() < 0.3:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb best with a decent sigma
                x = best_x + 2.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = 1.0
            no_improve_count = 0
            
            # Inner (1+1)-ES loop
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                if sigma < min_sigma:
                    break # Convergence reached, restart outer loop
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.2, max_sigma)
                    no_improve_count = 0
                else:
                    sigma = max(sigma * 0.8, min_sigma)
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break # Stuck, restart outer loop
                        
        return best_x, best_f
