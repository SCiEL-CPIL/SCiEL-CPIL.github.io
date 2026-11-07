import numpy as np
import time

class Optimizer:
    """
    (1+1)-ES with 1/5 success rule, randomized simplex initialization,
    and a final polishing phase for BBOB f13 (sharp ridge).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        # Parameters for 1/5 success rule
        window_size = 10
        successes = []
        sigma = 2.0
        
        # Randomized simplex initialization: try a few random starting points
        # to better align with rotated ridges
        n_init = 5
        init_budget = min(n_init, budget // 10)
        
        for i in range(init_budget):
            if total_evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            total_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Use the best initial point as starting point for ES
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        successes = []
        
        # Track stagnation for restarts
        no_improve_count = 0
        max_no_improve = 30
        
        # Main loop
        while total_evals < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_evals += 1
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update current best for ES
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
            else:
                successes.append(0)
            
            # Adapt sigma based on 1/5 success rule
            if len(successes) >= window_size:
                successes = successes[-window_size:]
                success_rate = np.mean(successes)
                
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 1.0 / 1.1
                
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart if stuck
            if no_improve_count >= max_no_improve and sigma < 0.5:
                # Restart from a random point near the best
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                total_evals += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    no_improve_count = 0
                sigma = 1.5
                successes = []
                no_improve_count = 0
        
        # Final polishing phase: local refinement around best_x
        if total_evals < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.5, 0.05)
            polish_budget = min(budget - total_evals, 50)
            for i in range(polish_budget):
                if time.time() > deadline or total_evals >= budget:
                    break
                y = best_x + polish_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_evals += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                # Gradually reduce polish sigma
                polish_sigma *= 0.95
        
        return best_x, best_f
