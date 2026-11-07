import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        dim = self.dim
        budget = self.budget
        
        # Counters
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Helper to evaluate and update best
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            # Time check
            if time.time() - start_time > 280: # Safety margin
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Scan using Sobol Sequence
        # Allocate ~30% of budget to global scan
        scan_budget = max(50, budget // 3)
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=True)
        # We need scan_budget points
        # Sobol requires power of 2 for optimal distribution, but we can just generate
        # and take the first N.
        # Actually, sampler.random(n) works for any n.
        points = sampler.random(scan_budget)
        # Scale from [0, 1] to [-5, 5]
        points = points * 10.0 - 5.0
        
        for p in points:
            if time.time() - start_time > 280:
                break
            fx = eval_func(p)
            if fx is None:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            return np.zeros(dim), np.inf

        # Phase 2: Local Refinement using (1+1)-ES
        # Start from best point found
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma: estimate from the spread of the scan? 
        # Or just a fixed reasonable value.
        # The domain is [-5, 5]. A step of 1.0 is reasonable.
        sigma = 1.0
        
        # 1/5 success rule variables
        successes = 0
        n_successes = 0
        window_size = 15 # Adapt every 15 steps
        
        remaining_budget = budget - calls_made
        
        for i in range(remaining_budget):
            if time.time() - start_time > 280:
                break
            
            # Generate candidate
            step = sigma * np.random.randn(dim)
            y = np.clip(x + step, -5, 5)
            
            fy = eval_func(y)
            if fy is None:
                break
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            else:
                successes += 0
                
            n_successes += 1
            
            if n_successes == window_size:
                # Adapt sigma
                if successes / n_successes > 0.2: # More than 1/5 success rate
                    sigma *= 1.5
                else:
                    sigma *= 0.75
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-4, 10.0)
                
                successes = 0
                n_successes = 0

        return best_x, best_f
