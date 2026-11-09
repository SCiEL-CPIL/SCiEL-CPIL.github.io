import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Lattice Search (3^5 = 243 points)
        grid_vals = [-5.0, 0.0, 5.0]
        grid_points = [np.array(combo) for combo in itertools.product(grid_vals, repeat=n)]
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1
            
        # Phase 2: Adaptive (1+1)-ES with success-rate step size control
        x = best_x.copy()
        f = best_f
        sigma = 0.8  # Initial step size
        rng = np.random.default_rng(42)
        
        # Success rate tracking
        success_count = 0
        total_count = 0
        window_size = 10
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            z = rng.standard_normal(n)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls_used += 1
            total_count += 1
            
            if f_new < f:
                x = x_new
                f = f_new
                success_count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                success_count = max(0, success_count - 1)
            
            # Adapt step size based on success rate over recent window
            if total_count % window_size == 0:
                success_rate = success_count / window_size
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.3:
                    sigma *= 0.8
                # else keep same
                
                # Reset window
                success_count = 0
                total_count = 0
            
            # If sigma is very small, try a restart from best with larger sigma
            if sigma < 0.01:
                sigma = 0.5
                x = best_x.copy()
                f = best_f
            
        return best_x, best_f
