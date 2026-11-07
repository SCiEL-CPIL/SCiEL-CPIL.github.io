import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        start_time = time.time()
        deadline = start_time + 290
        
        max_calls = self.budget
        calls = 0
        
        # Sample from [-4, 4] as minimizer is known to be in this box
        x = rng.uniform(-4, 4, self.dim)
        try:
            fx = func(x)
        except Exception:
            return x, np.inf # Or handle first call failure
        calls += 1
        
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.0
        window_size = 10
        success_count = 0
        total_in_window = 0
        stagnation_count = 0
        stagnation_threshold = 15
        
        while calls < max_calls:
            if time.time() > deadline:
                break
            
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
            except Exception:
                fy = np.inf
            calls += 1
            
            if fy < fx:
                success_count += 1
                x = y
                fx = fy
                stagnation_count = 0
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                stagnation_count += 1
                
            total_in_window += 1
            
            if total_in_window >= window_size:
                p_success = success_count / window_size
                if p_success > 0.2:
                    sigma *= 1.5
                else:
                    sigma *= 0.7
                success_count = 0
                total_in_window = 0
                
            if stagnation_count > stagnation_threshold or sigma < 0.05:
                if calls >= max_calls:
                    break
                x = rng.uniform(-4, 4, self.dim)
                try:
                    fx = func(x)
                except Exception:
                    fx = np.inf
                calls += 1
                sigma = 1.0
                stagnation_count = 0
                success_count = 0
                total_in_window = 0

        return best_x, best_fx
