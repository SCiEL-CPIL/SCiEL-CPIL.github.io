import numpy as np
import time
from collections import deque

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        sigma = 2.0
        min_sigma = 1e-8
        max_sigma = 10.0
        
        success_history = deque(maxlen=10)
        no_improve_count = 0
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
                
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                success_history.append(1)
                no_improve_count = 0
            else:
                success_history.append(0)
                no_improve_count += 1
                
            if len(success_history) == 10:
                rate = np.mean(success_history)
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                
                if sigma < min_sigma:
                    sigma = min_sigma
                elif sigma > max_sigma:
                    sigma = max_sigma

            # Early termination if converged
            if no_improve_count > 100 and sigma < 1e-5:
                break

        return x, fx
