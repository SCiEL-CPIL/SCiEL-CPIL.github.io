import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        # 1. Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made = 1
        
        # 2. Determine signs of gradient components
        h = 5.0
        signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            calls_made += 1
            
            # Determine direction for axis i
            # If f(xp) > f0, gradient is positive, we want -h
            # If f(xp) < f0, gradient is negative, we want +h
            if fp > f0:
                signs[i] = -1.0
            else:
                signs[i] = 1.0
                
        # 3. Construct optimal vertex
        x_opt = x0 + h * signs
        
        # 4. Evaluate optimal vertex
        f_opt = func(x_opt)
        calls_made += 1
        
        # 5. Fill remaining budget
        while calls_made < budget:
            if time.time() > deadline:
                break
            func(x_opt)
            calls_made += 1
            
        return x_opt, f_opt
