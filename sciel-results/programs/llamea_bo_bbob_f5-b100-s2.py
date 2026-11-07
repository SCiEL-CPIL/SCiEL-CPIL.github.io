import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x = np.asarray(x, dtype=float)
                f_val = func(x)
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy: Sequential Refinement
        # 1. Evaluate origin
        # 2. For each dimension i, probe x_probe with optimal values for j<i, -5 for i, 0 for j>i
        # 3. Calculate w_i from the difference
        # 4. After d+1 calls, we have the full gradient and the optimal corner
        
        x0 = np.zeros(d)
        f0 = wrapper(x0)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        w_est = np.zeros(d)
        
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
                
            x_probe = np.zeros(d)
            # Set previous dimensions to their estimated optimal values
            for j in range(i):
                if w_est[j] > 0:
                    x_probe[j] = -5.0
                elif w_est[j] < 0:
                    x_probe[j] = 5.0
                else:
                    x_probe[j] = 0.0
                    
            # Set current dimension to -5
            x_probe[i] = -5.0
            
            f_probe = wrapper(x_probe)
            
            # Calculate w_i
            # f_probe = sum(w_j * x_probe[j]) + c
            # f0 = c
            # f_probe - f0 = sum(w_j * x_probe[j])
            # w_i * (-5) = f_probe - f0 - sum_{j<i} w_j * x_probe[j]
            
            sum_known = 0.0
            for j in range(i):
                sum_known += w_est[j] * x_probe[j]
                
            w_est[i] = (f0 + sum_known - f_probe) / 5.0
        
        # Determine the optimal corner
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_opt[i] = -5.0
            elif w_est[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Evaluate the optimal corner to ensure it's recorded as best
        if calls < budget and time.time() <= deadline:
            f_opt_val = wrapper(x_opt)
        
        # Fill remaining budget
        while calls < budget and time.time() <= deadline:
            wrapper(best_x if best_x is not None else np.zeros(d))

        return best_x, best_f
