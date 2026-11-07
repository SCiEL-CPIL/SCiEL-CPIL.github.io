import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x_arr = np.asarray(x, dtype=float).flatten()
                f_val = float(func(x_arr))
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Phase 1: Random Vertex Exploration
        # Try a few random vertices to potentially hit or get close to the optimum early.
        rng = np.random.default_rng(42)
        n_random = 10
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            # Random vertex: each coordinate is -5 or 5
            x_rand = rng.choice([-5.0, 5.0], size=d)
            wrapper(x_rand)
            
        # Phase 2: Systematic Estimation
        # Evaluate 0
        x_zero = np.zeros(d)
        f_zero = wrapper(x_zero)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        # Estimate gradient w
        w_est = np.zeros(d)
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = wrapper(x_probe)
            w_est[i] = (f_probe - f_zero) / 5.0
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
            
        # Determine optimal vertex
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
                
        # Evaluate the optimal vertex
        f_opt = wrapper(x_opt)
        
        # Phase 3: Fill remaining budget with calls to x_opt
        while calls < budget and time.time() < deadline:
            wrapper(x_opt)
            
        return best_x, best_f
