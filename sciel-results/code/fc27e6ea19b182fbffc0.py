import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin
        
        d = self.dim
        budget = self.budget
        
        # Multi-restart (1+1)-ES
        # For 5D, Schaffer's F7 is multimodal.
        # Let's use 5 restarts.
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        best_x_global = None
        best_f_global = float('inf')
        calls_made = 0
        
        rng = np.random.default_rng(0)
        
        for r in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5: # Leave some time for cleanup
                break
                
            # Initialize
            x = rng.uniform(-5, 5, d)
            f_x = func(x)
            calls_made += 1
            
            if f_x < best_f_global:
                best_f_global = f_x
                best_x_global = x.copy()
            
            sigma = 1.0
            
            # Run (1+1)-ES for this restart
            while calls_made < budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate perturbation
                z = rng.standard_normal(d)
                x_new = x + sigma * z
                
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < f_x:
                    # Accept
                    x = x_new
                    f_x = f_new
                    sigma *= 1.05 # Increase step size slightly
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                else:
                    # Reject
                    sigma /= 1.05 # Decrease step size slightly
                    # Prevent sigma from becoming too small
                    if sigma < 1e-5:
                        sigma = 1e-5
                    # Prevent sigma from becoming too large
                    if sigma > 10:
                        sigma = 10

        return best_x_global, best_f_global
