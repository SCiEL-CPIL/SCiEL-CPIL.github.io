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
        deadline = start_time + 280.0
        
        rng = np.random.default_rng()
        
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        calls = 1
        
        lam = 10  # Number of offspring per generation
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            improved = False
            best_child_x = None
            best_child_fx = fx
            
            # Generate lam offspring
            for _ in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < best_child_fx:
                    best_child_fx = fy
                    best_child_x = y
                    improved = True  # This flag logic is slightly off, we need to compare to parent fx
                    
            # Update parent
            if best_child_x is not None and best_child_fx < fx:
                x = best_child_x
                fx = best_child_fx
                sigma *= 1.05
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.95
                if sigma < 1e-5:
                    # Restart sigma or x?
                    # If stuck, maybe random jump?
                    # Let's just keep going.
                    pass
        
        return best_x, best_fx
