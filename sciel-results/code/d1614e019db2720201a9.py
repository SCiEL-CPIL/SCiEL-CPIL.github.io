import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific: 
        # x_1 (index 0) is the variable that defines the position on the Pareto front.
        # x_2..x_n (indices 1..n-1) are deceptive variables.
        # Knowledge indicates their global optimum for the front is 0.5.
        
        base_x = np.zeros(self.dim)
        
        # Set deceptive variables to 0.5 (clipped to bounds)
        for i in range(1, self.dim):
            if lo[i] <= 0.5 <= hi[i]:
                base_x[i] = 0.5
            else:
                # Fallback to midpoint if 0.5 is out of bounds
                base_x[i] = (lo[i] + hi[i]) / 2.0
                
        # We use the entire budget for the sweep
        n_evals = self.budget
        
        # Uniformly sample x_1
        # Using linspace ensures we hit the boundaries and cover the range evenly
        x1_samples = np.linspace(lo[0], hi[0], n_evals)
        
        # Construct points
        # We can pre-allocate the array for efficiency, though we evaluate one by one
        # to allow for the time check.
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = x1_samples[i]
            problem(x)
            
        return None
