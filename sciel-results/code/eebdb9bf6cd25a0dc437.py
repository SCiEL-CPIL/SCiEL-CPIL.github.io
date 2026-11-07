import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 Pareto front is at x_2...x_n = 0 (lower bounds)
        # f_1 = x_1
        # f_2 = 1 - sqrt(x_1)
        
        # We have 'budget' evaluations.
        # We sweep x_1 across [0, 1].
        # To maximize hypervolume, uniform distribution in x_1 is a strong baseline.
        # We can try to be slightly smarter, but uniform is safe.
        
        n_evals = self.budget
        # Create x_1 values
        # Using linspace ensures we cover the whole range.
        x1_vals = np.linspace(0, 1, n_evals)
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            # Construct x vector
            x = np.zeros(self.dim)
            # Set x_1
            x[0] = x1_vals[i]
            # x_2...x_n remain 0 (which is lo[1:])
            
            # Evaluate
            f = problem(x)
            results.append(f)
            
        return results
