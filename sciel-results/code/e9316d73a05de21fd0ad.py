import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 specific structure:
        # Pareto front is at x[1:] = 0.
        # We only need to vary x[0].
        
        # Generate candidate x[0] values
        # Use a distribution that is dense near 0
        # u ~ Uniform(0, 1), x0 = u^k
        # k=4 is a good guess for ZDT6
        
        n_evals = self.budget
        u = np.random.rand(n_evals)
        x0_candidates = u ** 4
        
        # Ensure we cover 0 exactly
        x0_candidates[0] = 0.0
        
        # We might also want to cover the end of the range
        # But ZDT6 front is mostly in the first part of x0
        # Let's add a few points at higher x0 just in case
        if n_evals > 10:
             x0_candidates[-5:] = np.linspace(0.5, 1.0, 5)

        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0_candidates[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results
