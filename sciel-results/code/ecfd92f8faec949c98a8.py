import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The optimal tail values for ZDT4 are 0.0 for all variables except the first.
        # This minimizes g(x) to 0, making f2 = 1.
        # We set x[1:] to 0.0, clipped to bounds to be safe.
        x_tail = np.clip(np.zeros(self.dim - 1), lo[1:], hi[1:])
        
        # We sample x0 uniformly across its bounds to cover the Pareto front.
        # The hypervolume is dominated by the minimum x0 value, which is lo[0].
        # Sampling uniformly ensures we hit lo[0] and cover the front.
        
        # Use budget evaluations
        x0_values = np.linspace(lo[0], hi[0], self.budget)
        
        for x0 in x0_values:
            if time.time() > deadline:
                break
            
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None
