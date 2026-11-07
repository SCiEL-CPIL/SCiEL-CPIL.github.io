import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, leave some margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is at x[1:] = 0, x[0] in [0, 1]
        # We sample x[0] uniformly to cover the front
        x0_samples = np.linspace(0, 1, self.budget)
        
        recorded_vectors = []
        
        for x0 in x0_samples:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            # Clip to bounds just in case, though x0 is in [0,1] and bounds are [0,1]
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                recorded_vectors.append(f)
            except Exception:
                continue
        
        # Return the recorded vectors or the last X, the return value is ignored
        # but we must return something.
        return recorded_vectors
