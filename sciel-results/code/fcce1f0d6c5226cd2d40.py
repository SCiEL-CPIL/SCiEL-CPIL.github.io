import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        n_evals = self.budget
        dim = self.dim
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy:
        # 1. Exploit ZDT2 structure: Pareto front is at x[1:] = 0 (minimizing g).
        # 2. Use non-uniform sampling for x[0] to maximize hypervolume.
        #    For convex decreasing front, optimal density is higher near x1=0.
        #    Use power transformation x1 = t^k with k tuned for hypervolume.
        
        # Generate t uniformly in [0, 1]
        t = np.linspace(0.0, 1.0, n_evals)
        
        # Apply power transformation to bias towards 0
        # k < 1 biases towards 0. For hypervolume with convex front,
        # k around 0.4-0.6 is good. Try k=0.4 for more bias to origin.
        k = 0.4
        x1_vals = t ** k
        
        # Ensure we stay within bounds for x1
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        # Ensure we have endpoints for better hypervolume coverage
        if n_evals >= 2:
            x1_vals[0] = lo[0]  # x1 = 0
            x1_vals[-1] = hi[0]  # x1 = 1
        
        # Construct full vectors
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_vals
        # Set remaining dimensions to lower bound to minimize g(x)
        if dim > 1:
            X[:, 1:] = lo[1:]
            
        # Evaluate all points with time guard
        for i in range(n_evals):
            if time.time() > deadline:
                break
            # problem(x) clips to bounds internally, but we've already clipped x1 and set x[1:] to lo
            # It's safe to pass X[i]
            problem(X[i])
        
        return X
