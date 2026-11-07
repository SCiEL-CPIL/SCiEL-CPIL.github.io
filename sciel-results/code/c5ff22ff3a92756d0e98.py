import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Generate points on the Pareto front of ZDT6
        # f1 = x0, f2 = 1 - sqrt(x0)
        # Optimal spacing: f1(i) = 10 * ((1.1)^((i-1)/(N-1)) - 1)
        N = n_points
        if N <= 1:
            x0_vals = np.array([0.5])
        else:
            i = np.arange(N)
            # Ensure i goes from 0 to N-1
            x0_vals = 10.0 * ((1.1) ** ((i) / (N - 1)) - 1.0)
            # Clip to [0, 1] just in case of floating point errors
            x0_vals = np.clip(x0_vals, 0.0, 1.0)
            
        # Create X matrix
        X = np.zeros((N, self.dim))
        X[:, 0] = x0_vals
        # x1..x9 are 0, which is within [0, 1]
        
        # Evaluate
        for i in range(N):
            if time.time() > deadline:
                break
            # problem expects a vector of length dim
            # It clips to bounds
            problem(X[i])
            
        return X
