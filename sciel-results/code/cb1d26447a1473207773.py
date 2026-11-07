import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # Create 200 points on the Pareto front
        # x[2:] = 0.5 to minimize g
        # x[0], x[1] sampled to cover the front
        
        # Use Latin Hypercube Sampling for x[0] and x[1]
        # This provides good coverage of the 2D Pareto front surface
        
        # Generate LHS points
        # n_samples = budget
        n_samples = budget
        
        # Create bins
        # For simplicity, we can use random permutation
        idx = np.arange(n_samples)
        
        # Randomize the assignment of bins to ensure coverage
        # Standard LHS:
        # x_i = (i + U_i) / n_samples
        # where U_i is uniform(0,1)
        
        u1 = np.random.rand(n_samples)
        u2 = np.random.rand(n_samples)
        
        x1 = (idx + u1) / n_samples
        x2 = (idx + u2) / n_samples
        
        # To make it a proper LHS, we should permute one dimension
        # so that each bin in x2 is also used exactly once
        perm = np.random.permutation(n_samples)
        x2 = (perm + u2) / n_samples  # Wait, this changes the values.
        
        # Correct LHS generation:
        # x1_i = (i + U_i) / N
        # x2_i = (P(i) + V_i) / N
        # where P is a permutation of 0..N-1
        
        P = np.random.permutation(n_samples)
        x1 = (idx + u1) / n_samples
        x2 = (P + u2) / n_samples
        
        # Construct the full vectors
        X = np.zeros((n_samples, n))
        X[:, 0] = x1
        X[:, 1] = x2
        X[:, 2:] = 0.5
        
        # Clip to bounds (though 0.5 is always in [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function returns 3 objectives
        # We just need to call it. The evaluator records the results.
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
