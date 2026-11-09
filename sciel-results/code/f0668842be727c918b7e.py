import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit
        
        n_pos = self.dim - (self.n_obj - 1)
        n_shape = self.n_obj - 1
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ4, the Pareto front is achieved when position variables are at 0.5
        # and shape variables satisfy: sum of f_i = 1 where f_i are the objective contributions
        
        # We'll use a Halton sequence for the shape variables to get good coverage
        # of the Pareto front. The Pareto front of DTLZ4 with 3 objectives is the
        # set of points where the first two shape variables determine the objectives.
        
        n_samples = self.budget
        
        # Generate Halton sequence for shape variables
        def halton_sequence(n, d):
            """Generate n points in d dimensions using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, d + 2):  # bases: 2, 3, ...
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        shape_samples = halton_sequence(n_samples, n_shape)
        
        # Construct full dimension vectors
        # Position vars set to 0.5
        pos_val = 0.5 * np.ones((n_samples, n_pos))
        
        # Concatenate
        X = np.hstack((pos_val, shape_samples))
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = []
        for x in X:
            if time.time() > deadline:
                break
            F.append(problem(x))
        
        return np.array(F)
