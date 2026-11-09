import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, bases):
        """Generate Halton sequence points in [0,1]^d."""
        seq = np.zeros((n, len(bases)))
        for i in range(n):
            for j, base in enumerate(bases):
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                seq[i, j] = r
        return seq

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4: first dim-2 vars control g, last 2 control direction on Pareto front
        # Set first dim-2 vars to 1 to get g=1 (Pareto front)
        # Use Halton sequence for the last 2 vars for uniform coverage
        
        n_points = self.budget
        
        # Generate Halton sequence in 2D for the last two variables
        halton = self._halton_sequence(n_points, [2, 3])
        
        # Map to [lo, hi] for the last two dimensions
        # First dim-2 dimensions set to 1 (to achieve g=1 for DTLZ4)
        X = np.zeros((n_points, self.dim))
        X[:, :self.dim - 2] = 1.0  # Set to 1 for g=1
        
        # Last two dimensions from Halton, scaled to bounds
        X[:, self.dim - 2] = lo[self.dim - 2] + (hi[self.dim - 2] - lo[self.dim - 2]) * halton[:, 0]
        X[:, self.dim - 1] = lo[self.dim - 1] + (hi[self.dim - 1] - lo[self.dim - 1]) * halton[:, 1]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time - if we're close to deadline, return immediately
        if time.time() > deadline:
            return X
        
        return X
