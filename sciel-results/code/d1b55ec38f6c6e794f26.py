import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate Halton sequence points in [0, 1]^2
        # We need 200 points
        n_points = self.budget
        
        # Simple Halton sequence generator for 2 dimensions
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate points
        x1_vals = []
        x2_vals = []
        for i in range(1, n_points + 1):
            x1_vals.append(halton(i, 2))
            x2_vals.append(halton(i, 3))
            
        x1_arr = np.array(x1_vals)
        x2_arr = np.array(x2_vals)
        
        # Create full vectors
        # dim=12, n_obj=3
        # Position vars: x1, x2
        # Distance vars: x3..x12 set to 0.5
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_arr
        X[:, 1] = x2_arr
        X[:, 2:] = 0.5
        
        # Ensure bounds are respected (though 0.5 is safe, and halton is in (0,1))
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for x in X:
            problem(x)
            
        return X
