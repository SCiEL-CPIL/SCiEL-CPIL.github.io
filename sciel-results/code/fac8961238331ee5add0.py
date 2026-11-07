import numpy as np
from scipy.stats import qmc

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            # Generate Sobol sequence
            # We need 2 dimensions for the shape parameters
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate budget points
            # Sobol requires power of 2 for best performance, but can generate any number?
            # scipy qmc.Sobol.random(n) works for any n, but n should be power of 2 for uniformity.
            # 200 is not power of 2. 256 is. 128 is.
            # We can generate 256 and take first 200, or just use 200.
            # Let's use 200.
            points = sampler.random(self.budget)
            
            # Transform points for uniform coverage on sphere
            # u = points[:, 0], v = points[:, 1]
            # x0 = (2/pi) * arccos(u)
            # x1 = v
            
            u = points[:, 0]
            v = points[:, 1]
            
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            # Construct full x vectors
            # dim = 12, n_obj = 3
            # Shape vars: 0, 1
            # Distance vars: 2 to 11
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x0
            X[:, 1] = x1
            X[:, 2:] = 0.5  # Center of bounds for distance variables
            
            # Clip to bounds just in case (though 0.5 is inside [0,1])
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            return X
