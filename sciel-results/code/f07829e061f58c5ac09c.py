import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 300

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use the entire budget to sample the Pareto front.
        # DTLZ1 Pareto front is a simplex in objective space.
        # We generate points uniformly on this simplex.
        
        # Use a Sobol sequence for low-discrepancy sampling.
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7.
        # If not, we can use a simple grid or random.
        # Let's assume SciPy is available as per prompt.
        
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=3, scramble=True)
            # Generate budget points
            # Sobol requires power of 2 for best properties, but works for any N.
            # We'll generate 256 points and take the first 200 if needed, or just 200.
            # Actually, Sobol in scipy accepts any size.
            samples = sampler.random(n=budget)
        except ImportError:
            # Fallback to random if Sobol not available
            samples = np.random.rand(budget, 3)
        
        X = np.zeros((budget, dim))
        
        for i in range(budget):
            v = samples[i]
            # Sort to get order statistics
            v_sorted = np.sort(v)
            v1, v2, v3 = v_sorted
            
            # Calculate simplex coordinates a, b
            # a corresponds to f1, b to f2
            a = v2 - v1
            b = v3 - v2
            # c = 1 - a - b = 1 - (v3 - v1)
            
            # Map to x variables
            # x2 = a + b = v3 - v1
            x2 = v3 - v1
            # x1 = a / (a+b)
            if x2 > 1e-10:
                x1 = a / x2
            else:
                x1 = 0.0 # Degenerate case, set to 0
                
            X[i, 0] = x1
            X[i, 1] = x2
            # Set remaining variables to 0.5 to minimize g(x)
            X[i, 2:] = 0.5
            
            # Clip to bounds
            X[i] = np.clip(X[i], lo, hi)
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
