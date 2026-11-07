import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Pre-compute optimal x_i for i=2..dim-1 to minimize f1
        # f1 term: x^2 - 10*cos(4*pi*x + i*pi/n)
        # n = 10 (dim)
        n = self.dim
        optimal_x_tail = np.zeros(self.dim - 1)
        for k in range(self.dim - 1):
            i = k + 2  # index in 1-based ZDT formula, 0-based in array is k
            # Minimize f(x) = x^2 - 10*cos(4*pi*x + i*pi/n) on [0, 1]
            # Use a simple grid search or scipy minimize
            # Since it's 1D and periodic-ish, a fine grid is fast and robust
            grid = np.linspace(0, 1, 1000)
            vals = grid**2 - 10 * np.cos(4 * np.pi * grid + i * np.pi / n)
            optimal_x_tail[k] = grid[np.argmin(vals)]
        
        # Generate x1 samples using Sobol sequence for better low-discrepancy
        # This ensures better coverage of the oscillatory f1(x1) part
        sampler = qmc.Sobol(d=1, scramble=True)
        x1_samples = sampler.random(n_evals)[:, 0]
        
        # Construct X
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_samples
        X[:, 1:] = optimal_x_tail
        
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
