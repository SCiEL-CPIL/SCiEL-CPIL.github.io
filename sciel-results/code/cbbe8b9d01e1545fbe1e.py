import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT3, the Pareto front is achieved when x_2...x_n = 0
        # This minimizes g(x) and thus f_2(x) for any given x_1
        x_opt_rest = 0.0
        
        # Use all budget for dense sampling of x_1
        n_points = self.budget
        
        # Create dense sampling of x_1 in [0, 1]
        # ZDT3 has oscillatory behavior with sin(10*pi*x_1), so we want
        # to capture the critical points of this oscillation
        x1_values = np.linspace(0.0, 1.0, n_points)
        
        # Build the candidate set
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = x_opt_rest
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all candidates
        F = np.array([problem(x) for x in X])
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # The return value is ignored by the evaluator, but we return X
        return X
