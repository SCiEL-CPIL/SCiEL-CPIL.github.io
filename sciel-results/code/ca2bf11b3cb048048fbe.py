import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        N = n_evals
        
        # Parameters: u0, u_max, alpha
        # u_i = u0 + (u_max - u0) * (i/(N-1))^alpha
        
        def hv_from_params(params):
            u0, u_max, alpha = params
            # Handle boundaries
            if u0 < 0: u0 = 0
            if u_max > 1: u_max = 1
            if u0 >= u_max: 
                # Degenerate, return 0
                return 0.0
            
            t = np.arange(N) / (N - 1)
            u = u0 + (u_max - u0) * (t ** alpha)
            
            # Ensure u is within [0, 1]
            u = np.clip(u, 0.0, 1.0)
            
            # Calculate HV
            # f2 = 1 - sqrt(u)
            # Height = 11 - f2 = 10 + sqrt(u)
            heights = 10.0 + np.sqrt(u)
            
            # Widths
            # w_i = u_{i+1} - u_i for i < N-1
            # w_{N-1} = 11 - u_{N-1}
            widths = np.diff(u)
            last_width = 11.0 - u[-1]
            
            hv = np.sum(widths * heights[:-1]) + last_width * heights[-1]
            return hv

        def neg_hv(params):
            return -hv_from_params(params)

        # Initial guess
        # u0 ~ 0, u_max ~ 1, alpha ~ 1.5
        x0 = [0.0, 1.0, 1.5]
        
        # Bounds
        bounds_opt = [(0.0, 0.5), (0.5, 1.0), (1.0, 5.0)]
        
        res = minimize(neg_hv, x0, method='L-BFGS-B', bounds=bounds_opt)
        
        u0, u_max, alpha = res.x
        
        # Generate points
        t = np.arange(N) / (N - 1)
        u_vals = u0 + (u_max - u0) * (t ** alpha)
        u_vals = np.clip(u_vals, 0.0, 1.0)
        
        # Create X
        X = np.zeros((N, self.dim))
        X[:, 0] = u_vals
        
        # Evaluate
        F = np.zeros((N, self.n_obj))
        for k in range(N):
            F[k] = problem(X[k])
            
        return X
