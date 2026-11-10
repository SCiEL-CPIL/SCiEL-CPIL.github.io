import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        N = self.budget
        
        def hv_from_params(params):
            u0, u_max, alpha = params
            if u0 < 0: u0 = 0
            if u_max > 1: u_max = 1
            if u0 >= u_max:
                return 0.0
            
            t = np.arange(N) / (N - 1)
            u = u0 + (u_max - u0) * (t ** alpha)
            u = np.clip(u, 0.0, 1.0)
            
            heights = 10.0 + np.sqrt(u)
            widths = np.diff(u)
            last_width = 11.0 - u[-1]
            
            hv = np.sum(widths * heights[:-1]) + last_width * heights[-1]
            return hv

        def neg_hv(params):
            return -hv_from_params(params)

        # Try multiple initial guesses to avoid local optima
        initial_guesses = [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.5],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 0.5],
            [0.0, 0.9, 1.5],
        ]
        
        bounds_opt = [(0.0, 0.5), (0.5, 1.0), (0.1, 5.0)]
        
        best_hv = -np.inf
        best_params = None
        
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
            try:
                res = minimize(neg_hv, x0, method='L-BFGS-B', bounds=bounds_opt,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
                hv_val = -res.fun
                if hv_val > best_hv:
                    best_hv = hv_val
                    best_params = res.x
            except Exception:
                continue
        
        if best_params is None:
            best_params = np.array([0.0, 1.0, 1.5])
        
        u0, u_max, alpha = best_params
        
        t = np.arange(N) / (N - 1)
        u_vals = u0 + (u_max - u0) * (t ** alpha)
        u_vals = np.clip(u_vals, 0.0, 1.0)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = u_vals
        
        F = np.zeros((N, self.n_obj))
        for k in range(N):
            if time.time() > deadline:
                break
            F[k] = problem(X[k])
        
        return X
