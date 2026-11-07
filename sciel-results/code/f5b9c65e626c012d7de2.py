import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        max_time = 320.0  # Safety margin before 330s kill
        
        def time_ok():
            return time.time() - start_time < max_time

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return best_f
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Adaptive Trajectory (Rank-One CMA-ES style)
        # We use a 1+1-ES with rank-one covariance adaptation.
        
        # Initialize with a few random starts to avoid bad local minima
        n_starts = 5
        start_xs = []
        start_fs = []
        
        for i in range(n_starts):
            x0 = np.random.uniform(-5, 5, dim)
            f0 = wrapped_func(x0)
            start_xs.append(x0)
            start_fs.append(f0)
            
        # Pick the best start
        best_idx = np.argmin(start_fs)
        x = start_xs[best_idx].copy()
        f_x = start_fs[best_idx]
        
        # CMA-ES parameters
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        
        # Adaptation rates
        cs = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
        # nu for rank-one update
        nu = 2 / ((dim + 1.3) ** 2 + 1)
        
        while calls < budget and time_ok():
            # Sample z from N(0, C)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                z_std = np.random.randn(dim)
                z = eigenvectors @ (np.sqrt(eigenvalues) * z_std)
            except np.linalg.LinAlgError:
                z = np.random.randn(dim)
                C = np.eye(dim)
                
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = wrapped_func(x_new)
            
            if f_new < f_x:
                # Success
                y = (x_new - x) / sigma
                p_sigma = (1 - cs) * p_sigma + cs * y
                C = (1 - nu) * C + nu * np.outer(y, y)
                C = (C + C.T) / 2
                
                x = x_new
                f_x = f_new
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            if sigma < 1e-6:
                break

        # Phase 2: Local Refinement with Nelder-Mead
        if time_ok() and calls < budget and best_x is not None:
            remaining_budget = budget - calls
            if remaining_budget > 10:
                try:
                    result_nm = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result_nm.fun < best_f:
                        best_f = result_nm.fun
                        best_x = result_nm.x.copy()
                except Exception:
                    pass

        return best_x, best_f
