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
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Ensure x is a numpy array of correct shape and within bounds
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # Use a small budget to find a good starting point for the local optimizer.
        # Bent Cigar has a narrow valley. Random search might miss it, but a few
        # steps of (1+1)-ES or just sampling can help. 
        # We'll use a simple random sampling with a few (1+1)-ES steps.
        
        # Initial point
        x = rng.uniform(-4, 4, dim)
        f = safe_func(x)
        
        # Brief (1+1)-ES to get closer to the basin
        sigma = 1.0
        es_steps = min(50, budget // 10) # Use up to 50 or 10% budget for exploration
        for _ in range(es_steps):
            if time.time() > deadline or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            f_y = safe_func(y)
            if f_y < f:
                x, f = y, f_y
                sigma *= 1.1
            else:
                sigma *= 0.9
        
        # Phase 2: Local Refinement with Powell's Method
        # Powell's method is chosen because it adapts to the anisotropic curvature
        # of the Bent Cigar function, leading to super-linear convergence in practice.
        
        # We will run Powell's method. SciPy's minimize handles the iterations.
        # We need to be careful with the number of function evaluations.
        # Powell's method typically uses O(dim * N) evaluations.
        # We'll set a reasonable limit on iterations or let it run until budget/time is up.
        
        # Define bounds for SciPy
        bounds = [(-5, 5)] * dim
        
        # Run minimize
        # method='Powell' is derivative-free and good for anisotropic functions
        try:
            result = minimize(
                safe_func, 
                x, 
                method='Powell',
                bounds=bounds,
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
        except Exception:
            pass
        
        # If Powell didn't use all budget, we can try another local method or restart
        # But Powell is usually very efficient. Let's check if we have budget left.
        
        # If we have significant budget left, we can try Nelder-Mead from the best point found
        if calls < budget - 20 and time.time() < deadline - 10:
            x_start = best_x if best_x is not None else x
            f_start = best_f if best_f != float('inf') else safe_func(x_start)
            
            # Nelder-Mead
            try:
                result_nm = minimize(
                    safe_func, 
                    x_start, 
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass
        
        return best_x, best_f
