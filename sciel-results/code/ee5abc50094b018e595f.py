import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

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
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f_val = float(func(x))
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Sampling with Space-Filling Design
        # Use Halton sequence for better coverage than pure random, especially in low dims.
        # Also include the origin as a likely candidate for the global minimum of bi-Rastrigin.
        
        n_samples = 30
        if budget < n_samples + 10:
            n_samples = max(5, budget // 2)
            
        # Generate Halton points
        try:
            sampler = qmc.Halton(d=dim, scramble=True)
            halton_points = sampler.random(n_samples)
            points = halton_points * 10 - 5
        except Exception:
            points = np.random.uniform(-5, 5, (n_samples, dim))
            
        # Ensure origin is included
        points[0] = 0.0

        # Evaluate initial points
        sample_vals = []
        for p in points:
            if not time_ok() or calls >= budget:
                break
            f_val = wrapped_func(p)
            sample_vals.append(f_val)
        
        if not sample_vals:
            return best_x, best_f

        # Sort samples by value to identify promising basins
        sorted_indices = np.argsort(sample_vals)

        # Phase 2: Local Refinement with Rank-One CMA-ES
        # The current program's strength is its adaptive step size and covariance update.
        # We apply this to the top few promising points found in Phase 1.
        
        n_refine = min(5, len(sample_vals))
        
        # CMA-ES parameters
        cs = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
        nu = 2 / ((dim + 1.3) ** 2 + 1)
        
        for idx in sorted_indices[:n_refine]:
            if not time_ok() or calls >= budget:
                break
            
            x = points[idx].copy()
            f_x = sample_vals[idx]
            
            sigma = 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            
            # Run a short CMA-ES loop from this start
            # Limit the number of steps per start to ensure we explore multiple basins
            max_steps_per_start = 50 
            
            steps = 0
            while steps < max_steps_per_start and calls < budget and time_ok():
                steps += 1
                
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

        # Phase 3: Final Polish with Nelder-Mead
        # Use the global best found so far and run a final Nelder-Mead to squeeze out the last bit of precision.
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
