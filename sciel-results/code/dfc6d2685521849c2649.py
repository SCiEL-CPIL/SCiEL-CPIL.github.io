"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: Multi-start Powell's Method. Powell's method is a derivative-free
direct search algorithm that is effective for multimodal functions like
Lunacek bi-Rastrigin. It adapts search directions to align with the valleys
of the function, which helps escape the local minima traps more effectively
than simple (1+1)-ES.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin for 300s limit
        d = self.dim
        budget = self.budget

        # Best solution tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls_made
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Generate diverse starting points using Sobol sequence
        n_starts = 15
        rng = np.random.default_rng(42)
        
        # Sobol for diverse starts
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_samples = sampler.random(n_starts)
            start_points = sobol_samples * 10.0 - 5.0
        except Exception:
            start_points = rng.uniform(-5, 5, size=(n_starts, d))

        # Ensure we try the origin and a few other key points
        start_points[0] = np.zeros(d)
        # The local minimum of Lunacek is often shifted, but origin is a safe bet for global
        # We can also add a point far from origin to cover the other basin
        start_points[1] = np.ones(d) * 2.5
        start_points[2] = np.ones(d) * -2.5

        # Run Powell's Method from each starting point
        for start_idx in range(n_starts):
            if calls_made >= budget - 10:
                break
            if time.time() > deadline:
                break

            x0 = start_points[start_idx].copy()
            
            # We use Powell's method via scipy.optimize.minimize
            # maxiter is set to limit the number of iterations, but we also monitor calls
            # Powell's method in scipy is 'Powell'
            try:
                # We wrap the function to count calls and stop if budget is exceeded
                # However, scipy's minimize doesn't easily allow early termination based on external call count
                # So we limit maxiter and rely on the budget check in the outer loop
                # A single Powell run can use many calls. We need to be careful.
                # Let's estimate: 500 budget / 15 starts = ~33 calls per start.
                # Powell might use 5-10 calls per iteration. So ~5 iterations per start.
                
                # To be safer, we can implement a custom Powell or use a very low maxiter
                # and then re-optimize the best result.
                
                res = minimize(
                    eval_func, 
                    x0, 
                    method='Powell', 
                    options={'maxiter': 10, 'maxfev': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # Update best if Powell found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            # Check time and budget
            if time.time() > deadline - 5.0:
                break

        # If we have budget left, do a final refinement on the best point found
        if best_x is not None and calls_made < budget - 20 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    eval_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 20, 'maxfev': 100, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
