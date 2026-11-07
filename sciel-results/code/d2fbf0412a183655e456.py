"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

class Optimizer:
    """
    Optimizer using Multi-Start Powell's Method (derivative-free).
    
    Powell's method performs conjugate direction line searches without gradient
    estimates, achieving superlinear convergence for smooth functions. This
    structural change from quasi-Newton L-BFGS-B avoids Hessian approximation
    overhead and may converge faster in early iterations for AOCC scoring.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 1.0

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate starting points using Halton sequence for low-discrepancy coverage
        def halton_sequence(n, dims):
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = d + 2  # Start with prime bases 2, 3, 5...
                    result = 0.0
                    denom = 1
                    while True:
                        denom *= f
                        result += (i % f) / denom
                        i //= f
                        if i == 0:
                            break
                    point.append(result)
                points.append(point)
            return np.array(points)

        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Number of restarts: Powell's method in 5D typically converges in 20-60 function evals
        n_restarts = 12
        halton_points = halton_sequence(n_restarts, dim)
        start_points = map_to_box(halton_points)
        
        # Include origin for safety
        start_points = np.vstack([np.zeros(dim), start_points])
        
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim

        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                # Use Powell's method (derivative-free)
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell', 
                    bounds=bounds,
                    options={'maxiter': 30, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
                if res.success or res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except (ValueError, Exception) as e:
                if "Budget Exceeded" in str(e):
                    break
                else:
                    pass

        # If we have budget left, do a final polish from the best found so far
        if calls < budget and time_left() and best_f < np.inf:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    bounds=bounds,
                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                update_best(res.x, res.fun)
            except (ValueError, Exception):
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
