"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sphere-specific optimizer: quadratic fit init + (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Sphere-specific quadratic fit using 2*dim+1 points
        # f(x) = a*||x-c||^2 + offset, but we fit f(x) = a*||x||^2 + w^T x + c
        # Actually for sphere: f(x) = c0 * ||x - x*||^2 + f_opt
        # We can fit: f(x) = a * ||x||^2 + w^T x + b
        # The minimum is at x* = -w/(2a)
        
        # Use center 0 and radius r
        r = 3.0
        points = [np.zeros(dim)]
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = r
            points.append(e_i)
            points.append(-e_i)
        
        # Evaluate all points
        evals = []
        for p in points:
            evals.append(func(p))
        
        # Fit f(x) = a*||x||^2 + w^T x + b
        # At x=0: f(0) = b
        # At x=e_i*r: f = a*r^2 + w_i*r + b
        # At x=-e_i*r: f = a*r^2 - w_i*r + b
        # So: w_i = (f(e_i*r) - f(-e_i*r)) / (2*r)
        #     a = (f(e_i*r) + f(-e_i*r) - 2*b) / (2*r^2)
        
        b = evals[0]
        w = np.zeros(dim)
        a_sum = 0.0
        for i in range(dim):
            w[i] = (evals[1 + 2*i] - evals[2 + 2*i]) / (2.0 * r)
            a_sum += (evals[1 + 2*i] + evals[2 + 2*i] - 2.0 * b) / (2.0 * r**2)
        a = a_sum / dim
        
        # If a <= 0, the fit is degenerate; fall back to centroid of best points
        if a > 1e-10:
            x0 = -w / (2.0 * a)
        else:
            # Find the best evaluated point
            best_idx = np.argmin(evals)
            x0 = points[best_idx].copy()
        
        # Clip to box
        x0 = np.clip(x0, -5.0, 5.0)
        fx0 = func(x0)
        
        calls_made = 2 * dim + 1 + 1  # 2*dim+1 fit points + 1 for x0
        
        # Phase 2: (1+1)-ES with adaptive sigma
        sigma = 1.0
        best_x = x0.copy()
        best_f = fx0
        
        for i in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma *= 1.25
            else:
                sigma *= 0.85
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
