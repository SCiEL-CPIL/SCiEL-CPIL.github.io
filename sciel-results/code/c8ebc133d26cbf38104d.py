"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    f5 is a linear function: f(x) = f0 + c^T (R(x - shift))
    where R is a rotation, shift is a translation, c is a fixed vector.
    
    Strategy:
    1. Use a two-point difference method to estimate the gradient.
    2. Jump to the optimal corner.
    3. Use remaining budget for a simple local search to verify.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # Phase 1: Estimate gradient using two-point differences
        # For a linear function, we can use any two points
        # We'll use the origin and a point far away
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        
        # Choose a step size that stays within the box
        # We'll use a step of 4.0 in each direction
        # This gives us points at +/-4 which are within [-5, 5]
        h = 4.0
        
        # Estimate gradient components using forward differences from origin
        # f(h*e_i) - f(0) = grad[i] * h
        grad = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = np.zeros(dim)
            xp[i] = h
            
            fp = eval_point(xp)
            grad[i] = (fp - f0) / h
        
        # Determine the corner that minimizes the linear function
        # For linear f(x) ≈ f0 + grad^T x, min is at
        # x_i = -5 if grad[i] > 0, else +5
        corner = np.where(grad > 0, -5.0, 5.0)
        
        # Evaluate the corner
        f_corner = eval_point(corner)
        
        # Phase 2: If we have budget left, do a simple local search
        # to verify and potentially refine
        if calls_made < budget and time.time() < deadline:
            # Simple (1+1)-ES around the best point
            sigma = 0.5
            x = best_x.copy()
            
            # Run a limited number of iterations to avoid wasting budget
            max_iters = min(budget - calls_made, 100)
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                if sigma < 1e-10:
                    break
                
                y = x + sigma * np.random.randn(dim)
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                
                if fy < best_f:
                    x = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
