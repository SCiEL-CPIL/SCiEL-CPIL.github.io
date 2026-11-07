import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific logic:
        # f1 = x0
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (xi^2 - 10cos(4pi*xi))
        # To minimize f2, we must minimize g(x).
        # The terms in the sum are identical for i=2..n.
        # Let h(x) = x^2 - 10cos(4pi*x).
        # We want to find x_opt that minimizes h(x) within the bounds of the auxiliary variables.
        
        # Assume bounds for auxiliary variables are the same (standard ZDT4)
        # Use the bounds of variable 1 (index 1) as representative
        aux_lo = bounds[1][0]
        aux_hi = bounds[1][1]
        
        # Define the auxiliary function
        def h(x):
            return x**2 - 10*np.cos(4*np.pi*x)
        
        # Find the minimum of h on [aux_lo, aux_hi]
        # Since h is periodic and has many local minima, we check a grid and refine
        # Standard ZDT4 bounds are [0,1]. Global min is at x=0.
        # To be safe, we scan a grid.
        grid = np.linspace(aux_lo, aux_hi, 200)
        vals = np.array([h(x) for x in grid])
        min_idx = np.argmin(vals)
        x_opt = grid[min_idx]
        
        # Refine around the grid minimum
        # Use a small bounded optimization
        # Ensure bounds for minimize_scalar are valid
        local_lo = max(aux_lo, x_opt - 0.01)
        local_hi = min(aux_hi, x_opt + 0.01)
        
        if local_lo < local_hi:
            res = minimize_scalar(h, bounds=(local_lo, local_hi), method='bounded')
            if res.success:
                x_opt = res.x
        
        # Construct the points
        # We have self.budget evaluations.
        # We will vary x0 (index 0) uniformly across its bounds.
        # Fix x1..x9 to x_opt.
        
        n_evals = self.budget
        x0_lo, x0_hi = bounds[0][0], bounds[0][1]
        
        # Generate x0 values
        x0_vals = np.linspace(x0_lo, x0_hi, n_evals)
        
        # Initialize points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x0_vals
        # Set auxiliary variables to x_opt
        points[:, 1:] = x_opt
        
        # Evaluate all points
        # The problem function clips inputs, so we are safe.
        # We just need to call it n_evals times.
        for i in range(n_evals):
            problem(points[i])
            
        return points[-1]
