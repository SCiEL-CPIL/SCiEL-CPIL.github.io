# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i, evaluate at 5*e_i to determine gradient component v_i.
    3. Construct the optimal vertex x* by setting each coordinate to -5 or 5 based on the sign of the gradient.
    4. Evaluate x* to find the global minimum (since f is linear, min is at a vertex).
    5. Stall with the best point found (which is x*) for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Track gradient components
        v = np.zeros(d)
        
        # Step 2: Probe axes to estimate gradient
        for i in range(d):
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            
            # Update gradient component
            v[i] = (fi - f0) / 5.0
            
            # Update best if this axis point is better
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
        
        # Step 3: Construct and evaluate the optimal vertex
        if np.all(v != 0) or calls_made >= d + 1: # If we have enough info
            x_vertex = np.zeros(d)
            for i in range(d):
                if v[i] > 0:
                    x_vertex[i] = -5.0
                else:
                    x_vertex[i] = 5.0
            
            if time.time() < deadline and calls_made < self.budget:
                fv = func(x_vertex)
                calls_made += 1
                if fv < f_best:
                    f_best = fv
                    x_best = x_vertex.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
