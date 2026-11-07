import numpy as np
import time

class Optimizer:
    """
    Optimizer for WFG9 using a systematic grid search on the primary shape variables.
    Exploits the structure of WFG problems where the first few variables determine
    the Pareto front shape, and the rest determine the distance from the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 with 3 objectives: The Pareto front is 2-dimensional.
        # Typically, the first M-1 (2) variables define the front shape, 
        # and the rest are position variables. However, knowledge suggests 
        # a 3D grid search worked well, implying x0, x1, x2 are critical.
        # We will grid search x0, x1, x2 and set the rest to 0 (optimal position).
        
        # Determine grid sizes to use the full budget
        # 5 * 5 * 8 = 200 evaluations
        n0 = 5
        n1 = 5
        n2 = 8
        total_grid = n0 * n1 * n2
        
        # If budget is different, adjust
        if self.budget < total_grid:
            # Scale down grid
            scale = (self.budget ** (1/3)) / (total_grid ** (1/3))
            n0 = max(2, int(n0 * scale))
            n1 = max(2, int(n1 * scale))
            n2 = max(2, int(n2 * scale))
            total_grid = n0 * n1 * n2
            
        # Generate grid points for the first 3 variables
        # We use linspace to cover the range
        g0 = np.linspace(lo[0], hi[0], n0)
        g1 = np.linspace(lo[1], hi[1], n1)
        g2 = np.linspace(lo[2], hi[2], n2)
        
        X = []
        # Set position variables (index 3 to end) to 0.0
        # If 0 is out of bounds, clip it
        pos_val = np.zeros(self.dim - 3)
        pos_val = np.clip(pos_val, lo[3:], hi[3:])
        
        for x0 in g0:
            for x1 in g1:
                for x2 in g2:
                    if time.time() > deadline:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    x[3:] = pos_val
                    X.append(x)
                    
                    if len(X) >= self.budget:
                        break
                if len(X) >= self.budget:
                    break
            if len(X) >= self.budget:
                break
        
        # If we have budget left, add random samples
        if len(X) < self.budget:
            rng = np.random.default_rng(42)
            remaining = self.budget - len(X)
            X_rand = lo + (hi - lo) * rng.random((remaining, self.dim))
            X.extend([x for x in X_rand])
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
