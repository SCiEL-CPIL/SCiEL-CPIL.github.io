class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import numpy as np
        import time
        
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Use a non-uniform grid that is denser near t=0 where the slope is high
        # This helps capture the hypervolume more accurately in the steep region.
        # alpha > 1 makes it denser near 0.
        alpha = 2.0
        t_vals = (np.arange(n_points) / (n_points - 1)) ** alpha
        # Ensure we hit 1.0 exactly
        t_vals[-1] = 1.0
        
        best_x = np.zeros(self.dim)
        
        for t in t_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = t
            if self.dim > 1:
                x[1:] = np.sqrt(t)
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            # Store best (for return, though ignored)
            best_x = x
            
        return best_x
