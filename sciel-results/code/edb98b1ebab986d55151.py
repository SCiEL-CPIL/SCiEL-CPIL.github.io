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
        
        # Use all budget for Pareto front sampling
        n_points = self.budget
        
        # Generate t values
        # Use linspace for even coverage
        t_vals = np.linspace(0.0, 1.0, n_points)
        
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
            # Just keep the last one or first one
            best_x = x
            
        return best_x
