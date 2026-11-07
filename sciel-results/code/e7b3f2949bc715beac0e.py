import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Deadline guard
        import time
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s margin

        # Center of the sphere: The lower bound (lo) is a good guess for WFG9
        # as f1 = sum(x) is minimized near 0.
        center = lo.copy()
        
        # Max radius: distance from center to the furthest corner (hi)
        max_radius = np.linalg.norm(hi - lo)
        
        # Non-dominated set
        nd_x = []
        nd_f = []
        
        def is_dominated(f_new, nd_f_list):
            for f in nd_f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        for t in range(n):
            # Check time
            if time.time() > deadline:
                break
                
            # Calculate current radius
            # Linear shrink from max_radius to a small value
            # Use a power law or linear? Linear is suggested.
            progress = t / max(1, n - 1)
            # Add a small epsilon to avoid zero radius
            current_radius = max_radius * (1.0 - progress) + 1e-5 * max_radius
            
            # Sample point in hyper-sphere
            # 1. Random direction
            direction = np.random.randn(dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.ones(dim) / np.sqrt(dim)
                
            # 2. Random distance (uniform in volume)
            # r = R * u^(1/dim)
            u = np.random.random()
            r = current_radius * (u ** (1.0 / dim))
            
            x = center + direction * r
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = np.array(problem(x))
            
            # Update non-dominated set
            if not is_dominated(f, nd_f):
                # Check if it dominates any existing
                new_nd_f = []
                new_nd_x = []
                for i in range(len(nd_f)):
                    if not dominates(f, nd_f[i]):
                        new_nd_f.append(nd_f[i])
                        new_nd_x.append(nd_x[i])
                
                nd_f.append(f)
                nd_x.append(x)
                
                nd_f = new_nd_f + [f]
                nd_x = new_nd_x + [x]

        return nd_x # Return list of x? The contract says "Every returned objective vector is recorded and the return value of __call__ is ignored." 
                    # So return value doesn't matter for scoring, but must not crash.
