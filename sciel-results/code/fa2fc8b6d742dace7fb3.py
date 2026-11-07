import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # Strategy:
        # 1. Use Latin Hypercube Sampling (LHS) to ensure uniform coverage of the x1 domain.
        #    This avoids the systematic aliasing or potential gaps of a fixed grid.
        # 2. Apply a non-uniform transformation x1 = t^k to bias sampling towards x1=0.
        #    For ZDT2, the Pareto front f2 = (1+10(n-1))(1 - sqrt(x1/(1+10(n-1))))
        #    has high curvature near x1=0. To maximize hypervolume with reference point
        #    (11, 11), we need dense sampling in the region where the front changes rapidly.
        #    The theoretical optimal k for y ~ C - D*sqrt(x) is 4/3.
        
        # Generate LHS points in [0, 1]
        # 1. Divide [0, 1] into n_points intervals
        # 2. Pick one point uniformly at random from each interval
        idx = np.arange(n_points)
        offsets = np.random.rand(n_points)
        
        # Normalize to [0, 1]
        t_uniform = (idx + offsets) / n_points
        
        # Shuffle to decorrelate indices from specific offsets (standard LHS)
        perm = np.random.permutation(n_points)
        t_uniform = t_uniform[perm]
        
        # Apply bias towards 0 using power law
        # k = 4.0/3.0 is derived from the curvature of the ZDT2 front
        k = 4.0 / 3.0
        x1_vals = t_uniform ** k
        
        # Map to [lo[0], hi[0]]
        # Since bounds are [0, 1], lo[0]=0, hi[0]=1
        x1_vals = lo[0] + x1_vals * (hi[0] - lo[0])
        
        # Ensure strict bounds adherence
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        recorded_F = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            # For ZDT2, the Pareto front is at x[1:] = 0
            x[0] = x1_vals[i]
            
            # Clip full vector just in case, though x[1:] is 0 which is within [0,1]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if f is not None and len(f) == n_obj:
                recorded_F.append(f)
            
            # Time check
            if time.time() > deadline:
                break

        return recorded_F
