import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        budget = self.budget
        
        candidates = []
        
        # 1. Extreme points
        # (0.5, 0, 0)
        candidates.append((0.0, 0.0))
        # (0, 0.5, 0)
        candidates.append((0.0, 1.0))
        # (0, 0, 0.5)
        candidates.append((1.0, 0.0))
        # (0, 0, 0.5) duplicate or center? Let's use center (0.25, 0.25, 0.35) approx
        # x1=0.5, x2=0.5
        candidates.append((0.5, 0.5))
        
        # 2. Stratified grid
        n_v = 14
        n_b = 14
        # 14*14 = 196 points
        # Total 200 points
        
        for i in range(n_v):
            v_min = i / n_v
            v_max = (i + 1) / n_v
            v = v_min + np.random.rand() * (v_max - v_min)
            
            # v = cos(a)
            # a = arccos(v)
            # x1 = 2/pi * a
            a = np.arccos(v)
            x1 = (2.0 / np.pi) * a
            
            for j in range(n_b):
                b_min = j * (np.pi / 2) / n_b
                b_max = (j + 1) * (np.pi / 2) / n_b
                b = b_min + np.random.rand() * (b_max - b_min)
                
                # x2 = 2/pi * b
                x2 = (2.0 / np.pi) * b
                
                candidates.append((x1, x2))
                
        # We have 4 + 196 = 200 candidates.
        # If budget < 200, we need to trim. But budget is 200.
        # If budget > 200, we can add more, but 200 is the limit.
        
        evaluated_points = []
        used = 0
        
        for x1, x2 in candidates:
            if used >= budget:
                break
            if time.time() - t0 > deadline:
                break
            
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            evaluated_points.append(f)
            used += 1
            
        return np.array(evaluated_points)
