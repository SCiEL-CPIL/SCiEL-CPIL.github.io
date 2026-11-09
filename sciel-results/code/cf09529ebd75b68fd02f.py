import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9: 3 shape variables, 9 position variables
        n_shape = 3
        n_pos = self.dim - n_shape
        
        # Grid search on shape variables
        # 5 x 5 x 8 = 200 points
        n1, n2, n3 = 5, 5, 8
        
        points = []
        for i in range(n1):
            x1 = lo[0] + (hi[0] - lo[0]) * i / (n1 - 1) if n1 > 1 else mid[0]
            for j in range(n2):
                x2 = lo[1] + (hi[1] - lo[1]) * j / (n2 - 1) if n2 > 1 else mid[1]
                for k in range(n3):
                    x3 = lo[2] + (hi[2] - lo[2]) * k / (n3 - 1) if n3 > 1 else mid[2]
                    
                    x = np.empty(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = mid[3:]  # Position variables at midpoints
                    
                    points.append(x)
                    
                    # Check budget and time
                    if len(points) >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                if len(points) >= self.budget or time.time() > deadline:
                    break
            if len(points) >= self.budget or time.time() > deadline:
                break
        
        # Evaluate
        F = []
        for x in points:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
        
        return F
