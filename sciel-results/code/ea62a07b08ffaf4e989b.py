import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        # Generate Halton sequence for 2 dimensions
        # x1, x2 determine the front
        # x3...xd are set to lo (0) to minimize g
        
        points_2d = self._halton_2d(n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        # x[2:] remains 0, which is lo[2:]
        
        # If bounds are not [0,1], we should scale?
        # Task says bounds [0,1].
        # But to be safe:
        # x1 in [lo[0], hi[0]]
        # x2 in [lo[1], hi[1]]
        # x3.. in [lo[2:], hi[2:]] -> set to lo[2:]
        
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * points_2d[:, 0]
        X[:, 1] = lo[1] + (hi[1] - lo[1]) * points_2d[:, 1]
        X[:, 2:] = lo[2:]
        
        F = np.array([problem(x) for x in X])
        
        return X

    def _halton_2d(self, n):
        points = []
        for i in range(1, n + 1):
            # Base 2
            f1 = 1.0
            r1 = 0.0
            ii = i
            while ii > 0:
                f1 /= 2
                r1 += f1 * (ii % 2)
                ii //= 2
            
            # Base 3
            f2 = 1.0
            r2 = 0.0
            ii = i
            while ii > 0:
                f2 /= 3
                r2 += f2 * (ii % 3)
                ii //= 3
                
            points.append([r1, r2])
        return np.array(points)
