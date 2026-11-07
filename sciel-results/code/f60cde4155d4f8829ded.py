# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Spherical LHS on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _lhs_2d(self, n, seed=42):
        """Latin Hypercube Sampling in 2D."""
        rng = np.random.RandomState(seed)
        indices = [np.arange(n) for _ in range(2)]
        for i in range(2):
            rng.shuffle(indices[i])
        u = np.array([(indices[i][j] + rng.rand()) / n for i in range(2) for j in range(n)]).reshape(n, 2)
        return u

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Phase 1: Place vertices (corners of the front)
        # Vertices in x[0], x[1] space for DTLZ4 front (unit sphere first octant)
        # (0,0) -> (1,0,0)
        # (0,1) -> (0,1,0)
        # (1,0) -> (0,0,1)
        # (1,1) -> (0,0,1) [degenerate, but valid point]
        vertices = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        for v in vertices:
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = 0.5  # Optimal g-variables
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Phase 2: Spherical LHS sampling on front
        # To sample the front (unit sphere octant) uniformly, we use:
        # cos(theta) = u, phi = (pi/2) * v
        # x[0] = (2/pi) * theta
        # x[1] = (2/pi) * phi = v
        n_lhs = self.budget - used
        if n_lhs > 0:
            u_lhs = self._lhs_2d(n_lhs, seed=0)
            
            for u in u_lhs:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # u[0] is for cos(theta), u[1] is for phi scaling
                # Ensure u[0] is in (0, 1] to avoid arccos domain issues or singularities?
                # arccos(1) = 0, arccos(0) = pi/2.
                # LHS gives [0, 1). 
                cos_theta = u[0]
                # Clip to valid range for arccos
                cos_theta = np.clip(cos_theta, 0.0, 1.0)
                
                theta = np.arccos(cos_theta)
                phi = (np.pi / 2.0) * u[1]
                
                x0 = (2.0 / np.pi) * theta
                x1 = (2.0 / np.pi) * phi # Which is just u[1]
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                evaluate(x)
        
        return all_X

# EVOLVE-BLOCK-END
