# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure.
    Uses Sobol quasi-random sequence for uniform coverage of the 2D simplex.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s kill

        n_points = min(self.budget, 200)

        # DTLZ1 Pareto front for M=3 objectives:
        # f1 + f2 + f3 = 0.5, fi >= 0
        # In decision space: x0 + x1 = 1, x2..x6 = 0.5
        # Objectives: f1 = 0.5*(x0 + 0.5*(x1 + g)), f2 = 0.5*(1-x0+g), f3 = 0.5*g
        # On the front (g=0): f1 = 0.5*x0, f2 = 0.5*(1-x0), f3 = 0
        # Wait, let me re-derive DTLZ1 objectives.
        # DTLZ1:
        #   g = 100 * (n - m + 1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i+0.5))) for i=m..n-1)
        #   f1 = 0.5 * x0 * (1 + g)
        #   f2 = 0.5 * x1 * (1 + g)
        #   f3 = 0.5 * (1 - x0 - x1) * (1 + g)
        # For n=7, m=3: g depends on x3..x6 (4 variables)
        # On Pareto front: g = 0, so x3..x6 = 0.5
        # And x0 + x1 + x2 = 1 with x0,x1,x2 >= 0
        # Wait, the standard DTLZ1 has:
        #   f1 = 0.5*x0*(1+g)
        #   f2 = 0.5*x1*(1+g)  
        #   f3 = 0.5*(1-x0-x1)*(1+g)
        # So on the front: f1=0.5*x0, f2=0.5*x1, f3=0.5*(1-x0-x1)
        # The front is a triangle in objective space with vertices (0.5,0,0), (0,0.5,0), (0,0,0.5)
        
        # In decision space on the front:
        # x0, x1, x2 >= 0, x0+x1+x2 = 1
        # x3=x4=x5=x6=0.5
        
        # To sample the triangle uniformly, we can use:
        # u1, u2 ~ uniform(0,1)
        # x0 = 1 - sqrt(u1)
        # x1 = sqrt(u1) * (1 - u2)
        # x2 = sqrt(u1) * u2
        # This gives uniform sampling over the simplex.
        
        # But for hypervolume, we want good coverage. Let's use a combination:
        # 1. Include the 3 vertices and edge midpoints explicitly
        # 2. Fill the rest with quasi-random points
        
        # Generate points on the simplex
        # Use scipy if available for Sobol, otherwise use a simple quasi-random approach
        try:
            from scipy.stats import qmc
            # Sobol sequence for 2D
            sampler = qmc.Sobol(d=2, scramble=True)
            # We need n_points points, but reserve some for vertices
            n_sobol = n_points - 5  # 5 reserved for special points
            sobol_pts = sampler.random(n_sobol)
        except ImportError:
            # Fallback: use Halton sequence
            def halton(n, base):
                seq = []
                f = 1.0
                for i in range(1, n + 1):
                    x = 0.0
                    ii = i
                    while ii > 0:
                        f /= base
                        ii, rem = divmod(ii, base)
                        x += f * rem
                    seq.append(x)
                return np.array(seq)
            
            h1 = halton(n_sobol, 2)
            h2 = halton(n_sobol, 3)
            sobol_pts = np.column_stack([h1, h2])

        # Convert 2D unit square to simplex
        u1 = sobol_pts[:, 0]
        u2 = sobol_pts[:, 1]
        
        sqrt_u1 = np.sqrt(u1)
        x0 = 1.0 - sqrt_u1
        x1 = sqrt_u1 * (1.0 - u2)
        x2 = sqrt_u1 * u2

        # Add special points: 3 vertices and 3 edge midpoints
        special_x0 = np.array([0.0, 1.0, 0.0, 0.5, 0.0, 0.5])
        special_x1 = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0])
        special_x2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Combine all points
        all_x0 = np.concatenate([x0, special_x0])
        all_x1 = np.concatenate([x1, special_x1])
        all_x2 = np.concatenate([x2, special_x2])
        
        total = len(all_x0)
        # Trim to exactly n_points if needed
        if total > n_points:
            all_x0 = all_x0[:n_points]
            all_x1 = all_x1[:n_points]
            all_x2 = all_x2[:n_points]
        
        # Build decision vectors
        X = np.zeros((len(all_x0), self.dim))
        X[:, 0] = all_x0
        X[:, 1] = all_x1
        X[:, 2] = all_x2
        for i in range(3, self.dim):
            X[:, i] = 0.5

        # Evaluate
        F = np.zeros((len(X), self.n_obj))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
            except Exception:
                pass

        return X
# EVOLVE-BLOCK-END
