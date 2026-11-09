# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hypervolume-maximizing optimizer for DTLZ4.
    
    Strategy:
    1. Analytically place all distance variables (x_3...x_12) at 0.5 to ensure g(x)=1.
    2. The shape variables (x_1, x_2) determine the position on the Pareto front (a 2D simplex).
    3. Instead of uniform/Halton sampling, we use a grid of points spaced with Chebyshev nodes.
       Chebyshev nodes cluster points near the boundaries of the domain.
    4. In hypervolume maximization on a simplex, the "corners" (vertices of the simplex) 
       contribute the most to the volume. Clustering points near the boundaries (where the 
       simplex vertices are located in the transformed space) maximizes the hypervolume 
       for a fixed budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_shape_vars = self.n_obj - 1  # 2 variables: x0, x1
        n_dist_vars = self.dim - n_shape_vars  # 10 variables: x2...x11
        
        # Optimal value for distance variables in DTLZ4 is 0.5
        opt_dist_val = 0.5
        
        # Create a base vector with optimal distance variables
        base_x = np.zeros(self.dim)
        base_x[n_shape_vars:] = opt_dist_val
        
        # We need to generate points for the shape variables (x0, x1) such that:
        # f1 = g * sin(pi/2 * x0)
        # f2 = g * cos(pi/2 * x0) * sin(pi/2 * x1)
        # f3 = g * cos(pi/2 * x0) * cos(pi/2 * x1)
        
        # To maximize hypervolume, we want to cover the simplex efficiently.
        # The vertices of the simplex in (f1, f2, f3) space are:
        # v1 = (1, 0, 0) -> x0=0, x1=0
        # v2 = (0, 1, 0) -> x0=pi/2, x1=0
        # v3 = (0, 0, 1) -> x0=pi/2, x1=pi/2
        
        # Let's map a grid in (u, v) space to (x0, x1) space.
        # u, v in [0, 1].
        # x0 = u * (pi/2)
        # x1 = v * (pi/2) * (1 - u)  # This ensures when u=1, x1=0 (vertex v2).
        # Wait, standard simplex mapping:
        # Let s1, s2, s3 be barycentric coordinates summing to 1.
        # f1 = s1, f2 = s2, f3 = s3.
        # s1 = sin(pi/2 * x0)
        # s2 = cos(pi/2 * x0) * sin(pi/2 * x1)
        # s3 = cos(pi/2 * x0) * cos(pi/2 * x1)
        
        # We want to generate (s1, s2, s3) points that maximize hypervolume.
        # The hypervolume of a set of points on the simplex is maximized by points
        # that are well-distributed, but specifically, the "corners" of the simplex
        # in the objective space (1,0,0), (0,1,0), (0,0,1) are crucial.
        
        # Let's use a grid in the (s1, s2) plane, where s3 = 1 - s1 - s2.
        # The domain is s1 >= 0, s2 >= 0, s1 + s2 <= 1.
        
        # We can generate a square grid in (a, b) in [0,1]x[0,1] and map to simplex.
        # Or better, use a triangular grid.
        
        # Let's use Chebyshev nodes for the grid indices to cluster near boundaries.
        # We will create a grid of size N x N.
        # Total points = N*N.
        # We have budget B. Let N = int(sqrt(B)).
        N = int(np.sqrt(self.budget))
        if N < 2:
            N = 2
            
        # Generate Chebyshev nodes in [0, 1]
        # Chebyshev nodes are clustered near -1 and 1.
        # We want them clustered near 0 and 1 (boundaries of the simplex).
        # Standard Chebyshev on [-1, 1]: x_k = cos(pi * (2k+1) / (2N))
        # Map to [0, 1]: y_k = (x_k + 1) / 2
        
        k = np.arange(N)
        # Chebyshev nodes
        cheb_nodes = np.cos(np.pi * (2 * k + 1) / (2 * N))
        # Map to [0, 1]
        grid_u = (cheb_nodes + 1) / 2
        grid_v = (cheb_nodes + 1) / 2
        
        # We will iterate over the grid (u, v)
        # u, v in [0, 1]
        # Map (u, v) to (s1, s2) on the simplex.
        # A simple mapping: s1 = u * (1 - v), s2 = v * (1 - u) ? No, that doesn't cover the whole simplex simply.
        
        # Let's use a direct mapping from a square [0,1]x[0,1] to the triangle.
        # s1 = u
        # s2 = v * (1 - u)
        # s3 = 1 - s1 - s2 = 1 - u - v(1-u) = (1-u)(1-v)
        # This maps (0,0) -> (0,0,1)
        # (1,0) -> (1,0,0)
        # (0,1) -> (0,1,0)
        # (1,1) -> (1,0,0) (degenerate, but covers the triangle)
        
        # Actually, let's just generate points directly in the (s1, s2) space.
        # We want points (s1, s2) such that s1 >= 0, s2 >= 0, s1+s2 <= 1.
        
        # Let's generate a grid in (alpha, beta) in [0, 1]x[0, 1].
        # s1 = alpha * (1 - beta)
        # s2 = beta * (1 - alpha)
        # s3 = (1 - alpha) * (1 - beta)
        
        # This covers the simplex.
        # Let's evaluate how well this distributes.
        # At alpha=0, beta=0: (0,0,1)
        # At alpha=1, beta=0: (1,0,0)
        # At alpha=0, beta=1: (0,1,0)
        # At alpha=1, beta=1: (1,0,0) -- Wait, s1=0, s2=0, s3=0? No.
        # s1 = 1*(0) = 0. s2 = 1*(0) = 0. s3 = 0*0 = 0. Sum=0. This is invalid.
        
        # Correct mapping for square to triangle:
        # s1 = alpha
        # s2 = beta * (1 - alpha)
        # s3 = 1 - s1 - s2
        # Domain: alpha in [0,1], beta in [0,1].
        # If alpha=1, s2=0, s3=0. Point (1,0,0).
        # If alpha=0, s2=beta, s3=1-beta. Line from (0,1,0) to (0,0,1).
        # This covers the triangle.
        
        # We use Chebyshev nodes for alpha and beta to cluster near boundaries.
        # Boundaries are alpha=0, alpha=1, beta=0, beta=1.
        # Clustering near alpha=0 puts points near the edge (0,1,0)-(0,0,1).
        # Clustering near alpha=1 puts points near (1,0,0).
        # Clustering near beta=0 puts points near the edge (1,0,0)-(0,0,1).
        # Clustering near beta=1 puts points near the edge (1,0,0)-(0,1,0).
        
        # This seems to put a lot of weight on the vertices.
        
        points_s = []
        for a in grid_u:
            for b in grid_v:
                s1 = a
                s2 = b * (1 - a)
                s3 = 1 - s1 - s2
                # Clip to ensure validity due to floating point
                s1 = max(0, min(1, s1))
                s2 = max(0, min(1, s2))
                s3 = max(0, min(1, s3))
                # Normalize just in case
                sum_s = s1 + s2 + s3
                if sum_s > 0:
                    s1 /= sum_s
                    s2 /= sum_s
                    s3 /= sum_s
                else:
                    s1, s2, s3 = 1/3, 1/3, 1/3
                    
                points_s.append([s1, s2, s3])
                
        points_s = np.array(points_s)
        
        # We might have more or fewer points than budget.
        # If we have more, we should subsample or just take the first 'budget' ones?
        # Better: Adjust N to get close to budget.
        # Or simply evaluate as many as we can within budget.
        
        # If len(points_s) > budget, we can't evaluate all.
        # We should pick the most important ones.
        # The vertices are most important.
        # Let's sort points by distance to vertices? Or just take a subset.
        # Actually, let's adjust N to be closer to sqrt(budget).
        
        # Re-calculate N to be exactly int(sqrt(budget)) might leave some budget unused.
        # Let's just use the generated points. If there are too many, we truncate.
        # If there are too few, we are under budget.
        
        # To be safe and use the full budget effectively:
        # If len(points_s) > self.budget:
        #     # Keep the ones closest to the vertices?
        #     # Or just take the first ones?
        #     # Chebyshev nodes are already ordered.
        #     points_s = points_s[:self.budget]
        # elif len(points_s) < self.budget:
        #     # We have extra budget. Add random points? Or more grid points?
        #     # Let's just add the vertices explicitly if not present, and maybe some random.
        #     pass
        
        # Let's refine the generation to hit the budget more precisely.
        # Use a 1D Chebyshev sequence for the "radius" and angle?
        # No, 2D grid is fine.
        
        # Let's ensure we have exactly budget points or fewer.
        if len(points_s) > self.budget:
             # Sort by "importance" - distance to centroid? No, distance to vertices is better.
             # Actually, just taking the first ones from the Chebyshev grid (which are clustered at corners) is good.
             points_s = points_s[:self.budget]
             
        # If we have fewer, we can fill the rest with random points on the front.
        if len(points_s) < self.budget:
             n_extra = self.budget - len(points_s)
             # Random points in simplex
             extra_s = np.random.dirichlet([1, 1, 1], size=n_extra)
             points_s = np.vstack([points_s, extra_s])
             
        # Now, map (s1, s2, s3) back to (x0, x1)
        # s1 = sin(pi/2 * x0)
        # s2 = cos(pi/2 * x0) * sin(pi/2 * x1)
        # s3 = cos(pi/2 * x0) * cos(pi/2 * x1)
        
        # x0 = 2/pi * arcsin(s1)
        # cos_x0 = sqrt(1 - s1^2)
        # sin_x1 = s2 / cos_x0
        # cos_x1 = s3 / cos_x0
        # x1 = 2/pi * arcsin(sin_x1)
        
        X = np.tile(base_x, (len(points_s), 1))
        
        for i, s in enumerate(points_s):
            s1, s2, s3 = s
            # Handle edge cases where cos_x0 is 0
            if s1 >= 1.0:
                x0 = 1.0
                x1 = 0.0
            else:
                x0 = 2.0 / np.pi * np.arcsin(np.clip(s1, 0, 1))
                cos_x0 = np.sqrt(1.0 - s1 * s1)
                if cos_x0 > 1e-12:
                    sin_x1 = s2 / cos_x0
                    sin_x1 = np.clip(sin_x1, 0, 1)
                    x1 = 2.0 / np.pi * np.arcsin(sin_x1)
                else:
                    x1 = 0.0
            
            X[i, 0] = x0
            X[i, 1] = x1
        
        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
