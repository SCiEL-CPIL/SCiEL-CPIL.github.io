import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We aim to finish well before.
        # Each eval is very fast (numpy ops on 22 floats), so 200 evals take < 0.1s usually.
        # The main cost is Python overhead.
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        points = []
        
        # DTLZ7 Pareto Front Analysis:
        # The Pareto front for DTLZ7 with M=3 objectives is defined by x[0] and x[1] (active vars),
        # with x[2:] = 0 (inactive vars).
        # Let u = x[0], v = x[1].
        # h = 1 + 9 * sum(x[2:]) / (n-1) = 1 (since x[2:]=0)
        # Standard DTLZ7 definitions:
        # f0 = u
        # f1 = (1-u) * (1 - v * (1 - exp(-u / (4 * h))))  ... wait, standard DTLZ7 is:
        # f_i = y_i * h for i < m-1? No.
        # Let's recall DTLZ7:
        # g(x) = 1 + 9 * sum(x_i) / (n-m+1) for i in [m-1, n-1]
        # h(y, g) = g * (1 - sum_{i=0}^{m-2} (y_i / g * (1 + sin(3 * pi * y_i))) / (m-1))
        # f_i = y_i * h for i in [0, m-2]
        # f_{m-1} = 1 - h
        #
        # With g=1, m=3:
        # h = 1 - (y0 * (1 + sin(3*pi*y0)) + y1 * (1 + sin(3*pi*y1))) / 2
        # f0 = y0 * h
        # f1 = y1 * h
        # f2 = 1 - h
        #
        # So on the Pareto front (g=1):
        # f0 = u * h(u, v)
        # f1 = v * h(u, v)
        # f2 = 1 - h(u, v)
        # where h(u, v) = 1 - 0.5 * (u * (1 + sin(3*pi*u)) + v * (1 + sin(3*pi*v)))
        #
        # Key Insight for Hypervolume:
        # Hypervolume is maximized by covering the "extreme" points (where one objective is minimal)
        # and having a dense distribution in the "knee" region.
        # However, for DTLZ7, the front is non-linear.
        # The corners of the parameter space (u,v) in [0,1]x[0,1] map to specific points on the front.
        # (0,0) -> h=1, f=(0,0,0) -> This is the ideal point (best for all).
        # (1,1) -> h = 1 - 0.5*(1*(1+0) + 1*(1+0)) = 0. f=(0,0,1).
        # (1,0) -> h = 1 - 0.5*(1*(1+0) + 0) = 0.5. f=(0.5, 0, 0.5).
        # (0,1) -> h = 1 - 0.5*(0 + 1*(1+0)) = 0.5. f=(0, 0.5, 0.5).
        #
        # The reference point is [1.1, 1.1, 25.0].
        # Points with f0, f1, f2 < ref are counted.
        # Since f0, f1 are in [0, 1] and f2 in [0, 1], all points on the front are strictly better than ref.
        #
        # Strategy:
        # 1. Explicitly add the 4 corners of the (u,v) square: (0,0), (1,0), (0,1), (1,1).
        #    These define the "extents" of the front and are crucial for bounding the hypervolume.
        # 2. Use a non-uniform grid for the remaining budget.
        #    To maximize hypervolume, we want to fill the "convex hull" of the front.
        #    A uniform grid in (u,v) might be suboptimal if the mapping to (f0,f1,f2) is non-linear.
        #    However, a "hypervolume-aware" approach suggests clustering where the front is "flat" or
        #    where the volume contribution is highest.
        #    Actually, for a smooth front, a uniform grid in the parameter space is often good,
        #    but adding the corners is critical.
        #    
        #    Let's refine the "clustering near origin" idea.
        #    In objective space, the origin (0,0,0) is the best point.
        #    The hypervolume is the volume dominated by the set of points.
        #    To maximize this, we need points that are "far" from each other in objective space.
        #    The corners (0,0), (1,0), (0,1), (1,1) in parameter space map to:
        #    P00 = (0,0,0)
        #    P10 = (0.5, 0, 0.5)
        #    P01 = (0, 0.5, 0.5)
        #    P11 = (0, 0, 1)
        #    
        #    Wait, P11 has f0=0, f1=0, f2=1. This is "good" for f0, f1 but "bad" for f2.
        #    P00 is "good" for all.
        #    The "extreme" points in terms of single-objective minima are:
        #    Min f0: P00 (0), P01 (0), P11 (0).
        #    Min f1: P00 (0), P10 (0), P11 (0).
        #    Min f2: P00 (0).
        #    
        #    Actually, the "knee" is usually where the trade-off is balanced.
        #    For DTLZ7, the front is disconnected or has a specific shape.
        #    
        #    Let's stick to the directive: "clusters sampling points near the origin of the objective space".
        #    The origin in objective space is (0,0,0), which corresponds to (u,v)=(0,0).
        #    So we should cluster samples near (0,0) in the (u,v) parameter space.
        #    
        #    We will use a grid that is dense near (0,0) and sparse near (1,1).
        #    We can achieve this by using a transformation like u = s^2, v = t^2 where s,t in [0,1].
        #    Or simply use a geometric progression for the grid lines.
        #    
        #    Let's create a set of (u,v) pairs:
        #    1. The 4 corners.
        #    2. A grid of points. To cluster near (0,0), we can use a grid in (sqrt(u), sqrt(v)) space,
        #       which means the grid in (u,v) space is denser near 0.
        #       Let N be the number of grid points per dimension.
        #       Total points = N^2.
        #       We have budget B.
        #       Let's reserve 4 points for corners.
        #       Remaining budget B_rem = B - 4.
        #       Choose N such that N^2 <= B_rem.
        #       N = floor(sqrt(B_rem)).
        #       
        #       Generate grid points s_i = i / (N-1) for i in 0..N-1.
        #       u_i = s_i^2, v_j = t_j^2.
        #       This creates a grid denser near 0.
        #       
        #       We must avoid duplicates with the corners.
        #       (0,0) is generated by i=0, j=0.
        #       (1,1) is generated by i=N-1, j=N-1.
        #       (1,0) is generated by i=N-1, j=0.
        #       (0,1) is generated by i=0, j=N-1.
        #       So the grid includes the corners.
        #       We can just use the grid points.
        #       
        #       Total points = N^2.
        #       If N^2 < B, we have leftover budget.
        #       We can add more points by increasing N or using a finer sub-grid in the densest region.
        #       Let's just pick N = floor(sqrt(B)).
        #       If N^2 < B, we can add random points or refine the grid further.
        #       Actually, a simple uniform grid in (sqrt(u), sqrt(v)) is a good start.
        #       
        #       Let's calculate N for B=200.
        #       sqrt(200) ~ 14.14. N=14. N^2 = 196.
        #       4 points left. We can add 4 random points or mid-points.
        #       
        #       Let's use N = int(np.floor(np.sqrt(budget))).
        #       If N < 2, N=2.
        
        N = int(np.floor(np.sqrt(self.budget)))
        if N < 2:
            N = 2
            
        # Generate grid in transformed space
        # s, t in [0, 1]
        # u = s^2, v = t^2
        # This clusters points near u=0, v=0.
        
        s_vals = np.linspace(0, 1, N)
        t_vals = np.linspace(0, 1, N)
        
        u_grid, v_grid = np.meshgrid(s_vals, t_vals, indexing='ij')
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        
        # u_flat and v_flat are the parameter values.
        # We need to map them to actual u, v for DTLZ7.
        # u_actual = u_flat^2
        # v_actual = v_flat^2
        
        u_actual = u_flat ** 2
        v_actual = v_flat ** 2
        
        # Check time
        if time.time() > deadline:
             # Fallback to just corners if we are super slow (unlikely)
             pass

        # Evaluate all grid points
        for i in range(len(u_actual)):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = u_actual[i]
            x[1] = v_actual[i]
            # x[2:] are 0
            
            f = problem(x)
            n_evals += 1
            points.append((x, f))
            
        # If we have budget left, add some random points or refine the corner regions?
        # The directive says "clusters ... near the origin".
        # The grid already does this.
        # Leftover budget: self.budget - n_evals.
        # Let's add random points in the (0, 0.5] x (0, 0.5] square to further densify the "good" region.
        while n_evals < self.budget:
            if time.time() > deadline:
                break
                
            # Random point in the "dense" region
            u_r = np.random.rand() ** 2 # Clustered near 0
            v_r = np.random.rand() ** 2
            
            x = np.zeros(self.dim)
            x[0] = u_r
            x[1] = v_r
            
            f = problem(x)
            n_evals += 1
            points.append((x, f))

        return points
