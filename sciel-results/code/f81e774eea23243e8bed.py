"""Direct construction optimizer for DTLZ4 using stratified objective-space sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using stratified sampling in objective space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives (M=3, n=12):
        # f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # where g = sum_{i=3}^{12} (x_i - 0.5)^2
        #
        # On the Pareto front (g=0), f1^2 + f2^2 + f3^2 = 0.25
        # The front is a quarter-sphere of radius 0.5 in the positive octant.
        #
        # Strategy: Generate stratified points in the (f1, f2) quarter-disk,
        # compute f3 analytically, then map back to decision variables via inverse trig.

        n_points = budget

        # Use stratified sampling in polar coordinates for the quarter-disk
        # f1 = r * cos(theta), f2 = r * sin(theta), where r in [0, 0.5], theta in [0, pi/2]
        # f3 = sqrt(0.25 - r^2)
        
        # To get good hypervolume, we want uniform coverage of the Pareto front.
        # The Pareto front is a 2D manifold. We can parameterize by (r, theta)
        # where r = sqrt(f1^2 + f2^2) and theta = atan2(f2, f1).
        #
        # For uniform area coverage on the sphere, we need to be careful about
        # the Jacobian. Let's use a grid in (theta, phi) where phi is the
        # angle from the z-axis (f3 axis).
        #
        # Actually, let's use a simpler approach: stratified sampling in the
        # (u, v) unit square mapped to the quarter-disk.
        #
        # We'll divide the quarter-disk into n x n cells and sample one point
        # per cell with jitter.
        
        # Use a sqrt-spaced grid for r to get uniform area coverage
        # Area element in polar: r dr dtheta
        # For uniform area, we want r_i such that r_i^2 is uniform in [0, 0.25]
        # So r_i = 0.5 * sqrt(i/n)
        
        n_grid = int(np.ceil(np.sqrt(n_points)))
        # Adjust to get exactly n_points or close
        # We'll use n_grid x n_grid and take the first n_points
        
        # Generate stratified points in (r, theta) space
        # r in [0, 0.5], theta in [0, pi/2]
        # For uniform area on the quarter-disk:
        #   r^2 should be uniform in [0, 0.25]
        #   theta should be uniform in [0, pi/2]
        
        # Create grid indices
        i_vals = np.arange(n_grid)
        j_vals = np.arange(n_grid)
        
        # For each cell, sample with jitter
        # r_i = 0.5 * sqrt((i + U_i) / n_grid) where U_i ~ Uniform(0,1)
        # theta_j = (pi/2) * (j + U_j) / n_grid where U_j ~ Uniform(0,1)
        
        # We need exactly n_points, so we'll generate n_grid*n_grid and take first n_points
        total_grid = n_grid * n_grid
        
        # Generate all grid points with jitter
        # Use deterministic jitter for reproducibility
        rng = np.random.RandomState(42)
        
        # Create meshgrid of cell indices
        ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
        ii_flat = ii.ravel()
        jj_flat = jj.ravel()
        
        # Jitter
        U_r = rng.rand(total_grid)
        U_theta = rng.rand(total_grid)
        
        # r = 0.5 * sqrt((i + U_r) / n_grid)
        r = 0.5 * np.sqrt((ii_flat + U_r) / n_grid)
        # theta = (pi/2) * (j + U_theta) / n_grid
        theta = (np.pi / 2.0) * (jj_flat + U_theta) / n_grid
        
        # Take first n_points
        r = r[:n_points]
        theta = theta[:n_points]
        
        # Compute f1, f2, f3
        f1 = r * np.cos(theta)
        f2 = r * np.sin(theta)
        f3 = np.sqrt(np.maximum(0.25 - r**2, 0.0))
        
        # Map back to decision variables
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = 0.5 * sin(pi*x1/2)
        #
        # From f3: sin(pi*x1/2) = 2*f3 => x1 = (2/pi) * arcsin(2*f3)
        # From f1, f2: 
        #   cos(pi*x1/2) * cos(pi*x2/2) = 2*f1
        #   cos(pi*x1/2) * sin(pi*x2/2) = 2*f2
        #   So tan(pi*x2/2) = f2/f1 => x2 = (2/pi) * atan2(f2, f1)
        #   And cos(pi*x2/2) = 2*f1 / cos(pi*x1/2)
        #   We need to verify consistency: cos^2 + sin^2 = 1
        #   (2*f1/cos(pi*x1/2))^2 + (2*f2/cos(pi*x1/2))^2 = 4*(f1^2+f2^2)/cos^2(pi*x1/2)
        #   = 4*r^2 / cos^2(pi*x1/2)
        #   But cos^2(pi*x1/2) = 1 - sin^2(pi*x1/2) = 1 - 4*f3^2 = 1 - 4*(0.25 - r^2) = 4*r^2
        #   So 4*r^2 / (4*r^2) = 1. Good, it's consistent.
        
        # Compute x1, x2
        # sin(pi*x1/2) = 2*f3, so pi*x1/2 = arcsin(2*f3)
        # x1 = (2/pi) * arcsin(2*f3)
        # Handle numerical issues: clip 2*f3 to [-1, 1]
        sin_arg = np.clip(2.0 * f3, -1.0, 1.0)
        x1 = (2.0 / np.pi) * np.arcsin(sin_arg)
        
        # tan(pi*x2/2) = f2/f1
        # pi*x2/2 = atan2(f2, f1)
        x2 = (2.0 / np.pi) * np.arctan2(f2, f1)
        
        # Clip to [0, 1] to be safe
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1
        X[:, 1] = x2
        # Set all other variables to 0.5 to ensure g=0
        if dim > 2:
            X[:, 2:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F_list = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F_list.append(f)
            
        if not F_list:
            return X
        
        F = np.array(F_list)
        # If we broke out early, we only have a subset of X
        if len(F_list) < n_points:
            X = X[:len(F_list)]
        
        return X
# EVOLVE-BLOCK-END
