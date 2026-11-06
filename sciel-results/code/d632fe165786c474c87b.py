"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Place 300 equal circles inside the unit circle.
    Uses a multi-start local optimization approach:
    1. Initialize with a hexagonal lattice (good starting point).
    2. Use scipy.optimize.minimize with L-BFGS-B to maximize the minimum distance.
    3. The objective function minimizes the negative of the minimum pair distance,
       while ensuring all circles stay inside the container.
    """
    # Step 1: Generate initial points using hexagonal lattice
    def hexagonal_init(n):
        # Estimate spacing for n points in unit disk
        # Area of unit disk = pi, area per point ~ pi/n
        # Hexagonal packing density ~ pi/(2*sqrt(3)) per unit area for radius r
        # So n * (2*r)^2 * sqrt(3)/2 ~ pi  =>  r ~ sqrt(pi/(2*sqrt(3)*n))
        r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * n))
        spacing = 2 * r_est * 1.05  # slightly larger to ensure we get enough points
        
        points = []
        # Generate hexagonal lattice points
        # Hexagonal lattice: rows offset by spacing/2, row spacing = spacing*sqrt(3)/2
        row_spacing = spacing * np.sqrt(3) / 2
        max_radius = 1.0
        
        # Determine how many rows/cols we need
        n_rows = int(2 * max_radius / row_spacing) + 4
        n_cols = int(2 * max_radius / spacing) + 4
        
        for j in range(-n_rows, n_rows + 1):
            y = j * row_spacing
            offset = (j % 2) * spacing / 2
            for i in range(-n_cols, n_cols + 1):
                x = i * spacing + offset
                r = np.sqrt(x**2 + y**2)
                if r <= max_radius - r_est * 0.5:  # leave some margin
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n points, select the n closest to center
        if len(points) > n:
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            points = points[idx]
        elif len(points) < n:
            # Fallback: use a grid
            n_grid = int(np.ceil(np.sqrt(n)))
            x = np.linspace(-0.9, 0.9, n_grid)
            y = np.linspace(-0.9, 0.9, n_grid)
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack([xx.ravel(), yy.ravel()])
            points = points[:n]
        
        return points


    def min_pair_distance(pts):
        """Compute the minimum pairwise distance."""
        n = len(pts)
        min_d = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
        return min_d


    def objective(x, n):
        """Objective: minimize negative min distance, with penalty for going outside."""
        pts = x.reshape(-1, 2)
        
        # Check if any point is outside the unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        outside = np.max(np.maximum(0, radii - 1.0))
        
        # Compute min pair distance efficiently using broadcasting for small N
        # For N=300, this is 300*299/2 ~ 45k pairs, which is fine
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        # Penalty for being outside
        penalty = 100.0 * outside**2
        
        return -min_d + penalty


    def gradient(x, n):
        """Approximate gradient using finite differences."""
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (objective(xp, n) - objective(xm, n)) / (2 * eps)
        return grad


    # Step 2: Optimize
    initial_points = hexagonal_init(N)
    
    # Flatten to 1D
    x0 = initial_points.ravel()
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Use L-BFGS-B with finite difference gradient
    start_time = time.time()
    timeout = 110  # leave 10s buffer
    
    best_x = x0.copy()
    best_val = objective(x0, N)
    
    # Try a few iterations
    result = minimize(
        objective,
        x0,
        args=(N,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() - start_time > timeout:
        # If we've spent too much time, use the best we have
        pass
    else:
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    
    # Convert back to points
    pts = best_x.reshape(-1, 2)
    
    # Ensure all points are inside the unit circle
    radii = np.hypot(pts[:, 0], pts[:, 1])
    for i in range(len(pts)):
        if radii[i] > 1.0:
            # Scale back to unit circle
            scale = 1.0 / radii[i]
            pts[i] *= scale
    
    return pts


# EVOLVE-BLOCK-END
