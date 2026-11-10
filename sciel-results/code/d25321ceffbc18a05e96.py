import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590  # 600s limit, keep 10s margin

    def score(points):
        # points shape (n, 2)
        # Calculate all pairwise distances
        d = pdist(points)
        if d.max() == 0:
            return 0.0
        d_min = d.min()
        d_max = d.max()
        return (d_min / d_max) ** 2

    def objective(points_flat):
        # points_flat shape (2*n,)
        points = points_flat.reshape(n, 2)
        # We want to maximize score, so minimize negative score
        # To help convergence, we might fix scale? 
        # The ratio is scale invariant, so the optimizer can move points 
        # to infinity or zero. 
        # To prevent this, we can add a small penalty for large norms or fix the centroid/scale.
        # However, Nelder-Mead usually handles this if initialized well, 
        # but it's safer to fix the scale.
        # Let's fix the sum of squared distances from centroid to be 1.
        # Actually, just fixing the centroid to 0 is not enough for scale invariance.
        # We can normalize the points inside the objective? No, that breaks differentiability/smoothness for some optimizers, but NM is derivative-free.
        # Better: Fix the scale in the parameterization?
        # Or simply add a penalty term to keep points bounded.
        # Let's try: minimize -score(points) + 0.01 * (np.sum(points**2) - n)**2
        # This encourages the points to have a specific "size".
        
        pts = points_flat.reshape(n, 2)
        # Center the points
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Calculate score on centered points (scale invariant, so centering doesn't change score)
        s = score(pts_centered)
        
        # Penalty to fix scale: we want sum of squares of centered points to be close to n (avg sq dist from centroid = 1)
        scale_penalty = (np.sum(pts_centered**2) - n) ** 2
        
        return -s + 0.1 * scale_penalty

    best_score = -1
    best_points = None

    # Initializations
    inits = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid)

    # 2. Random uniform in circle
    np.random.seed(123)
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        inits.append(pts)

    # 3. Hexagonal-like packing
    # Generate a hexagonal grid and pick the 16 closest to center
    # Hex grid points
    hex_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Convert axial to cartesian (pointy top)
            x_h = q + 0.5 * r
            y_h = (np.sqrt(3)/2) * r
            hex_points.append((x_h, y_h))
    hex_points = np.array(hex_points)
    # Sort by distance to origin
    dists = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists)[:n]
    hex_init = hex_points[idx]
    inits.append(hex_init)

    # 4. Another random Gaussian
    np.random.seed(456)
    gauss_init = np.random.randn(n, 2)
    inits.append(gauss_init)

    for init in inits:
        if time.time() > deadline:
            break
            
        # Flatten for optimizer
        init_flat = init.ravel()
        
        # Run Nelder-Mead
        # maxiter needs to be sufficient but not too long
        # 32 dimensions
        res = minimize(
            objective, 
            init_flat, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        # Extract points
        opt_pts = res.x.reshape(n, 2)
        # Center them
        opt_pts -= np.mean(opt_pts, axis=0)
        
        # Calculate actual score
        s = score(opt_pts)
        
        if s > best_score:
            best_score = s
            best_points = opt_pts

    # If optimization failed or didn't run, return a default valid structure
    if best_points is None:
        # Fallback to grid
        best_points = inits[0]
        best_points -= np.mean(best_points, axis=0)

    return best_points
