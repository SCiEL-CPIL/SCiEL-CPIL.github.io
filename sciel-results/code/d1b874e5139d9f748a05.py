import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Returns 14 points in 3D maximizing (d_min/d_max)^2.
    Uses a structured initial guess (subset of 3D cross polytope / cube)
    plus deterministic optimization.
    """
    deadline = time.time() + 600  # safety margin before 630s kill

    n = 14
    d = 3

    # Start with a structured configuration:
    # Use a mix of cube vertices and face centers to get a good spread
    # Actually, let's try a better structured initial guess:
    # 14 points can be thought of as a subset of a 3D lattice or polytope.
    # A good approach: start with points on a sphere or in a symmetric arrangement.
    # Let's use a deterministic grid-based initialization with jitter.

    # Create a 2x2x4 grid? That's 16 points. We need 14.
    # Or 3x3x3 is 27.
    # Let's try a spherical code initialization.
    
    # For better performance, use a known good structure:
    # The 14-vertex polytope in 3D... 
    # Actually, let's just use a well-distributed random initialization with fixed seed
    # and then optimize.

    # Better: use a grid initialization
    # 2 x 3 x 3 = 18 points, too many
    # 2 x 2 x 4 = 16, we can remove 2
    # Let's try: vertices of a rectangular box
    
    # Initialize with points on a cube + some interior points
    # Or better: use a deterministic "Fibonacci sphere" style distribution
    
    # For 14 points, a good starting point is to place them symmetrically.
    # Let's use a combination of:
    # - 8 cube vertices
    # - 6 face centers
    # That's 14 points!
    
    # Cube vertices at (±1, ±1, ±1)
    cube_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    
    # Face centers: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    face_centers = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    
    points = np.vstack([cube_verts, face_centers])
    
    # Now optimize using gradient-free methods or direct search
    # Use a simple iterative improvement: for each pair, adjust points to increase min distance
    
    # Compute current metric
    def compute_metric(pts):
        if np.any(~np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if dists.size == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2
    
    best_points = points.copy()
    best_metric = compute_metric(best_points)
    
    # Simple local optimization: perturb each point and keep if improves
    rng = np.random.default_rng(42)
    step_size = 0.1
    
    for iteration in range(500):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            for _ in range(20):  # Try multiple perturbations per point
                perturbed = best_points.copy()
                noise = rng.normal(0, step_size, size=d)
                perturbed[i] += noise
                
                metric = compute_metric(perturbed)
                if metric > best_metric:
                    best_points = perturbed
                    best_metric = metric
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
        else:
            step_size = max(step_size * 0.95, 0.01)
    
    return best_points
