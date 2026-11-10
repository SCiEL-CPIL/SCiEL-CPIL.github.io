# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(x, k=10):
    """
    Objective function to minimize.
    We want to maximize the minimum distance relative to the maximum.
    
    Strategy:
    1. Normalize the points so that the maximum distance is 1. 
       This makes the problem scale-invariant and focuses optimization on shape.
    2. Maximize the sum of the smallest k pairwise distances.
       Maximizing the sum of the smallest distances ensures that no pair is 
       arbitrarily close, effectively pushing up the d_min.
       Since the scale is fixed (d_max=1), maximizing d_min is equivalent to 
       maximizing the ratio d_min/d_max.
       
    We minimize the negative of this sum.
    """
    n = len(x)
    points = x.reshape((n, 2))
    
    # Compute pairwise distances
    d = pdist(points)
    
    # Handle degenerate cases
    if np.allclose(d, 0):
        return 1e10
    
    d_max = np.max(d)
    if d_max < 1e-8:
        return 1e10
        
    # Normalize distances so d_max = 1
    d_norm = d / d_max
    
    # We want to maximize the sum of the smallest k distances
    # Sorting is O(n log n) but n=120 is small. 
    # Alternatively, we can use a smooth approximation, but sorting is fine for 16 points.
    d_sorted = np.sort(d_norm)
    
    # Sum of the smallest k
    # Using k=10 ensures we are not just optimizing the single minimum, 
    # but a robust "core" minimum distance, which helps with global packing.
    obj_val = -np.sum(d_sorted[:k])
    
    return obj_val

def _optimize_from_seed(seed_points, max_iter=500, time_limit=10.0):
    """
    Run L-BFGS-B optimization starting from a seed configuration.
    """
    n = len(seed_points)
    x0 = seed_points.flatten()
    
    # Use L-BFGS-B as it handles bounds well and is efficient for this size
    # We don't strictly need bounds if we normalize, but keeping points in a 
    # reasonable box helps convergence. Since we normalize inside the objective,
    # the absolute scale doesn't matter, but relative positions do.
    
    # Gradient is not provided, so L-BFGS-B will use numerical differentiation.
    # This is acceptable for n=32 variables.
    
    res = minimize(
        _smooth_objective, 
        x0, 
        args=(10,), # k=10
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Reshape result
    opt_points = res.x.reshape((n, 2))
    
    # Calculate actual score
    score = _compute_score(opt_points)
    
    return opt_points, score

def _generate_seeds(n_points=16, n_seeds=20, rng=None):
    """
    Generate diverse initial configurations.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    seeds = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    seeds.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 3. Hexagonal Lattice
    pts_hex = []
    for row in range(4):
        for col in range(4):
            x = col * 1.0
            y = row * np.sqrt(3)/2
            if col % 2 == 1:
                y += np.sqrt(3)/4
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    pts_hex -= pts_hex.mean(axis=0)
    seeds.append(pts_hex)
    
    # 4. Two Concentric Circles
    r1, r2 = 0.5, 1.5
    a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    a2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
    p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
    seeds.append(np.vstack([p1, p2]))
    
    # 5. Random Seeds
    for _ in range(n_seeds):
        pts = rng.standard_normal((n_points, 2))
        # Normalize to unit scale to help optimizer
        d = pdist(pts)
        if np.max(d) > 0:
            pts = pts / np.max(d)
        seeds.append(pts)
        
    return seeds

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    n_points = 16
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Generate seeds
    seeds = _generate_seeds(n_points=n_points, n_seeds=30, rng=rng)
    
    # Sort seeds by initial score to prioritize promising starts? 
    # Or just iterate. Let's iterate.
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        # Check time budget for this optimization
        # L-BFGS-B is usually fast for 32 vars. 
        # We allow up to 10 seconds per seed if we have time, but usually it's <1s.
        
        try:
            opt_pts, score = _optimize_from_seed(seed, max_iter=1000)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
                
                # If we find a very good score, we might want to refine further
                # but L-BFGS-B is quite robust.
                
        except Exception as e:
            # If optimization fails, skip
            continue
            
    # If we still have time, try a few more random restarts with tighter tolerance
    while time.time() < deadline - 10:
        seed = rng.standard_normal((n_points, 2))
        d = pdist(seed)
        if np.max(d) > 0:
            seed = seed / np.max(d)
            
        try:
            opt_pts, score = _optimize_from_seed(seed, max_iter=500)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
