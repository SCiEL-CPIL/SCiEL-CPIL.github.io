import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

def _compute_score(points):
    """
    Compute the squared ratio of max to min pairwise distance.
    """
    if len(points) < 2:
        return 0.0
    # Compute pairwise distances
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_min < 1e-9:
        return 1e10
    return (d_max / d_min) ** 2

def _objective(params):
    """
    Objective function for optimization.
    params: array of shape (32,)
    """
    points = params.reshape(N, 2)
    
    # Regularization to fix translation and scale
    # Center of mass penalty
    com = np.mean(points, axis=0)
    trans_penalty = np.sum(com**2) * 1e-4
    
    # Scale penalty: keep sum of squared distances to origin near 1
    scale = np.sum(points**2)
    scale_penalty = (scale - 1.0)**2 * 1e-4
    
    score = _compute_score(points)
    
    return score + trans_penalty + scale_penalty

def _hexagonal_init(n):
    """
    Generate a hexagonal packing of n points.
    """
    points = []
    # Hexagonal lattice vectors
    a = 1.0
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    
    # Generate enough points and take the first n
    # Order by distance from origin
    candidates = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * v1 + j * v2
            candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance to origin
    dists = np.sum(candidates**2, axis=1)
    indices = np.argsort(dists)
    points = candidates[indices[:n]]
    return points

def _grid_init(n):
    """
    Generate a grid of n points.
    """
    side = int(np.ceil(np.sqrt(n)))
    # Create grid
    coords = np.arange(side)
    x, y = np.meshgrid(coords, coords)
    points = np.column_stack((x.ravel(), y.ravel()))[:n]
    # Center it
    points -= np.mean(points, axis=0)
    # Scale it
    norm = np.sqrt(np.sum(points**2))
    if norm > 0:
        points /= norm
    return points

def _spiral_init(n):
    """
    Generate a spiral of n points.
    """
    points = []
    r = 0
    theta = 0
    for i in range(n):
        r = 0.2 * np.sqrt(i + 1)
        theta = i * 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    norm = np.sqrt(np.sum(points**2))
    if norm > 0:
        points /= norm
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Initializers
    initializers = []
    initializers.append(_grid_init(N))
    initializers.append(_hexagonal_init(N))
    initializers.append(_spiral_init(N))
    
    # Add some random initializers
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(-1, 1, (N, 2))
        pts -= np.mean(pts, axis=0)
        norm = np.sqrt(np.sum(pts**2))
        if norm > 0:
            pts /= norm
        initializers.append(pts)
    
    # Optimize each initializer
    for i, init_points in enumerate(initializers):
        if time.time() > deadline:
            break
            
        init_params = init_points.flatten()
        
        # Use Powell method
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Powell', 
                options={'maxiter': 2000, 'maxfev': 10000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.success or res.fun < best_score:
                # Check if this is a valid improvement
                # We need to re-evaluate the raw score without penalties
                pts = res.x.reshape(N, 2)
                raw_score = _compute_score(pts)
                
                # Check distinctness
                # If points are too close, raw_score is huge, so it's fine.
                
                if raw_score < best_score:
                    best_score = raw_score
                    best_points = pts
        except Exception:
            pass

    # If we still have time, try a few more random restarts or refine the best one
    if time.time() < deadline - 10 and best_points is not None:
        # Refine the best point found
        try:
            init_params = best_points.flatten()
            res = minimize(
                _objective, 
                init_params, 
                method='Powell', 
                options={'maxiter': 5000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts = res.x.reshape(N, 2)
            raw_score = _compute_score(pts)
            if raw_score < best_score:
                best_score = raw_score
                best_points = pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to grid
        best_points = _grid_init(N)
        
    return best_points
