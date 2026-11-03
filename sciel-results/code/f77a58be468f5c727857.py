"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# EVOLVE-BLOCK-START

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _get_facet_data(points):
    """
    Compute convex hull and return facet normals and distances h.
    Returns:
        h: array of distances from origin to facet planes (shape M,)
        normals: array of outward unit normals (shape M, 3)
    """
    try:
        hull = ConvexHull(points)
    except:
        return None, None
    
    simplices = hull.simplices
    # Get vertices for each facet
    # simplices is (M, 3)
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    
    # Compute normals
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    
    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Handle zero normals (degenerate facets)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms
    
    # Ensure normals point away from origin
    # Distance from origin to plane: h = dot(normal, p0)
    h = np.einsum('ij,ij->i', normals, p0)
    
    # If h is negative, normal points inward. Flip it.
    mask = h < 0
    if np.any(mask):
        normals[mask] = -normals[mask]
        h[mask] = -h[mask]
        
    # Clip h to [-1, 1] for safety, though it should be in (0, 1]
    h = np.clip(h, -1.0, 1.0)
    
    return h, normals

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    h, _ = _get_facet_data(points)
    if h is None:
        return 180.0
    if len(h) == 0:
        return 180.0
    # Covering radius is max over facets of arccos(h)
    # Since arccos is decreasing, min h gives max angle
    min_h = np.min(h)
    # If min_h is close to 1, angle is 0. If min_h is close to 0, angle is 90.
    # Actually, h is cos(angle). So angle = arccos(h).
    # We want to minimize the max angle.
    angle = np.arccos(min_h)
    return angle * 180.0 / np.pi

def _smooth_surrogate(points, temperature=0.05):
    """
    Smooth surrogate for the covering radius.
    Minimizes the soft-max of the angles, which is equivalent to maximizing the soft-min of h.
    soft_min_h = T * log(sum(exp(-h/T)))
    We want to minimize -soft_min_h (to maximize h) or maximize soft_min_h.
    Let's return -soft_min_h so we can use a minimizer.
    """
    h, _ = _get_facet_data(points)
    if h is None:
        return 1e6
    if len(h) == 0:
        return 1e6
        
    # soft_min_h = T * log(sum(exp(-h/T)))
    # For numerical stability
    neg_h_over_T = -h / temperature
    max_val = np.max(neg_h_over_T)
    log_sum = max_val + np.log(np.sum(np.exp(neg_h_over_T - max_val)))
    soft_min_h = temperature * log_sum
    
    # We want to maximize soft_min_h, so minimize -soft_min_h
    return -soft_min_h

def _fibonacci_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(jitter=0.1):
    pts = _fibonacci_points()
    noise = np.random.normal(0, jitter, pts.shape)
    return _normalize(pts + noise)

def _lloyd_iteration(points, max_iter=50, time_limit=10.0, damping=0.5):
    """
    Spherical Lloyd's algorithm using vectorized facet data.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        h, normals = _get_facet_data(points)
        if h is None:
            break
            
        # Map facets to points
        # We need to accumulate the normal vector for each point from its adjacent facets.
        # This requires knowing which facets are adjacent to which points.
        # ConvexHull.simplices gives indices.
        # We can use a sparse matrix or just loop if N is small (50).
        # Let's loop for simplicity and correctness, it's fast enough for 50 points.
        
        new_points = points.copy()
        point_normals = np.zeros((N, 3))
        
        # Build point to facets map
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(ConvexHull(points).simplices): # Re-computing hull is bad. 
             # We already have normals, but we lost the simplices mapping in _get_facet_data.
             # Let's refactor _get_facet_data to return simplices too or just re-compute here?
             # Better: modify _get_facet_data to return simplices.
             pass
        
        # Refactor: Get data once
        try:
            hull = ConvexHull(points)
        except:
            break
            
        simplices = hull.simplices
        p0 = points[simplices[:, 0]]
        p1 = points[simplices[:, 1]]
        p2 = points[simplices[:, 2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normals = normals / norms
        
        h = np.einsum('ij,ij->i', normals, p0)
        mask = h < 0
        if np.any(mask):
            normals[mask] = -normals[mask]
            h[mask] = -h[mask]
            
        # Accumulate normals for each point
        for fi in range(len(simplices)):
            for pi in simplices[fi]:
                point_normals[pi] += normals[fi]
                
        # Move points
        for pi in range(N):
            if np.linalg.norm(point_normals[pi]) > 1e-10:
                target = point_normals[pi] / np.linalg.norm(point_normals[pi])
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
                
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            break
            
    return best_points

def _optimize_smooth(points, time_limit=30.0):
    """
    Optimize using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        # Adaptive temperature
        elapsed = time.time() - start_time
        # Start with larger T, decrease over time
        T = 0.1 * (0.5 ** (elapsed / 10.0))
        return _smooth_surrogate(pts, temperature=T)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def _nelder_mead_optimize(points, time_limit=15.0):
    """
    Nelder-Mead optimization on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Initialization
    np.random.seed(42)
    init_points = _fibonacci_points()
    
    # Stage 1: Lloyd's to get into a good basin
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 10:
        points_lloyd = _lloyd_iteration(init_points, max_iter=50, time_limit=min(remaining - 5, 10))
    else:
        points_lloyd = init_points
        
    # Stage 2: Smooth Gradient Optimization
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 20:
        points_smooth = _optimize_smooth(points_lloyd, time_limit=min(remaining - 10, 40))
    else:
        points_smooth = points_lloyd
        
    # Stage 3: Nelder-Mead Polish on True Objective
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 10:
        points_final = _nelder_mead_optimize(points_smooth, time_limit=min(remaining - 5, 20))
    else:
        points_final = points_smooth
        
    # Final check and normalization
    points_final = _normalize(points_final)
    
    return points_final

# EVOLVE-BLOCK-END
