import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 120.0  # Safety margin from 150s

def _compute_covering_radius(points):
    """
    Fast computation of covering radius using nearest-neighbor distances.
    For well-distributed points on sphere, the covering radius is approximately
    the maximum nearest-neighbor distance.
    """
    # Compute pairwise angular distances
    # dot[i,j] = points[i] . points[j]
    dot = np.clip(points @ points.T, -1.0, 1.0)
    
    # For each point, find the nearest neighbor (excluding self)
    # The covering radius is bounded by the max nearest-neighbor distance
    # But for a better estimate, we use the circumradius approach
    
    # Fast proxy: max over all pairs of the angular distance to nearest other point
    # For each point i, the nearest other point is argmin_j dot[i,j] for j != i
    # Set diagonal to 1 (max dot = min angle)
    dot_no_self = dot.copy()
    np.fill_diagonal(dot_no_self, 1.0)
    
    # Nearest neighbor dot product for each point
    nn_dot = np.max(dot_no_self, axis=1)
    
    # Angular distance to nearest neighbor
    nn_angles = np.arccos(nn_dot)
    
    # The covering radius is at least half the max gap between nearest neighbors
    # But a better estimate uses the fact that the worst point is roughly
    # at the centroid of the largest empty cap
    
    # For a quick objective, use the max nearest-neighbor angle as a lower bound
    # and add a small correction
    max_nn_angle = np.max(nn_angles)
    
    # Better: compute the actual covering radius via hull for final evaluation,
    # but use this fast proxy during optimization
    return max_nn_angle

def _compute_circumradii(points):
    """
    Compute the squared circumradii of the facets of the convex hull of points.
    Returns the sum of squared radii and the max squared radius.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # If hull fails, return a large penalty
        return 1e10, 1e10
    
    sum_sq_rad = 0.0
    max_sq_rad = 0.0
    
    # For each facet (simplex)
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the normal to the plane containing v1, v2, v3
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            sum_sq_rad += 1e10
            max_sq_rad = max(max_sq_rad, 1e10)
            continue
            
        c = n / norm_n
        
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0, 1.0)
        
        if dot < 0:
            c = -c
            dot = -dot
            
        theta = np.arccos(dot)
        theta_sq = theta * theta
        
        sum_sq_rad += theta_sq
        if theta_sq > max_sq_rad:
            max_sq_rad = theta_sq
            
    return sum_sq_rad, max_sq_rad

def _objective(x):
    """
    Objective function for scipy.optimize.
    x is a flat array of shape (3*N,).
    Uses fast covering radius proxy.
    """
    points = x.reshape(N, 3)
    
    # Penalty for not being on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 100.0
    
    # Compute covering radius proxy (fast)
    cover_rad = _compute_covering_radius(points)
    
    # We want to minimize the covering radius
    # Use squared for smoothness
    return cover_rad * cover_rad + sphere_penalty

def _objective_hull(x):
    """
    Objective using hull-based circumradii (slower but more accurate).
    """
    points = x.reshape(N, 3)
    
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 100.0
    
    sum_sq_rad, _ = _compute_circumradii(points)
    
    return sum_sq_rad + sphere_penalty

def _optimize_from_start(start_points, max_iter=200):
    """
    Run local optimization from a given start.
    Uses L-BFGS-B for better performance.
    """
    x0 = start_points.flatten()
    
    start_time = time.time()
    try:
        res = scipy.optimize.minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return start_points, _objective(x0)
    
    elapsed = time.time() - start_time
    x_opt = res.x.reshape(N, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    
    score = _objective(x_opt.flatten())
    
    return x_opt, score

def _optimize_hull_refine(points, max_iter=50):
    """
    Refine using hull-based objective for better accuracy.
    """
    x0 = points.flatten()
    
    start_time = time.time()
    try:
        res = scipy.optimize.minimize(
            _objective_hull, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return points
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    
    return x_opt

def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate n random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _icosahedron_based(n):
    """Generate points starting from icosahedron vertices and expanding."""
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 12:
        # Select 12 closest to fibonacci
        fib = _fibonacci_sphere(12)
        # Just use first n icosahedron vertices
        return verts[:n]
    
    # Start with icosahedron and add points in largest gaps
    points = verts.copy()
    
    for _ in range(n - 12):
        # Find the point on sphere farthest from current points
        # Use fibonacci spiral as candidate locations
        candidates = _fibonacci_sphere(100)
        # For each candidate, compute min distance to existing points
        best_candidate = None
        best_min_dist = -1
        for cand in candidates:
            dists = np.arccos(np.clip(points @ cand, -1.0, 1.0))
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = cand
        
        if best_candidate is not None:
            points = np.vstack([points, best_candidate])
    
    return points[:n]

def construct():
    start_time = time.time()
    best_points = None
    best_score = np.inf
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))
    
    # 3. Random points with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        starts.append(_random_sphere(N, seed=seed))
    
    # 4. Rotated fibonacci
    for angle in [0.1, 0.5, 1.0]:
        fib = _fibonacci_sphere(N)
        # Rotate around z-axis
        theta = angle
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        starts.append(fib @ R.T)
    
    # Phase 1: Fast optimization with nearest-neighbor proxy
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 30:
            break
            
        points, score = _optimize_from_start(start, max_iter=150)
        
        if score < best_score:
            best_score = score
            best_points = points
    
    # Phase 2: Hull-based refinement of best solution
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 15:
        refined = _optimize_hull_refine(best_points, max_iter=30)
        hull_score, _ = _compute_circumradii(refined)
        if hull_score < 1e6:  # Valid hull
            best_points = refined
    
    # Phase 3: If time remains, try more random restarts
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        for seed in range(20, 40):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            start = _random_sphere(N, seed=seed)
            points, score = _optimize_from_start(start, max_iter=100)
            if score < best_score:
                best_score = score
                best_points = points
        
        # Final hull refinement
        if time.time() - start_time < TIME_LIMIT - 10:
            refined = _optimize_hull_refine(best_points, max_iter=20)
            hull_score, _ = _compute_circumradii(refined)
            if hull_score < 1e6:
                best_points = refined
    
    # Final normalization
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
