import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 50

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _covering_radius_exact(points):
    """
    Compute the exact covering radius in radians.
    Returns 180/pi if hull fails.
    """
    if len(points) < 3:
        return np.pi
    try:
        hull = ConvexHull(points)
    except:
        return np.pi
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Filter out degenerate facets
    valid = norms[:, 0] > 1e-10
    if not np.any(valid):
        return np.pi
        
    normals = normals[valid] / norms[valid]
    p1 = p1[valid]
    
    # Ensure normals point outward
    dots = np.sum(normals * p1, axis=1)
    normals = np.where(dots[:, None] < 0, -normals, normals)
    
    # h = dot(normal, p1) = cos(circumradius)
    h = np.sum(normals * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    
    # Circumradius angle = arccos(h)
    angles = np.arccos(h)
    
    return np.max(angles)

def _smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp of facet circumradii.
    This provides gradients for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return 1.0  # Large penalty
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return 1.0
        
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = norms[:, 0] > 1e-10
    if not np.any(valid):
        return 1.0
        
    normals = normals[valid] / norms[valid]
    p1 = p1[valid]
    
    dots = np.sum(normals * p1, axis=1)
    normals = np.where(dots[:, None] < 0, -normals, normals)
    
    h = np.sum(normals * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    # Log-sum-exp to approximate max
    max_angle = np.max(angles)
    logsumexp = max_angle + np.log(np.sum(np.exp(angles - max_angle)))
    return logsumexp

def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)

def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = list(verts)
    
    # Add 38 more points
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        # Check minimum distance to existing points
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
            
    return np.array(points[:N])

def _lloyd_relaxation(points, max_iter=50, time_limit=5.0):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) to improve initial config.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius_exact(points)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
            
        new_points = points.copy()
        
        # Map point index to list of facet indices
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
                
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
                
            normal_sum = np.zeros(3)
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                    
                normal_sum += n
                
            if np.linalg.norm(normal_sum) > 1e-10:
                target = normal_sum / np.linalg.norm(normal_sum)
                # Move towards the centroid of the adjacent facet normals
                damping = 0.5
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
                
        new_points = _normalize(new_points)
        new_cost = _covering_radius_exact(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
        else:
            break
            
    return points

def _refine_with_lbfgs(points, time_limit=10.0):
    """
    Refine points using L-BFGS-B on the smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_surrogate(params, beta=100.0)
        
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def _refine_with_nm(points, time_limit=10.0):
    """
    Final polish with Nelder-Mead on the exact objective.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_exact(pts)
        
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0 # Safety margin for 150s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
        
    # Generate diverse initializations
    initializations = [
        _fibonacci_points(),
        _random_points(seed=1),
        _random_points(seed=2),
        _random_points(seed=3),
        _icosahedron_based_points(),
    ]
    
    # Add some perturbed Fibonacci
    rng = np.random.RandomState(42)
    fib = _fibonacci_points()
    for seed in [0, 1, 2]:
        noise = rng.randn(N, 3) * 0.05
        initializations.append(_normalize(fib + noise))
        
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Lloyd Relaxation + L-BFGS
    # Allocate time for each initialization
    n_inits = len(initializations)
    time_per_init = min(10.0, (total_time_limit * 0.6) / n_inits)
    
    for init_pts in initializations:
        if _time_left() < 20:
            break
            
        # 1. Lloyd Relaxation
        t_start = time.time()
        pts = _lloyd_relaxation(init_pts, max_iter=30, time_limit=min(5.0, _time_left() - 15))
        cost = _covering_radius_exact(pts)
        
        # 2. L-BFGS on smooth surrogate
        if _time_left() > 15:
            t_limit_lbfgs = min(5.0, _time_left() - 10)
            pts = _refine_with_lbfgs(pts, time_limit=t_limit_lbfgs)
            cost = _covering_radius_exact(pts)
            
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius_exact(best_points)
        
    # Phase 2: Final Polish with Nelder-Mead on exact objective
    if _time_left() > 10:
        t_limit_nm = min(20.0, _time_left() - 5)
        pts = _refine_with_nm(best_points, time_limit=t_limit_nm)
        cost = _covering_radius_exact(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    # Phase 3: One more round of L-BFGS + NM if time permits
    if _time_left() > 15:
        t_limit_lbfgs2 = min(10.0, _time_left() - 10)
        pts = _refine_with_lbfgs(best_points, time_limit=t_limit_lbfgs2)
        cost = _covering_radius_exact(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
        if _time_left() > 5:
            t_limit_nm2 = min(5.0, _time_left() - 2)
            pts = _refine_with_nm(best_points, time_limit=t_limit_nm2)
            cost = _covering_radius_exact(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()

    return _normalize(best_points)
