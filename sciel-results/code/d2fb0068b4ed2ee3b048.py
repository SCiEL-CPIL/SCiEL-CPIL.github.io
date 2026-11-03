"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _compute_facets(points):
    """
    Compute facet normals and heights for a convex hull.
    Returns (normals, h, simplices, valid) or None on failure.
    """
    try:
        hull = ConvexHull(points)
    except:
        return None
    
    simplices = hull.simplices
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norm_n = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = norm_n.squeeze() > 1e-10
    if not np.any(valid):
        return None
    
    normals = normals / norm_n
    dots = np.einsum('ij,ij->i', normals, p1)
    normals = np.where(dots[:, np.newaxis] < 0, -normals, normals)
    
    h = np.einsum('ij,ij->i', normals, p1)
    h = np.clip(h, -1.0, 1.0)
    
    return normals, h, simplices, valid


def _covering_radius(points):
    """Compute the covering radius in degrees."""
    if len(points) < 3:
        return 180.0
    result = _compute_facets(points)
    if result is None:
        return 180.0
    normals, h, simplices, valid = result
    min_h = np.min(h[valid])
    angle = np.arccos(min_h)
    return float(angle * 180.0 / np.pi)


def _covering_radius_radians(points):
    """Covering radius in radians."""
    return _covering_radius(points) * np.pi / 180.0


def _subgradient_covering(points, tol=1e-6):
    """
    Compute subgradient of the covering radius w.r.t. each point.
    The covering radius is arccos(min_h). For active facets (h close to min_h),
    the gradient w.r.t. point p_i is -n_f / sin(theta) for each active facet f
    containing p_i, where n_f is the facet normal and theta is the covering angle.
    
    Returns (grad, cost_rad) where grad has shape (N, 3).
    """
    result = _compute_facets(points)
    if result is None:
        return None, None
    normals, h, simplices, valid = result
    
    min_h = np.min(h[valid])
    if min_h >= 1.0 - 1e-12:
        return np.zeros((N, 3)), 0.0
    
    theta = np.arccos(min_h)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        sin_theta = 1e-10
    
    # Active facets: those with h close to min_h
    active = valid & (h < min_h + tol)
    
    # Gradient: for each active facet, add -n_f / sin_theta to each vertex
    grad = np.zeros((N, 3))
    for fi in range(len(simplices)):
        if not active[fi]:
            continue
        n_f = normals[fi]
        for pi in simplices[fi]:
            # Gradient of arccos(h) w.r.t. p_i: -n_f / sin(theta)
            # But we need to project to the tangent plane of the sphere
            grad[pi] += -n_f / sin_theta
    
    # Project gradients to tangent plane: g_tan = g - (g·p)*p
    for i in range(N):
        dot_gp = np.dot(grad[i], points[i])
        grad[i] = grad[i] - dot_gp * points[i]
    
    return grad, theta


def _subgradient_descent(points, max_iter=200, time_limit=30, lr=0.05):
    """
    Subgradient descent on the covering radius using active facet normals.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius_radians(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad, theta = _subgradient_covering(points)
        if grad is None:
            break
        
        # Adaptive learning rate
        step = lr / (1.0 + i * 0.01)
        
        # Take step and project to sphere
        new_points = points - step * grad
        new_points = _normalize(new_points)
        
        new_cost = _covering_radius_radians(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Halve learning rate
            lr = max(lr * 0.5, 1e-6)
            # Try smaller step
            new_points = points - (step * 0.1) * grad
            new_points = _normalize(new_points)
            new_cost = _covering_radius_radians(new_points)
            if new_cost < current_cost:
                current_cost = new_cost
                points = new_points
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = points.copy()
            else:
                break
    
    return best_points


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


def _jittered_fibonacci(seed=0, jitter=0.1):
    """Fibonacci points with random jitter for diversity."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * jitter
    pts = pts + noise
    return _normalize(pts)


def _spherical_lloyd(points, max_iter=50, time_limit=10, damping=0.5):
    """Spherical Lloyd's algorithm (Voronoi relaxation)."""
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        result = _compute_facets(points)
        if result is None:
            break
        normals, h, simplices, valid = result
        
        normal_sums = np.zeros((N, 3))
        for fi, simplex in enumerate(simplices):
            if not valid[fi]:
                continue
            for pi in simplex:
                normal_sums[pi] += normals[fi]
        
        new_points = points.copy()
        for pi in range(N):
            ns = normal_sums[pi]
            ns_norm = np.linalg.norm(ns)
            if ns_norm > 1e-10:
                target = ns / ns_norm
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


def _diff_evolve_objective(params):
    """Objective for differential evolution."""
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    return _covering_radius(pts)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 140.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Generate diverse initializations
    initializations = [
        _fibonacci_points(),
        _jittered_fibonacci(seed=1, jitter=0.05),
        _jittered_fibonacci(seed=2, jitter=0.1),
        _jittered_fibonacci(seed=3, jitter=0.15),
        _random_points(seed=42),
        _random_points(seed=123),
    ]
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Quick Lloyd + subgradient descent on each initialization
    for init_pts in initializations:
        if _time_left() < 30:
            break
        
        # Lloyd relaxation
        pts = _spherical_lloyd(init_pts, max_iter=30, time_limit=min(8, _time_left() - 25), damping=0.5)
        cost = _covering_radius(pts)
        
        # Subgradient descent polish
        if _time_left() > 25:
            pts2 = _subgradient_descent(pts, max_iter=100, time_limit=min(10, _time_left() - 20), lr=0.05)
            cost2 = _covering_radius(pts2)
            if cost2 < cost:
                pts = pts2
                cost = cost2
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Differential evolution for basin-hopping (uses remaining time)
    if _time_left() > 40:
        de_time = min(30, _time_left() - 30)
        try:
            # Use best point as a seed in the initial population
            de_result = differential_evolution(
                _diff_evolve_objective,
                bounds=[(-1.0, 1.0)] * (N * 3),
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                init='sobol',
                polish=False,
                updating='immediate',
                callback=None,
                workers=1,
            )
            if de_result.success or de_result.fun < best_cost:
                de_pts = _normalize(de_result.x.reshape(N, 3))
                de_cost = _covering_radius(de_pts)
                if de_cost < best_cost:
                    best_cost = de_cost
                    best_points = de_pts
        except:
            pass
    
    # Phase 3: Final subgradient descent polish
    if _time_left() > 15:
        pts = _subgradient_descent(best_points, max_iter=150, time_limit=min(10, _time_left() - 10), lr=0.03)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Final Lloyd polish
    if _time_left() > 10:
        pts = _spherical_lloyd(best_points, max_iter=30, time_limit=min(5, _time_left() - 6), damping=0.2)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Final subgradient micro-polish
    if _time_left() > 8:
        pts = _subgradient_descent(best_points, max_iter=100, time_limit=min(3, _time_left() - 4), lr=0.01)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    return best_points
# EVOLVE-BLOCK-END
