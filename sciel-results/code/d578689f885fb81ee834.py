"""Spherical Voronoi relaxation with direct max-circumradius minimization."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_circumcenter(a, b, c):
    """Compute the spherical circumcenter of triangle (a,b,c) on unit sphere.
    
    The circumcenter p satisfies p·a = p·b = p·c.
    p is parallel to (a-b) × (a-c).
    """
    u = a - b
    v = a - c
    p_dir = np.cross(u, v)
    p_norm = np.linalg.norm(p_dir)
    
    if p_norm < 1e-14:
        # Degenerate: use normalized centroid
        centroid = a + b + c
        cn = np.linalg.norm(centroid)
        if cn < 1e-14:
            return None
        return centroid / cn
    
    p = p_dir / p_norm
    
    # Ensure correct orientation: circumcenter should be near centroid
    centroid = (a + b + c) / 3.0
    if np.dot(p, centroid) < 0:
        p = -p
    
    return p

def _facet_circumradius_rad(a, b, c):
    """Compute circumradius in radians."""
    cc = _spherical_circumcenter(a, b, c)
    if cc is None:
        return 0.0
    dot_val = np.clip(np.dot(cc, a), -1.0, 1.0)
    return np.arccos(dot_val)

def _compute_covering_radius_deg(points):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        cr = _facet_circumradius_rad(a, b, c)
        if cr > max_cr:
            max_cr = cr
    
    return np.degrees(max_cr)

def _slerp(p1, p2, t):
    """Spherical linear interpolation between p1 and p2 with parameter t in [0,1]."""
    dot_val = np.clip(np.dot(p1, p2), -1.0, 1.0)
    angle = np.arccos(dot_val)
    
    if angle < 1e-14:
        return p1.copy()
    
    sin_angle = np.sin(angle)
    if sin_angle < 1e-14:
        return p1.copy()
    
    coeff1 = np.sin((1 - t) * angle) / sin_angle
    coeff2 = np.sin(t * angle) / sin_angle
    result = coeff1 * p1 + coeff2 * p2
    norm = np.linalg.norm(result)
    if norm < 1e-14:
        return p1.copy()
    return result / norm

def _fibonacci_init(n, offset=0.5):
    """Fibonacci spiral initialization."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _random_init(n, seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _voronoi_centroid(points, point_idx):
    """Compute the weighted Voronoi centroid for a point using its incident facets.
    
    Weight each facet by area * circumradius to bias toward reducing large cells.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points[point_idx].copy()
    
    # Find all facets containing this point
    point_facets = []
    for simplex in hull.simplices:
        if point_idx in simplex:
            point_facets.append(simplex)
    
    if not point_facets:
        return points[point_idx].copy()
    
    weighted_sum = np.zeros(3)
    total_weight = 0.0
    
    for simplex in point_facets:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        cc = _spherical_circumcenter(a, b, c)
        if cc is None:
            continue
        
        # Compute spherical area
        triple = np.dot(a, np.cross(b, c))
        denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
        area = 2.0 * np.arctan2(abs(triple), denom)
        
        cr = _facet_circumradius_rad(a, b, c)
        
        if area > 1e-12 and cr > 1e-12:
            weight = area * cr
            weighted_sum += weight * cc
            total_weight += weight
    
    if total_weight < 1e-14:
        return points[point_idx].copy()
    
    centroid = weighted_sum / total_weight
    norm = np.linalg.norm(centroid)
    if norm < 1e-14:
        return points[point_idx].copy()
    
    return centroid / norm

def _lloyd_step(points, damping=0.5):
    """One Lloyd step: move each point toward its Voronoi centroid."""
    new_points = np.zeros_like(points)
    for i in range(len(points)):
        target = _voronoi_centroid(points, i)
        new_points[i] = _slerp(points[i], target, damping)
    
    return new_points

def _direct_max_cr_optimize(points, max_time=30.0):
    """Directly optimize to reduce the maximum facet circumradius.
    
    Repeatedly find the worst facet and move its vertices toward the circumcenter.
    Use adaptive step sizes and accept only improvements.
    """
    start_time = time.time()
    current = points.copy()
    current_radius = _compute_covering_radius_deg(current)
    
    base_step = 0.1
    step = base_step
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        if no_improve > max_no_improve:
            break
        
        try:
            hull = ConvexHull(current)
        except Exception:
            break
        
        # Find worst facet
        max_cr_rad = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = current[simplex[0]], current[simplex[1]], current[simplex[2]]
            cr = _facet_circumradius_rad(a, b, c)
            if cr > max_cr_rad:
                max_cr_rad = cr
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Get circumcenter of worst facet
        a, b, c = current[worst_simplex[0]], current[worst_simplex[1]], current[worst_simplex[2]]
        cc = _spherical_circumcenter(a, b, c)
        if cc is None:
            break
        
        # Try moving vertices toward circumcenter
        trial = current.copy()
        for idx in worst_simplex:
            trial[idx] = _slerp(current[idx], cc, step)
        
        # Check separation
        sep_ok = True
        for i in range(len(trial)):
            for j in range(i + 1, len(trial)):
                if np.linalg.norm(trial[i] - trial[j]) < 1e-10:
                    sep_ok = False
                    break
            if not sep_ok:
                break
        
        if not sep_ok:
            step *= 0.5
            if step < 1e-8:
                break
            continue
        
        new_radius = _compute_covering_radius_deg(trial)
        
        if new_radius < current_radius - 1e-10:
            current = trial
            current_radius = new_radius
            no_improve = 0
            # Slowly increase step if improving
            step = min(base_step, step * 1.05)
        else:
            no_improve += 1
            step *= 0.7
            if step < 1e-8:
                break
    
    return current

def _combined_relax(points, max_time=60.0):
    """Combined relaxation: Lloyd steps followed by direct max-CR optimization."""
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_radius = _compute_covering_radius_deg(best)
    
    # Phase 1: Lloyd relaxation
    lloyd_end = start_time + max_time * 0.4
    damping = 0.5
    while time.time() < lloyd_end:
        new_points = _lloyd_step(current, damping=damping)
        
        # Check separation
        sep_ok = True
        for i in range(len(new_points)):
            for j in range(i + 1, len(new_points)):
                if np.linalg.norm(new_points[i] - new_points[j]) < 1e-10:
                    sep_ok = False
                    break
            if not sep_ok:
                break
        
        if not sep_ok:
            damping *= 0.8
            if damping < 0.01:
                break
            continue
        
        radius = _compute_covering_radius_deg(new_points)
        if radius < best_radius:
            best_radius = radius
            best = new_points.copy()
        
        current = new_points
        
        # Reduce damping slowly
        damping = max(0.1, damping * 0.995)
    
    # Phase 2: Direct max-CR optimization
    remaining = max_time - (time.time() - start_time)
    if remaining > 5.0:
        optimized = _direct_max_cr_optimize(current, max_time=remaining - 2.0)
        opt_radius = _compute_covering_radius_deg(optimized)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best = optimized.copy()
    
    return best, best_radius

def construct():
    """Construct 50 points on the unit sphere using spherical Voronoi relaxation."""
    start_time = time.time()
    max_total_time = 110.0
    
    best_points = None
    best_radius = float('inf')
    
    # Multiple initializations
    initializations = [
        _fibonacci_init(N, offset=0.5),
        _fibonacci_init(N, offset=0.3),
        _fibonacci_init(N, offset=0.7),
        _random_init(N, seed=42),
        _random_init(N, seed=123),
        _random_init(N, seed=7),
    ]
    
    # Time budget per initialization
    time_per_init = max_total_time / len(initializations)
    
    for init_points in initializations:
        elapsed = time.time() - start_time
        remaining = max_total_time - elapsed
        if remaining < 15.0:
            break
        
        alloc_time = min(time_per_init, remaining - 5.0)
        
        result, radius = _combined_relax(init_points, max_time=alloc_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = result.copy()
    
    # Final polish on the best
    remaining = max_total_time - (time.time() - start_time)
    if remaining > 10.0 and best_points is not None:
        final_result = _direct_max_cr_optimize(best_points, max_time=remaining - 3.0)
        final_radius = _compute_covering_radius_deg(final_result)
        if final_radius < best_radius:
            best_points = final_result
            best_radius = final_radius
    
    return best_points
