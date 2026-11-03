"""Spherical Lloyd's algorithm with damped relaxation and improved initialization."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_triangle_circumcenter(a, b, c):
    """Compute the spherical circumcenter of triangle (a,b,c) on unit sphere.
    
    The circumcenter p satisfies p·a = p·b = p·c.
    We solve for p in the plane spanned by (a-b) and (a-c).
    """
    # Solve linear system: p·(a-b) = 0, p·(a-c) = 0
    # p is orthogonal to both (a-b) and (a-c)
    u = a - b
    v = a - c
    
    # p is parallel to u × v
    p_dir = np.cross(u, v)
    p_norm = np.linalg.norm(p_dir)
    
    if p_norm < 1e-12:
        # Degenerate case: use normalized centroid
        centroid = a + b + c
        cn = np.linalg.norm(centroid)
        if cn < 1e-12:
            return None
        return centroid / cn
    
    p = p_dir / p_norm
    
    # Ensure p is on the same side as the outward normal of the triangle
    # Outward normal from origin through triangle
    normal = np.cross(b - a, c - a)
    nn = np.linalg.norm(normal)
    if nn > 1e-12:
        normal = normal / nn
        # The circumcenter should be in the direction of the normal (or opposite)
        # Check which side: the circumcenter of a small triangle is near the centroid
        centroid = (a + b + c) / 3.0
        if np.dot(p, centroid) < 0:
            p = -p
    
    return p

def _spherical_triangle_area(a, b, c):
    """Compute spherical area of triangle using vector formula."""
    # Area = 2 * arctan(|a·(b×c)| / (1 + a·b + b·c + c·a))
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    
    if abs(denom) < 1e-14 and abs(triple) < 1e-14:
        return 0.0
    
    return 2.0 * np.arctan2(abs(triple), denom)

def _facet_circumradius(a, b, c):
    """Compute circumradius (angular distance from circumcenter to vertices) in radians."""
    cc = _spherical_triangle_circumcenter(a, b, c)
    if cc is None:
        return 0.0
    
    # Angular distance from cc to a
    dot_val = np.clip(np.dot(cc, a), -1.0, 1.0)
    return np.arccos(dot_val)

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        cr = _facet_circumradius(a, b, c)
        if cr > max_cr:
            max_cr = cr
    
    return np.degrees(max_cr)

def _lloyd_step_damped(points, damping=0.5):
    """Perform one damped Lloyd's step.
    
    For each point, compute the area-weighted centroid of incident facet circumcenters,
    then move toward it with damping to prevent overshoot.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    n_points = len(points)
    targets = np.zeros_like(points)
    
    # Map each point to its incident facets
    point_facets = [[] for _ in range(n_points)]
    for simplex in hull.simplices:
        for v in simplex:
            point_facets[v].append(simplex)
    
    for i in range(n_points):
        facets = point_facets[i]
        if not facets:
            targets[i] = points[i]
            continue
        
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for simplex in facets:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            cc = _spherical_triangle_circumcenter(a, b, c)
            area = _spherical_triangle_area(a, b, c)
            
            if cc is not None and area > 1e-12:
                weighted_sum += area * cc
                total_weight += area
        
        if total_weight > 1e-12:
            centroid = weighted_sum / total_weight
            cn = np.linalg.norm(centroid)
            if cn > 1e-12:
                targets[i] = centroid / cn
            else:
                targets[i] = points[i]
        else:
            targets[i] = points[i]
    
    # Apply damped update: move toward target by damping factor
    new_points = np.zeros_like(points)
    for i in range(n_points):
        # Spherical interpolation with damping
        dot_val = np.clip(np.dot(points[i], targets[i]), -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        if angle < 1e-12:
            new_points[i] = points[i]
        else:
            # Move by damping * angle along the geodesic
            move_angle = damping * angle
            sin_move = np.sin(move_angle)
            sin_total = np.sin(angle)
            
            if sin_total < 1e-12:
                new_points[i] = points[i]
            else:
                coeff1 = np.cos(move_angle)
                coeff2 = sin_move / sin_total
                new_points[i] = coeff1 * points[i] + coeff2 * targets[i]
                nn = np.linalg.norm(new_points[i])
                if nn > 1e-12:
                    new_points[i] = new_points[i] / nn
                else:
                    new_points[i] = points[i]
    
    return new_points

def _separation_check(points, min_sep=1e-8):
    """Check minimum pairwise separation. Returns True if all points are sufficiently separated."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_sep:
                return False
    return True

def _local_optimize_covering(points, max_time=10.0):
    """Local optimization: iteratively reduce the maximum facet circumradius.
    
    Strategy: find the worst facet, move its vertices toward the circumcenter
    with small steps, accept only if the max circumradius decreases.
    """
    start_time = time.time()
    current = points.copy()
    current_radius = _compute_covering_radius(current)
    
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        if no_improve_count > 20:
            break
        
        try:
            hull = ConvexHull(current)
        except Exception:
            break
        
        # Find worst facet
        max_cr = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = current[simplex[0]], current[simplex[1]], current[simplex[2]]
            cr = _facet_circumradius(a, b, c)
            if cr > max_cr:
                max_cr = cr
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Get circumcenter of worst facet
        a, b, c = current[worst_simplex[0]], current[worst_simplex[1]], current[worst_simplex[2]]
        cc = _spherical_triangle_circumcenter(a, b, c)
        if cc is None:
            break
        
        # Move vertices toward circumcenter
        trial = current.copy()
        for idx in worst_simplex:
            # Spherical interpolation toward cc
            dot_val = np.clip(np.dot(trial[idx], cc), -1.0, 1.1)
            angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
            if angle > 1e-12:
                sin_angle = np.sin(angle)
                if sin_angle > 1e-12:
                    coeff1 = np.cos(step_size * angle)
                    coeff2 = np.sin(step_size * angle) / sin_angle
                    new_pos = coeff1 * trial[idx] + coeff2 * cc
                    nn = np.linalg.norm(new_pos)
                    if nn > 1e-12:
                        trial[idx] = new_pos / nn
        
        # Normalize
        norms = np.linalg.norm(trial, axis=1, keepdims=True)
        trial = trial / norms
        
        if not _separation_check(trial, min_sep=1e-8):
            step_size *= 0.5
            if step_size < 1e-6:
                break
            continue
        
        new_radius = _compute_covering_radius(trial)
        if new_radius < current_radius - 1e-8:
            current = trial
            current_radius = new_radius
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return current

def _fibonacci_initialization(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    """Construct 50 points on the unit sphere using spherical Lloyd's algorithm."""
    # Fibonacci spiral initialization
    points = _fibonacci_initialization(N)
    
    start_time = time.time()
    max_total_time = 110.0
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Lloyd's relaxation with decreasing damping
    damping = 0.5
    iterations_per_phase = 50
    
    for phase in range(6):
        if time.time() - start_time > max_total_time - 15.0:
            break
        
        damping = max(0.1, 0.5 * (0.7 ** phase))
        
        for _ in range(iterations_per_phase):
            if time.time() - start_time > max_total_time - 15.0:
                break
            
            new_points = _lloyd_step_damped(points, damping=damping)
            
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            
            if not _separation_check(new_points, min_sep=1e-10):
                continue
            
            radius = _compute_covering_radius(new_points)
            
            if radius < best_radius:
                best_radius = radius
                best_points = new_points.copy()
            
            points = new_points
    
    # Phase 2: Local optimization on best configuration
    remaining_time = max_total_time - (time.time() - start_time)
    if remaining_time > 5.0:
        optimized = _local_optimize_covering(best_points, max_time=min(remaining_time - 2.0, 20.0))
        opt_radius = _compute_covering_radius(optimized)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    return best_points
