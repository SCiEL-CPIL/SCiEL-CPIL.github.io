"""Spherical Lloyd's algorithm with adaptive step size and centroid-based relaxation."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_triangle_area(a, b, c):
    """Compute spherical area of triangle using vector formula."""
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    
    if abs(denom) < 1e-14 and abs(triple) < 1e-14:
        return 0.0
    
    return 2.0 * np.arctan2(abs(triple), denom)

def _spherical_triangle_centroid(a, b, c):
    """Compute approximate spherical centroid of triangle (a,b,c).
    
    Uses the normalized sum of vertices. This is accurate for small triangles 
    and robust for larger ones in the context of Lloyd's relaxation.
    """
    s = a + b + c
    n = np.linalg.norm(s)
    if n < 1e-12:
        # Degenerate case, fallback to mean
        return (a + b + c) / 3.0
    return s / n

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Circumradius is the angular distance from circumcenter to vertices
        # Circumcenter p satisfies p.a = p.b = p.c
        u = a - b
        v = a - c
        p_dir = np.cross(u, v)
        p_norm = np.linalg.norm(p_dir)
        
        if p_norm < 1e-12:
            continue
            
        p = p_dir / p_norm
        
        # Ensure p is on the correct side (outward)
        normal = np.cross(b - a, c - a)
        nn = np.linalg.norm(normal)
        if nn > 1e-12:
            normal = normal / nn
            centroid = (a + b + c) / 3.0
            if np.dot(p, centroid) < 0:
                p = -p
        
        # Angular distance from p to a
        dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
        cr = np.arccos(dot_val)
        
        if cr > max_cr:
            max_cr = cr
    
    return np.degrees(max_cr)

def _slerp(p, q, t):
    """Spherical linear interpolation between p and q by fraction t."""
    dot_val = np.clip(np.dot(p, q), -1.0, 1.0)
    angle = np.arccos(dot_val)
    
    if angle < 1e-12:
        return p
        
    sin_angle = np.sin(angle)
    if sin_angle < 1e-12:
        return p
        
    coeff1 = np.cos(t * angle)
    coeff2 = np.sin(t * angle) / sin_angle
    
    new_p = coeff1 * p + coeff2 * q
    n = np.linalg.norm(new_p)
    if n < 1e-12:
        return p
    return new_p / n

def _lloyd_step(points, step_size):
    """Perform one Lloyd's step.
    
    Moves each point to the centroid of its Voronoi cell, 
    interpolated by step_size.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    n_points = len(points)
    targets = np.zeros_like(points)
    
    # Initialize targets with current points
    for i in range(n_points):
        targets[i] = points[i]
    
    # Map each point to its incident facets
    point_facets = [[] for _ in range(n_points)]
    for simplex in hull.simplices:
        for v in simplex:
            point_facets[v].append(simplex)
    
    for i in range(n_points):
        facets = point_facets[i]
        if not facets:
            continue
        
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for simplex in facets:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            area = _spherical_triangle_area(a, b, c)
            
            if area > 1e-12:
                centroid = _spherical_triangle_centroid(a, b, c)
                weighted_sum += area * centroid
                total_weight += area
        
        if total_weight > 1e-12:
            target = weighted_sum / total_weight
            tn = np.linalg.norm(target)
            if tn > 1e-12:
                targets[i] = target / tn
    
    # Apply slerp update
    new_points = np.zeros_like(points)
    for i in range(n_points):
        new_points[i] = _slerp(points[i], targets[i], step_size)
        
    return new_points

def _separation_check(points, min_sep=1e-8):
    """Check minimum pairwise separation."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_sep:
                return False
    return True

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
    points = _fibonacci_initialization(N)
    
    start_time = time.time()
    max_total_time = 115.0
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Adaptive Lloyd's
    step_size = 0.5
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_total_time:
        new_points = _lloyd_step(points, step_size)
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        if not _separation_check(new_points, min_sep=1e-9):
            step_size *= 0.5
            if step_size < 1e-6:
                break
            continue
            
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = new_points.copy()
            no_improve = 0
            # Optional: slightly increase step size if we are far from optimum? 
            # Keep it simple: maintain or decrease.
        else:
            no_improve += 1
            step_size *= 0.8
            
            if no_improve > max_no_improve:
                break
                
    # Final Polish: Greedy local search on worst facets
    # This helps fine-tune the covering radius
    current = best_points.copy()
    current_radius = best_radius
    step = 0.02
    polish_time_limit = time.time() + 5.0
    
    while time.time() < polish_time_limit:
        try:
            hull = ConvexHull(current)
        except Exception:
            break
            
        max_cr = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = current[simplex[0]], current[simplex[1]], current[simplex[2]]
            u = a - b
            v = a - c
            p_dir = np.cross(u, v)
            p_norm = np.linalg.norm(p_dir)
            if p_norm < 1e-12:
                continue
            p = p_dir / p_norm
            normal = np.cross(b - a, c - a)
            nn = np.linalg.norm(normal)
            if nn > 1e-12:
                normal = normal / nn
                centroid = (a + b + c) / 3.0
                if np.dot(p, centroid) < 0:
                    p = -p
            dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
            cr = np.arccos(dot_val)
            if cr > max_cr:
                max_cr = cr
                worst_simplex = simplex
                
        if worst_simplex is None:
            break
            
        a, b, c = current[worst_simplex[0]], current[worst_simplex[1]], current[worst_simplex[2]]
        u = a - b
        v = a - c
        p_dir = np.cross(u, v)
        p_norm = np.linalg.norm(p_dir)
        if p_norm < 1e-12:
            break
        cc = p_dir / p_norm
        normal = np.cross(b - a, c - a)
        nn = np.linalg.norm(normal)
        if nn > 1e-12:
            normal = normal / nn
            centroid = (a + b + c) / 3.0
            if np.dot(cc, centroid) < 0:
                cc = -cc
                
        trial = current.copy()
        for idx in worst_simplex:
            trial[idx] = _slerp(trial[idx], cc, step)
            
        norms = np.linalg.norm(trial, axis=1, keepdims=True)
        trial = trial / norms
        
        if not _separation_check(trial, min_sep=1e-9):
            step *= 0.5
            if step < 1e-6:
                break
            continue
            
        trial_radius = _compute_covering_radius(trial)
        if trial_radius < current_radius - 1e-8:
            current = trial
            current_radius = trial_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current.copy()
        else:
            step *= 0.8
            if step < 1e-6:
                break
                
    return best_points
