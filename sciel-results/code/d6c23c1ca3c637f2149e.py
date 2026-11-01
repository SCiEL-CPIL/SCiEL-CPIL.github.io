"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def _compute_hull_radius_and_worst(points):
    """
    Compute the covering radius and identify the worst facet.
    Returns: (radius_in_degrees, worst_facet_indices, worst_facet_circumcenter)
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in facets:
            verts = points[simplex]
            if len(verts) == 3:
                a, b, c = verts
                # Circumcenter of spherical triangle
                # x . a = x . b = x . c = 1/R
                # x is parallel to (a-b) x (a-c) ? No, that's the normal to the plane.
                # Let's solve the linear system for the circumcenter direction.
                # c . a = c . b => c . (a-b) = 0
                # c . a = c . c => c . (a-c) = 0
                # So c is orthogonal to a-b and a-c.
                # c is parallel to (a-b) x (a-c).
                
                n = np.cross(a - b, a - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                c_dir = n / n_norm
                
                # Check if c_dir is the correct direction (should be close to centroid)
                centroid = (a + b + c) / 3.0
                if np.dot(c_dir, centroid) < 0:
                    c_dir = -c_dir
                
                # Compute distance
                d = np.arccos(np.clip(np.dot(c_dir, a), -1.0, 1.0))
                
                if d > max_r:
                    max_r = d
                    worst_facet = simplex
                    worst_center = c_dir
                    
        return max_r * 180.0 / np.pi, worst_facet, worst_center
    except:
        return np.inf, None, None

def _geometric_local_search(points, max_iter=100, tol=1e-6):
    """
    Perform local search by moving points to reduce the circumradius of the worst facet.
    For the worst facet (a, b, c), we want to move a, b, c such that the circumradius decreases.
    The circumcenter is equidistant from a, b, c.
    If we move a, b, c slightly towards the circumcenter (on the sphere), the circumradius decreases?
    Actually, moving vertices towards the circumcenter on the sphere decreases the side lengths, 
    and thus the circumradius.
    
    However, we must maintain the convex hull structure.
    A robust way: For each vertex in the worst facet, move it slightly towards the circumcenter.
    """
    points = _normalize(points)
    best_radius, worst_facet, worst_center = _compute_hull_radius_and_worst(points)
    
    if worst_facet is None:
        return points, best_radius

    for _ in range(max_iter):
        radius, worst_facet, worst_center = _compute_hull_radius_and_worst(points)
        if worst_facet is None:
            break
            
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        # Get vertices of worst facet
        verts_idx = worst_facet
        verts = points[verts_idx]
        
        # Move each vertex towards the circumcenter
        # Step size: small, e.g., 0.01 radians
        step_size = 0.05 
        
        new_points = points.copy()
        for idx in verts_idx:
            v = points[idx]
            # Move v towards worst_center on the sphere
            # Use geodesic step
            cos_d = np.clip(np.dot(v, worst_center), -1.0, 1.0)
            d = np.arccos(cos_d)
            if d < 1e-6:
                continue
            # Step along the great circle towards worst_center
            # v_new = (sin(d-step)/sin(d)) * v + (sin(step)/sin(d)) * worst_center
            # But we want to move a fraction of the way.
            # Let's just move a fixed angular distance.
            # If d is small, we can approximate.
            # Simpler: v_new = normalize(v + alpha * (worst_center - v))
            # This is not a great circle step, but it works for small steps.
            
            # Better: Spherical interpolation (SLERP)
            # v_new = slerp(v, worst_center, t) where t is step_size / d
            t = min(1.0, step_size / d)
            if t > 0:
                sin_d = np.sin(d)
                if sin_d < 1e-6:
                    continue
                a1 = np.sin((1 - t) * d) / sin_d
                a2 = np.sin(t * d) / sin_d
                new_v = a1 * v + a2 * worst_center
                new_points[idx] = new_v / np.linalg.norm(new_v)
        
        # Normalize all to be safe
        new_points = _normalize(new_points)
        
        # Check if the new configuration is valid and improved
        new_radius, _, _ = _compute_hull_radius_and_worst(new_points)
        
        if new_radius < radius - tol:
            points = new_points
        else:
            # If no improvement, try a smaller step or break
            step_size *= 0.5
            if step_size < 1e-4:
                break
                
    return points, best_radius

def _random_perturbation(points, sigma=0.1):
    """
    Apply a random perturbation to the points.
    """
    random_dirs = np.random.randn(N, 3)
    random_dirs = _normalize(random_dirs)
    perturbed = points + sigma * random_dirs
    return _normalize(perturbed)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145 seconds to be safe (limit is 150)
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius, _, _ = _compute_hull_radius_and_worst(points)
    
    # Multi-start basin hopping with geometric local search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random perturbation
        current_points = _random_perturbation(points, sigma=0.3)
        
        # Geometric local search
        current_points, current_radius = _geometric_local_search(current_points, max_iter=50)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
            points = current_points.copy()
        else:
            # Accept with some probability if it's close
            if current_radius < best_radius * 1.05:
                points = current_points.copy()
                
    return best_points
# EVOLVE-BLOCK-END
