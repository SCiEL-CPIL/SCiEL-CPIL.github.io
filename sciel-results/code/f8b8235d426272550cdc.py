import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_metrics(points):
    """
    Computes the covering radius and identifies the worst facet.
    Returns: (radius, worst_facet_indices, circumcenter)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points), return large radius
        return np.pi, None, None
    
    max_radius = 0.0
    worst_facet = None
    worst_center = None
    
    # hull.simplices gives indices of vertices for each facet
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute circumcenter of the triangle on the sphere
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c_circ = n / norm_n
        
        # Ensure c_circ is the correct one (closer to the triangle vertices)
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c_circ, centroid) < 0:
            c_circ = -c_circ
            
        # Calculate radius (angle)
        cos_theta = np.clip(np.dot(c_circ, v1), -1.0, 1.0)
        radius = np.arccos(cos_theta)
        
        if radius > max_radius:
            max_radius = radius
            worst_facet = simplex
            worst_center = c_circ
            
    return max_radius, worst_facet, worst_center

def _lloyd_step(points, step_size=0.1):
    """
    Perform one Lloyd's algorithm step on the sphere.
    Moves each point towards the centroid of its Voronoi cell (spherical facet).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    new_points = points.copy()
    
    # For each point, find the facets it belongs to
    # Sum the circumcenters (or centroids) of those facets
    # A more accurate Lloyd step uses the spherical centroid of the Voronoi cell.
    # Approximation: Average the circumcenters of the facets surrounding the point.
    
    # Build a map: point index -> list of facet indices
    point_to_facets = {i: [] for i in range(N)}
    for i, simplex in enumerate(hull.simplices):
        for idx in simplex:
            point_to_facets[idx].append(i)
    
    for i in range(N):
        if not point_to_facets[i]:
            continue
            
        # Compute the spherical centroid of the Voronoi cell
        # Approximate by averaging the circumcenters of the adjacent facets
        # Weighted by the area? For now, simple average.
        
        # Get circumcenters for adjacent facets
        centers = []
        for facet_idx in point_to_facets[i]:
            simplex = hull.simplices[facet_idx]
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c_circ = n / norm_n
            
            # Ensure correct sign
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c_circ, centroid) < 0:
                c_circ = -c_circ
                
            centers.append(c_circ)
            
        if not centers:
            continue
            
        # Average the centers
        avg_center = np.mean(centers, axis=0)
        avg_norm = np.linalg.norm(avg_center)
        if avg_norm < 1e-10:
            continue
        avg_center /= avg_norm
        
        # Move point towards the centroid
        # p_new = normalize(p + step_size * (centroid - p))
        p_new = points[i] + step_size * (avg_center - points[i])
        p_norm = np.linalg.norm(p_new)
        if p_norm > 1e-10:
            new_points[i] = p_new / p_norm
            
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_metrics(best_points)
    
    # Optimization parameters
    step_size = 0.1
    decay = 0.999
    
    # Phase 1: Lloyd's algorithm
    lloyd_count = 0
    while time.time() < deadline - 5.0: # Leave 5s for final polish
        new_points = _lloyd_step(points, step_size)
        
        if np.allclose(points, new_points, atol=1e-8):
            # Converged
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_metrics(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            
        points = new_points
        lloyd_count += 1
        
        # Decay step size slowly
        step_size *= decay
        if step_size < 0.01:
            step_size = 0.01
            
    # Phase 2: Gradient descent on worst facet
    step_size = 0.01
    decay = 0.999
    
    while time.time() < deadline:
        current_radius, worst_facet, worst_center = _compute_covering_metrics(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if worst_facet is None:
            break
            
        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = points[v1_idx], points[v2_idx], points[v3_idx]
        
        # Move points towards the circumcenter
        alpha = step_size
        p1_new = v1 + alpha * (worst_center - v1)
        p2_new = v2 + alpha * (worst_center - v2)
        p3_new = v3 + alpha * (worst_center - v3)
        
        # Normalize
        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)
        
        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            points[v1_idx] = p1_new / n1
            points[v2_idx] = p2_new / n2
            points[v3_idx] = p3_new / n3
        
        # Decay step size to ensure convergence
        step_size *= decay
        
        # If step size is too small, break
        if step_size < 1e-6:
            break
            
    return best_points
