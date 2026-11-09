"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _covering_radius(points):
    """
    Compute the exact covering radius (in radians) using the convex hull.
    The covering radius is the maximum circumradius of the spherical facets.
    Each facet of the convex hull corresponds to a Voronoi cell on the sphere.
    The circumradius of the spherical triangle (or polygon) formed by the facet vertices
    is the distance from the center of that empty cap to the nearest point.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # simplex contains indices of points forming a triangular face
            v1, v2, v3 = points[simplex]
            
            # The normal to the plane of the triangle
            normal = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            # Ensure normal points away from origin (outward)
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            
            # The center of the spherical cap is `normal` (which is a unit vector).
            # The radius is the angle between `normal` and any of the vertices, e.g., v1.
            
            dot = np.dot(normal, v1)
            # Clip for numerical stability
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            if angle > max_r:
                max_r = angle
                
        return max_r
    except Exception:
        return np.pi # Fallback if hull fails (e.g., coplanar points)

def _worst_facet(points):
    """
    Find the facet with the largest circumradius.
    Returns (radius, simplex_indices, circumcenter) or (0, None, None) if none found.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            
            normal = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            
            dot = np.dot(normal, v1)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            if angle > max_r:
                max_r = angle
                worst_simplex = simplex
                worst_center = normal
                
        return max_r, worst_simplex, worst_center
    except Exception:
        return 0.0, None, None

def _repulsive_force_step(points, p=20, eta=0.01):
    """
    Perform one step of repulsive force optimization.
    Maximizes the minimum distance between points.
    """
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    dists[mask] = np.maximum(dists[mask], 1e-10)
    
    # Weights for repulsion
    weights = p * (dists ** (-p - 2))
    np.fill_diagonal(weights, 0.0)
    
    # Forces
    sum_weights = np.sum(weights, axis=1)
    weighted_points = np.sum(weights[:, :, np.newaxis] * points[np.newaxis, :, :], axis=1)
    forces = points * sum_weights[:, np.newaxis] - weighted_points
    
    # Update
    points = points + eta * forces
    
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def _lloyd_step(points, alpha=0.1):
    """
    Perform one step of Lloyd's algorithm on the sphere.
    Moves each point to the centroid of its Voronoi cell (approximated by the average of adjacent vertices).
    Actually, for covering radius minimization, we want to move points away from the centers of the largest caps.
    A simpler, more robust approach: move points towards the center of their Voronoi cell?
    No, Lloyd's algorithm for k-means moves points to the centroid of the Voronoi cell.
    For sphere covering, the "centroid" is the point that minimizes the maximum distance to the vertices of the cell?
    Actually, the standard Lloyd iteration for sphere covering is:
    1. Compute Voronoi cells.
    2. For each cell, find the point that minimizes the maximum distance to the cell vertices (the circumcenter of the cell's facets? No, the center of the largest empty cap in the cell).
    3. Move the point there.
    
    However, computing the exact "center" that minimizes the max distance in a spherical polygon is non-trivial.
    A common approximation is to move the point towards the circumcenter of the facet with the largest radius?
    Or, simpler: use the gradient of the covering radius.
    
    Let's use a hybrid:
    Identify the worst facet. Move its vertices towards the circumcenter of that facet.
    This is similar to inspiration 1 but applied to all facets or just the worst ones.
    
    Let's implement a "worst-facet correction" step.
    """
    # This function is a placeholder for now, we will implement the specific logic in construct
    return points

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Initial covering radius
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Optimization parameters
    p = 20
    eta = 0.05
    min_eta = 1e-5
    decay = 0.999
    
    # Precompute mask for non-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    
    step = 0
    # We will alternate between global repulsion (to spread points) and local correction (to fix worst caps)
    
    while time.time() - start_time < time_limit:
        # 1. Global Repulsion Step
        # This helps escape local minima where points are clustered
        points = _repulsive_force_step(points, p, eta)
        
        # 2. Local Correction Step (Inspiration 1 strategy)
        # Find the worst facet and move its vertices towards its circumcenter
        cr, worst_simplex, worst_center = _worst_facet(points)
        
        if worst_simplex is not None and worst_center is not None:
            v1_idx, v2_idx, v3_idx = worst_simplex
            v1, v2, v3 = points[v1_idx], points[v2_idx], points[v3_idx]
            
            # Move points towards the circumcenter
            # We want to reduce the circumradius.
            # Moving towards the center reduces the distance to the center.
            # But we must stay on the sphere.
            # p_new = normalize(p + alpha * (c - p))
            
            # Use a smaller alpha for local correction to avoid oscillation
            alpha = 0.01 * (eta / 0.05) # Scale with global eta
            
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
        
        # Evaluate covering radius periodically to save time
        # Every 10 steps or if eta is very small
        if step % 10 == 0 or eta < min_eta * 10:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        
        # Decay step size
        eta *= decay
        
        step += 1
        
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
    return best_points

# EVOLVE-BLOCK-END
