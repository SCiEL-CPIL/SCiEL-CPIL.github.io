import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return v / norms[:, np.newaxis]

def _covering_radius(points):
    """
    Compute the covering radius (max circumradius of convex hull facets) in radians.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails (e.g., coplanar points), return a large value
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter is the normalized cross product of (v2-v1) x (v3-v1) ? 
        # No, the circumcenter of a spherical triangle is the intersection of the sphere 
        # and the plane passing through v1, v2, v3.
        # The normal to the plane is n = (v2-v1) x (v3-v1)
        # The circumcenter c is parallel to n.
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue # Degenerate triangle
            
        c = n / norm_n
        
        # The angular distance from c to v1 is the circumradius
        # cos(angle) = c . v1
        cos_angle = np.dot(c, v1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if angle > max_radius:
            max_radius = angle
            
    return max_radius

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local search optimization
    # We will iteratively move the vertices of the worst facet towards its circumcenter.
    
    # Parameters
    alpha = 0.05 # Step size
    max_iter = 20000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute convex hull and find worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_radius = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            c = n / norm_n
            cos_angle = np.dot(c, v1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle > max_radius:
                max_radius = angle
                worst_simplex = simplex
                worst_center = c
                
        if worst_simplex is None:
            break
            
        # Check convergence
        if max_radius < 11.3165625 * np.pi / 180.0:
            # We have reached the best known value, can stop or polish
            break
            
        # Move vertices of the worst simplex towards the center
        # This reduces the circumradius of this facet
        for idx in worst_simplex:
            points[idx] = (1.0 - alpha) * points[idx] + alpha * worst_center
            
        # Normalize all points back to the sphere
        points = _normalize(points)
        
        # To help escape local minima, we can occasionally add small noise
        # or move other points slightly. 
        # For now, let's stick to the direct improvement.
        
    # Final normalization
    points = _normalize(points)
    
    return points

# EVOLVE-BLOCK-END
