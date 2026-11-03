import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    Returns:
        max_radius: float
        radii: (M,) array of circumradii for each facet
        simplices: (M, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback if hull fails
        return np.pi, np.array([]), np.array([])
    
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[idx] = np.pi / 2
            continue
            
        u = n / norm_n
        
        # Ensure u points towards the centroid (inside the spherical triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # Circumradius is the angle between u and any vertex
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    max_radius = np.max(radii) if len(radii) > 0 else np.pi
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    best_points = None
    best_radius = np.pi
    
    # Use a Fibonacci sphere as initial guess for better start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try a few restarts with different noise levels
    for restart in range(6):
        if time.time() - start_time > time_limit:
            break
            
        # For restart 0, use pure Fibonacci. For others, add noise
        if restart == 0:
            points = init_points.copy()
        else:
            np.random.seed(42 + restart)
            points = init_points + np.random.randn(N, 3) * 0.5
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization loop
        alpha = 0.05
        no_improve_count = 0
        current_radius = np.pi
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            max_radius, radii, simplices = compute_covering_radius_and_facets(points)
            
            if len(simplices) == 0:
                break
                
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
                # Slightly increase alpha if improving significantly?
                # Or keep constant.
            else:
                no_improve_count += 1
                # If no improvement, reduce alpha
                if no_improve_count > 20:
                    alpha *= 0.9
                    if alpha < 1e-4:
                        break
                
            # Find the largest facet
            max_idx = np.argmax(radii)
            simplex = simplices[max_idx]
            i, j, k = simplex
            a = points[i]
            b = points[j]
            c = points[k]
            
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            u = n / norm_n
            centroid = (a + b + c) / 3.0
            if np.dot(u, centroid) < 0:
                u = -u
            
            # Move vertices towards u
            # Use a step size relative to the distance to u
            # dist is roughly radii[max_idx]
            step = alpha * radii[max_idx]
            
            points[i] += step * (u - a)
            points[j] += step * (u - b)
            points[k] += step * (u - c)
            
            # Normalize
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # End of restart loop
        if time.time() - start_time > time_limit:
            break

    if best_points is None:
        # Fallback to Fibonacci if optimization failed completely
        best_points = init_points
        
    return best_points
