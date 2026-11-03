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
        # Fallback if hull fails (shouldn't happen for random points)
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
            # Degenerate triangle, radius is effectively pi/2 or undefined
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
    time_limit = 110.0  # Leave 10s buffer
    
    best_points = None
    best_radius = np.pi
    
    # Try a few random restarts
    for restart in range(4):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random points
        np.random.seed(42 + restart)
        points = np.random.randn(N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization loop
        alpha = 0.02
        no_improve_count = 0
        current_radius = np.pi
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            max_radius, radii, simplices = compute_covering_radius_and_facets(points)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > 100:
                # Perturb to escape local minimum
                points += np.random.randn(N, 3) * 0.01
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                no_improve_count = 0
                continue
            
            if len(simplices) == 0:
                break
                
            # Find the top 5 largest facets to shrink
            # This helps balance the configuration better than just the single largest
            top_k = 5
            if len(radii) < top_k:
                top_k = len(radii)
                
            # Get indices of the largest radii
            # We want the indices in the simplices array
            sorted_indices = np.argsort(radii)[::-1][:top_k]
            
            # Apply updates
            # To avoid race conditions if facets share vertices, we sum the updates
            updates = np.zeros_like(points)
            
            for idx in sorted_indices:
                simplex = simplices[idx]
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
                updates[i] += alpha * (u - a)
                updates[j] += alpha * (u - b)
                updates[k] += alpha * (u - c)
                
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # End of restart loop
        if time.time() - start_time > time_limit:
            break

    if best_points is None:
        # Fallback to Fibonacci if optimization failed completely
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
