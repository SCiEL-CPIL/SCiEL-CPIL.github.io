import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_circumcenter(a, b, c):
    """
    Compute the circumcenter of a spherical triangle defined by unit vectors a, b, c.
    """
    v1 = b - a
    v2 = c - a
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    if norm < 1e-10:
        # Degenerate triangle, return centroid
        centroid = (a + b + c) / 3.0
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-10:
            return a
        return centroid / norm_c
    u = cross / norm
    centroid = (a + b + c) / 3.0
    if np.dot(u, centroid) < 0:
        u = -u
    return u

def get_facet_circumcenters(points, hull):
    """
    Compute circumcenter and radius for each facet of the hull.
    """
    simplices = hull.simplices
    num_facets = len(simplices)
    circumcenters = np.zeros((num_facets, 3))
    radii = np.zeros(num_facets)
    
    for i, simplex in enumerate(simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        cc = compute_circumcenter(a, b, c)
        circumcenters[i] = cc
        dot_val = np.dot(cc, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    return circumcenters, radii

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_radius = np.pi
    
    # Initial points: Fibonacci sphere
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    init_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    # Try multiple restarts
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            points = init_points.copy()
        else:
            np.random.seed(42 + restart)
            points = init_points + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Iterative optimization using Voronoi Facet Circumcenters
        max_iters = 50
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, add small noise and retry
                points += np.random.randn(N, 3) * 0.01
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
                
            circumcenters, radii = get_facet_circumcenters(points, hull)
            current_max_radius = np.max(radii)
            
            if current_max_radius < best_radius:
                best_radius = current_max_radius
                best_points = points.copy()
                
            # Check for convergence
            if current_max_radius < 0.1975: # ~11.3 degrees in radians
                break
                
            # Update points: Move each point towards the average of the circumcenters of its incident facets
            simplices = hull.simplices
            sum_vecs = np.zeros((N, 3))
            counts = np.zeros(N)
            
            for i_facet, simplex in enumerate(simplices):
                cc = circumcenters[i_facet]
                for p_idx in simplex:
                    sum_vecs[p_idx] += cc
                    counts[p_idx] += 1
            
            for i in range(N):
                if counts[i] > 0:
                    new_p = sum_vecs[i] / counts[i]
                    norm = np.linalg.norm(new_p)
                    if norm > 1e-10:
                        new_p = new_p / norm
                        # Blend to avoid overshooting
                        points[i] = 0.5 * points[i] + 0.5 * new_p
                        points[i] /= np.linalg.norm(points[i])
                    else:
                        points[i] = points[i] + np.random.randn(3) * 0.01
                        points[i] /= np.linalg.norm(points[i])
            
            # Normalize all points again for safety
            points /= np.linalg.norm(points, axis=1, keepdims=True)

    if best_points is None:
        best_points = init_points
        
    return best_points
